"""
Adopted from Andreas F Clenow's "Trend Following" method(s). 

Further readings:

https://www.followingthetrend.com/stocks-on-the-move/ 
https://www.followingthetrend.com/trading-evolved/

"""


import asyncio
import sys
import time
import uuid
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
from liualgotrader.analytics import analysis
from liualgotrader.common import config
from liualgotrader.common.data_loader import DataLoader
from liualgotrader.common.market_data import (m_and_a_data,
                                              sp500_historical_constituents)
from liualgotrader.common.tlog import tlog
from liualgotrader.common.trading_data import (buy_indicators, buy_time,
                                               cool_down, last_used_strategy,
                                               latest_cost_basis,
                                               latest_scalp_basis, open_orders,
                                               sell_indicators, stop_prices,
                                               target_prices)
from liualgotrader.common.types import AssetType
from liualgotrader.strategies.base import Strategy, StrategyType
from pandas import DataFrame as df
from pytz import timezone

sys.path.append("..")
from liualgotrader.models.accounts import Accounts
from liualgotrader.models.portfolio import Portfolio

from trades.common.trend import Trend as TrendLogic

nyc = timezone("America/New_York")


class TrendFollow(Strategy):
    name = "trend_follow"

    def __init__(
        self,
        batch_id: str,
        data_loader: DataLoader,
        rebalance_rate: str,
        stock_count: int,
        index: str,
        rank_days: int,
        debug: bool,
        portfolio_id: str,
        ref_run_id: str = None,
        reinvest: bool = False,
        volatility_threshold: float = 0.03,
    ):
        self.portfolio_id = portfolio_id
        self.last_rebalance: str
        self.trend_logic: Optional[TrendLogic] = None
        self.index = index
        self.stock_count = stock_count
        self.rank_days = rank_days
        self.debug = debug
        self.reinvest = reinvest
        self.volatility_threshold = volatility_threshold
        self.asset_type: AssetType
        self.key: str
        if rebalance_rate in {"daily", "hourly", "weekly"}:
            self.rebalance_rate = rebalance_rate
        else:
            raise AssertionError(
                f"rebalance schedule can be either daily/hourly not {rebalance_rate}"
            )

        super().__init__(
            name=self.name,
            type=StrategyType.SWING,
            batch_id=batch_id,
            ref_run_id=ref_run_id,
            data_loader=data_loader,
            schedule=[],
        )

    async def buy_callback(
        self, symbol: str, price: float, qty: float, now: datetime = None
    ) -> None:
        if self.account_id:
            await Accounts.add_transaction(
                account_id=self.account_id, amount=-price * qty, tstamp=now
            )
            print(
                "buy",
                -price * qty,
                "balance post buy",
                await Accounts.get_balance(self.account_id),
            )

    async def sell_callback(
        self, symbol: str, price: float, qty: float, now: datetime = None
    ) -> None:
        if self.account_id:
            await Accounts.add_transaction(
                account_id=self.account_id, amount=price * qty, tstamp=now
            )
            print(
                "sell",
                price * qty,
                "balance post sell",
                await Accounts.get_balance(self.account_id),
            )

    async def create(self) -> bool:
        if not await super().create():
            return False

        if await Portfolio.is_associated(
            portfolio_id=self.portfolio_id, batch_id=self.batch_id
        ):
            return False

        try:
            await Portfolio.associate_batch_id_to_profile(
                portfolio_id=self.portfolio_id, batch_id=self.batch_id
            )
        except Exception:
            return False
        portfolio = await Portfolio.load_by_portfolio_id(self.portfolio_id)
        self.account_id = portfolio.account_id
        self.portfolio_size = portfolio.portfolio_size
        self.asset_type = portfolio.asset_type

        tlog(
            f"strategy {self.name} created for portfolio_id {self.portfolio_id}"
        )
        return True

    async def check_adjustment(self, now: date, symbols: List) -> bool:
        return bool(
            str(now) in m_and_a_data.index
            and m_and_a_data.loc[str(now), "from_symbol"]
        )

    async def apply_adjustment(
        self, now: date, symbols_position: Dict[str, float]
    ) -> float:
        if not await self.check_adjustment(now, list(symbols_position.keys())):
            return 0.0

        symbol = m_and_a_data.loc[str(now), "from_symbol"]
        position = symbols_position[symbol]

        if position > 0:
            new_symbol = m_and_a_data.loc[str(now), "to_symbol"]
            conversation_rate = m_and_a_data.loc[str(now), "convert_price"]
            cash_rate = m_and_a_data.loc[str(now), "cash_per_share"]

            cash = position * cash_rate
            symbols_position[new_symbol] = position * conversation_rate

            symbols_position.pop(symbol)
            print(
                symbol,
                position,
                new_symbol,
                cash_rate,
                cash,
                symbols_position[new_symbol],
            )

            return cash

        return 0.0

    async def rebalance(
        self,
        data_loader: DataLoader,
        symbols_position: Dict[str, float],
        now: datetime,
        carrier=None,
    ) -> Dict[str, Dict]:
        await self.set_global_var(self.key, str(now))
        cash = await Accounts.get_balance(self.account_id)
        cash += await self.apply_adjustment(now.date(), symbols_position)
        invested_amount = sum(
            symbols_position[symbol] * data_loader[symbol].close[now]
            for symbol in symbols_position
        )
        tlog(
            f"starting rebalance for {now} w ${cash} cash + ${invested_amount} equity"
        )
        self.trend_logic = TrendLogic(
            symbols=(await sp500_historical_constituents(str(now.date()))),
            portfolio_size=cash + invested_amount,
            stock_count=self.stock_count,
            rank_days=self.rank_days,
            debug=self.debug,
            volatility_threshold=self.volatility_threshold,
            data_loader=data_loader,
        )
        new_profile = await self.trend_logic.run(now)
        tlog(f"{new_profile}")
        sell_positions = {
            symbol: symbols_position[symbol]
            for symbol in symbols_position
            if symbol not in new_profile.symbol.tolist()
        }
        sell_positions.update(
            {
                symbol: (
                    symbols_position[symbol]
                    - new_profile[new_profile.symbol == symbol].qty.values[0]
                )
                for symbol in symbols_position
                if (
                    symbol in new_profile.symbol.tolist()
                    and new_profile[new_profile.symbol == symbol].qty.values[0]
                    < symbols_position[symbol]
                )
            }
        )
        sell_amount = sum(
            sell_positions[symbol] * data_loader[symbol].close[now]
            for symbol in sell_positions
        )

        tlog(f"cash:{cash} sell_amount:{sell_amount}")
        buy_positions = {
            symbol: new_profile[new_profile.symbol == symbol].qty.values[0]
            for symbol in new_profile.symbol.tolist()
            if symbol not in symbols_position and symbol not in sell_positions
        }
        buy_positions.update(
            {
                symbol: (
                    new_profile[new_profile.symbol == symbol].qty.values[0]
                    - symbols_position[symbol]
                )
                for symbol in symbols_position
                if symbol in new_profile.symbol.tolist()
                and new_profile[new_profile.symbol == symbol].qty.values[0]
                > symbols_position[symbol]
            }
        )

        tlog(f"sell: {sell_positions} buy: {buy_positions}")

        actions = {
            symbol: {
                "side": "sell",
                "qty": sell_positions[symbol],
                "type": "market",
            }
            for symbol in sell_positions
            if sell_positions[symbol] > 0
        }
        actions.update(
            {
                symbol: {
                    "side": "buy",
                    "qty": buy_positions[symbol],
                    "type": "market",
                }
                for symbol in buy_positions
                if buy_positions[symbol] > 0
            }
        )
        tlog("rebalance completed")
        return actions

    async def should_rebalance(self, now: datetime) -> bool:
        last_rebalance = await self.get_global_var(self.key)
        if not last_rebalance:  # ignore: type
            return True

        if self.rebalance_rate == "hourly" and now - datetime.fromisoformat(
            last_rebalance
        ) >= timedelta(hours=1):
            return True
        elif self.rebalance_rate == "daily" and now - datetime.fromisoformat(
            last_rebalance
        ) >= timedelta(days=1):
            return True
        elif self.rebalance_rate == "weekly" and now - datetime.fromisoformat(
            last_rebalance
        ) >= timedelta(days=7):
            return True

        return False

    async def load_symbol_position(self) -> Dict[str, float]:
        trades = analysis.load_trades_by_portfolio(self.portfolio_id)

        if not len(trades):
            return {}
        new_df = pd.DataFrame()
        new_df["symbol"] = trades.symbol.unique()
        new_df["qty"] = new_df.symbol.apply(
            lambda x: (
                trades[
                    (trades.symbol == x) & (trades.operation == "buy")
                ].qty.sum()
            )
            - trades[
                (trades.symbol == x) & (trades.operation == "sell")
            ].qty.sum()
        )
        new_df = new_df.loc[new_df.qty != 0]

        rc_dict: Dict[str, float] = {}
        for _, row in new_df.iterrows():
            rc_dict[row.symbol] = float(row.qty)

        return rc_dict

    async def run_all(
        self,
        symbols_position: Dict[str, float],
        data_loader: DataLoader,
        now: datetime,
        portfolio_value: float = None,
        trading_api: tradeapi = None,
        debug: bool = False,
        backtesting: bool = False,
    ) -> Dict[str, Dict]:
        self.key = f"{self.portfolio_id}-{self.name}-last-rebalance"
        if await self.should_rebalance(now):
            tlog("time for rebalance")
            portfolio_symbols_position = await self.load_symbol_position()
            tlog(f"current positions {portfolio_symbols_position}")
            return await self.rebalance(
                data_loader, portfolio_symbols_position, now
            )

        else:
            tlog(f"skip rebalance {now}")

        return {}

    async def should_run_all(self):
        return True
