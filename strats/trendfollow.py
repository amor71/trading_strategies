import asyncio
import sys
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import alpaca_trade_api as tradeapi
import numpy as np
from liualgotrader.common import config
from liualgotrader.common.data_loader import DataLoader
from liualgotrader.common.market_data import index_data
from liualgotrader.common.tlog import tlog
from liualgotrader.common.trading_data import (buy_indicators, buy_time,
                                               cool_down, last_used_strategy,
                                               latest_cost_basis,
                                               latest_scalp_basis, open_orders,
                                               sell_indicators, stop_prices,
                                               target_prices)
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
        portfolio_size: int,
        rebalance_rate: str,
        stock_count: int,
        index: str,
        rank_days: int,
        debug: bool,
        ref_run_id: str = None,
        account_id: int = None,
        reinvest: bool = False,
        portfolio_id: str = None,
    ):
        self.context: str
        self.portfolio_id = portfolio_id
        self.portfolio_size = portfolio_size
        self.last_rebalance: str
        self.trend_logic: Optional[TrendLogic] = None
        self.index = index
        self.stock_count = stock_count
        self.rank_days = rank_days
        self.debug = debug
        self.account_id = account_id
        self.reinvest = reinvest
        if rebalance_rate not in ["daily", "hourly", "weekly"]:
            raise AssertionError(
                f"rebalance schedule can be either daily/hourly not {rebalance_rate}"
            )
        else:
            self.rebalance_rate = rebalance_rate

        super().__init__(
            name=self.name,
            type=StrategyType.SWING,
            batch_id=batch_id,
            ref_run_id=ref_run_id,
            data_loader=data_loader,
            schedule=[],
        )

    async def buy_callback(self, symbol: str, price: float, qty: int) -> None:
        if self.account_id:
            await Accounts.add_transaction(
                account_id=self.account_id, amount=-price * qty
            )

    async def sell_callback(self, symbol: str, price: float, qty: int) -> None:
        if self.account_id:
            await Accounts.add_transaction(
                account_id=self.account_id, amount=price * qty
            )

    async def create(self) -> bool:
        await super().create()

        tlog(f"strategy {self.name} created")
        if not self.portfolio_id or not len(self.portfolio_id):
            self.portfolio_id = str(uuid.uuid4())
        if not await Portfolio.exists(self.portfolio_id):
            await Portfolio.save(
                portfolio_id=self.portfolio_id,
                portfolio_size=self.portfolio_size,
                stock_count=self.stock_count,
                parameters={"rank_days": self.rank_days, "index": self.index},
            )
            print(f"create new Portfolio w/ id {self.portfolio_id}")
        try:
            await Portfolio.associate_batch_id_to_profile(
                portfolio_id=self.portfolio_id, batch_id=self.batch_id
            )
        except Exception:
            print("Probably already associated...")
            return False

        return True

    async def rebalance(
        self,
        data_loader: DataLoader,
        symbols_position: Dict[str, int],
        now: datetime,
    ) -> Dict[str, Dict]:
        await self.set_global_var("last_rebalance", str(now), self.context)
        cash = await Accounts.get_balance(self.account_id)
        invested_amount = sum(
            symbols_position[symbol] * data_loader[symbol].close[now]
            for symbol in symbols_position
        )
        tlog(
            f"starting rebalance for {now} w ${cash} cash + ${invested_amount} equity"
        )
        self.trend_logic = TrendLogic(
            symbols=(await index_data(self.index)).Symbol.tolist(),
            portfolio_size=cash + invested_amount,
            stock_count=self.stock_count,
            rank_days=self.rank_days,
            debug=self.debug,
        )
        symbols_position = {
            symbol: symbols_position[symbol]
            for symbol in symbols_position
            if symbols_position[symbol]
        }
        new_profile = await self.trend_logic.run(now)

        sell_symbols = [
            symbol
            for symbol in symbols_position
            if symbol not in new_profile.symbol.tolist()
        ]
        keep_symbols = [
            symbol
            for symbol in symbols_position
            if symbol in new_profile.symbol.tolist()
        ]
        sell_amount = sum(
            symbols_position[symbol] * data_loader[symbol].close[now]
            for symbol in sell_symbols
        )

        print(f"cash:{cash} sell_amount:{sell_amount}")
        money_left = cash + sell_amount

        buy_symbols = new_profile[
            ~new_profile.symbol.isin(sell_symbols + keep_symbols)
        ].sort_values(by="score", ascending=False)
        buy_symbols["accumulative"] = buy_symbols.est.cumsum()
        buy_symbols = buy_symbols[buy_symbols.accumulative <= money_left]

        tlog(f"sell: {sell_symbols} buy: {buy_symbols}")

        actions = {}
        actions.update(
            {
                symbol: {
                    "side": "sell",
                    "qty": symbols_position[symbol],
                    "type": "market",
                }
                for symbol in sell_symbols
            }
        )
        actions.update(
            {
                symbol: {
                    "side": "buy",
                    "qty": int(
                        buy_symbols.loc[buy_symbols.symbol == symbol, "qty"]
                    ),
                    "type": "market",
                }
                for symbol in buy_symbols.symbol.unique().tolist()
            }
        )
        tlog("rebalance completed")
        return actions

    async def should_rebalance(self, now: datetime) -> bool:
        last_rebalance = await self.get_global_var(
            "last_rebalance", self.context
        )
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

    async def run_all(
        self,
        symbols_position: Dict[str, int],
        data_loader: DataLoader,
        now: datetime,
        portfolio_value: float = None,
        trading_api: tradeapi = None,
        debug: bool = False,
        backtesting: bool = False,
    ) -> Dict[str, Dict]:
        self.context = self.batch_id if backtesting else self.name
        if await self.should_rebalance(now):
            return await self.rebalance(data_loader, symbols_position, now)
        return {}

    async def should_run_all(self):
        return True
