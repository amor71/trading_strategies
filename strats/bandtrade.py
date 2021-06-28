import asyncio
import sys
import uuid
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
from liualgotrader.common import config
from liualgotrader.common.data_loader import DataLoader
from liualgotrader.common.tlog import tlog
from liualgotrader.common.trading_data import (buy_indicators, buy_time,
                                               cool_down, last_used_strategy,
                                               latest_cost_basis,
                                               latest_scalp_basis, open_orders,
                                               sell_indicators, stop_prices,
                                               target_prices)
from liualgotrader.fincalcs.support_resistance import find_stop
from liualgotrader.models.accounts import Accounts
from liualgotrader.models.portfolio import Portfolio
from liualgotrader.strategies.base import Strategy, StrategyType
from pandas import DataFrame as df
from pytz import timezone
from talib import BBANDS, MACD, RSI, MA_Type


class BandTrade(Strategy):
    def __init__(
        self,
        batch_id: str,
        data_loader: DataLoader,
        portfolio_id: str,
        ref_run_id: int = None,
    ):
        self.name = type(self).__name__
        self.portfolio_id = portfolio_id
        super().__init__(
            name=type(self).__name__,
            type=StrategyType.SWING,
            batch_id=batch_id,
            ref_run_id=ref_run_id,
            schedule=[],
            data_loader=data_loader,
        )

    async def buy_callback(
        self, symbol: str, price: float, qty: int, now: datetime = None
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
        self, symbol: str, price: float, qty: int, now: datetime = None
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

        tlog(f"strategy {self.name} created")
        try:
            await Portfolio.associate_batch_id_to_profile(
                portfolio_id=self.portfolio_id, batch_id=self.batch_id
            )
        except Exception:
            tlog("Probably already associated...")
            return False

        self.account_id, self.portfolio_size = await Portfolio.load_details(
            self.portfolio_id
        )

        return True

    async def is_buy_time(self, now: datetime):
        return (
            time(hour=14, minute=30) >= now.time() >= time(hour=9, minute=30)
        )

    async def is_sell_time(self, now: datetime):
        return True

    async def handle_buy_side(
        self,
        symbols_position: Dict[str, float],
        data_loader: DataLoader,
        now: datetime,
    ) -> Dict[str, Dict]:
        actions = {}

        for symbol in symbols_position:
            current_price = data_loader[symbol].close[now]
            serie = (
                self.data_loader[symbol]
                .close[now - timedelta(days=30) : now]  # type:ignore
                .between_time("9:30", "16:00")
            )

            if not len(serie):
                serie = self.data_loader[symbol].close[
                    now - timedelta(days=30) : now  # type:ignore
                ]

            resampled_close = serie.resample("1D").last().dropna()
            bband = BBANDS(
                resampled_close,
                timeperiod=7,
                nbdevdn=1,
                nbdevup=1,
                matype=MA_Type.SMA,
            )

            yesterday_lower_band = bband[2][-2]
            today_lower_band = bband[2][-1]
            yesterday_close = resampled_close[-2]

            today_open = self.data_loader[symbol].open[
                config.market_open.replace(second=0, microsecond=0)
            ]

            if (
                yesterday_close < yesterday_lower_band
                and today_open > yesterday_close
                and current_price > today_lower_band
            ):
                yesterday_upper_band = bband[0][-2]
                if current_price > yesterday_upper_band:
                    return {}

                buy_indicators[symbol] = {
                    "lower_band": bband[2][-2:].tolist(),
                }
                cash = await Accounts.get_balance(self.account_id)
                shares_to_buy = cash // current_price
                tlog(
                    f"[{self.name}][{now}] Submitting buy for {shares_to_buy} shares of {symbol} at {current_price}"
                )
                tlog(f"indicators:{buy_indicators[symbol]}")
                actions[symbol] = {
                    "side": "buy",
                    "qty": str(shares_to_buy),
                    "type": "limit",
                    "limit_price": str(current_price),
                }

        return actions

    async def handle_sell_side(
        self,
        symbols_position: Dict[str, float],
        data_loader: DataLoader,
        now: datetime,
    ) -> Dict[str, Dict]:
        actions = {}

        for symbol, position in symbols_position.items():
            if position == 0:
                continue

            current_price = data_loader[symbol].close[now]
            serie = (
                self.data_loader[symbol]
                .close[now - timedelta(days=30) : now]  # type:ignore
                .between_time("9:30", "16:00")
            )

            if not len(serie):
                serie = self.data_loader[symbol].close[
                    now - timedelta(days=30) : now  # type:ignore
                ]

            self.resampled_close[symbol] = serie.resample("1D").last().dropna()
            self.bband[symbol] = BBANDS(
                self.resampled_close[symbol],
                timeperiod=7,
                nbdevdn=1,
                nbdevup=1,
                matype=MA_Type.EMA,
            )

            yesterday_upper_band = self.bband[symbol][0][-2]
            if current_price > yesterday_upper_band:
                sell_indicators[symbol] = {
                    "upper_band": self.bband[symbol][0][-2:].tolist(),
                    "lower_band": self.bband[symbol][2][-2:].tolist(),
                }

                tlog(
                    f"[{self.name}][{now}] Submitting sell for {position} shares of {symbol} at market"
                )
                tlog(f"indicators:{sell_indicators[symbol]}")
                actions[symbol] = {
                    "side": "sell",
                    "qty": str(position),
                    "type": "market",
                }

        return actions

    async def should_run_all(self):
        return True

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

        actions = {}
        # print(now, symbols_position)
        if await self.is_buy_time(now) and not open_orders:
            actions.update(
                await self.handle_buy_side(
                    symbols_position=symbols_position,
                    data_loader=data_loader,
                    now=now,
                )
            )

        if (
            await self.is_sell_time(now)
            and (
                len(symbols_position)
                or any(symbols_position[x] for x in symbols_position)
            )
            and not open_orders
        ):
            actions.update(
                await self.handle_sell_side(
                    symbols_position=symbols_position,
                    data_loader=data_loader,
                    now=now,
                )
            )

        return actions
