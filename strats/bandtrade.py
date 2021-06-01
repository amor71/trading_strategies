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
    whipsawed: Dict = {}
    bband: Dict = {}
    resampled_close: Dict = {}

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

    async def should_cool_down(self, symbol: str, now: datetime):
        if (
            symbol in cool_down
            and cool_down[symbol]
            and cool_down[symbol] >= now.replace(second=0, microsecond=0)  # type: ignore
        ):
            return True

        cool_down[symbol] = None
        return False

    async def is_buy_time(self, now: datetime):
        return (
            time(hour=14, minute=30) >= now.time() >= time(hour=9, minute=30)
        )

    async def is_sell_time(self, now: datetime):
        return True

    async def run(
        self,
        symbol: str,
        shortable: bool,
        position: int,
        now: datetime,
        minute_history: df = None,
        portfolio_value: float = None,
        trading_api: tradeapi = None,
        debug: bool = False,
        backtesting: bool = False,
    ) -> Tuple[bool, Dict]:
        current_price = (
            minute_history.close[-1]
            if minute_history is not None
            else self.data_loader[symbol].open[now]
        )
        # tlog(f"{now} {current_price}")
        if (
            await self.is_buy_time(now)
            and not position
            and not open_orders.get(symbol, None)
            and not await self.should_cool_down(symbol, now)
        ):
            # Calculate 7 day Bolinger Band, w 1 std
            if minute_history is not None:
                self.resampled_close[symbol] = (
                    minute_history.between_time("9:30", "16:00")
                    .close.resample("1D")
                    .last()
                    .dropna()
                )
                # print(self.resampled_close[symbol])
            else:
                self.resampled_close[symbol] = (
                    self.data_loader[symbol]
                    .close[now - timedelta(days=30) : now]  # type: ignore
                    .between_time("9:30", "16:00")
                    .resample("1D")
                    .last()
                    .dropna()
                )

            self.bband[symbol] = BBANDS(
                self.resampled_close[symbol],
                timeperiod=7,
                nbdevdn=1,
                nbdevup=1,
                matype=MA_Type.EMA,
            )

            # print(self.resampled_close[symbol])  # , self.bband[symbol])

            # if previous day finish below band,
            # and current day open above previous day close
            # and cross above band -> buy
            yesterday_lower_band = self.bband[symbol][2][-2]
            today_lower_band = self.bband[symbol][2][-1]
            yesterday_close = self.resampled_close[symbol][-2]

            # print(
            #    f"yesterday_lower_band:{yesterday_lower_band} today_lower_band:{today_lower_band} yesterday_close:{yesterday_close}"
            # )
            if minute_history is not None:
                start_day_index = minute_history.index.get_loc(
                    config.market_open.replace(second=0, microsecond=0),
                    method="nearest",
                )
                today_open = minute_history.iloc[start_day_index].open
                # print("today open:", today_open)
            else:
                today_open = self.data_loader[symbol].open[
                    config.market_open.replace(second=0, microsecond=0)
                ]

            if (
                yesterday_close < yesterday_lower_band
                and today_open > yesterday_close
                and current_price > today_lower_band
            ):
                # check if not sell signal not triggered too
                # (if price pops above upper-band -> sell)
                yesterday_upper_band = self.bband[symbol][0][-2]
                if current_price > yesterday_upper_band:
                    return False, {}

                print(
                    config.market_close.replace(second=0, microsecond=0)
                    - timedelta(days=1),
                    self.data_loader[symbol].close[
                        config.market_close.replace(second=0, microsecond=0)
                        - timedelta(days=1)
                    ],
                )
                print(
                    f"{now}-{yesterday_close}<{yesterday_lower_band} {today_open}>{yesterday_close} {current_price} > {today_lower_band}"
                )
                buy_indicators[symbol] = {
                    "lower_band": self.bband[symbol][2][-2:].tolist(),
                }
                cash = await Accounts.get_balance(self.account_id)
                shares_to_buy = cash // current_price
                tlog(
                    f"[{self.name}][{now}] Submitting buy for {shares_to_buy} shares of {symbol} at {current_price}"
                )
                tlog(f"indicators:{buy_indicators[symbol]}")
                return (
                    True,
                    {
                        "side": "buy",
                        "qty": str(shares_to_buy),
                        "type": "limit",
                        "limit_price": str(current_price),
                    },
                )

        if (
            await self.is_sell_time(now)
            and position
            and last_used_strategy[symbol].name == self.name
            and not open_orders.get(symbol)
        ):
            # Calculate 7 day Bolinger Band, w 1 std
            if minute_history is not None:
                self.resampled_close[symbol] = (
                    minute_history.between_time("9:30", "16:00")
                    .close.resample("1D")
                    .last()
                    .dropna()
                )
            else:
                self.resampled_close[symbol] = (
                    self.data_loader[symbol]
                    .close[now - timedelta(days=30) : now]  # type: ignore
                    .between_time("9:30", "16:00")
                    .resample("1D")
                    .last()
                    .dropna()
                )
            self.bband[symbol] = BBANDS(
                self.resampled_close[symbol],
                timeperiod=7,
                nbdevdn=1,
                nbdevup=1,
                matype=MA_Type.EMA,
            )

            # if price pops above upper-band -> sell
            yesterday_upper_band = self.bband[symbol][0][-2]
            # print(current_price, yesterday_upper_band)
            if current_price > yesterday_upper_band:
                sell_indicators[symbol] = {
                    "upper_band": self.bband[symbol][0][-2:].tolist(),
                }

                tlog(
                    f"[{self.name}][{now}] Submitting sell for {position} shares of {symbol} at market"
                )
                tlog(f"indicators:{sell_indicators[symbol]}")
                return (
                    True,
                    {
                        "side": "sell",
                        "qty": str(position),
                        "type": "market",
                    },
                )

        return False, {}
