import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import alpaca_trade_api as tradeapi
import numpy as np
from liualgotrader.common import config
from liualgotrader.common.tlog import tlog
from liualgotrader.common.trading_data import (buy_indicators, buy_time,
                                               cool_down, last_used_strategy,
                                               latest_cost_basis,
                                               latest_scalp_basis, open_orders,
                                               sell_indicators, stop_prices,
                                               target_prices)
from liualgotrader.fincalcs.support_resistance import find_stop
from liualgotrader.strategies.base import Strategy, StrategyType
from pandas import DataFrame as df
from talib import BBANDS, MACD, RSI, MA_Type


class Gold(Strategy):
    name = "gold"
    whipsawed: Dict = {}
    bband: Dict = {}
    resampled_close: Dict = {}

    def __init__(
        self,
        batch_id: str,
        schedule: List[Dict],
        amount: int,
        ref_run_id: int = None,
    ):
        self.amount = amount
        super().__init__(
            name=self.name,
            type=StrategyType.SWING,
            batch_id=batch_id,
            ref_run_id=ref_run_id,
            schedule=schedule,
        )

    async def buy_callback(self, symbol: str, price: float, qty: int) -> None:
        pass

    async def sell_callback(self, symbol: str, price: float, qty: int) -> None:
        pass

    async def create(self) -> None:
        await super().create()
        tlog(f"strategy {self.name} created")

    async def should_cool_down(self, symbol: str, now: datetime):
        if (
            symbol in cool_down
            and cool_down[symbol]
            and cool_down[symbol] >= now.replace(second=0, microsecond=0)  # type: ignore
        ):
            return True

        cool_down[symbol] = None
        return False

    async def run(
        self,
        symbol: str,
        shortable: bool,
        position: int,
        minute_history: df,
        now: datetime,
        portfolio_value: float = None,
        trading_api: tradeapi = None,
        debug: bool = False,
        backtesting: bool = False,
    ) -> Tuple[bool, Dict]:
        if (
            await super().is_buy_time(now)
            and not position
            and not open_orders.get(symbol, None)
            and not await self.should_cool_down(symbol, now)
        ):
            if symbol not in self.bband:
                # Calculate 7 day Bolinger Band, w 1.5 std
                self.resampled_close[symbol] = (
                    minute_history.close.between_time("9:30", "16:00")
                    .resample("1D")
                    .last()
                )
                self.bband[symbol] = BBANDS(
                    self.resampled_close[symbol],
                    timeperiod=7,
                    nbdevdn=1.5,
                    nbdevup=1.5,
                    matype=MA_Type.EMA,
                )

            # if previous day finish below band,
            # and current day open above previous day close
            # and cross above band -> buy
            current_price = minute_history.close[-1]
            yesterday_lower_band = self.bband[symbol][2][-2]
            yesterday_close = self.resampled_close[symbol][-2]
            today_open = minute_history.open[
                config.market_open.replace(second=0, microsecond=0)
            ]
            if (
                yesterday_close < yesterday_lower_band
                and today_open > yesterday_close
                and current_price > yesterday_lower_band
            ):
                buy_indicators[symbol] = {
                    "lower_band": self.bband[symbol][2][-5:].tolist(),
                }
                shares_to_buy = self.amount // current_price
                tlog(
                    f"[{self.name}][{now}] Submitting buy for {shares_to_buy} shares of {symbol} at {current_price}"
                )
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
            await super().is_sell_time(now)
            and position > 0
            and last_used_strategy[symbol].name == self.name
            and not open_orders.get(symbol)
        ):
            if symbol not in self.bband:
                # Calculate 7 day Bolinger Band, w 1.5 std
                self.resampled_close[symbol] = (
                    minute_history.close.between_time("9:30", "16:00")
                    .resample("1D")
                    .last()
                )
                self.bband[symbol] = BBANDS(
                    self.resampled_close[symbol],
                    timeperiod=7,
                    nbdevdn=1.5,
                    nbdevup=1.5,
                    matype=MA_Type.EMA,
                )

            # if price pops above upper-band -> sell
            current_price = minute_history.close[-1]
            yesterday_upper_band = self.bband[symbol][0][-2]
            if current_price > yesterday_upper_band:
                sell_indicators[symbol] = {
                    "upper_band": self.bband[symbol][0][-5:].tolist(),
                }

                tlog(
                    f"[{self.name}][{now}] Submitting sell for {position} shares of {symbol} at market"
                )
                return (
                    True,
                    {
                        "side": "sell",
                        "qty": str(position),
                        "type": "market",
                    },
                )

        return False, {}
