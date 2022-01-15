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
from liualgotrader.fincalcs.vwap import add_daily_vwap
from liualgotrader.strategies.base import Strategy, StrategyType
from pandas import DataFrame as df
from pandas import Series
from pandas import Timestamp as ts
from pandas import concat
from talib import BBANDS, MACD, RSI


class ShortGapDown(Strategy):
    name = "ShortGapDown"
    was_above_vwap: Dict = {}

    def __init__(
        self,
        batch_id: str,
        schedule: List[Dict],
        ref_run_id: int = None,
        check_patterns: bool = False,
    ):
        self.check_patterns = check_patterns
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
        latest_cost_basis[symbol] = price

    async def create(self) -> None:
        await super().create()
        tlog(f"strategy {self.name} created")

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
        if not shortable:
            return False, {}

        data = minute_history.iloc[-1]

        if data.close > data.average:
            self.was_above_vwap[symbol] = True

        if (
            await super().is_buy_time(now)
            and not position
            and not open_orders.get(symbol, None)
        ):
            tlog(f"{self.name} CONSIDER SHORT SELL {symbol}")

        if (
            await super().is_sell_time(now)
            and position
            and last_used_strategy[symbol].name == self.name
            and not open_orders.get(symbol)
        ):
            tlog(f"{self.name} CONSIDER SHORT BUY {symbol}")

        return False, {}
