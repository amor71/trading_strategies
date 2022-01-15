import asyncio
import sys
import uuid
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple

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


class Hft(Strategy):
    def __init__(
        self,
        batch_id: str,
        data_loader: DataLoader,
        ref_run_id: int = None,
    ):
        self.name = type(self).__name__
        super().__init__(
            name=type(self).__name__,
            type=StrategyType.DAY_TRADE,
            batch_id=batch_id,
            ref_run_id=ref_run_id,
            schedule=[],
            data_loader=data_loader,
        )

    async def buy_callback(
        self, symbol: str, price: float, qty: int, now: datetime = None
    ) -> None:
        ...

    async def sell_callback(
        self, symbol: str, price: float, qty: int, now: datetime = None
    ) -> None:
        ...

    async def create(self) -> bool:
        if not await super().create():
            return False

        tlog(f"strategy {self.name} created")

        return True

    async def should_run_all(self):
        return False

    async def run(
        self,
        symbol: str,
        shortable: bool,
        position: float,
        now: datetime,
        minute_history: pd.DataFrame,
        portfolio_value: float = None,
        debug: bool = False,
        backtesting: bool = False,
    ) -> Tuple[bool, Dict]:
        print("HFT", symbol, minute_history)
        return False, {}
