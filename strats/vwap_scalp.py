import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import alpaca_trade_api as tradeapi
import pandas as pd
from liualgotrader.common import config
from liualgotrader.common.data_loader import DataLoader
from liualgotrader.common.tlog import tlog
from liualgotrader.fincalcs.trends import (SeriesTrendType, get_series_trend,
                                           volatility)
from liualgotrader.fincalcs.vwap import add_daily_vwap
from liualgotrader.strategies.base import Strategy, StrategyType


class VWAPScalp(Strategy):
    def __init__(
        self,
        batch_id: str,
        data_loader: DataLoader,
        portfolio_id: str,
        ref_run_id: str = None,
    ):
        self.total_volume: Dict[str, float] = {}
        self.total_price: Dict[str, float] = {}
        self.last_update: Dict[str, pd.Timestamp] = {}

        super().__init__(
            name=type(self).__name__,
            type=StrategyType.DAY_TRADE,
            batch_id=batch_id,
            ref_run_id=ref_run_id,
            data_loader=data_loader,
            schedule=[],
        )

    async def update_vwap(self, symbol: str, now: datetime) -> None:
        if not self.data_loader[symbol][now].average:
            add_daily_vwap(self.data_loader[symbol])

    async def crossed_vwap(self, symbol: str, now: datetime) -> bool:
        """return True if stock just crossed VWAP from below to above"""
        before = now - timedelta(minutes=5)
        return (
            self.data_loader[symbol].vwap[before]
            >= self.data_loader[symbol].close[before]
            and self.data_loader[symbol].vwap[now]
            < self.data_loader[symbol].close[now]
        )

    async def buy_signal(
        self, symbol: str, minute_history: pd.DataFrame, now: datetime
    ) -> bool:
        """Buy on cross of VWAP if positive trend below VWAP for last 20 minutes"""
        await self.update_vwap(symbol, now)

        print(
            "VWAP:",
            self.data_loader[symbol][now - timedelta(minutes=10) : now],  # type: ignore
        )
        if await self.crossed_vwap(symbol, minute_history):
            _, trend = get_series_trend(
                minute_history[now - timedelta(minutes=20) : now].close  # type: ignore
            )
            return trend in (
                SeriesTrendType.SHARP_UP,
                SeriesTrendType.UP,
            )

        return False

    async def calc_amount(self, symbol: str, now: datetime) -> float:
        # calculate size based on volatility
        volatility = 1.0 - volatility(self.data_loader, symbol, now)
        return await self.avaliable_funds()

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

        if position == 0.0 and await self.buy_signal(
            symbol, minute_history, now
        ):
            size = await self.calc_amount(symbol, now)

            tlog(
                f"making purchase of {symbol} at {size} @ {minute_history.close[-1]}"
            )

            return (
                True,
                {
                    "side": "buy",
                    "qty": size,
                    "type": "limit",
                    "limit_price": minute_history.close[-1],
                },
            )

        elif position != 0.0:
            tlog("we need to sell!")
            # scale-out
            # sell if not picking up.
            pass

        return False, {}
