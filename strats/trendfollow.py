import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

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

nyc = timezone("America/New_York")


class TrendFollow(Strategy):
    name = "trend_follow"

    def __init__(
        self,
        batch_id: str,
        data_loader: DataLoader,
        portfolio_id: str,
        portfolio_size: int,
        rebalance_rate: str,
        ref_run_id: str = None,
    ):
        self.context: str
        self.portfolio_id = portfolio_id
        self.portfolio_size = portfolio_size
        self.last_rebalance: str
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
        pass

    async def sell_callback(self, symbol: str, price: float, qty: int) -> None:
        pass

    async def create(self) -> None:
        await super().create()
        tlog(f"strategy {self.name} created")

    async def rebalance(self, now: datetime):
        await self.set_global_var("last_rebalance", str(now), self.context)

        print(
            f"RE-BALANCING {await self.get_global_var('last_rebalance', self.context)}"
        )
        return {}

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
    ) -> Dict[str, Tuple[bool, Dict]]:
        self.context = self.batch_id if backtesting else self.name
        if await self.should_rebalance(now):
            return await self.rebalance(now)
        return {}

    async def should_run_all(self):
        return True
