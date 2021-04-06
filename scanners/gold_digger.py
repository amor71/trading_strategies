"""follow the gold"""
from datetime import datetime, timedelta
from typing import List, Optional

import alpaca_trade_api as tradeapi
from liualgotrader.common.data_loader import DataLoader
from liualgotrader.common.tlog import tlog
from liualgotrader.scanners.base import Scanner


class GoldDigger(Scanner):
    name = "GoldDigger"
    golden_list = ["JNUG", "JDST"]

    def __init__(
        self,
        data_loader: DataLoader,
        recurrence: Optional[timedelta] = None,
        target_strategy_name: str = None,
    ):
        super().__init__(
            name=self.name,
            data_loader=data_loader,
            recurrence=recurrence,
            target_strategy_name=target_strategy_name,
        )

    async def run(self, back_time: datetime = None) -> List[str]:
        # tlog(f"{self.name}{back_time} picked {self.golden_list}")

        return self.golden_list
