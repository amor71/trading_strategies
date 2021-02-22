"""follow the gold"""
from datetime import datetime, timedelta
from typing import List, Optional

import alpaca_trade_api as tradeapi
from liualgotrader.common.tlog import tlog
from liualgotrader.scanners.base import Scanner


class GoldDigger(Scanner):
    name = "GoldDigger"
    golden_list = ["JNUG"]

    def __init__(
        self,
        data_api: tradeapi,
        recurrence: Optional[timedelta] = None,
        target_strategy_name: str = None,
    ):
        super().__init__(
            name=self.name,
            recurrence=recurrence,
            target_strategy_name=target_strategy_name,
            data_api=data_api,
        )

    async def run(self, back_time: datetime = None) -> List[str]:
        tlog(f"{self.name}{back_time} picked {self.golden_list}")

        return self.golden_list
