"""follow the gold"""
from datetime import timedelta
from typing import List, Optional

import alpaca_trade_api as tradeapi
from liualgotrader.common.tlog import tlog
from liualgotrader.scanners.base import Scanner


class GoldDigger(Scanner):
    name = "GoldDigger"
    golden_list = ["JNUG", "JDST"]

    def __init__(
        self, recurrence: Optional[timedelta], data_api: tradeapi, **args
    ):
        super().__init__(
            name=self.name,
            recurrence=recurrence,
            data_api=data_api,
        )

    def run(self) -> List[str]:
        tlog(f"return {self.golden_list}")
        return self.golden_list
