"""follow the gold"""
from datetime import datetime, timedelta
from typing import List, Optional

from liualgotrader.common.data_loader import DataLoader
from liualgotrader.common.tlog import tlog
from liualgotrader.data.data_factory import data_loader_factory
from liualgotrader.scanners.base import Scanner


class Load(Scanner):
    name = "Load"

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

        self.data_factory = data_loader_factory()

    async def run(self, back_time: datetime = None) -> List[str]:
        return self.data_factory.get_symbols
