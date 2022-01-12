"""follow the gold"""
from datetime import datetime, timedelta
from typing import List, Optional

from liualgotrader.common.data_loader import DataLoader
from liualgotrader.common.tlog import tlog
from liualgotrader.scanners.base import Scanner


class GoldDigger(Scanner):
    name = "GoldDigger"

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
        if not back_time:
            back_time = datetime.now()
        sma_50 = (
            self.data_loader["GDXJ"]
            .close[back_time - timedelta(days=90) : back_time]  # type: ignore
            .between_time("9:30", "16:00")
            .resample("1D")
            .last()
            .dropna()
            .rolling(50)
            .mean()
            .iloc[-1]
        )
        return (
            ["JNUG"]
            if self.data_loader["GDXJ"].close[back_time] > sma_50
            else ["JDST"]
        )
