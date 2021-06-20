"""follow the gold"""
from datetime import datetime, timedelta
from typing import List, Optional

import alpaca_trade_api as tradeapi
import pandas as pd
from liualgotrader.common.data_loader import DataLoader
from liualgotrader.common.tlog import tlog
from liualgotrader.fincalcs.trends import SeriesTrendType, get_series_trend
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
        if (
            self.data_loader["GDXJ"].close[back_time]
            > self.data_loader["GDXJ"]
            .close[back_time - timedelta(days=90) : back_time]  # type: ignore
            .rolling(50)
            .sum()[-1]
            / 50
        ):
            print("UP!", back_time)
            return ["JNUG"]
        else:
            print("DOWN!", back_time)
            return ["JDST"]
