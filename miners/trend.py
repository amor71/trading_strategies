import asyncio
import concurrent.futures
import sys
import uuid
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import alpaca_trade_api as tradeapi
import numpy as np
from liualgotrader.common import config
from liualgotrader.common.data_loader import DataLoader  # type: ignore
from liualgotrader.common.market_data import index_data
from liualgotrader.common.tlog import tlog
from liualgotrader.common.types import TimeScale
from liualgotrader.miners.base import Miner
from liualgotrader.models.portfolio import Portfolio as DBPortfolio
from pandas import DataFrame as df
from pytz import timezone
from scipy.stats import linregress
from stockstats import StockDataFrame
from tabulate import tabulate

sys.path.append("..")
from trades.common.trend import Trend as TrendLogic

nyc = timezone("America/New_York")


class Trend(Miner):
    def __init__(
        self,
        data: Dict,
        debug=False,
    ):
        try:
            self.index = data["index"]
            self.debug = debug
            self.portfolio_size = data["portfolio_size"]
            self.risk_factor = data["risk_factor"]
            self.rank_days = int(data["rank_days"])
        except Exception:
            raise ValueError(
                "[ERROR] Miner must receive all valid parameter(s)"
            )
        super().__init__(name="PortfolioBuilder")

        if self.debug:
            tlog(f"{self.name} running in debug mode")

    async def save_portfolio(self, df: df) -> str:
        portfolio_id = str(uuid.uuid4())
        tlog(f"Saving portfolio {portfolio_id}")
        await DBPortfolio.save(id=portfolio_id, df=df)
        tlog("Done.")
        return portfolio_id

    async def display_portfolio(self, df: df) -> None:
        if self.debug:
            print(f"FINAL:\n{tabulate(df, headers='keys', tablefmt='psql')}")

    async def run(self) -> bool:
        self.trend_logic = TrendLogic(
            symbols=(await index_data(self.index)).Symbol.tolist(),
            portfolio_size=self.portfolio_size,
            risk_factor=self.risk_factor,
            rank_days=self.rank_days,
            debug=self.debug,
        )
        if self.debug:
            tlog(f"symbols: {self.trend_logic.symbols}")

        df = await self.trend_logic.run(
            nyc.localize(datetime.utcnow()) + timedelta(days=1)
        )
        portfolio_id = await self.save_portfolio(df)
        await self.display_portfolio(df)

        print(
            "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
        )
        tlog(f"PORTFOLIO_ID:{portfolio_id}")
        print(
            "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
        )

        return True
