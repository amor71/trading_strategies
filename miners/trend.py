import asyncio
import concurrent.futures
import sys
import traceback
import uuid
from datetime import date, datetime, timedelta
from multiprocessing import Queue
from queue import Empty, Full
from typing import Dict, List, Optional, Tuple

import numpy as np
from liualgotrader.common import config
from liualgotrader.common.data_loader import DataLoader  # type: ignore
from liualgotrader.common.market_data import sp500_historical_constituents
from liualgotrader.common.tlog import tlog
from liualgotrader.common.types import QueueMapper, TimeScale
from liualgotrader.miners.base import Miner
from liualgotrader.models.algo_run import AlgoRun
from liualgotrader.models.new_trades import NewTrade
from liualgotrader.models.portfolio import Portfolio as DBPortfolio
from liualgotrader.trading.base import Trader
from liualgotrader.trading.trader_factory import trader_factory
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
        print("here...")
        try:
            self.index = data["index"]
            self.debug = debug
            self.portfolio_size = data["portfolio_size"]
            self.rank_days = int(data["rank_days"])
            self.stock_count = int(data["stock_count"])
            self.volatility_threshold = float(data["volatility_threshold"])
            self.short = data.get("short", False)
            self.top = data.get("top", 40)
        except Exception:
            raise ValueError(
                "[ERROR] Miner must receive all valid parameter(s)"
            )
        super().__init__(name="PortfolioBuilder")

        if self.debug:
            tlog(f"{self.name} running in debug mode")

    async def save_portfolio(self) -> str:
        portfolio_id = str(uuid.uuid4())
        tlog(f"Saving portfolio {portfolio_id}")
        await DBPortfolio.save(
            portfolio_id,
            self.portfolio_size,
            self.stock_count,
            {"rank_days": self.rank_days},
        )
        tlog("Done.")
        return portfolio_id

    async def display_portfolio(self, df: df) -> None:
        if self.debug:
            print(f"FINAL:\n{tabulate(df, headers='keys', tablefmt='psql')}")

    async def execute_portfolio(
        self, portfolio_id: str, df: df, now: datetime
    ) -> None:
        tlog("Executing portfolio buys")
        trader = trader_factory()

        algo_run = await trader.create_session(self.name)
        await DBPortfolio.associate_batch_id_to_profile(
            portfolio_id, algo_run.batch_id
        )

        orders = [
            await trader.submit_order(
                symbol=row.symbol,
                qty=row.qty,
                side="buy",
                order_type="market",
                time_in_force="day",
            )
            for _, row in df.iterrows()
        ]

        open_orders = []
        while True:
            for order in orders:
                (
                    order_completed,
                    executed_price,
                ) = await trader.is_order_completed(order)
                if order_completed:
                    db_trade = NewTrade(
                        algo_run_id=algo_run.run_id,
                        symbol=order.symbol,
                        qty=int(order.qty),
                        operation="buy",
                        price=executed_price,
                        indicators={},
                    )
                    await db_trade.save(
                        config.db_conn_pool,
                        str(now),
                        0.0,
                        0.0,
                    )
                else:
                    open_orders.append(order)
            if not len(open_orders):
                break
            await asyncio.sleep(5.0)
            orders = open_orders

    async def run(self) -> bool:
        trader = trader_factory()
        self.trend_logic = TrendLogic(
            symbols=await sp500_historical_constituents(str(datetime.now())),
            portfolio_size=self.portfolio_size,
            rank_days=self.rank_days,
            debug=self.debug,
            stock_count=self.stock_count,
            volatility_threshold=self.volatility_threshold,
            data_loader=DataLoader(),
            trader=trader,
            top=self.top,
        )
        if self.debug:
            tlog(f"symbols: {self.trend_logic.symbols}")

        df = (
            await self.trend_logic.run_short(
                nyc.localize(datetime.utcnow()) + timedelta(days=1)
            )
            if self.short
            else await self.trend_logic.run(
                nyc.localize(datetime.utcnow()) + timedelta(days=1)
            )
        )

        portfolio_id = await self.save_portfolio()
        await self.display_portfolio(df)

        # await self.execute_portfolio(
        #    portfolio_id, df, nyc.localize(datetime.utcnow())
        # )
        print(
            "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
        )
        tlog(f"PORTFOLIO_ID:{portfolio_id}")
        print(
            "-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
        )

        return True
