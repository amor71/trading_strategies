import asyncio
import concurrent.futures
import traceback
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
from scipy.stats import linregress
from stockstats import StockDataFrame
from tabulate import tabulate


class Trend:
    portfolio: df = df(columns=["symbol", "slope", "r", "score"])
    data_bars: Dict[str, df] = {}

    def __init__(
        self,
        symbols: List[str],
        portfolio_size: int,
        risk_factor: float,
        rank_days: int,
        debug=False,
    ):
        try:
            self.rank_days = rank_days
            self.debug = debug
            self.portfolio_size = portfolio_size
            self.risk_factor = risk_factor
            self.data_loader = DataLoader(TimeScale.day)
            self.symbols = symbols
        except Exception:
            raise ValueError(
                "[ERROR] Miner must receive all valid parameter(s)"
            )

    def load_data_for_symbol(self, symbol: str, now: datetime) -> None:
        try:
            self.data_bars[symbol] = self.data_loader[symbol][
                date.today() - timedelta(days=int(200 * 7 / 5)) : now  # type: ignore
            ]
        except Exception:
            tlog(f"[ERROR] could not load all data points for {symbol}")
            traceback.print_exc()
            self.data_bars[symbol] = None

    async def load_data(self, symbols: List[str], now: datetime) -> None:
        tlog("Data loading started")
        if not len(symbols):
            raise Exception(
                "load_data() received an empty list of symbols to load. aborting"
            )
        # We can use a with statement to ensure threads are cleaned up promptly
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the load operations and mark each future with its URL
            futures = {
                executor.submit(self.load_data_for_symbol, symbol, now): symbol
                for symbol in symbols
            }
            for _ in concurrent.futures.as_completed(futures):
                pass
        tlog(
            f"Data loading completed, loaded data for {len(self.data_bars)} symbols"
        )

    def calc_symbol_momentum(
        self, symbol: str, now: datetime
    ) -> Optional[Dict]:
        d = self.data_bars[symbol]
        _df = df(d)
        deltas = np.log(_df.close[-self.rank_days :])  # type: ignore
        slope, _, r_value, _, _ = linregress(np.arange(len(deltas)), deltas)
        if slope > 0:
            annualized_slope = (np.power(np.exp(slope), 252) - 1) * 100
            score = annualized_slope * (r_value ** 2)

            return dict(
                {
                    "symbol": symbol,
                    "slope": annualized_slope,
                    "r": r_value,
                    "score": score,
                },
            )
        else:
            return None

    async def calc_momentum(self, now: datetime) -> None:
        if not len(self.data_bars):
            raise Exception("calc_momentum() can't run without data. aborting")

        tlog("Trend ranking calculation started")
        symbols = [
            symbol
            for symbol in self.data_bars.keys()
            if not self.data_bars[symbol].empty
        ]

        l: List[Dict] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the load operations and mark each future with its URL
            futures = {
                executor.submit(self.calc_symbol_momentum, symbol, now): symbol
                for symbol in symbols
            }
            for future in concurrent.futures.as_completed(futures):
                data = future.result()
                if data:
                    l.append(data)  # , ignore_index=True)

        self.portfolio = df.from_records(l).sort_values(
            by="score", ascending=False
        )
        tlog(
            f"Trend ranking calculation completed w/ {len(self.portfolio)} trending stocks"
        )

    def apply_filters_symbol(self, symbol: str, now: datetime) -> bool:
        indicator_calculator = StockDataFrame(self.data_bars[symbol])
        sma_100 = indicator_calculator["close_100_sma"]
        if self.data_bars[symbol].close[-1] < sma_100[-1]:
            return False

        return True

        # filter stocks moving > 15% in last 90 days
        last = self.data_bars[symbol].close[
            -1
        ]  # self.data_bars[row.symbol].close[-90:].max()
        low = self.data_bars[symbol].close[-90:].min()  # type: ignore
        return last / low <= 1.25

    async def apply_filters(self, now: datetime) -> None:
        tlog("Applying filters")
        pre_filter_len = len(self.portfolio)
        symbols = [
            symbol
            for symbol in self.data_bars.keys()
            if not self.data_bars[symbol].empty
        ]

        pass_filter: list = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the load operations and mark each future with its URL
            futures = {
                executor.submit(self.apply_filters_symbol, symbol, now): symbol
                for symbol in symbols
            }
            for future in concurrent.futures.as_completed(futures):
                filter = future.result()
                if filter:
                    pass_filter.append(futures[future])

        self.portfolio = self.portfolio[
            self.portfolio.symbol.isin(pass_filter)
        ]
        tlog(f"filters removed {pre_filter_len-len(self.portfolio)}")

    async def calc_balance(self) -> None:
        for _, row in self.portfolio.iterrows():
            volatility = (
                self.data_bars[row.symbol]
                .close.pct_change()
                .rolling(20)
                .std()
                .iloc[-1]
            )
            qty = int(self.portfolio_size * self.risk_factor // volatility)
            self.portfolio.loc[
                self.portfolio.symbol == row.symbol, "volatility"
            ] = volatility
            self.portfolio.loc[
                self.portfolio.symbol == row.symbol, "qty"
            ] = qty
            self.portfolio.loc[self.portfolio.symbol == row.symbol, "est"] = (
                qty * self.data_bars[row.symbol].close[-1]
            )
        self.portfolio = self.portfolio.loc[self.portfolio.qty > 0]
        self.portfolio["accumulative"] = self.portfolio.est.cumsum()

    async def run(self, now: datetime) -> df:
        await self.load_data(self.symbols, now)
        await self.calc_momentum(now)
        await self.apply_filters(now)
        await self.calc_balance()
        return self.portfolio
