"""
Adopted from Andreas F Clenow's "Trend Following" method(s). 

Further readings:

https://www.followingthetrend.com/stocks-on-the-move/ 
https://www.followingthetrend.com/trading-evolved/

"""

import asyncio
import concurrent.futures
import math
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
    def __init__(
        self,
        symbols: List[str],
        portfolio_size: int,
        rank_days: int,
        stock_count: int,
        volatility_threshold: float,
        debug=False,
    ):
        try:
            self.rank_days = rank_days
            self.debug = debug
            self.portfolio_size = portfolio_size
            self.data_loader = DataLoader(TimeScale.day)
            self.symbols = symbols
            self.stock_count = stock_count
            self.volatility_threshold = volatility_threshold
        except Exception:
            raise ValueError(
                "[ERROR] Miner must receive all valid parameter(s)"
            )

        self.portfolio: df = df(columns=["symbol", "slope", "r", "score"])
        self.data_bars: Dict[str, df] = {}

    def load_data_for_symbol(self, symbol: str, now: datetime) -> None:
        try:
            self.data_bars[symbol] = self.data_loader[symbol][
                now.date() - timedelta(days=int(100 * 7 / 5)) : now  # type: ignore
            ]

        except Exception:
            tlog(f"[ERROR] could not load all data points for {symbol}")
            traceback.print_exc()

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

    def calc_symbol_momentum(self, symbol: str) -> Optional[Dict]:
        d = self.data_bars[symbol]
        _df = df(d)
        deltas = np.log(_df.close[-self.rank_days :])  # type: ignore
        slope, _, r_value, _, _ = linregress(np.arange(len(deltas)), deltas)
        if slope > 0:
            annualized_slope = (np.power(np.exp(slope), 252) - 1) * 100
            score = annualized_slope * (r_value ** 2)
            volatility = (
                1
                - self.data_bars[symbol]
                .close.pct_change()
                .rolling(20)
                .std()
                .iloc[-1]
            )
            return dict(
                {
                    "symbol": symbol,
                    "slope": annualized_slope,
                    "r": r_value,
                    "score": score,
                    "volatility": volatility,
                },
            )
        else:
            return None

    def calc_symbol_negative_momentum(self, symbol: str) -> Optional[Dict]:
        d = self.data_bars[symbol]
        _df = df(d)
        deltas = np.log(_df.close[-self.rank_days :])  # type: ignore
        slope, _, r_value, _, _ = linregress(np.arange(len(deltas)), deltas)
        if slope < 0:
            annualized_slope = (np.power(np.exp(slope), 252) - 1) * 100
            score = annualized_slope * (r_value ** 2)
            volatility = (
                1
                - self.data_bars[symbol]
                .close.pct_change()
                .rolling(20)
                .std()
                .iloc[-1]
            )
            return dict(
                {
                    "symbol": symbol,
                    "slope": annualized_slope,
                    "r": r_value,
                    "score": score,
                    "volatility": volatility,
                },
            )
        else:
            return None

    async def calc_momentum(self) -> None:
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
                executor.submit(self.calc_symbol_momentum, symbol): symbol
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
        print(self.portfolio)

    async def calc_negative_momentum(self) -> None:
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
                executor.submit(
                    self.calc_symbol_negative_momentum, symbol
                ): symbol
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

    def apply_filters_symbol(self, symbol: str) -> bool:
        indicator_calculator = StockDataFrame(self.data_bars[symbol])
        sma_100 = indicator_calculator["close_100_sma"]
        if self.data_bars[symbol].close[-1] < sma_100[-1]:
            return False

        if (
            self.portfolio.loc[
                self.portfolio.symbol == symbol
            ].volatility.values
            < 1 - self.volatility_threshold
        ):
            return False

        return True
        """
            # filter stocks moving > 15% in last 90 days
            last = self.data_bars[symbol].close[
                -1
            ]  # self.data_bars[row.symbol].close[-90:].max()
            start = self.data_bars[symbol].close[-90]
            return last / start <= 1.50
        """

    def apply_filters_symbol_for_short(self, symbol: str) -> bool:
        indicator_calculator = StockDataFrame(self.data_bars[symbol])
        sma_100 = indicator_calculator["close_100_sma"]
        if self.data_bars[symbol].close[-1] >= sma_100[-1]:
            return False

        if (
            self.portfolio.loc[
                self.portfolio.symbol == symbol
            ].volatility.values
            < 1 - self.volatility_threshold
        ):
            return False

        return True

    async def apply_filters(self) -> None:
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
                executor.submit(self.apply_filters_symbol, symbol): symbol
                for symbol in symbols
            }
            for future in concurrent.futures.as_completed(futures):
                filter = future.result()
                if filter:
                    pass_filter.append(futures[future])

        self.portfolio = self.portfolio[
            self.portfolio.symbol.isin(pass_filter)
        ]
        tlog(
            f"filters removed {pre_filter_len-len(self.portfolio)} new portfolio length {len(self.portfolio)}"
        )
        self.portfolio = self.portfolio.head(self.stock_count)
        tlog(
            f"taking top {self.stock_count} by score, new portfolio length {len(self.portfolio)}"
        )

    async def apply_filters_for_short(self) -> None:
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
                executor.submit(
                    self.apply_filters_symbol_for_short, symbol
                ): symbol
                for symbol in symbols
            }
            for future in concurrent.futures.as_completed(futures):
                filter = future.result()
                if filter:
                    pass_filter.append(futures[future])

        self.portfolio = self.portfolio[
            self.portfolio.symbol.isin(pass_filter)
        ]
        tlog(
            f"filters removed {pre_filter_len-len(self.portfolio)} new portfolio length {len(self.portfolio)}"
        )
        self.portfolio = self.portfolio.tail(self.stock_count)
        tlog(
            f"taking top {self.stock_count} by score, new portfolio length {len(self.portfolio)}"
        )

    async def calc_balance(self) -> None:
        tlog(
            f"portfolio size {self.portfolio_size} w/ length {len(self.portfolio)}"
        )

        sum_vol = self.portfolio.volatility.sum()
        for _, row in self.portfolio.iterrows():
            qty = int(
                self.portfolio_size
                * row.volatility
                / sum_vol
                / self.data_bars[row.symbol].close[-1]
            )
            if qty == 0:
                qty = 1
            self.portfolio.loc[
                self.portfolio.symbol == row.symbol, "qty"
            ] = qty
            self.portfolio.loc[self.portfolio.symbol == row.symbol, "est"] = (
                qty * self.data_bars[row.symbol].close[-1]
            )

        if len(self.portfolio) > 0:
            self.portfolio = self.portfolio.loc[self.portfolio.qty > 0]
            self.portfolio["accumulative"] = self.portfolio.est.cumsum()

    async def run(self, now: datetime) -> df:
        await self.load_data(self.symbols, now)
        await self.calc_momentum()
        await self.apply_filters()
        await self.calc_balance()
        return self.portfolio

    async def run_short(self, now: datetime) -> df:
        await self.load_data(self.symbols, now)
        await self.calc_negative_momentum()
        await self.apply_filters_for_short()
        await self.calc_balance()
        return self.portfolio
