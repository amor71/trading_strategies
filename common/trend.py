"""
Adopted from Andreas F Clenow's "Trend Following" method(s). 

Further readings:

https://www.followingthetrend.com/stocks-on-the-move/ 
https://www.followingthetrend.com/trading-evolved/

"""
import asyncio
import concurrent.futures
import json
import math
import sys
import time
import traceback
import uuid
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
from liualgotrader.common import config
from liualgotrader.common.data_loader import DataLoader  # type: ignore
from liualgotrader.common.market_data import get_trading_day
from liualgotrader.common.tlog import tlog
from liualgotrader.common.types import TimeScale
from liualgotrader.miners.base import Miner
from liualgotrader.models.portfolio import Portfolio as DBPortfolio
from liualgotrader.trading.base import Trader
from pandas import DataFrame as df
from scipy.stats import linregress
from stockstats import StockDataFrame
from tabulate import tabulate


class Trend:
    def __init__(
        self,
        symbols: List[str],
        portfolio_size: float,
        rank_days: int,
        stock_count: int,
        volatility_threshold: float,
        data_loader: DataLoader,
        trader: Trader,
        top: int,
        debug=False,
    ):
        try:
            self.rank_days = rank_days
            self.debug = debug
            self.portfolio_size = portfolio_size
            tlog(f"data_loader:{data_loader}")
            self.data_loader = DataLoader(TimeScale.day)
            self.symbols = symbols
            self.stock_count = stock_count
            self.volatility_threshold = volatility_threshold
            self.trader: Trader = trader
            self.top = top
        except Exception:
            raise ValueError(
                "[ERROR] Miner must receive all valid parameter(s)"
            )

        self.portfolio: df = df(columns=["symbol", "slope", "r", "score"])

    async def load_data(self, symbols: List[str], now: datetime) -> None:
        tlog(f"Data loading started for {now}")
        t0 = time.time()
        start = await get_trading_day(now=now, offset=200)

        print(f"{start}:{now}")
        self.data_loader.pre_fetch(symbols=symbols, end=now, start=start)
        t1 = time.time()

        tlog(
            f"Data loading completed, loaded data for {len(self.data_loader)} symbols in {t1-t0} seconds"
        )

    def calc_symbol_momentum(
        self, symbol: str, now: datetime
    ) -> Optional[Dict]:
        np.seterr(all="raise")
        try:
            if (
                len(self.data_loader[symbol].close[-self.rank_days : now])  # type: ignore
                < self.rank_days - 10
            ):
                tlog(
                    f"missing data for {symbol} only {len(self.data_loader[symbol].close[-self.rank_days:now])}"  # type: ignore
                )
                return None

            deltas = np.log(self.data_loader[symbol].close[-self.rank_days : now].tolist())  # type: ignore
        except Exception as e:
            tlog(f"np.log-> Exception {e} for {symbol}, {now}")  # type: ignore
            return None

        try:
            slope, _, r_value, _, _ = linregress(
                np.arange(len(deltas)), deltas
            )
        except Exception:
            tlog(
                f"linregress-> {symbol}, {now}, {self.data_loader[symbol].close[-self.rank_days : now]}"  # type: ignore
            )
            raise

        if slope > 0:
            annualized_slope = (np.power(np.exp(slope), 252) - 1) * 100
            score = annualized_slope * (r_value ** 2)
            volatility = (
                1
                - self.data_loader[symbol]
                .close[-self.rank_days : now]  # type: ignore
                .pct_change()
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
        d = self.data_loader[symbol]
        _df = df(d)
        deltas = np.log(_df.close[-self.rank_days :])  # type: ignore
        slope, _, r_value, _, _ = linregress(np.arange(len(deltas)), deltas)
        if slope < 0:
            annualized_slope = (np.power(np.exp(slope), 252) - 1) * 100
            score = annualized_slope * (r_value ** 2)
            volatility = (
                1
                - self.data_loader[symbol]
                .close[-self.rank_days : now]  # type: ignore
                .pct_change()
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

    async def calc_momentum(self, now: datetime) -> None:
        if not len(self.data_loader):
            raise ValueError(
                "calc_momentum() can't run without data. aborting"
            )

        tlog("Trend ranking calculation started")
        symbols = self.symbols

        l: List[Dict] = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the load operations and mark each future with its URL
            futures = {
                executor.submit(self.calc_symbol_momentum, symbol, now): symbol
                for symbol in symbols
            }
            l.extend(
                data
                for future in concurrent.futures.as_completed(futures)
                if (data := future.result())
            )

        self.portfolio = (
            df.from_records(l)
            .sort_values(by="score", ascending=False)
            .head(self.top)
        )
        tlog(
            f"Trend ranking calculation completed w/ {len(self.portfolio)} trending stocks"
        )
        print(self.portfolio)

    async def calc_negative_momentum(self) -> None:
        if not len(self.data_loader):
            raise ValueError(
                "calc_momentum() can't run without data. aborting"
            )

        tlog("Trend ranking calculation started")
        symbols = [
            symbol
            for symbol in self.data_loader.keys()
            if not self.data_loader[symbol].empty
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
            l.extend(
                data
                for future in concurrent.futures.as_completed(futures)
                if (data := future.result())
            )

        self.portfolio = (
            df.from_records(l)
            .sort_values(by="score", ascending=False)
            .head(self.top)
        )
        tlog(
            f"Trend ranking calculation completed w/ {len(self.portfolio)} trending stocks"
        )

    def apply_filters_symbol(self, symbol: str, now: datetime) -> bool:
        df = self.data_loader[symbol].symbol_data[:now]  # type:ignore
        indicator_calculator = StockDataFrame(df)
        sma_100 = indicator_calculator["close_100_sma"]
        if df.empty or df.close[-1] < sma_100[-1]:
            return False

        if (
            self.portfolio.loc[
                self.portfolio.symbol == symbol
            ].volatility.values
            < 1 - self.volatility_threshold
        ):
            return False

        return True

    def apply_filters_symbol_for_short(self, symbol: str) -> bool:
        indicator_calculator = StockDataFrame(self.data_loader[symbol])
        sma_100 = indicator_calculator["close_100_sma"]
        if self.data_loader[symbol].close[-1] >= sma_100[-1]:
            return False

        if (
            self.portfolio.loc[
                self.portfolio.symbol == symbol
            ].volatility.values
            < 1 - self.volatility_threshold
        ):
            return False

        return True

    async def apply_filters(self, now: datetime) -> None:
        tlog("Applying filters")

        pre_filter_len = len(self.portfolio)
        symbols = self.data_loader.keys()
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
            for symbol in self.data_loader.keys()
            if not self.data_loader[symbol].empty
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

    async def calc_balance(self, now: datetime) -> None:
        tlog(
            f"portfolio size {self.portfolio_size} w/ length {len(self.portfolio)}"
        )
        sum_vol = self.portfolio.volatility.sum()
        for _, row in self.portfolio.iterrows():
            df = self.data_loader[row.symbol].symbol_data[:now]  # type: ignore
            qty = round(
                float(
                    self.portfolio_size
                    * row.volatility
                    / sum_vol
                    / df.close[-1]
                ),
                1,
            )

            if not await self.trader.is_fractionable(row.symbol):
                qty = math.ceil(qty - 1.0)
                if qty <= 0:
                    continue

            self.portfolio.loc[
                self.portfolio.symbol == row.symbol, "qty"
            ] = qty
            self.portfolio.loc[self.portfolio.symbol == row.symbol, "est"] = (
                qty * df.close[-1]
            )

        if len(self.portfolio) > 0:
            self.portfolio = self.portfolio.loc[self.portfolio.qty > 0]
            self.portfolio["accumulative"] = self.portfolio.est.cumsum()

    async def run(self, now: datetime, carrier=None) -> df:
        await self.load_data(self.symbols, now)
        await self.calc_momentum(now)
        await self.apply_filters(now)
        await self.calc_balance(now)
        return self.portfolio

    async def run_short(self, now: datetime, carrier=None) -> df:
        await self.load_data(self.symbols, now)
        await self.calc_negative_momentum()
        await self.apply_filters_for_short()
        await self.calc_balance(now)
        return self.portfolio
