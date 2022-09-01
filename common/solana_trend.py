"""
Adopted from Andreas F Clenow's "Trend Following" method(s). 

Further readings:

https://www.followingthetrend.com/stocks-on-the-move/ 
https://www.followingthetrend.com/trading-evolved/

"""
import base64
import concurrent.futures
import json
import math
import struct
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from liualgotrader.common import config
from liualgotrader.common.data_loader import DataLoader  # type: ignore
from liualgotrader.common.market_data import get_trading_day
from liualgotrader.common.tlog import tlog
from liualgotrader.common.types import DataConnectorType, TimeScale
from liualgotrader.data.data_factory import data_loader_factory
from liualgotrader.miners.base import Miner
from liualgotrader.trading.base import Trader
from pandas import DataFrame as df
from scipy.stats import linregress
from solana.keypair import Keypair
from solana.publickey import PublicKey
from solana.rpc.async_api import AsyncClient
from solana.rpc.core import RPCException
from solana.system_program import (CreateAccountWithSeedParams,
                                   create_account_with_seed)
from solana.transaction import AccountMeta, Transaction, TransactionInstruction
from stockstats import StockDataFrame
from tabulate import tabulate

solana_network: str = "https://api.devnet.solana.com"
programId: str = "64ZdvpvU73ig1NVd36xNGqpy5JyAN2kCnVoF7M4wJ53e"
wallet_filename = "/Users/amichayoren/.config/solana/id.json"


def load_wallet(wallet_filename: str) -> Keypair:
    with open(wallet_filename) as f:
        data = json.load(f)

        return Keypair.from_secret_key(bytes(data))


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

    async def load_data(
        self, symbols: List[str], now: datetime
    ) -> Dict[str, pd.DataFrame]:
        tlog(f"Data loading started for {now.date()}")
        t0 = time.time()
        start = await get_trading_day(now=now.date(), offset=200)

        scale: TimeScale = TimeScale.day
        connector: DataConnectorType = config.data_connector
        data_api = data_loader_factory(connector)
        data = data_api.get_symbols_data(
            symbols=symbols, start=start, end=now.date(), scale=scale
        )

        t1 = time.time()
        tlog(f"Data loading completed, loaded data in {t1-t0} seconds")
        return data

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

    async def _set_response_account(
        self, client: AsyncClient, payer: Keypair, program_publicKey: PublicKey
    ) -> PublicKey:
        response_size = 4

        rent_lamports = (
            await client.get_minimum_balance_for_rent_exemption(response_size)
        )["result"]

        response_key = PublicKey.create_with_seed(
            payer.public_key, "hello5", program_publicKey
        )

        instruction = create_account_with_seed(
            CreateAccountWithSeedParams(
                from_pubkey=payer.public_key,
                new_account_pubkey=response_key,
                base_pubkey=payer.public_key,
                seed="hello5",
                lamports=rent_lamports,
                space=response_size,
                program_id=program_publicKey,
            )
        )
        trans = Transaction().add(instruction)
        try:
            result = await client.send_transaction(trans, payer)
        except RPCException:
            print("account already exists, skipping")

        return response_key

    def _symbol_to_bytes(
        self, data: Dict[str, pd.DataFrame], symbol: str
    ) -> bytes:
        return bytes(int(x) for x in data[symbol].close.to_list())

    async def _parse_response(
        self, client: AsyncClient, response_key: PublicKey
    ) -> float:
        base64_result = (await client.get_account_info(response_key))[
            "result"
        ]["value"]["data"]
        return struct.unpack("f", base64.b64decode(base64_result[0]))[
            0
        ]  # type ignore

    async def _get_score(
        self,
        client: AsyncClient,
        program_key: PublicKey,
        response_key: PublicKey,
        payer: Keypair,
        data: Dict[str, pd.DataFrame],
        symbol: str,
    ) -> float:
        payload_to_contract: bytes = self._symbol_to_bytes(data, symbol)

        instruction = TransactionInstruction(
            keys=[
                AccountMeta(
                    pubkey=response_key,
                    is_signer=False,
                    is_writable=True,
                ),
            ],
            program_id=program_key,
            data=payload_to_contract,
        )
        recent_blockhash = (await client.get_recent_blockhash())["result"][
            "value"
        ]["blockhash"]
        trans = Transaction(
            recent_blockhash=recent_blockhash, fee_payer=payer.public_key
        ).add(instruction)

        trans_result = await client.send_transaction(trans, payer)
        await client.confirm_transaction(trans_result["result"], "confirmed")

        return await self._parse_response(client, response_key)

    async def calc_momentum(
        self, data: Dict[str, pd.DataFrame], now: datetime
    ) -> None:
        tlog("Trend ranking calculation started")
        symbols = self.symbols

        async with AsyncClient(solana_network) as client:
            res = await client.is_connected()
            print(f"Connectivity to {solana_network}:{res}")

            program_key = PublicKey(programId)
            payer = load_wallet(wallet_filename)

            current_balance = (await client.get_balance(payer.public_key))[
                "result"
            ]["value"]
            tlog(f"SOLANA Wallet Balance:{current_balance} SOL")

            response_key = await self._set_response_account(
                client, payer, program_key
            )
            score: float = await self._get_score(
                client=client,
                payer=payer,
                program_key=program_key,
                response_key=response_key,
                data=data,
                symbol="LW",
            )

            tlog(f"Score for 'LW': {score}")

        raise Exception()

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
        data = await self.load_data(self.symbols, now)
        await self.calc_momentum(data, now)
        await self.apply_filters(now)
        await self.calc_balance(now)
        return self.portfolio

    async def run_short(self, now: datetime, carrier=None) -> df:
        await self.load_data(self.symbols, now)
        await self.calc_negative_momentum()
        await self.apply_filters_for_short()
        await self.calc_balance(now)
        return self.portfolio
