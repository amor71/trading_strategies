"""
Adopted from Andreas F Clenow's "Trend Following" method(s). 

Further readings:

https://www.followingthetrend.com/stocks-on-the-move/ 
https://www.followingthetrend.com/trading-evolved/

"""
import asyncio
import base64
import concurrent.futures
import json
import math
import random
import struct
import time
from datetime import datetime
from typing import Dict, List, Optional
from unittest.util import strclass

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
from solana.exceptions import SolanaRpcException
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
vola_program_id: str = "3GXYHwCLxy1i7ymZVQcj2DaLeD3x9GUEcAyN9KTX8e8h"
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
        start = await get_trading_day(now=now.date(), offset=199)

        scale: TimeScale = TimeScale.day
        connector: DataConnectorType = config.data_connector
        data_api = data_loader_factory(connector)
        data = data_api.get_symbols_data(
            symbols=symbols, start=start, end=now.date(), scale=scale
        )

        t1 = time.time()
        tlog(f"Data loading completed, loaded data in {t1-t0} seconds")
        return data

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
            await self.retry(
                client.get_minimum_balance_for_rent_exemption, response_size
            )
        )["result"]

        response_key = PublicKey.create_with_seed(
            payer.public_key, "hello", program_publicKey
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
            result = await self.retry(client.send_transaction, trans, payer)
        except RPCException:
            None

        return response_key

    def _symbol_to_bytes(self, data: pd.DataFrame) -> bytes:
        raw_data = [int(x) for x in data.close.to_list()]

        prefix = 0
        if max(raw_data) >= 256:
            bytes_data: List = []

            for val in raw_data:
                bytes_data.append(val // 256)
                bytes_data.append(val % 256)

            raw_data = bytes_data
            prefix = 1

        return bytes([prefix] + [int(x) for x in raw_data])

    async def _parse_response(
        self, client: AsyncClient, response_key: PublicKey
    ) -> float:
        base64_result = (
            await self.retry(client.get_account_info, response_key)
        )["result"]["value"]["data"]
        return struct.unpack("f", base64.b64decode(base64_result[0]))[
            0
        ]  # type ignore

    async def _get_program_result(
        self,
        client: AsyncClient,
        program_key: PublicKey,
        response_key: PublicKey,
        payer: Keypair,
        data: pd.DataFrame,
        symbol: str,
    ) -> Optional[float]:

        payload_to_contract: bytes = self._symbol_to_bytes(data)
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
        recent_blockhash = (await self.retry(client.get_recent_blockhash))[
            "result"
        ]["value"]["blockhash"]
        trans = Transaction(
            recent_blockhash=recent_blockhash, fee_payer=payer.public_key
        ).add(instruction)

        try:
            trans_result = await self.retry(
                client.send_transaction, trans, payer
            )
        except RPCException as e:
            tlog(
                f"SOLANA ERROR {e} for {symbol} payload of {len(payload_to_contract)} bytes"
            )
            return None

        await self.retry(
            client.confirm_transaction, trans_result["result"], "confirmed"
        )

        return await self._parse_response(client, response_key)

    async def calc_symbol_momentum(
        self, data: pd.DataFrame, symbol: str
    ) -> Optional[Dict]:
        client = AsyncClient(solana_network)
        while not await client.is_connected():
            await asyncio.sleep(10)
            client = AsyncClient(solana_network)

        program_key = PublicKey(programId)
        vola_program_key = PublicKey(vola_program_id)
        payer = load_wallet(wallet_filename)

        response_key = await self._set_response_account(
            client, payer, program_key
        )
        score: Optional[float] = await self._get_program_result(
            client=client,
            payer=payer,
            program_key=program_key,
            response_key=response_key,
            data=data,
            symbol=symbol,
        )

        if score and score > 0.0:
            vola_response_key = await self._set_response_account(
                client, payer, vola_program_key
            )

            vola: Optional[float] = await self._get_program_result(
                client=client,
                payer=payer,
                program_key=vola_program_key,
                response_key=vola_response_key,
                data=data,
                symbol=symbol,
            )

            if vola:
                tlog(f"Score for {symbol}: {score}, vola {1.0 - vola}")
                return dict(
                    {
                        "symbol": symbol,
                        "score": score,
                        "volatility": 1.0 - vola,
                    }
                )

        return None

    async def retry(self, coro, *args):
        result = None
        while not result:
            try:
                return await coro(*args)
            except SolanaRpcException as e:
                tlog(f"Error {e} for {coro} retrying after short nap")
                await asyncio.sleep(10 + random.randint(0, 10))

        return result

    async def calc_momentum(
        self, data: Dict[str, pd.DataFrame], now: datetime
    ) -> None:
        tlog("Trend ranking calculation started")
        symbols = list(data.keys())

        chunk_size = 3
        chunked_symbols = [
            symbols[i : i + chunk_size]
            for i in range(0, len(symbols), chunk_size)
        ]

        l: List[Dict] = []

        for chunk in chunked_symbols:
            l += await asyncio.gather(
                *[
                    asyncio.create_task(
                        self.calc_symbol_momentum(data[symbol], symbol)
                    )
                    for symbol in chunk
                ]
            )

        l = [x for x in l if x]
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

    def apply_filters_symbol(
        self, df: pd.DataFrame, symbol: str, now: datetime
    ) -> bool:
        return True
        indicator_calculator = StockDataFrame(df)
        sma_100 = indicator_calculator["close_100_sma"]
        if df.empty or df.close[-1] < sma_100[-1]:
            return False

        return (
            self.portfolio.loc[self.portfolio.symbol == symbol].volatility
            >= 1 - self.volatility_threshold
        )

    def apply_filters_symbol_for_short(self, symbol: str) -> bool:
        indicator_calculator = StockDataFrame(self.data_loader[symbol])
        sma_100 = indicator_calculator["close_100_sma"]
        if self.data_loader[symbol].close[-1] >= sma_100[-1]:
            return False

        if (
            self.portfolio.loc[self.portfolio.symbol == symbol].volatility
            < 1 - self.volatility_threshold
        ):
            tlog(
                f"filter {self.portfolio.loc[self.portfolio.symbol == symbol].volatility} < {1 - self.volatility_threshold}"
            )
            return False

        return True

    async def apply_filters(
        self, data: Dict[str, pd.DataFrame], now: datetime
    ) -> None:
        tlog("Applying filters")

        pre_filter_len = len(self.portfolio)
        symbols = data.keys()
        pass_filter: list = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Start the load operations and mark each future with its URL
            futures = {
                executor.submit(
                    self.apply_filters_symbol, data[symbol], symbol, now
                ): symbol
                for symbol in symbols
            }
            for future in concurrent.futures.as_completed(futures):
                filter = future.result()
                print(futures[future], filter)
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

    async def calc_balance(
        self, data: Dict[str, pd.DataFrame], now: datetime
    ) -> None:
        tlog(
            f"portfolio size {self.portfolio_size} w/ length {len(self.portfolio)}"
        )
        sum_vol = self.portfolio.volatility.sum()
        for _, row in self.portfolio.iterrows():
            df = data[row.symbol].symbol_data[:now]  # type: ignore
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
        await self.apply_filters(data, now)
        await self.calc_balance(data, now)
        return self.portfolio
