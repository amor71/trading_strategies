import asyncio
import sys
import uuid
from datetime import datetime, time, timedelta
from typing import Dict, List, Tuple

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
from liualgotrader.common.data_loader import DataLoader
from liualgotrader.common.tlog import tlog
from liualgotrader.common.trading_data import (buy_indicators, buy_time,
                                               cool_down, last_used_strategy,
                                               latest_cost_basis,
                                               latest_scalp_basis, open_orders,
                                               sell_indicators, stop_prices,
                                               target_prices)
from liualgotrader.common.types import AssetType
from liualgotrader.fincalcs.support_resistance import find_stop
from liualgotrader.models.accounts import Accounts
from liualgotrader.models.portfolio import Portfolio
from liualgotrader.strategies.base import Strategy, StrategyType
from pandas import DataFrame as df
from pytz import timezone
from talib import BBANDS, MACD, RSI, MA_Type


class Crypto(Strategy):
    def __init__(
        self,
        batch_id: str,
        data_loader: DataLoader,
        portfolio_id: str,
        ref_run_id: int = None,
    ):
        self.name = type(self).__name__
        self.portfolio_id = portfolio_id
        self.asset_type: AssetType
        self.last_buy_price = 0.0
        super().__init__(
            name=type(self).__name__,
            type=StrategyType.SWING,
            batch_id=batch_id,
            ref_run_id=ref_run_id,
            schedule=[],
            data_loader=data_loader,
            fractional=True,
        )

    async def buy_callback(
        self,
        symbol: str,
        price: float,
        qty: float,
        now: datetime = None,
        trade_fee: float = 0.0,
    ) -> None:
        if self.account_id:
            self.last_buy_price = price
            amount_to_withdraw = price * qty
            if not await Accounts.check_if_enough_balance_to_withdraw(
                self.account_id, amount_to_withdraw + trade_fee
            ):
                raise AssertionError(
                    f"account {self.account_id} does not have enough balance for transaction"
                )

            await Accounts.add_transaction(
                account_id=self.account_id,
                amount=-amount_to_withdraw,
                tstamp=now,
            )
            await Accounts.add_transaction(
                account_id=self.account_id, amount=-trade_fee, tstamp=now
            )
            print(
                "buy",
                -price * qty,
                "fee",
                trade_fee,
                "balance post buy",
                await Accounts.get_balance(self.account_id),
            )

    async def sell_callback(
        self,
        symbol: str,
        price: float,
        qty: float,
        now: datetime = None,
        trade_fee: float = 0.0,
    ) -> None:
        if self.account_id:
            if not await Accounts.check_if_enough_balance_to_withdraw(
                self.account_id, trade_fee
            ):
                raise AssertionError(
                    f"account {self.account_id} does not have enough balance for transaction"
                )

            await Accounts.add_transaction(
                account_id=self.account_id, amount=-trade_fee, tstamp=now
            )
            await Accounts.add_transaction(
                account_id=self.account_id, amount=price * qty, tstamp=now
            )
            print(
                "sell",
                price * qty,
                "fee",
                trade_fee,
                "balance post sell",
                await Accounts.get_balance(self.account_id),
            )
            self.last_buy_price = 0.0

    async def create(self) -> bool:
        if not await super().create():
            return False

        tlog(f"strategy {self.name} created")
        # try:
        #    await Portfolio.associate_batch_id_to_profile(
        #        portfolio_id=self.portfolio_id, batch_id=self.batch_id
        #    )
        # except Exception:
        #    tlog("Probably already associated...")
        #    return False

        portfolio = await Portfolio.load_by_portfolio_id(self.portfolio_id)
        self.account_id = portfolio.account_id
        self.portfolio_size = portfolio.portfolio_size
        self.asset_type = portfolio.asset_type

        return True

    async def is_buy_time(self, now: datetime):
        return (
            time(hour=14, minute=30) >= now.time() >= time(hour=9, minute=30)
            if self.asset_type == AssetType.US_EQUITIES
            else True
        )

    async def is_sell_time(self, now: datetime):
        return True

    def calc_close(self, symbol: str, data_loader: DataLoader, now: datetime):
        data = self.data_loader[symbol].close[
            now - timedelta(days=2) : now  # type:ignore
        ]
        return data.resample("15min").last()

    def calc_open(self, symbol: str, data_loader: DataLoader, now: datetime):
        data = self.data_loader[symbol].open[
            now - timedelta(days=2) : now  # type:ignore
        ]
        return data.resample("15min").last()

    async def handle_buy_side(
        self,
        symbols_position: Dict[str, float],
        data_loader: DataLoader,
        now: datetime,
        trade_fee_precentage: float,
    ) -> Dict[str, Dict]:
        actions = {}

        for symbol, position in symbols_position.items():
            if position != 0:
                continue

            # sma_50 = (
            #    data_loader[symbol]
            #    .close[now - timedelta(days=100) : now]  # type: ignore
            #    .resample("1D")
            #    .last()
            #    .rolling(50)
            #    .mean()
            #    .dropna()
            #    .iloc[-1]
            # )

            sma_20 = (
                data_loader[symbol]
                .close[now - timedelta(days=40) : now]  # type: ignore
                .resample("1D")
                .last()
                .rolling(20)
                .mean()
                .dropna()
                .iloc[-1]
            )

            current_price = data_loader[symbol].close[now]

            # if current_price < sma_50 and current_price < sma_20:
            #    continue

            # tlog(f"{symbol} -> {current_price}")
            resampled_close = self.calc_close(symbol, data_loader, now)
            resampled_open = self.calc_open(symbol, data_loader, now)
            bband = BBANDS(
                resampled_close,
                timeperiod=7,
                nbdevdn=1,
                nbdevup=1,
                matype=MA_Type.EMA,
            )
            yesterday_lower_band = bband[2][-2]
            today_lower_band = bband[2][-1]
            yesterday_close = resampled_close[-2]
            today_open = resampled_open[-1]

            if (
                yesterday_close < yesterday_lower_band
                and today_open > yesterday_close
                and current_price > today_lower_band
                and current_price > today_open
            ):
                yesterday_upper_band = bband[0][-2]
                if current_price > yesterday_upper_band:
                    return {}

                buy_indicators[symbol] = {
                    "lower_band": bband[2][-2:].tolist(),
                    "resampled_close": resampled_close[-2:].tolist(),
                    "resampled_open": resampled_open[-2:].tolist(),
                    "current_price": current_price,
                }
                shares_to_buy = await self.calc_qty(
                    current_price,
                    trade_fee_precentage,
                )
                tlog(
                    f"[{self.name}][{now}] Submitting buy for {shares_to_buy} shares of {symbol} at {current_price}"
                )
                tlog(f"indicators:{buy_indicators[symbol]}")
                actions[symbol] = {
                    "side": "buy",
                    "qty": str(shares_to_buy),
                    "type": "limit",
                    "limit_price": str(current_price),
                }

        return actions

    async def handle_sell_side(
        self,
        symbols_position: Dict[str, float],
        data_loader: DataLoader,
        now: datetime,
        trade_fee_precentage: float,
    ) -> Dict[str, Dict]:
        actions = {}

        for symbol, position in symbols_position.items():
            if position == 0:
                continue

            # sma_20 = (
            #    data_loader[symbol]
            #    .close[now - timedelta(days=40) : now]  # type: ignore
            #    .resample("1D")
            #    .last()
            #    .rolling(20)
            #    .mean()
            #    .dropna()
            #    .iloc[-1]
            # )
            current_price = data_loader[symbol].close[now]

            if False:  # current_price < self.last_buy_price:
                sell_indicators[symbol] = {
                    "last_buy_price": self.last_buy_price,
                    "current_price": current_price,
                }

                tlog(
                    f"[{self.name}][{now}] Submitting sell for {position} shares of {symbol} at market"
                )
                tlog(f"indicators:{sell_indicators[symbol]}")
                actions[symbol] = {
                    "side": "sell",
                    "qty": str(position),
                    "type": "limit",
                    "limit_price": str(current_price),
                }
                return actions

            resampled_close = self.calc_close(symbol, data_loader, now)
            bband = BBANDS(
                resampled_close,
                timeperiod=7,
                nbdevdn=1,
                nbdevup=1,
                matype=MA_Type.EMA,
            )

            today_upper_band = bband[0][-1]

            # print(
            #    f"\ncurrent_price > yesterday_upper_band : {current_price > yesterday_upper_band}({current_price} < {yesterday_upper_band})"
            # )

            if (
                current_price > today_upper_band
                and current_price > self.last_buy_price
            ):
                sell_indicators[symbol] = {
                    "upper_band": bband[0][-2:].tolist(),
                    "lower_band": bband[2][-2:].tolist(),
                    "current_price": current_price,
                }

                tlog(
                    f"[{self.name}][{now}] Submitting sell for {position} shares of {symbol} at market"
                )
                tlog(f"indicators:{sell_indicators[symbol]}")
                actions[symbol] = {
                    "side": "sell",
                    "qty": str(position),
                    "type": "limit",
                    "limit_price": str(current_price),
                }

        return actions

    async def should_run_all(self):
        return False

    async def run(
        self,
        symbol: str,
        shortable: bool,
        position: float,
        now: datetime,
        minute_history: pd.DataFrame,
        portfolio_value: float = None,
        debug: bool = False,
        backtesting: bool = False,
    ) -> Tuple[bool, Dict]:
        fee_buy_percentage: float = 0.0
        fee_sell_percentage: float = 0.0
        symbols_position = {symbol: position}
        actions = {}
        if await self.is_buy_time(now) and not open_orders:
            actions.update(
                await self.handle_buy_side(
                    symbols_position={symbol: position},
                    data_loader=self.data_loader,
                    now=now,
                    trade_fee_precentage=fee_buy_percentage / 100.0,
                )
            )

        if (
            await self.is_sell_time(now)
            and (
                len(symbols_position)
                or any(symbols_position[x] for x in symbols_position)
            )
            and not open_orders
        ):
            actions.update(
                await self.handle_sell_side(
                    symbols_position=symbols_position,
                    data_loader=self.data_loader,
                    now=now,
                    trade_fee_precentage=fee_sell_percentage / 100.0,
                )
            )

        return (True, actions[symbol]) if symbol in actions else (False, {})

    async def run_all(
        self,
        symbols_position: Dict[str, float],
        data_loader: DataLoader,
        now: datetime,
        portfolio_value: float = None,
        trading_api: tradeapi = None,
        debug: bool = False,
        backtesting: bool = False,
        fee_buy_percentage: float = 0.0,
        fee_sell_percentage: float = 0.0,
    ) -> Dict[str, Dict]:
        tlog("run_all here!")
        actions = {}
        if await self.is_buy_time(now) and not open_orders:
            actions.update(
                await self.handle_buy_side(
                    symbols_position=symbols_position,
                    data_loader=data_loader,
                    now=now,
                    trade_fee_precentage=fee_buy_percentage / 100.0,
                )
            )

        if (
            await self.is_sell_time(now)
            and (
                len(symbols_position)
                or any(symbols_position[x] for x in symbols_position)
            )
            and not open_orders
        ):
            actions.update(
                await self.handle_sell_side(
                    symbols_position=symbols_position,
                    data_loader=data_loader,
                    now=now,
                    trade_fee_precentage=fee_sell_percentage / 100.0,
                )
            )

        return actions
