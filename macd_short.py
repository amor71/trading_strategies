import asyncio
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Tuple

import alpaca_trade_api as tradeapi
from liualgotrader.common import config
from liualgotrader.common.tlog import tlog
from liualgotrader.common.trading_data import (buy_indicators, cool_down,
                                               last_used_strategy,
                                               latest_cost_basis, open_orders,
                                               sell_indicators, stop_prices,
                                               target_prices)
from liualgotrader.strategies.base import Strategy, StrategyType
from pandas import DataFrame as df
from talib import MACD


class MACDTrend(Enum):
    UP = 1
    DOWN = 2
    SIDEWAYS = 3


class MACDShort(Strategy):
    name = "macd_short"

    def __init__(
        self,
        batch_id: str,
        schedule: List[Dict],
        ref_run_id: int = None,
        check_patterns: bool = False,
    ):
        self.check_patterns = check_patterns
        super().__init__(
            name=self.name,
            type=StrategyType.DAY_TRADE,
            batch_id=batch_id,
            ref_run_id=ref_run_id,
            schedule=schedule,
        )

    async def buy_callback(self, symbol: str, price: float, qty: int) -> None:
        pass

    async def sell_callback(self, symbol: str, price: float, qty: int) -> None:
        latest_cost_basis[symbol] = price

    async def create(self) -> None:
        await super().create()
        tlog(f"strategy {self.name} created")

    async def run(
        self,
        symbol: str,
        shortable: bool,
        position: int,
        minute_history: df,
        now: datetime,
        portfolio_value: float = None,
        trading_api: tradeapi = None,
        debug: bool = False,
        backtesting: bool = False,
    ) -> Tuple[bool, Dict]:
        data = minute_history.iloc[-1]

        if (
            await super().is_buy_time(now)
            and not position
            and not open_orders.get(symbol, None)
            and data.volume > 1000
        ):
            close = (
                minute_history["close"].dropna().between_time("9:30", "16:00")
            )

            # Check for buy signals
            lbound = config.market_open.replace(second=0, microsecond=0)
            ubound = lbound + timedelta(minutes=15)
            try:
                high_15m = minute_history[lbound:ubound]["high"].max()  # type: ignore
                min_15m = minute_history[lbound:ubound]["low"].min()  # type: ignore
            except Exception as e:
                tlog(
                    f"{symbol}[{now}] failed to aggregate {lbound}:{ubound} {minute_history}"
                )
                return False, {}

            if (high_15m - min_15m) / min_15m > 0.10:
                return False, {}

            macds = MACD(close, 13, 21)
            macd = macds[0]
            macd_signal = macds[1]
            macd_hist = macds[2]

            macd_above_signal = (
                macd[-2] > macd_signal[-2] and macd[-3] > macd_signal[-3]
            )
            hist_trending_down = macd_hist[-1] < macd_hist[-2] < macd_hist[-3]

            macd_treding_down = macd[-1] < macd[-2] < macd[-3]
            to_buy_short = False
            reason: List[str] = []
            if (
                macd_above_signal
                and hist_trending_down
                and macd_treding_down
                and macd[-2] > 0
                and macd[-1] > 0
                and data.close < data.open
                and data.vwap < data.open
            ):
                to_buy_short = True
                reason.append(
                    "MACD positive & above signal, MACD hist trending down, MACD trended down"
                )

            if to_buy_short:
                stop_price = data.close * 1.03
                target_price = data.average * 0.90

                target_prices[symbol] = target_price
                stop_prices[symbol] = stop_price

                if portfolio_value is None:
                    if trading_api:

                        retry = 3
                        while retry > 0:
                            try:
                                portfolio_value = float(
                                    trading_api.get_account().portfolio_value
                                )
                                break
                            except ConnectionError as e:
                                tlog(
                                    f"[{symbol}][{now}[Error] get_account() failed w/ {e}, retrying {retry} more times"
                                )
                                await asyncio.sleep(0)
                                retry -= 1

                        if not portfolio_value:
                            tlog(
                                "f[{symbol}][{now}[Error] failed to get portfolio_value"
                            )
                            return False, {}
                    else:
                        raise Exception(
                            f"{self.name}: both portfolio_value and trading_api can't be None"
                        )

                shares_to_buy = (
                    portfolio_value * config.risk * 10 // data.close
                )
                if not shares_to_buy:
                    shares_to_buy = 1

                buy_price = data.close
                tlog(
                    f"[{self.name}][{now}] Submitting buy short for {-shares_to_buy} shares of {symbol} at {buy_price} target {target_prices[symbol]} stop {stop_prices[symbol]}"
                )

                sell_indicators[symbol] = {
                    "reason": reason,
                    "macd": macd[-4:].tolist(),
                    "macd_signal": macd_signal[-4:].tolist(),
                    "hist_signal": macd_hist[-4:].tolist(),
                }

                return (
                    True,
                    {
                        "side": "sell",
                        "qty": str(-shares_to_buy),
                        "type": "market",
                    },
                )
        elif (
            await super().is_sell_time(now)
            and position
            and symbol in latest_cost_basis
            and last_used_strategy[symbol].name == self.name
            and not open_orders.get(symbol)
            and data.volume > 1000
        ):
            close = (
                minute_history["close"].dropna().between_time("9:30", "16:00")
            )
            macds = MACD(close, 13, 21)
            macd = macds[0]
            macd_signal = macds[1]
            macd_hist = macds[2]

            hist_change_trend = macd_hist[-2] > macd_hist[-3]
            below_signal = macd[-1] < macd_signal[-1]
            crossing_above_signal = (
                macd[-1] > macd_signal[-1] and macd_signal[-2] >= macd[-2]
            )
            to_sell = False
            reason = []
            if crossing_above_signal:
                to_sell = True
                reason.append("MACD crossing above signal")
            if (
                below_signal
                and hist_change_trend
                and data.close < latest_cost_basis[symbol]
                and macd[-1] > macd[-2]
            ):
                to_sell = True
                reason.append("reversing direction")
            elif data.close >= stop_prices[symbol]:
                to_sell = True
                reason.append("stopped")
            elif data.close <= target_prices[symbol]:
                reason.append("target reached")

            if to_sell:
                buy_indicators[symbol] = {
                    "close_5m": close[-5:].tolist(),
                    "reason": reason,
                }

                tlog(
                    f"[{self.name}][{now}] Submitting sell short for {position} shares of {symbol} at market {data.close} with reason:{reason}"
                )
                return (
                    True,
                    {
                        "side": "buy",
                        "qty": str(-position),
                        "type": "market",
                    },
                )

        return False, {}
