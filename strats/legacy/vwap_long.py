from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import alpaca_trade_api as tradeapi
import talib
from deprecated import deprecated
from google.cloud import error_reporting
from liualgotrader.common import config
from liualgotrader.common.tlog import tlog
from liualgotrader.common.trading_data import (buy_indicators,
                                               last_used_strategy,
                                               latest_cost_basis, open_orders,
                                               sell_indicators, stop_prices,
                                               target_prices)
from liualgotrader.fincalcs.support_resistance import find_stop
from liualgotrader.fincalcs.vwap import add_daily_vwap
from liualgotrader.strategies.base import Strategy
from pandas import DataFrame as df
from pandas import Series
from pandas import Timestamp as ts
from pandas import concat
from tabulate import tabulate


@deprecated()
class VWAPLong(Strategy):
    name = "vwap_long"

    def __init__(self, batch_id: str, ref_run_id: int = None):
        super().__init__(
            name=self.name, batch_id=batch_id, ref_run_id=ref_run_id
        )

    async def buy_callback(self, symbol: str, price: float, qty: int) -> None:
        latest_cost_basis[symbol] = price

    async def sell_callback(self, symbol: str, price: float, qty: int) -> None:
        latest_cost_basis[symbol] = price

    async def create(self) -> None:
        await super().create()
        tlog(f"strategy {self.name} created")

    # async def is_buy_time(self, now: datetime):
    #    return (
    #        True
    #        if 45
    #        > (now - config.market_open).seconds // 60
    #        > config.market_cool_down_minutes
    #        or config.bypass_market_schedule
    #        else False
    #    )

    async def run(
        self,
        symbol: str,
        position: int,
        minute_history: df,
        now: datetime,
        portfolio_value: float = None,
        trading_api: tradeapi = None,
        debug: bool = False,
        backtesting: bool = False,
    ) -> Tuple[bool, Dict]:
        data = minute_history.iloc[-1]
        prev_minute = minute_history.iloc[-2]
        if await self.is_buy_time(now) and not position:
            # Check for buy signals
            lbound = config.market_open
            ubound = lbound + timedelta(minutes=15)
            try:
                high_15m = minute_history[lbound:ubound]["high"].max()  # type: ignore

                if data.vwap < high_15m:
                    return False, {}
            except Exception as e:
                # Because we're aggregating on the fly, sometimes the datetime
                # index can get messy until it's healed by the minute bars
                tlog(
                    f"[{self.name}] error aggregation {e} - maybe should use nearest?"
                )
                return False, {}

            back_time = ts(config.market_open)
            back_time_index = minute_history["close"].index.get_loc(
                back_time, method="nearest"
            )
            close = (
                minute_history["close"][back_time_index:-1]
                .dropna()
                .between_time("9:30", "16:00")
                .resample("5min")
                .last()
            ).dropna()
            open = (
                minute_history["open"][back_time_index:-1]
                .dropna()
                .between_time("9:30", "16:00")
                .resample("5min")
                .first()
            ).dropna()
            high = (
                minute_history["high"][back_time_index:-1]
                .dropna()
                .between_time("9:30", "16:00")
                .resample("5min")
                .max()
            ).dropna()
            low = (
                minute_history["low"][back_time_index:-1]
                .dropna()
                .between_time("9:30", "16:00")
                .resample("5min")
                .min()
            ).dropna()
            volume = (
                minute_history["volume"][back_time_index:-1]
                .dropna()
                .between_time("9:30", "16:00")
                .resample("5min")
                .sum()
            ).dropna()

            _df = concat(
                [
                    open.rename("open"),
                    high.rename("high"),
                    low.rename("low"),
                    close.rename("close"),
                    volume.rename("volume"),
                ],
                axis=1,
            )

            if not add_daily_vwap(_df):
                tlog(f"[{now}]failed add_daily_vwap")
                return False, {}

            if debug:
                tlog(
                    f"\n[{now}]{symbol} {tabulate(_df[-10:], headers='keys', tablefmt='psql')}"
                )
            vwap_series = _df["average"]

            if (
                # data.vwap > close_series[-1] > close_series[-2]
                # and round(data.average, 2) > round(vwap_series[-1], 2)
                # and data.vwap > data.average
                # and
                data.low > data.average
                and close[-1] > vwap_series[-1] > vwap_series[-2] > low[-2]
                and close[-1] > high[-2]
                and prev_minute.close > prev_minute.open
                and data.close > data.open
                and low[-2] < vwap_series[-2] - 0.2
            ):
                stop_price = find_stop(
                    data.close if not data.vwap else data.vwap,
                    minute_history,
                    now,
                )
                # upperband, middleband, lowerband = BBANDS(
                #    minute_history["close"], timeperiod=20
                # )

                # stop_price = min(
                #    prev_minute.close,
                #    data.average - 0.01,
                #    lowerband[-1] - 0.03,
                # )
                target = (
                    3 * (data.close - stop_price) + data.close
                )  # upperband[-1]

                # if target - stop_price < 0.05:
                #    tlog(
                #        f"{symbol} target price {target} too close to stop price {stop_price}"
                #    )
                #    return False, {}
                # if target - data.close < 0.05:
                #    tlog(
                #        f"{symbol} target price {target} too close to close price {data.close}"
                #    )
                #    return False, {}

                stop_prices[symbol] = stop_price
                target_prices[symbol] = target

                patterns: Dict[ts, Dict[int, List[str]]] = {}
                pattern_functions = talib.get_function_groups()[
                    "Pattern Recognition"
                ]
                for pattern in pattern_functions:
                    pattern_value = getattr(talib, pattern)(
                        open, high, low, close
                    )
                    result = pattern_value.to_numpy().nonzero()
                    if result[0].size > 0:
                        for timestamp, value in pattern_value.iloc[
                            result
                        ].items():
                            t = ts(timestamp)
                            if t not in patterns:
                                patterns[t] = {}
                            if value not in patterns[t]:
                                patterns[t][value] = [pattern]
                            else:
                                patterns[t][value].append(pattern)

                tlog(f"{symbol} found conditions for VWAP strategy now:{now}")
                candle_s = Series(patterns)
                candle_s = candle_s.sort_index()

                tlog(f"{symbol} 5-min VWAP {vwap_series}")
                tlog(f"{symbol} 5-min close values {close}")
                tlog(f"{symbol} {candle_s}")
                tlog(
                    f"\n{tabulate(minute_history[-10:], headers='keys', tablefmt='psql')}"
                )

                if candle_s.size > 0 and -100 in candle_s[-1]:
                    tlog(
                        f"{symbol} Bullish pattern does not exists -> should skip"
                    )
                    # return False, {}

                if portfolio_value is None:
                    if trading_api:
                        portfolio_value = float(
                            trading_api.get_account().portfolio_value
                        )
                    else:
                        raise Exception(
                            "VWAPLong.run(): both portfolio_value and trading_api can't be None"
                        )

                shares_to_buy = (
                    portfolio_value
                    * 20.0
                    * config.risk
                    // data.close
                    # // (data.close - stop_prices[symbol])
                )
                print(
                    f"shares to buy {shares_to_buy} {data.close} {stop_prices[symbol]}"
                )
                if not shares_to_buy:
                    shares_to_buy = 1
                shares_to_buy -= position

                if shares_to_buy > 0:
                    tlog(
                        f"[{self.name}] Submitting buy for {shares_to_buy} shares of {symbol} at {data.close} target {target_prices[symbol]} stop {stop_prices[symbol]}"
                    )
                    buy_indicators[symbol] = {
                        # bbrand_lower": lowerband[-5:].tolist(),
                        # "bbrand_middle": middleband[-5:].tolist(),
                        # "bbrand_upper": upperband[-5:].tolist(),
                        "average": round(data.average, 2),
                        "vwap": round(data.vwap, 2),
                        "patterns": candle_s.to_json(),
                    }

                    return (
                        True,
                        {
                            "side": "buy",
                            "qty": str(shares_to_buy),
                            "type": "limit",
                            "limit_price": str(data.close),
                        },
                    )
            elif debug:
                tlog(f"[{now}]{symbol} failed vwap strategy")
                if not (data.low > data.average):
                    tlog(
                        f"[{now}]{symbol} failed data.low {data.low} > data.average {data.average}"
                    )
                if not (
                    close[-1] > vwap_series[-1] > vwap_series[-2] > low[-2]
                ):
                    tlog(
                        f"[{now}]{symbol} failed close[-1] {close[-1]} > vwap_series[-1] {vwap_series[-1]} > vwap_series[-2]{ vwap_series[-2]} > low[-2] {low[-2]}"
                    )
                if not (prev_minute.close > prev_minute.open):
                    tlog(
                        f"[{now}]{symbol} failed prev_minute.close {prev_minute.close} > prev_minute.open {prev_minute.open}"
                    )
                if not (close[-1] > high[-2]):
                    tlog(
                        f"[{now}]{symbol} failed close[-1] {close[-1]} > high[-2] {high[-2]}"
                    )
                if not (data.close > data.open):
                    tlog(
                        f"[{now}]{symbol} failed data.close {data.close} > data.open {data.open}"
                    )
                if not low[-2] < vwap_series[-2] - 0.2:
                    tlog(
                        f"[{now}]{symbol} failed low[-2] {low[-2]} < vwap_series[-2] {vwap_series[-2] } - 0.2"
                    )

        elif (
            await super().is_sell_time(now)
            and position > 0
            and symbol in latest_cost_basis
            and last_used_strategy[symbol].name == self.name
        ):
            if open_orders.get(symbol) is not None:
                tlog(f"vwap_long: open order for {symbol} exists, skipping")
                return False, {}

            if data.vwap <= data.average - 0.02:
                sell_indicators[symbol] = {
                    "reason": "below VWAP",
                    "average": data.average,
                    "vwap": data.vwap,
                }
                return (
                    True,
                    {"side": "sell", "qty": str(position), "type": "market"},
                )

        return False, {}


"""
            elif doji(data.open, data.close, data.high, data.low):
                sell_indicators[symbol] = {
                    "reason": "doji",
                    "average": data.average,
                    "vwap": data.vwap,
                }
                return (
                    True,
                    {"side": "sell", "qty": str(position), "type": "market",},
                )
"""
