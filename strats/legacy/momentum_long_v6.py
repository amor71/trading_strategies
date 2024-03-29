import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import alpaca_trade_api as tradeapi
import numpy as np
from liualgotrader.common import config
from liualgotrader.common.data_loader import DataLoader
from liualgotrader.common.tlog import tlog
from liualgotrader.common.trading_data import (buy_indicators, buy_time,
                                               cool_down, last_used_strategy,
                                               latest_cost_basis,
                                               latest_scalp_basis, open_orders,
                                               sell_indicators, stop_prices,
                                               target_prices)
from liualgotrader.fincalcs.support_resistance import find_stop
from liualgotrader.strategies.base import Strategy, StrategyType
from pandas import DataFrame as df
from talib import BBANDS, MACD, RSI


class MomentumLongV6(Strategy):
    name = "momentum_long_v6"
    whipsawed: Dict = {}
    top_up: Dict = {}
    max15: Dict = {}

    def __init__(
        self,
        batch_id: str,
        schedule: List[Dict],
        ref_run_id: int = None,
        check_patterns: bool = False,
        data_loader: DataLoader = None,
    ):
        self.check_patterns = check_patterns
        super().__init__(
            name=self.name,
            type=StrategyType.DAY_TRADE,
            batch_id=batch_id,
            ref_run_id=ref_run_id,
            schedule=schedule,
            data_loader=data_loader,
        )

    async def buy_callback(self, symbol: str, price: float, qty: int) -> None:
        latest_scalp_basis[symbol] = latest_cost_basis[symbol] = price

    async def sell_callback(self, symbol: str, price: float, qty: int) -> None:
        latest_scalp_basis[symbol] = price

    async def create(self) -> bool:
        tlog(f"strategy {self.name} created")
        return await super().create()

    async def should_cool_down(self, symbol: str, now: datetime):
        if (
            symbol in cool_down
            and cool_down[symbol]
            and cool_down[symbol] >= now.replace(second=0, microsecond=0)  # type: ignore
        ):
            return True

        cool_down[symbol] = None
        return False

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

        morning_rush = (
            True if (now - config.market_open).seconds // 60 < 30 else False
        )

        if (
            await super().is_buy_time(now)
            and not position
            and not open_orders.get(symbol, None)
            and not await self.should_cool_down(symbol, now)
        ):
            # Check for buy signals
            if symbol not in self.max15:
                lbound = config.market_open.replace(second=0, microsecond=0)
                ubound = lbound + timedelta(minutes=15)
                try:
                    self.max15[symbol] = minute_history[lbound:ubound]["high"].max()  # type: ignore
                    # print(
                    #    f"max={self.max15[symbol]} {lbound}, {ubound}, {minute_history}"
                    # )
                except Exception as e:
                    tlog(
                        f"{symbol}[{now}] failed to aggregate {lbound}:{ubound} {minute_history}"
                    )
                    return False, {}

            if data.close > self.max15[symbol]:

                close = (
                    minute_history["close"]
                    .dropna()
                    .between_time("9:30", "16:00")
                )

                if (
                    data.vwap < data.open and data.vwap
                ) or data.close < data.open:
                    if debug:
                        tlog(f"[{self.name}][{now}] price action not positive")
                    return False, {}

                macds = MACD(close)
                macd = macds[0].round(3)

                daiy_max_macd = (
                    macd[
                        now.replace(  # type: ignore
                            hour=9, minute=30, second=0, microsecond=0
                        ) :
                    ]
                    .between_time("9:30", "16:00")
                    .max()
                )
                macd_signal = macds[1].round(3)
                macd_hist = macds[2].round(3)

                # print(macd)
                macd_trending = macd[-3] < macd[-2] < macd[-1]
                macd_above_signal = macd[-1] > macd_signal[-1]

                # print(f"{symbol} {data.close} {self.max15[symbol]} {macd_signal}")

                macd_upper_crossover = (
                    macd[-2] > macd_signal[-2] >= macd_signal[-3] > macd[-3]
                )
                macd_hist_trending = (
                    macd_hist[-4]
                    < macd_hist[-3]
                    < macd_hist[-2]
                    < macd_hist[-1]
                )

                to_buy = False
                reason = []
                if (
                    macd[-1] < 0
                    and macd_upper_crossover
                    and macd_trending
                    and macd_above_signal
                ):
                    to_buy = True
                    reason.append("MACD crossover")

                if (
                    macd_hist_trending
                    and macd_hist[-3] <= 0 < macd_hist[-2]
                    and macd[-1] < daiy_max_macd
                ):
                    reason.append("MACD histogram reversal")

                if macd[-2] > 0 >= macd[-3] and macd_trending:
                    macd2 = MACD(close, 40, 60)[0]
                    if macd2[-1] >= 0 and np.diff(macd2)[-1] >= 0:
                        if (
                            macd_hist_trending
                            and macd_hist[-3] <= 0 < macd_hist[-2]
                            and macd[-1] < daiy_max_macd
                        ):
                            to_buy = True
                            reason.append("MACD zero-cross")

                            if debug:
                                tlog(
                                    f"[{self.name}][{now}] slow macd confirmed trend"
                                )

                if to_buy:
                    print("to buy!")
                    # check RSI does not indicate overbought
                    rsi = RSI(close, 14)

                    if debug:
                        tlog(
                            f"[{self.name}][{now}] {symbol} RSI={round(rsi[-1], 2)}"
                        )

                    rsi_limit = 75
                    if rsi[-1] < rsi_limit:
                        if debug:
                            tlog(
                                f"[{self.name}][{now}] {symbol} RSI {round(rsi[-1], 2)} <= {rsi_limit}"
                            )
                    else:
                        tlog(
                            f"[{self.name}][{now}] {symbol} RSI over-bought, cool down for 5 min"
                        )
                        cool_down[symbol] = now.replace(
                            second=0, microsecond=0
                        ) + timedelta(minutes=5)

                        return False, {}

                    stop_price = find_stop(
                        data.close if not data.vwap else data.vwap,
                        minute_history,
                        now,
                    )
                    stop_price = (
                        stop_price - max(0.05, data.close * 0.02)
                        if stop_price
                        else data.close * config.default_stop
                    )
                    target_price = 3 * (data.close - stop_price) + data.close
                    target_prices[symbol] = target_price
                    stop_prices[symbol] = stop_price

                    if portfolio_value is None:
                        print("5555!")
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
                        portfolio_value
                        * config.risk
                        // (data.close - stop_prices[symbol])
                    )
                    if not shares_to_buy:
                        shares_to_buy = 1
                    shares_to_buy -= position
                    if shares_to_buy > 0:
                        self.whipsawed[symbol] = False

                        buy_price = max(data.close, data.vwap)
                        tlog(
                            f"[{self.name}][{now}] Submitting buy for {shares_to_buy} shares of {symbol} at {buy_price} target {target_prices[symbol]} stop {stop_prices[symbol]}"
                        )

                        buy_indicators[symbol] = {
                            "macd": macd[-5:].tolist(),
                            "macd_signal": macd_signal[-5:].tolist(),
                            "vwap": data.vwap,
                            "avg": data.average,
                            "reason": reason,
                            "rsi": rsi[-5:].tolist(),
                        }
                        self.top_up[symbol] = now
                        return (
                            True,
                            {
                                "side": "buy",
                                "qty": str(shares_to_buy),
                                "type": "limit",
                                "limit_price": str(buy_price),
                            }
                            if not morning_rush
                            else {
                                "side": "buy",
                                "qty": str(shares_to_buy),
                                "type": "market",
                            },
                        )
            else:
                if debug:
                    tlog(f"[{self.name}][{now}] {data.close} < 15min high ")

        elif (
            await super().is_buy_time(now)
            and position
            and not open_orders.get(symbol, None)
            and not await self.should_cool_down(symbol, now)
            and symbol in last_used_strategy
            and last_used_strategy[symbol].name == self.name
            and data.volume > 500
        ):
            if symbol not in self.top_up:
                return False, {}

            if (
                symbol not in latest_scalp_basis
                or latest_scalp_basis[symbol] > latest_cost_basis[symbol]
            ):
                return False, {}

            if not (
                (now - self.top_up[symbol]).total_seconds()
                > timedelta(minutes=10).total_seconds()
            ):
                return False, {}

            if (
                data.close > data.open
                and data.close > latest_scalp_basis[symbol]
            ):
                close = (
                    minute_history["close"]
                    .dropna()
                    .between_time("9:30", "16:00")
                )

                macds = MACD(close)
                macd = macds[0]
                macd_signal = macds[1]

                if not (
                    macd[-1] > macd_signal[-1]
                    and macd[-1] > macd[-2] > macd_signal[-2]
                ):
                    return False, {}

                rsi = RSI(close, 14)

                if not (rsi[-2] < rsi[-1] < 75):
                    return False, {}

                movement = (
                    data.close - latest_scalp_basis[symbol]
                ) / latest_scalp_basis[symbol]

                if movement < 0.005:
                    return False, {}

                shares_to_buy = int(position * 0.20)
                buy_price = max(data.close, data.vwap)
                reason = ["additional buy"]
                tlog(
                    f"[{self.name}][{now}] Submitting additional buy for {shares_to_buy} shares of {symbol} at {buy_price}"
                )

                self.top_up[symbol] = now

                buy_indicators[symbol] = {
                    "macd": macd[-5:].tolist(),
                    "macd_signal": macd_signal[-5:].tolist(),
                    "vwap": data.vwap,
                    "avg": data.average,
                    "reason": reason,
                    "rsi": rsi[-5:].tolist(),
                }

                return (
                    True,
                    {
                        "side": "buy",
                        "qty": str(shares_to_buy),
                        "type": "limit",
                        "limit_price": str(buy_price),
                    },
                )

        if (
            await super().is_sell_time(now)
            and position > 0
            and symbol in latest_cost_basis
            and last_used_strategy[symbol].name == self.name
            and not open_orders.get(symbol)
        ):
            if (
                not self.whipsawed.get(symbol, None)
                and data.close < latest_cost_basis[symbol] * 0.99
            ):
                self.whipsawed[symbol] = True

            serie = (
                minute_history["close"].dropna().between_time("9:30", "16:00")
            )

            if data.vwap:
                serie[-1] = data.vwap

            macds = MACD(
                serie,
                13,
                21,
            )

            macd = macds[0]
            macd_signal = macds[1]
            rsi = RSI(
                minute_history["close"].dropna().between_time("9:30", "16:00"),
                14,
            )

            if not latest_scalp_basis[symbol]:
                latest_scalp_basis[symbol] = latest_cost_basis[symbol] = 1.0
            movement = (
                data.close - latest_scalp_basis[symbol]
            ) / latest_scalp_basis[symbol]
            max_movement = (
                minute_history["close"][buy_time[symbol] :].max()
                - latest_scalp_basis[symbol]
            ) / latest_scalp_basis[symbol]
            macd_val = macd[-1]
            macd_signal_val = macd_signal[-1]

            round_factor = (
                2 if macd_val >= 0.01 or macd_signal_val >= 0.01 else 3
            )
            scalp_threshold = (
                target_prices[symbol] + latest_scalp_basis[symbol]
            ) / 2.0

            macd_below_signal = round(macd_val, round_factor) < round(
                macd_signal_val, round_factor
            )

            bail_out = (
                (
                    latest_scalp_basis[symbol] > latest_cost_basis[symbol]
                    or (max_movement > 0.02 and max_movement > movement)
                )
                and macd_below_signal
                and round(macd[-1], round_factor)
                < round(macd[-2], round_factor)
            )
            bail_on_whipsawed = (
                self.whipsawed.get(symbol, False)
                and movement > 0.01
                and macd_below_signal
                and round(macd[-1], round_factor)
                < round(macd[-2], round_factor)
            )
            scalp = movement > 0.04 or data.vwap > scalp_threshold
            below_cost_base = data.vwap < latest_cost_basis[symbol]

            rsi_limit = 79 if not morning_rush else 85
            to_sell = False
            partial_sell = False
            limit_sell = False
            sell_reasons = []
            if data.close <= stop_prices[symbol]:
                to_sell = True
                sell_reasons.append("stopped")
            elif (
                below_cost_base
                and round(macd_val, 2) < 0
                and rsi[-1] < rsi[-2]
                and round(macd[-1], round_factor)
                < round(macd[-2], round_factor)
                and data.vwap < 0.95 * data.average
            ):
                to_sell = True
                sell_reasons.append(
                    "below cost & macd negative & RSI trending down and too far from VWAP"
                )
            elif data.close >= target_prices[symbol] and macd[-1] <= 0:
                to_sell = True
                sell_reasons.append("above target & macd negative")
            elif (
                rsi[-1] >= rsi_limit and data.close > latest_cost_basis[symbol]
            ):
                to_sell = True
                sell_reasons.append("rsi max, cool-down for 5 minutes")
                cool_down[symbol] = now.replace(
                    second=0, microsecond=0
                ) + timedelta(minutes=5)
            elif bail_out:
                to_sell = True
                sell_reasons.append("bail")
            elif scalp:
                partial_sell = True
                to_sell = True
                sell_reasons.append("scale-out")
            elif bail_on_whipsawed:
                to_sell = True
                partial_sell = False
                limit_sell = True
                sell_reasons.append("bail post whipsawed")
            elif macd[-1] < macd_signal[-1] <= macd_signal[-2] < macd[-2]:
                sell_reasons.append("MACD cross signal from above")

            if to_sell:
                sell_indicators[symbol] = {
                    "rsi": rsi[-3:].tolist(),
                    "movement": movement,
                    "sell_macd": macd[-5:].tolist(),
                    "sell_macd_signal": macd_signal[-5:].tolist(),
                    "vwap": data.vwap,
                    "avg": data.average,
                    "reasons": " AND ".join(
                        [str(elem) for elem in sell_reasons]
                    ),
                }

                if not partial_sell:
                    if not limit_sell:
                        tlog(
                            f"[{self.name}][{now}] Submitting sell for {position} shares of {symbol} at market with reason:{sell_reasons}"
                        )
                        return (
                            True,
                            {
                                "side": "sell",
                                "qty": str(position),
                                "type": "market",
                            },
                        )
                    else:
                        tlog(
                            f"[{self.name}][{now}] Submitting sell for {position} shares of {symbol} at {data.close} with reason:{sell_reasons}"
                        )
                        return (
                            True,
                            {
                                "side": "sell",
                                "qty": str(position),
                                "type": "limit",
                                "limit_price": str(data.close),
                            },
                        )
                else:
                    qty = int(position / 2) if position > 1 else 1
                    tlog(
                        f"[{self.name}][{now}] Submitting sell for {str(qty)} shares of {symbol} at limit of {data.close }with reason:{sell_reasons}"
                    )
                    return (
                        True,
                        {
                            "side": "sell",
                            "qty": str(qty),
                            "type": "limit",
                            "limit_price": str(data.close),
                        },
                    )

        return False, {}
