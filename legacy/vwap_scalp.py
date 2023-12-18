from datetime import datetime, timedelta
from typing import Dict, Tuple

import alpaca_trade_api as tradeapi
from deprecated import deprecated
from liualgotrader.common import config
from liualgotrader.common.tlog import tlog
from liualgotrader.common.trading_data import (buy_indicators, down_cross,
                                               last_used_strategy,
                                               latest_cost_basis, open_orders,
                                               sell_indicators, stop_prices,
                                               target_prices)
from liualgotrader.fincalcs.support_resistance import find_stop
from liualgotrader.fincalcs.vwap import add_daily_vwap
from liualgotrader.strategies.base import Strategy, StrategyType
from pandas import DataFrame as df
from pandas import Timestamp as ts
from pandas import concat
from tabulate import tabulate


@deprecated()
class VWAPScalp(Strategy):
    name = "vwap_scalp"

    def __init__(self, batch_id: str, ref_run_id: int = None):
        super().__init__(
            name=self.name,
            batch_id=batch_id,
            type=StrategyType.DAY_TRADE,
            ref_run_id=ref_run_id,
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
        prev_2minutes = minute_history.iloc[-3]

        if await self.is_buy_time(now) and not position:
            # Check for buy signals

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

            if debug:
                tlog(
                    f"[{now}] {symbol} close:{round(data.close,2)} vwap:{round(vwap_series[-1],2)}"
                )

            if len(vwap_series) < 3:
                tlog(f"[{now}]{symbol}: missing vwap values {vwap_series}")
                return False, {}

            if close[-2] > vwap_series[-2] and close[-1] < vwap_series[-1]:
                down_cross[symbol] = vwap_series.index[-1].to_pydatetime()
                tlog(
                    f"[{now}] {symbol} down-crossing on 5-min bars at {down_cross[symbol]}"
                )
                return False, {}

            if (
                close[-2] > vwap_series[-2]
                and close[-3] < vwap_series[-3]
                and data.close > prev_minute.close
                and data.close > data.average
            ):
                if symbol not in down_cross:
                    tlog(
                        f"[{now}] {symbol} did not find download crossing in the past 15 min"
                    )
                    return False, {}
                if minute_history.index[-1].to_pydatetime() - down_cross[
                    symbol
                ] > timedelta(minutes=30):
                    tlog(
                        f"[{now}] {symbol} down-crossing too far {down_cross[symbol]} from now"
                    )
                    return False, {}
                stop_price = find_stop(
                    data.close if not data.vwap else data.vwap,
                    minute_history,
                    now,
                )
                target = data.close + 0.05

                stop_prices[symbol] = stop_price
                target_prices[symbol] = target

                tlog(
                    f"{symbol} found conditions for VWAP-Scalp strategy now:{now}"
                )

                tlog(
                    f"\n{tabulate(minute_history[-10:], headers='keys', tablefmt='psql')}"
                )

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
                if not shares_to_buy:
                    shares_to_buy = 1
                shares_to_buy -= position

                if shares_to_buy > 0:
                    tlog(
                        f"[{self.name}] Submitting buy for {shares_to_buy} shares of {symbol} at {data.close} target {target_prices[symbol]} stop {stop_prices[symbol]}"
                    )
                    buy_indicators[symbol] = {
                        "average": round(data.average, 2),
                        "vwap": round(data.vwap, 2),
                        "patterns": None,
                    }

                    return (
                        True,
                        {
                            "side": "buy",
                            "qty": str(shares_to_buy),
                            "type": "limit",
                            "limit_price": str(data.close + 0.01),
                        },
                    )

        elif (
            await super().is_sell_time(now)
            and position > 0
            and symbol in latest_cost_basis
            and last_used_strategy[symbol].name == self.name
        ):
            if open_orders.get(symbol) is not None:
                tlog(f"vwap_scalp: open order for {symbol} exists, skipping")
                return False, {}

            to_sell = False
            to_sell_market = False
            if data.vwap <= data.average - 0.05:
                to_sell = True
                reason = "below VWAP"
                to_sell_market = True
            elif data.close >= target_prices[symbol]:
                to_sell = True
                reason = "vwap scalp"
            elif (
                prev_minute.close < prev_minute.open and data.close < data.open
            ):
                to_sell = True
                reason = "vwap scalp no bears"

            if to_sell:
                sell_indicators[symbol] = {
                    "reason": reason,
                    "average": data.average,
                    "vwap": data.vwap,
                }
                return (
                    True,
                    {"side": "sell", "qty": str(position), "type": "market"}
                    if to_sell_market
                    else {
                        "side": "sell",
                        "qty": str(position),
                        "type": "limit",
                        "limit_price": str(data.close),
                    },
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
