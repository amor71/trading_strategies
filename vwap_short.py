import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import alpaca_trade_api as tradeapi
from liualgotrader.common import config
from liualgotrader.common.tlog import tlog
from liualgotrader.common.trading_data import (buy_indicators,
                                               last_used_strategy,
                                               latest_cost_basis, open_orders,
                                               sell_indicators, stop_prices,
                                               target_prices)
from liualgotrader.fincalcs.support_resistance import (StopRangeType,
                                                       find_supports)
from liualgotrader.fincalcs.vwap import add_daily_vwap, anchored_vwap
from liualgotrader.strategies.base import Strategy, StrategyType
from pandas import DataFrame as df
from pandas import concat
from scipy.stats import linregress, norm
from tabulate import tabulate
from talib import MACD


class VWAPShort(Strategy):
    name = "vwap_short"
    was_above_vwap: Dict = {}
    volume_test_time: Dict = {}

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
        if not shortable:
            return False, {}

        data = minute_history.iloc[-1]

        if data.close > data.average:
            self.was_above_vwap[symbol] = True

        if (
            await super().is_buy_time(now)
            and not position
            and not open_orders.get(symbol, None)
        ):
            if data.open > data.average:
                if debug:
                    tlog(f"{self.name} {symbol} trending up: {data}")
                return False, {}

            lbound = config.market_open.replace(second=0, microsecond=0)
            close = (
                minute_history["close"][lbound:]
                .dropna()
                .between_time("9:30", "16:00")
                .resample("5min")
                .last()
            ).dropna()
            open = (
                minute_history["open"][lbound:]
                .dropna()
                .between_time("9:30", "16:00")
                .resample("5min")
                .first()
            ).dropna()
            high = (
                minute_history["high"][lbound:]
                .dropna()
                .between_time("9:30", "16:00")
                .resample("5min")
                .max()
            ).dropna()
            low = (
                minute_history["low"][lbound:]
                .dropna()
                .between_time("9:30", "16:00")
                .resample("5min")
                .min()
            ).dropna()
            volume = (
                minute_history["volume"][lbound:]
                .dropna()
                .between_time("9:30", "16:00")
                .resample("5min")
                .sum()
            ).dropna()
            volume = volume[volume != 0]

            df = concat(
                [
                    open.rename("open"),
                    high.rename("high"),
                    low.rename("low"),
                    close.rename("close"),
                    volume.rename("volume"),
                ],
                axis=1,
            )
            if not add_daily_vwap(df):
                tlog(f"[{now}]{symbol} failed in add_daily_vwap")
                return False, {}

            vwap_series = df["average"]

            # calc macd on 5 min
            close_5min = (
                minute_history["close"]
                .dropna()
                .between_time("9:30", "16:00")
                .resample("5min")
                .last()
            ).dropna()

            if debug:
                tlog(
                    f"\n{tabulate(df[-10:], headers='keys', tablefmt='psql')}"
                )
            macds = MACD(close_5min)
            macd = macds[0].round(3)
            macd_signal = macds[1].round(3)
            macd_hist = macds[2].round(3)
            vwap_series = vwap_series.round(3)
            close = close.round(3)
            if (
                self.was_above_vwap.get(symbol, False)
                and close[-1] < vwap_series[-1]
                and close[-2] < vwap_series[-2]
                and close[-3] < vwap_series[-3]
                and close[-1] < open[-1]
                and close[-2] < open[-2]
                and close[-3] < open[-3]
                and macd[-1] < macd_signal[-1] < 0
                and macd[-1] < 0
                and macd_hist[-1] < macd_hist[-2] < macd_hist[-3] < 0
                and data.close < data.open
                and data.close
                < minute_history["close"][-2]
                < minute_history["close"][-3]
            ):
                stops = find_supports(
                    data.close, minute_history, now, StopRangeType.LAST_2_HOURS
                )

                if stops:
                    tlog(
                        f"[self.name]:{symbol}@{data.close} potential short-trap {stops}"
                    )
                    return False, {}

                tlog(
                    f"\n{tabulate(df[-10:], headers='keys', tablefmt='psql')}"
                )

                stop_price = vwap_series[-1] * 1.005
                target_price = round(
                    min(
                        data.close - 10 * (stop_price - data.close),
                        data.close * 0.98,
                    ),
                    2,
                )

                stop_prices[symbol] = round(stop_price, 2)
                target_prices[symbol] = round(target_price, 2)

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

                shares_to_buy = int(
                    max(
                        portfolio_value
                        * config.risk
                        // (data.close - stop_prices[symbol]),
                        -portfolio_value * 0.02 // data.close,
                    )
                )
                if not shares_to_buy:
                    shares_to_buy = 1

                buy_price = data.close
                tlog(
                    f"[{self.name}][{now}] Submitting buy short for {-shares_to_buy} shares of {symbol} at {buy_price} target {target_prices[symbol]} stop {stop_prices[symbol]}"
                )
                sell_indicators[symbol] = {
                    "vwap_series": vwap_series[-5:].tolist(),
                    "5-min-close": close[-5:].tolist(),
                    "vwap": data.vwap,
                    "avg": data.average,
                    "volume": minute_history["volume"][-5:].tolist(),
                    "stops": [] if not stops else stops.tolist(),
                }
                self.volume_test_time[symbol] = now
                return (
                    True,
                    {
                        "side": "sell",
                        "qty": str(-shares_to_buy),
                        "type": "market",
                    },
                )

        if (
            await super().is_sell_time(now)
            and position
            and last_used_strategy[symbol].name == self.name
            and not open_orders.get(symbol)
        ):
            volume = minute_history["volume"][
                self.volume_test_time[symbol] :
            ].dropna()
            mu, std = norm.fit(volume)
            a_vwap = anchored_vwap(
                minute_history, self.volume_test_time[symbol]
            )
            close_5min = (
                minute_history["close"]
                .dropna()
                .between_time("9:30", "16:00")
                .resample("5min")
                .last()
            ).dropna()
            to_sell: bool = False
            reason: str = ""

            macds = MACD(close_5min, 13, 21)
            macd = macds[0].round(2)
            macd_signal = macds[1].round(2)
            macd_hist = macds[2].round(2)
            close_5min = close_5min.round(2)
            movement = (
                data.close - latest_cost_basis[symbol]
            ) / latest_cost_basis[symbol]
            if (
                data.close >= stop_prices[symbol]
                and macd[-1] > macd_signal[-1]
            ):
                to_sell = True
                reason = "stopped"
            elif data.close <= target_prices[symbol]:
                to_sell = True
                reason = "target reached"
            elif (
                close_5min[-1]
                > close_5min[-2]
                > close_5min[-3]
                < close_5min[-4]
                and data.close < latest_cost_basis[symbol]
            ):
                to_sell = True
                reason = "reversing direction"
            elif (
                macd[-1] > macd_signal[-1]
                and data.close < latest_cost_basis[symbol]
            ):
                to_sell = True
                reason = "MACD changing trend"
            elif (
                0
                > macd_hist[-4]
                > macd_hist[-3]
                < macd_hist[-2]
                < macd_hist[-1]
                < 0
                and data.close < latest_cost_basis[symbol]
            ):
                to_sell = True
                reason = "MACD histogram trend reversal"
            elif (
                len(a_vwap) > 10
                and minute_history.close[-1] > a_vwap[-2]
                and minute_history.close[-2] > a_vwap[-2]
            ):
                slope_min, intercept_min, _, _, _ = linregress(
                    range(10), minute_history.close[-10:]
                )
                slope_a_vwap, intercept_a_vwap, _, _, _ = linregress(
                    range(10), a_vwap[-10:]
                )

                if round(slope_min, 2) > round(slope_a_vwap, 2):
                    to_sell = True
                    reason = f"deviate from anchored-vwap {round(slope_min, 2)}>{round(slope_a_vwap, 2)}"

            # elif data.volume > mu + 2 * std and data.close > data.open and data.vwap > data.open:
            #    to_sell = True
            #    reason = "suspicious spike in volume, may be short-trap"

            if to_sell:
                buy_indicators[symbol] = {
                    "close_5m": close_5min[-5:].tolist(),
                    "movement": movement,
                    "reason": reason,
                    "volume": minute_history.volume[-5:].tolist(),
                    "volume fit": (mu, std),
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
