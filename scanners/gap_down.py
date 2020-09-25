"""find short-able, liquid, large-cap stocks that gap down"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Optional

import alpaca_trade_api as tradeapi
from liualgotrader.common import config
from liualgotrader.common.tlog import tlog
from liualgotrader.scanners.base import Scanner
from pytz import timezone


class GapDown(Scanner):
    name = "GapDown"

    def __init__(
        self,
        data_api: tradeapi,
        recurrence: Optional[timedelta] = None,
        target_strategy_name: str = None,
        max_symbols: int = config.total_tickers,
    ):
        self.max_symbols = max_symbols
        super().__init__(
            name=self.name,
            recurrence=recurrence,
            target_strategy_name=target_strategy_name,
            data_api=data_api,
        )

    async def _wait_time(self) -> None:
        if not config.bypass_market_schedule and config.market_open:
            nyc = timezone("America/New_York")
            since_market_open = (
                datetime.today().astimezone(nyc) - config.market_open
            )

            if since_market_open.seconds // 60 < 10:
                tlog(f"{self.name} market open, wait {10} minutes")
                while since_market_open.seconds // 60 < 10:
                    await asyncio.sleep(1)
                    since_market_open = (
                        datetime.today().astimezone(nyc) - config.market_open
                    )

        tlog(f"Scanner {self.name} ready to run")

    def _get_short_able_trade_able_symbols(self) -> List[str]:
        assets = self.data_api.list_assets()
        tlog(
            f"{self.name} -> loaded list of {len(assets)} trade-able assets from Alpaca"
        )

        trade_able_symbols = [
            asset.symbol
            for asset in assets
            if asset.tradable and asset.shortable and asset.easy_to_borrow
        ]
        tlog(
            f"total number of trade-able symbols is {len(trade_able_symbols)}"
        )
        return trade_able_symbols

    async def run(self) -> List[str]:
        tlog(f"{self.name}: run(): started")
        await self._wait_time()

        try:
            while True:
                tickers = self.data_api.polygon.all_tickers()
                tlog(
                    f"{self.name} -> loaded {len(tickers)} tickers from Polygon"
                )
                if not len(tickers):
                    break
                trade_able_symbols = self._get_short_able_trade_able_symbols()

                unsorted = [
                    ticker
                    for ticker in tickers
                    if (
                        ticker.ticker in trade_able_symbols
                        and ticker.lastTrade["p"] >= 50.0
                        and ticker.prevDay["v"] * ticker.lastTrade["p"]
                        > 500000.0
                        and ticker.prevDay["l"] * 0.9 > ticker.day["o"]
                        and ticker.todaysChangePerc < 0
                        and (
                            ticker.day["v"] > 30000.0
                            or config.bypass_market_schedule
                        )
                    )
                ]
                if len(unsorted) > 0:
                    ticker_by_volume = sorted(
                        unsorted,
                        key=lambda ticker: float(ticker.day["v"]),
                        reverse=True,
                    )
                    tlog(
                        f"{self.name} -> picked {len(ticker_by_volume)} symbols"
                    )
                    return [x.ticker for x in ticker_by_volume][
                        : self.max_symbols
                    ]

                tlog("did not find gaping down stocks, retrying")

                await asyncio.sleep(30)
        except KeyboardInterrupt:
            tlog("KeyboardInterrupt")
            pass

        return []
