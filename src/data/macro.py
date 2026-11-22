"""Macro indicator collection via yfinance."""

from __future__ import annotations

import asyncio
import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pandas as pd
import yfinance as yf

from src.utils.logger import get_logger

log = get_logger(__name__)

# pandas 3.1 warns about yfinance's internal use of inplace=True on every
# download. Keep other pandas and yfinance warnings visible. Older pandas
# versions do not define Pandas4Warning.
_pandas4_warning = getattr(pd.errors, "Pandas4Warning", None)
if _pandas4_warning is not None:
    warnings.filterwarnings(
        "ignore",
        message=r"The inplace keyword in DataFrame\.(?:sort_index|drop|rename) is deprecated.*",
        category=_pandas4_warning,
        module=r"^yfinance(?:\.|$)",
    )

MACRO_TICKERS = {
    "dxy": "DX-Y.NYB",
    "sp500": "^GSPC",
    "gold": "GC=F",
    "treasury_10y": "^TNX",
    "vix": "^VIX",
}


@dataclass
class MacroIndicator:
    """Single macro indicator value."""

    name: str
    value: float
    change_pct: float
    prev_value: float


@dataclass
class MacroSnapshot:
    """All macro indicators."""

    indicators: dict[str, MacroIndicator] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def get(self, name: str) -> MacroIndicator | None:
        return self.indicators.get(name)


class MacroCollector:
    """Collects macro indicators via yfinance."""

    def __init__(self, tickers: dict[str, str] | None = None):
        self._tickers = tickers or MACRO_TICKERS
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def collect(self) -> MacroSnapshot:
        """Fetch all macro indicators.

        yfinance is synchronous, so we run it in a thread pool.
        """
        loop = asyncio.get_event_loop()
        try:
            snapshot = await loop.run_in_executor(self._executor, self._fetch_all)
            return snapshot
        except Exception as e:
            log.error("macro_fetch_failed", error=str(e))
            return MacroSnapshot()

    def _fetch_all(self) -> MacroSnapshot:
        """Synchronous fetch of all macro indicators."""
        snapshot = MacroSnapshot()

        for name, ticker_symbol in self._tickers.items():
            try:
                data = yf.download(
                    ticker_symbol,
                    period="5d",
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                )
                if data.empty or len(data) < 2:
                    continue

                close = data["Close"]
                # yfinance can return MultiIndex columns even for one ticker.
                if isinstance(close, pd.DataFrame):
                    if ticker_symbol in close.columns:
                        close = close[ticker_symbol]
                    elif len(close.columns) == 1:
                        close = close.iloc[:, 0]
                    else:
                        raise ValueError("Ambiguous Close columns")
                current = float(close.iloc[-1])
                prev = float(close.iloc[-2])
                if not math.isfinite(current) or not math.isfinite(prev) or prev == 0:
                    raise ValueError("Missing or invalid closing prices")
                change_pct = (current - prev) / prev * 100

                snapshot.indicators[name] = MacroIndicator(
                    name=name,
                    value=round(current, 4),
                    change_pct=round(change_pct, 2),
                    prev_value=round(prev, 4),
                )
            except Exception as e:
                log.warning("macro_ticker_failed", ticker=name, error=str(e))

        return snapshot

    def close(self) -> None:
        self._executor.shutdown(wait=False)
