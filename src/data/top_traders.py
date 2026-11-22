"""Top trader analysis from Polymarket leaderboard."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field

import aiohttp

from src.utils.logger import get_logger

log = get_logger(__name__)

# Minimum thresholds for a "qualified" trader
MIN_WIN_RATE = 0.55
MIN_PROFIT = 10_000  # $10K total profit
MIN_TRADES = 50


@dataclass
class TraderPosition:
    """A single position held by a top trader."""

    market_question: str
    side: str           # "yes" | "no"
    size_usd: float | None = None
    current_odds: float = 0.5
    condition_id: str = ""


@dataclass
class TopTrader:
    """A high-performing trader with their stats and positions."""

    username: str
    profit_total: float
    win_rate: float     # 0-1
    total_trades: int
    active_positions: list[TraderPosition] = field(default_factory=list)

    @property
    def weight(self) -> float:
        """Composite weight based on performance metrics."""
        return compute_trader_weight(self.win_rate, self.profit_total, self.total_trades)


@dataclass
class TraderConsensus:
    """Aggregated signal from multiple top traders on the same market."""

    market_question: str
    category: str
    bullish_count: int      # traders betting YES
    bearish_count: int      # traders betting NO
    total_tracked: int
    avg_win_rate: float     # weighted avg win rate
    confidence_score: float  # 0-1 weighted by win rate and profit
    description: str


@dataclass
class TopTradersSnapshot:
    """Collection of top trader data and consensus signals."""

    traders: list[TopTrader] = field(default_factory=list)
    consensus: list[TraderConsensus] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


def compute_trader_weight(win_rate: float, profit: float, trades: int) -> float:
    """Weight = win_rate * log(1 + profit) * sqrt(trades)."""
    if profit <= 0 or trades <= 0 or win_rate <= 0:
        return 0.0
    return win_rate * math.log(1 + profit) * math.sqrt(trades)


def compute_consensus(traders: list[TopTrader]) -> list[TraderConsensus]:
    """Compute consensus signals across qualified traders' positions."""
    if not traders:
        return []

    # Group positions by market question
    market_positions: dict[str, list[tuple[TopTrader, TraderPosition]]] = {}
    for trader in traders:
        for pos in trader.active_positions:
            key = pos.condition_id or pos.market_question.strip().lower()
            market_positions.setdefault(key, []).append((trader, pos))

    consensus_list: list[TraderConsensus] = []
    for _, entries in market_positions.items():
        if len(entries) < 2:
            continue

        bullish = [(t, p) for t, p in entries if p.side == "yes"]
        bearish = [(t, p) for t, p in entries if p.side == "no"]

        # Weighted averages
        total_weight = sum(t.weight for t, _ in entries)
        if total_weight == 0:
            continue

        bullish_weight = sum(t.weight for t, _ in bullish)
        bearish_weight = sum(t.weight for t, _ in bearish)

        weighted_wr = sum(t.win_rate * t.weight for t, _ in entries) / total_weight
        confidence = max(bullish_weight, bearish_weight) / total_weight

        question = entries[0][1].market_question
        from src.data.prediction_markets import _categorize
        category = _categorize(question)

        direction = "YES" if bullish_weight > bearish_weight else "NO"
        consensus_list.append(TraderConsensus(
            market_question=question,
            category=category,
            bullish_count=len(bullish),
            bearish_count=len(bearish),
            total_tracked=len(entries),
            avg_win_rate=round(weighted_wr, 3),
            confidence_score=round(confidence, 3),
            description=(
                f"{len(entries)} top traders on '{question[:60]}': "
                f"{len(bullish)} YES vs {len(bearish)} NO "
                f"(consensus: {direction}, confidence: {confidence:.0%})"
            ),
        ))

    # Sort by number of traders, then confidence
    consensus_list.sort(key=lambda c: (c.total_tracked, c.confidence_score), reverse=True)
    return consensus_list[:10]


class TopTraderCollector:
    """Fetches top trader data from Polymarket analytics."""

    # Polymarket profile API (public)
    POLYMARKET_PROFILE_API = "https://data-api.polymarket.com/v2/leaderboard"
    POSITIONS_API = "https://data-api.polymarket.com/v2/positions"
    USER_STATS_API = "https://data-api.polymarket.com/v2/user-stats"

    def __init__(self, session: aiohttp.ClientSession | None = None):
        self._session = session
        self._owns_session = session is None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            )
        return self._session

    async def collect(self) -> TopTradersSnapshot:
        """Fetch top trader data and compute consensus."""
        traders = await self._fetch_top_traders()
        qualified = [
            t for t in traders
            if t.win_rate >= MIN_WIN_RATE
            and t.profit_total >= MIN_PROFIT
            and t.total_trades >= MIN_TRADES
        ]
        consensus = compute_consensus(qualified)

        return TopTradersSnapshot(
            traders=qualified[:20],
            consensus=consensus,
        )

    async def _get_json(self, url: str, params: dict) -> dict:
        session = await self._get_session()
        async with session.get(url, params=params) as resp:
            if resp.status != 200:
                raise ValueError(f"HTTP {resp.status} from {url}")
            data = await resp.json()
            if not isinstance(data, dict):
                raise ValueError("Unexpected Polymarket response")
            return data

    async def _positions(self, user: str, status: str) -> list[dict]:
        # Never calculate a win rate or consensus from a truncated page.
        rows = []
        cursor = None
        for _ in range(10):
            params = {"user": user, "status": status, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            page = await self._get_json(self.POSITIONS_API, params)
            data = page.get("data")
            if not isinstance(data, list):
                raise ValueError("Missing positions data")
            rows.extend(data)
            pagination = page.get("pagination")
            if not isinstance(pagination, dict):
                raise ValueError("Missing positions pagination")
            cursor = pagination.get("next_cursor")
            if not cursor:
                if pagination.get("has_more"):
                    raise ValueError("Missing positions cursor")
                return rows
        raise ValueError("Positions exceed pagination limit")

    async def _fetch_top_traders(self) -> list[TopTrader]:
        """Enrich the leaderboard with verified account statistics and positions."""
        try:
            result = await self._get_json(self.POLYMARKET_PROFILE_API, {"limit": 50})
            items = result.get("data")
            if not isinstance(items, list):
                raise ValueError("Missing leaderboard data")
        except Exception as e:
            log.warning("top_traders_fetch_error", error=str(e))
            return []

        semaphore = asyncio.Semaphore(5)

        async def enrich(item: dict) -> TopTrader | None:
            async with semaphore:
                try:
                    user = item["user_id"]
                    stats_response = await self._get_json(self.USER_STATS_API, {"user": user})
                    stats = stats_response["data"]
                    all_time = stats["all_time_pnl"]
                    trades = int(all_time["trade_count"])
                    profit = float(all_time["realized_market_pnl"])
                    if not math.isfinite(profit) or trades < MIN_TRADES or profit < MIN_PROFIT:
                        return None
                    closed = await self._positions(user, "CLOSED")
                    # A closed market is a win when its combined realized PnL
                    # across outcome positions is positive. This is not a
                    # prediction-resolution win rate.
                    by_market: dict[str, float] = {}
                    for pos in closed:
                        condition = pos["condition_id"]
                        pnl = float(pos["realized_pnl"])
                        if not condition or not math.isfinite(pnl):
                            raise ValueError("Invalid closed position")
                        by_market[condition] = by_market.get(condition, 0.0) + pnl
                    if not by_market:
                        return None
                    win_rate = sum(pnl > 0 for pnl in by_market.values()) / len(by_market)
                    if win_rate < MIN_WIN_RATE:
                        return None
                    open_positions = await self._positions(user, "OPEN")
                    positions = []
                    for pos in open_positions:
                        outcome = str(pos.get("outcome", "")).lower()
                        # Named outcomes (e.g. teams) are not YES/NO signals.
                        if outcome not in ("yes", "no"):
                            continue
                        price = float(pos["current_price"])
                        size = float(pos["current_value"])
                        if not (math.isfinite(price) and 0 <= price <= 1 and math.isfinite(size) and size > 0):
                            continue
                        if not pos.get("condition_id") or not pos.get("title"):
                            continue
                        positions.append(TraderPosition(
                            market_question=pos["title"], side=outcome,
                            size_usd=size, current_odds=price, condition_id=pos["condition_id"],
                        ))
                    return TopTrader(username=item.get("user_name") or user,
                                     profit_total=profit, win_rate=win_rate,
                                     total_trades=trades, active_positions=positions)
                except (KeyError, TypeError, ValueError, aiohttp.ClientError, asyncio.TimeoutError) as e:
                    log.warning("top_trader_enrichment_failed", error=str(e))
                    return None

        enriched = await asyncio.gather(*(enrich(item) for item in items if isinstance(item, dict)))
        return [trader for trader in enriched if trader is not None]

    async def close(self) -> None:
        if self._owns_session and self._session:
            await self._session.close()
