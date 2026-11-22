"""Tests for news collection, scoring, and deduplication."""

from src.data.news import (
    NewsArticle,
    NewsSnapshot,
    categorize_news,
    classify_sentiment,
    compute_relevance_score,
    deduplicate,
)


def _make_article(
    title: str = "Bitcoin surges past $90K",
    source: str = "CoinDesk",
    category: str | None = None,
    sentiment: str | None = None,
    relevance: float | None = None,
    url: str | None = None,
) -> NewsArticle:
    return NewsArticle(
        title=title,
        source=source,
        category=category or categorize_news(title),
        sentiment=sentiment or classify_sentiment(title),
        published_at="2026-03-18T12:00:00Z",
        url=url or f"https://example.com/{hash(title)}",
        relevance_score=relevance if relevance is not None else compute_relevance_score(title),
    )


class TestComputeRelevanceScore:
    def test_high_relevance_bitcoin(self):
        score = compute_relevance_score("Bitcoin BTC price hits new ATH")
        assert score > 0.5

    def test_medium_relevance_fed(self):
        score = compute_relevance_score("Fed signals rate cut in next meeting")
        assert score > 0.3

    def test_low_relevance_unrelated(self):
        score = compute_relevance_score("New movie releases this weekend")
        assert score < 0.15

    def test_tier2_keywords(self):
        score = compute_relevance_score("Inflation data shows CPI rising, recession fears grow")
        assert score > 0.3

    def test_tier3_keywords(self):
        score = compute_relevance_score("Oil prices affect global economy")
        assert score > 0.1

    def test_empty_string(self):
        assert compute_relevance_score("") == 0.0

    def test_capped_at_one(self):
        score = compute_relevance_score(
            "bitcoin btc ethereum eth crypto fed fomc rate cut inflation recession tariff"
        )
        assert score <= 1.0


class TestClassifySentiment:
    def test_bullish(self):
        assert classify_sentiment("Bitcoin surges to new record high") == "bullish"

    def test_bearish(self):
        assert classify_sentiment("Crypto market crash, prices plunge") == "bearish"

    def test_neutral(self):
        assert classify_sentiment("Market moves sideways today") == "neutral"

    def test_mixed_defaults_neutral(self):
        # Equal bull and bear signals
        assert classify_sentiment("Markets rally then crash") in ("bullish", "bearish", "neutral")

    def test_strong_bullish(self):
        assert classify_sentiment("Prices soar, rally continues, gains accelerate") == "bullish"

    def test_strong_bearish(self):
        assert classify_sentiment("Crash warning, decline deepens, selloff risk") == "bearish"


class TestCategorizeNews:
    def test_crypto(self):
        assert categorize_news("Bitcoin hits new ATH") == "crypto"
        assert categorize_news("Ethereum DeFi protocol launches") == "crypto"

    def test_macro(self):
        assert categorize_news("Fed raises interest rate") == "macro"
        assert categorize_news("CPI inflation data released") == "macro"

    def test_geopolitics(self):
        assert categorize_news("NATO military response to conflict") == "geopolitics"

    def test_politics(self):
        assert categorize_news("New election results announced") == "politics"

    def test_markets_default(self):
        assert categorize_news("Stock market closes higher") == "markets"


class TestDeduplicate:
    def test_url_dedup(self):
        articles = [
            _make_article("Bitcoin surges", url="https://example.com/1"),
            _make_article("Bitcoin surges past 90K", url="https://example.com/1"),
        ]
        result = deduplicate(articles)
        assert len(result) == 1

    def test_fuzzy_title_dedup(self):
        articles = [
            _make_article("Bitcoin surges past $90,000 today", url="https://a.com/1"),
            _make_article("Bitcoin surges past $90,000 today!", url="https://b.com/2"),
        ]
        result = deduplicate(articles)
        assert len(result) == 1

    def test_different_articles_kept(self):
        articles = [
            _make_article("Bitcoin hits $90K", url="https://a.com/1"),
            _make_article("Fed cuts interest rate", url="https://b.com/2"),
            _make_article("Ethereum DeFi growth accelerates", url="https://c.com/3"),
        ]
        result = deduplicate(articles)
        assert len(result) == 3

    def test_empty_list(self):
        assert deduplicate([]) == []

    def test_no_url_still_dedupes_by_title(self):
        articles = [
            _make_article("Exact same title here", url=None),
            _make_article("Exact same title here", url=None),
        ]
        result = deduplicate(articles)
        assert len(result) == 1


class TestNewsSnapshot:
    def test_empty_snapshot(self):
        snap = NewsSnapshot()
        assert snap.articles == []
        assert snap.category_counts == {}
        assert snap.sentiment_summary == {}
        assert snap.top_headlines == []

    def test_snapshot_with_data(self):
        articles = [
            _make_article("BTC rally"),
            _make_article("ETH breakout"),
        ]
        snap = NewsSnapshot(
            articles=articles,
            category_counts={"crypto": 2},
            sentiment_summary={"bullish": 2, "bearish": 0, "neutral": 0},
            top_headlines=["BTC rally", "ETH breakout"],
        )
        assert len(snap.articles) == 2
        assert snap.category_counts["crypto"] == 2
        assert snap.sentiment_summary["bullish"] == 2


def test_freshness_filters_dates_and_rebuilds_sentiment():
    from datetime import datetime, timezone
    from src.data.news import filter_news_snapshot

    now = datetime(2026, 11, 23, 12, tzinfo=timezone.utc)
    fresh = _make_article("Bitcoin rally today", sentiment="bullish")
    fresh.published_at = "Mon, 23 Nov 2026 13:00:00 +0200"
    border = _make_article("Ethereum crash yesterday", sentiment="bearish")
    border.published_at = "2026-11-21T12:00:00Z"
    old = _make_article("Bitcoin older crash", sentiment="bearish")
    old.published_at = "2026-11-21T11:59:59Z"
    missing = _make_article("Crypto without date", sentiment="bearish")
    missing.published_at = "2026-11-23T11:00:00"
    future = _make_article("Crypto future report", sentiment="bullish")
    future.published_at = "2026-11-23T12:00:01Z"
    snapshot = filter_news_snapshot([fresh, border, old, missing, future], 48, now)
    assert {a.title for a in snapshot.articles} == {fresh.title, border.title}
    assert snapshot.sentiment_summary == {"bullish": 1, "bearish": 1, "neutral": 0}
    expired = filter_news_snapshot(snapshot.articles, 48, now.replace(hour=13))
    assert [a.title for a in expired.articles] == [fresh.title]
    assert expired.sentiment_summary["bearish"] == 0


def test_publication_time_formats_and_invalid():
    from datetime import datetime, timezone
    from src.data.news import parse_publication_time

    expected = datetime(2026, 11, 23, 12, tzinfo=timezone.utc)
    assert parse_publication_time("2026-11-23T14:00:00+02:00") == expected
    assert parse_publication_time(str(expected.timestamp())) == expected
    assert parse_publication_time("Mon, 23 Nov 2026 12:00:00 GMT") == expected
    assert parse_publication_time("2026-11-23 12:00:00") is None
    assert parse_publication_time("not a date") is None
    assert parse_publication_time("99999999999999999999999999999999999") is None


def test_prompt_contains_utc_publication_and_age():
    from datetime import datetime, timedelta, timezone
    from src.data.aggregator import MarketSnapshot
    from src.data.news import filter_news_snapshot
    from src.llm.context_builder import ContextBuilder

    article = _make_article("BTC rallies today", sentiment="bullish")
    article.published_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    news = filter_news_snapshot([article], 48)
    text = ContextBuilder()._format_news(MarketSnapshot(news=news))
    assert "UTC" in text and "2.0h ago" in text and "BTC rallies today" in text


async def test_cached_news_and_sentiment_are_refiltered_each_cycle():
    from datetime import datetime, timedelta, timezone
    from types import SimpleNamespace
    from src.data.aggregator import DataAggregator
    from src.data.sentiment import NewsItem, SentimentSnapshot

    aggregator = DataAggregator(market=None, websocket=SimpleNamespace(), onchain=None,
                                sentiment=None, macro=None, paper_mode=True,
                                news_max_age_hours=48)
    old = _make_article("BTC old crash", sentiment="bearish")
    old.published_at = (datetime.now(timezone.utc) - timedelta(hours=49)).isoformat()
    fresh = _make_article("ETH recent rally", sentiment="bullish")
    fresh.published_at = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    aggregator._cached_news = NewsSnapshot(articles=[old, fresh], sentiment_summary={"bearish": 1})
    aggregator._cached_sentiment = SentimentSnapshot(top_news=[
        NewsItem(title="old", source="source", sentiment="bearish", published_at=old.published_at),
        NewsItem(title="fresh", source="source", sentiment="bullish", published_at=fresh.published_at),
    ])
    snapshot = await aggregator.collect_snapshot(fetch_sentiment=False)
    assert [a.title for a in snapshot.news.articles] == [fresh.title]
    assert snapshot.news.sentiment_summary["bearish"] == 0
    assert [a.title for a in snapshot.sentiment.top_news] == ["fresh"]
    assert len(aggregator._cached_news.articles) == 2


def test_news_age_setting_default_override_and_validation(monkeypatch):
    from config.settings import Settings

    monkeypatch.delenv("NEWS_MAX_AGE_HOURS", raising=False)
    assert Settings().news_max_age_hours == 48
    monkeypatch.setenv("NEWS_MAX_AGE_HOURS", "6")
    assert Settings().news_max_age_hours == 6
    monkeypatch.setenv("NEWS_MAX_AGE_HOURS", "0")
    assert any("NEWS_MAX_AGE_HOURS" in error for error in Settings().validate())
