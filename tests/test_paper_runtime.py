from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

from config.settings import LLMSettings, Settings
from src.bot import TradingBot
from src.execution.executor import TradeExecutor
from src.execution.position_manager import PositionManager
from src.llm.providers.base import LLMProviderError
from src.llm.providers.openai_compat_provider import OpenAICompatProvider
from src.llm.response_parser import TradeAction
from src.llm.scheduler import LLMScheduler
from src.risk.risk_manager import ValidationResult
from src.storage.database import Database


def test_local_provider_does_not_require_key(monkeypatch):
    for key in ("LLM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(llm=LLMSettings(provider="openai-compat", api_key=""))
    assert not any("LLM API key" in error for error in settings.validate())


def test_daily_ignores_forced_and_model_intervals(monkeypatch):
    monkeypatch.setattr("src.llm.scheduler.time.time", lambda: 100000)
    scheduler = LLMScheduler(cadence="daily")
    scheduler.record_call()
    scheduler.set_llm_interval(5)
    scheduler.force_next_call()
    assert not scheduler.should_call(has_open_position=True, price_change_15m_pct=9)
    assert scheduler.get_next_call_in() == 86400
    monkeypatch.setattr("src.llm.scheduler.time.time", lambda: 186400)
    assert scheduler.should_call()


@pytest.mark.asyncio
async def test_paper_orders_and_restart(tmp_path):
    exchange = SimpleNamespace(market=lambda symbol: {"precision": {"amount": 0.001, "price": 0.01}},
                               create_order=AsyncMock(side_effect=AssertionError("real order")),
                               set_leverage=AsyncMock(side_effect=AssertionError("real leverage")))
    bot = TradingBot.__new__(TradingBot)
    bot.settings = SimpleNamespace(is_live=False, exchange=SimpleNamespace(quote_currency="USDT"))
    bot.position_manager = PositionManager(record_closed_trades=True)
    bot._paper_cash = 10000
    bot.db = Database(str(tmp_path / "paper.db"))
    await bot.db.initialize()
    await bot._load_paper_state()
    executor = TradeExecutor(exchange, bot.position_manager, paper_mode=True, taker_fee=0.001)
    action = TradeAction(action="open_long", symbol="BTC/USDT:USDT", size_pct=10,
                         leverage=2, stop_loss=90, take_profit=120, reasoning="test entry")
    result = await executor.execute(action, ValidationResult(approved=True, action=action), 100, 10000)
    assert result.success
    await bot._save_paper_state()
    await bot.db.close()
    bot.position_manager = PositionManager(record_closed_trades=True)
    bot._paper_cash = 1
    await bot.db.initialize()
    await bot._load_paper_state()
    assert bot.position_manager.get_position(action.symbol).stop_loss == 90
    close = TradeAction(action="close", symbol=action.symbol, reasoning="test close")
    result = await executor.__class__(exchange, bot.position_manager, paper_mode=True, taker_fee=0.001).execute(
        close, ValidationResult(approved=True, action=close), 110, 10000)
    assert result.success
    await bot._save_paper_state()
    assert (await bot._balance())["total"] == pytest.approx(10098)
    assert len(await bot.db.get_recent_trades()) == 1
    await bot.db.close()
    exchange.create_order.assert_not_awaited()
    exchange.set_leverage.assert_not_awaited()


@pytest.mark.asyncio
async def test_compat_bad_response_is_provider_error(monkeypatch):
    class Client:
        def __init__(self, **kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *args): pass
        async def post(self, url, **kwargs):
            assert url == "http://llama:8080/v1/chat/completions"
            assert "Authorization" not in kwargs["headers"]
            return httpx.Response(200, json={"choices": []})
    monkeypatch.setattr(httpx, "AsyncClient", Client)
    provider = OpenAICompatProvider(api_key="", model="test", base_url="http://llama:8080/v1")
    with pytest.raises(LLMProviderError):
        await provider.complete(system="s", user="u", max_output_tokens=20)
