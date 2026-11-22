import pytest

from src.data.aggregator import MarketSnapshot, SymbolSnapshot
from src.data.market import FundingData
from src.llm.context_builder import ContextBuilder


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), "unavailable"])
def test_missing_open_interest_does_not_break_prompt_or_create_zero(value):
    symbol = "BTC/USDT:USDT"
    snapshot = MarketSnapshot(symbols={symbol: SymbolSnapshot(
        symbol=symbol,
        funding=FundingData(symbol=symbol, funding_rate=0.0001, next_funding_time=None),
        open_interest={"open_interest_value": value},
    )})
    text = ContextBuilder()._format_derivatives(snapshot)
    assert "Funding:" in text
    assert "OI:" not in text


def test_available_open_interest_is_included():
    symbol = "BTC/USDT:USDT"
    snapshot = MarketSnapshot(symbols={symbol: SymbolSnapshot(
        symbol=symbol, open_interest={"open_interest_value": 123456.0},
    )})
    assert "OI: $123,456" in ContextBuilder()._format_derivatives(snapshot)
