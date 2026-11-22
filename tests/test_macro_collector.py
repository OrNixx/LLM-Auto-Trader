import pandas as pd
import pytest

from src.data.macro import MacroCollector


@pytest.mark.parametrize("multiindex", [False, True])
def test_yfinance_closes_single_ticker(monkeypatch, multiindex):
    data = pd.DataFrame({"Close": [100.0, 110.0]})
    if multiindex:
        data.columns = pd.MultiIndex.from_tuples([("Close", "TEST")])
    monkeypatch.setattr("src.data.macro.yf.download", lambda *args, **kwargs: data)
    collector = MacroCollector(tickers={"test": "TEST"})
    try:
        indicator = collector._fetch_all().get("test")
        assert indicator.value == 110
        assert indicator.prev_value == 100
        assert indicator.change_pct == 10
    finally:
        collector.close()


def test_missing_close_is_not_reported_as_data(monkeypatch):
    data = pd.DataFrame({"Close": [100.0, float("nan")]})
    monkeypatch.setattr("src.data.macro.yf.download", lambda *args, **kwargs: data)
    collector = MacroCollector(tickers={"test": "TEST"})
    try:
        assert not collector._fetch_all().indicators
    finally:
        collector.close()


def test_only_yfinance_inplace_deprecation_is_suppressed():
    import importlib
    import warnings

    import src.data.macro as macro

    warning_type = getattr(pd.errors, "Pandas4Warning", None)
    if warning_type is None:
        pytest.skip("Pandas4Warning is unavailable in this pandas version")
    message = "The inplace keyword in DataFrame.drop is deprecated and will be removed"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.reload(macro)
        warnings.warn_explicit(message, warning_type, "utils.py", 1, module="yfinance.utils")
        warnings.warn_explicit(message, warning_type, "other.py", 1, module="other")
        warnings.warn_explicit("different warning", warning_type, "utils.py", 2, module="yfinance.utils")
    assert [str(item.message) for item in caught] == [message, "different warning"]
