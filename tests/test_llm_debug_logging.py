import json
from unittest.mock import Mock

import pytest

from src.llm.client import TradingLLMClient
from tests.edge_cases.test_llm_providers import _MockProvider


@pytest.mark.asyncio
async def test_llm_trace_is_separate_and_matches_request_response(tmp_path, monkeypatch):
    from src.llm import client as module

    logger = Mock()
    monkeypatch.setattr(module, "log", logger)
    path = tmp_path / "nested" / "llm.jsonl"
    client = TradingLLMClient(provider=_MockProvider(response_text="raw response"),
                              system_prompt="system prompt", llm_log_file=str(path),
                              max_output_tokens=100, temperature=0.2)
    assert await client._call_api("user prompt") == "raw response"
    client.close()
    request, response = [json.loads(line) for line in path.read_text().splitlines()]
    assert (request["event"], request["system"], request["user"]) == (
        "llm_request", "system prompt", "user prompt")
    assert response["event"] == "llm_response"
    assert response["text"] == "raw response"
    assert request["request_id"] == response["request_id"]
    logger.debug.assert_not_called()


@pytest.mark.asyncio
async def test_llm_trace_disabled_by_default(tmp_path, monkeypatch):
    from src.llm import client as module

    logger = Mock()
    monkeypatch.setattr(module, "log", logger)
    client = TradingLLMClient(provider=_MockProvider())
    await client._call_api("private prompt")
    client.close()
    assert list(tmp_path.iterdir()) == []
    logger.debug.assert_not_called()
