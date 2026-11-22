from unittest.mock import Mock

import pytest

from src.llm.client import TradingLLMClient
from src.llm.providers.base import LLMCallResult, LLMUsage


@pytest.mark.asyncio
async def test_model_request_and_raw_response_are_logged(monkeypatch):
    from src.llm import client as module
    from tests.edge_cases.test_llm_providers import _MockProvider

    provider = _MockProvider(response_text='raw response')
    logger = Mock()
    monkeypatch.setattr(module, 'log', logger)
    client = TradingLLMClient(provider=provider, system_prompt='system prompt',
                              max_output_tokens=100, temperature=0.2)
    assert await client._call_api('user prompt') == 'raw response'
    logger.debug.assert_any_call('llm_request', model='mock-model', system='system prompt',
                                 user='user prompt', max_output_tokens=100, temperature=0.2)
    logger.debug.assert_any_call('llm_response', model='mock-model', text='raw response')
