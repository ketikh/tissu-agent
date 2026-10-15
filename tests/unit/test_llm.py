"""Unit tests for src/llm.py — Gemini outages must surface as an error,
never as a silent empty reply, while a deliberate STOP-with-nothing is kept."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai import types

import src.llm as llm


def _response(parts=None, finish=types.FinishReason.STOP, candidates=True, block=None):
    if not candidates:
        return SimpleNamespace(candidates=[], prompt_feedback=SimpleNamespace(block_reason=block))
    content = SimpleNamespace(parts=parts or [])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content, finish_reason=finish)],
                           prompt_feedback=None)


def _text_part(text: str):
    return SimpleNamespace(function_call=None, text=text)


class _FakeClient:
    """Returns the queued responses in order; an Exception instance is raised."""
    def __init__(self, outcomes):
        self._outcomes = list(outcomes)
        self.calls = 0
        self.models = SimpleNamespace(generate_content=self._generate)

    def _generate(self, model, contents, config):
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture
def gemini(monkeypatch):
    """Install a fake Gemini client and skip retry back-off sleeps."""
    holder = {}

    def install(outcomes):
        holder["client"] = _FakeClient(outcomes)
        monkeypatch.setattr(llm, "get_client", lambda: holder["client"])
        return holder["client"]

    async def no_sleep(seconds):
        return None
    monkeypatch.setattr(llm, "_sleep", no_sleep)
    return install


async def _call():
    return await llm.chat_with_tools("system", [{"role": "user", "content": "რა ღირს?"}], tools=[])


async def test_text_answer_is_returned(gemini):
    gemini([_response(parts=[_text_part("69₾ ✨")])])

    result = await _call()

    assert result["stop_reason"] == "end_turn"
    assert result["content"] == [{"type": "text", "text": "69₾ ✨"}]


async def test_outage_on_every_attempt_raises(gemini):
    client = gemini([Exception("503 UNAVAILABLE")] * 3)

    with pytest.raises(llm.LLMUnavailableError, match="503"):
        await _call()
    assert client.calls == 3


async def test_recovers_after_transient_error(gemini):
    gemini([Exception("429 RESOURCE_EXHAUSTED"), _response(parts=[_text_part("ok")])])

    result = await _call()

    assert result["content"][0]["text"] == "ok"


async def test_token_limit_with_no_content_raises(gemini):
    gemini([_response(parts=[], finish=types.FinishReason.MAX_TOKENS)] * 3)

    with pytest.raises(llm.LLMUnavailableError, match="MAX_TOKENS"):
        await _call()


async def test_blocked_prompt_raises(gemini):
    gemini([_response(candidates=False, block="SAFETY")] * 3)

    with pytest.raises(llm.LLMUnavailableError, match="SAFETY"):
        await _call()


async def test_deliberate_silence_is_preserved(gemini):
    gemini([_response(parts=[], finish=types.FinishReason.STOP)] * 3)

    result = await _call()

    assert result == {"stop_reason": "end_turn", "content": []}


async def test_non_retryable_error_propagates_immediately(gemini):
    client = gemini([Exception("400 INVALID_ARGUMENT")])

    with pytest.raises(Exception, match="400"):
        await _call()
    assert client.calls == 1
