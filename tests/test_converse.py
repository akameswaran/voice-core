"""Tests for voice_core.converse.ConversationEngine."""
import asyncio
import json
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call
import pytest
import respx
import httpx

# Add voice-core src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from voice_core.converse import ConversationEngine


TOPICS = [
    {"id": "daily", "label": "Daily Life", "description": "Everyday conversations"},
    {"id": "work", "label": "Work", "description": "Professional topics"},
]

LLM_CONFIG = {
    "url": "http://localhost:11434/v1",
    "model": "qwen2.5",
    "api_key": "test",
}

LLM_URL = "http://localhost:11434/v1/chat/completions"

OPENING_CONTENT = "That sounds wonderful! Tell me more."

LLM_RESPONSE_JSON = {
    "choices": [{"message": {"content": OPENING_CONTENT}}]
}


@pytest.fixture
def engine():
    return ConversationEngine(
        system_prompt="You are a friendly conversation partner.",
        topics=TOPICS,
        llm_config=LLM_CONFIG,
    )


class TestConversationEngineInit:
    def test_topics_stored(self, engine):
        assert engine.topics == TOPICS

    def test_initial_history_empty(self, engine):
        assert engine._history == []

    def test_turn_count_zero(self, engine):
        assert engine._turn_count == 0

    def test_no_analysis_fn_by_default(self, engine):
        assert engine._analysis_fn is None

    def test_no_tts_fn_by_default(self, engine):
        assert engine._tts_fn is None

    def test_no_analysis_ready_fn_by_default(self, engine):
        assert engine._analysis_ready_fn is None


class TestConversationEngineStart:
    @respx.mock
    async def test_start_calls_llm(self, engine):
        route = respx.post(LLM_URL).mock(
            return_value=httpx.Response(200, json=LLM_RESPONSE_JSON)
        )
        result = await engine.start("daily")
        assert route.called
        request_body = json.loads(route.calls[0].request.content)
        assert request_body["model"] == "qwen2.5"

    @respx.mock
    async def test_start_returns_opening_text(self, engine):
        respx.post(LLM_URL).mock(
            return_value=httpx.Response(200, json=LLM_RESPONSE_JSON)
        )
        result = await engine.start("daily")
        assert result["opening_text"] == OPENING_CONTENT
        assert "turn_id" in result

    @respx.mock
    async def test_start_adds_to_history(self, engine):
        respx.post(LLM_URL).mock(
            return_value=httpx.Response(200, json=LLM_RESPONSE_JSON)
        )
        await engine.start("daily")
        # System prompt + assistant opening
        assert len(engine._history) == 2
        assert engine._history[0]["role"] == "system"
        assert engine._history[1]["role"] == "assistant"

    @respx.mock
    async def test_start_no_audio_url_without_tts_fn(self, engine):
        respx.post(LLM_URL).mock(
            return_value=httpx.Response(200, json=LLM_RESPONSE_JSON)
        )
        result = await engine.start("daily")
        assert result.get("audio_url") is None


class TestConversationEngineProcessTurn:
    @pytest.fixture
    async def started_engine(self, engine):
        with respx.mock:
            respx.post(LLM_URL).mock(
                return_value=httpx.Response(200, json=LLM_RESPONSE_JSON)
            )
            await engine.start("daily")
        return engine

    @respx.mock
    async def test_process_turn_returns_response(self, started_engine, tmp_path):
        audio = tmp_path / "turn.wav"
        audio.write_bytes(b"fake wav")
        respx.post(LLM_URL).mock(
            return_value=httpx.Response(200, json=LLM_RESPONSE_JSON)
        )
        result = await started_engine.process_turn("Hello there!", audio)
        assert result["response_text"] == OPENING_CONTENT
        assert "turn_id" in result

    @respx.mock
    async def test_process_turn_adds_user_and_assistant_to_history(self, started_engine, tmp_path):
        audio = tmp_path / "turn.wav"
        audio.write_bytes(b"fake wav")
        history_before = len(started_engine._history)
        respx.post(LLM_URL).mock(
            return_value=httpx.Response(200, json=LLM_RESPONSE_JSON)
        )
        await started_engine.process_turn("Hello there!", audio)
        assert len(started_engine._history) == history_before + 2
        assert started_engine._history[-2]["role"] == "user"
        assert started_engine._history[-2]["content"] == "Hello there!"
        assert started_engine._history[-1]["role"] == "assistant"

    @respx.mock
    async def test_process_turn_dispatches_analysis_async(self, tmp_path):
        analysis_calls = []
        async def mock_analysis(audio_path):
            analysis_calls.append(audio_path)
            return {"gender_score": 72, "resonance": 65}

        ready_calls = []
        def mock_ready(turn_id, results):
            ready_calls.append((turn_id, results))

        eng = ConversationEngine(
            system_prompt="You are friendly.",
            topics=TOPICS,
            llm_config=LLM_CONFIG,
            analysis_fn=mock_analysis,
            analysis_ready_fn=mock_ready,
        )
        respx.post(LLM_URL).mock(
            return_value=httpx.Response(200, json=LLM_RESPONSE_JSON)
        )
        await eng.start("daily")
        audio = tmp_path / "turn.wav"
        audio.write_bytes(b"fake wav")
        result = await eng.process_turn("Hi", audio)
        # Give background task time to run
        await asyncio.sleep(0.05)
        assert len(analysis_calls) == 1
        assert len(ready_calls) == 1
        assert ready_calls[0][0] == result["turn_id"]
        assert ready_calls[0][1]["gender_score"] == 72

    @respx.mock
    async def test_history_trimmed_to_max(self, tmp_path):
        eng = ConversationEngine(
            system_prompt="Be friendly.",
            topics=TOPICS,
            llm_config=LLM_CONFIG,
            max_history=6,  # system + 2 turns * 2 messages = 5 max non-system
        )
        respx.post(LLM_URL).mock(
            return_value=httpx.Response(200, json=LLM_RESPONSE_JSON)
        )
        await eng.start("daily")
        for i in range(5):
            audio = tmp_path / f"t{i}.wav"
            audio.write_bytes(b"fake")
            await eng.process_turn(f"Turn {i}", audio)
        # Should not exceed max_history + 1 (system prompt)
        assert len(eng._history) <= 7  # 1 system + max_history


class TestConversationEngineEnd:
    @respx.mock
    async def test_end_returns_turn_count(self, engine, tmp_path):
        respx.post(LLM_URL).mock(
            return_value=httpx.Response(200, json=LLM_RESPONSE_JSON)
        )
        await engine.start("daily")
        audio = tmp_path / "t.wav"
        audio.write_bytes(b"x")
        await engine.process_turn("Hello", audio)
        result = await engine.end()
        assert result["turns"] == 1

    @respx.mock
    async def test_end_clears_history(self, engine):
        respx.post(LLM_URL).mock(
            return_value=httpx.Response(200, json=LLM_RESPONSE_JSON)
        )
        await engine.start("daily")
        await engine.end()
        assert engine._history == []
        assert engine._turn_count == 0
