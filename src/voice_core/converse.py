"""Shared ConversationEngine for multi-turn AI conversation coaching.

Apps inject:
  system_prompt    — AI persona (fully app-controlled)
  topics           — [{id, label, description}] for frontend topic picker
  llm_config       — {url, model, api_key} (OpenAI-compatible)
  analysis_fn      — async(audio_path) → dict, called after each user turn
  tts_fn           — async(text) → Path, returns audio file path
  analysis_ready_fn — (turn_id, results) → None, called when analysis completes
  max_history      — rolling window of non-system messages (default 40)

The WS handler owns: transcription, session timing, user storage, WebSocket transport.
The engine owns: LLM calls, history management, async analysis dispatch, TTS.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Awaitable, Callable


class ConversationEngine:
    def __init__(
        self,
        system_prompt: str,
        topics: list[dict],
        llm_config: dict,
        analysis_fn: Callable[[Path], Awaitable[dict]] | None = None,
        tts_fn: Callable[[str], Awaitable[Path]] | None = None,
        analysis_ready_fn: Callable[[str, dict], None] | None = None,
        max_history: int = 40,
    ):
        self._system_prompt = system_prompt
        self.topics = topics
        self._llm_config = llm_config
        self._analysis_fn = analysis_fn
        self._tts_fn = tts_fn
        self._analysis_ready_fn = analysis_ready_fn
        self._max_history = max_history
        self._history: list[dict] = []
        self._turn_count: int = 0

    async def start(self, topic_id: str) -> dict:
        """Begin a conversation on the given topic. Returns opening message."""
        topic = next((t for t in self.topics if t["id"] == topic_id), None)
        topic_label = topic["label"] if topic else topic_id

        self._history = [{"role": "system", "content": self._system_prompt}]
        self._turn_count = 0

        opening = await self._llm_call(
            f"Please start a conversation about: {topic_label}. "
            "Keep your opening short — one or two sentences."
        )
        turn_id = str(uuid.uuid4())
        self._history.append({"role": "assistant", "content": opening})

        result: dict = {"opening_text": opening, "turn_id": turn_id}
        if self._tts_fn:
            audio_path = await self._tts_fn(opening)
            result["audio_url"] = str(audio_path)

        return result

    async def process_turn(self, transcript: str, audio_path: Path) -> dict:
        """Process a user turn. transcript is pre-computed by the WS handler."""
        self._history.append({"role": "user", "content": transcript})

        response_text = await self._llm_call(None)  # None = use history as-is
        turn_id = str(uuid.uuid4())
        self._turn_count += 1
        self._history.append({"role": "assistant", "content": response_text})
        self._trim_history()

        result: dict = {"response_text": response_text, "turn_id": turn_id}
        if self._tts_fn:
            audio_path_tts = await self._tts_fn(response_text)
            result["audio_url"] = str(audio_path_tts)

        # Dispatch analysis as background task
        if self._analysis_fn and self._analysis_ready_fn:
            asyncio.create_task(
                self._run_analysis(turn_id, audio_path)
            )

        return result

    async def end(self) -> dict:
        """End the conversation. Returns {turns}."""
        turns = self._turn_count
        self._history = []
        self._turn_count = 0
        return {"turns": turns}

    async def _run_analysis(self, turn_id: str, audio_path: Path) -> None:
        try:
            results = await self._analysis_fn(audio_path)
            self._analysis_ready_fn(turn_id, results)
        except Exception as e:
            print(f"[converse] analysis failed for turn {turn_id}: {e}")

    async def _llm_call(self, user_message: str | None) -> str:
        """Call the LLM. If user_message is not None, it's appended to history first."""
        import httpx

        messages = list(self._history)
        if user_message is not None:
            messages.append({"role": "user", "content": user_message})

        payload = {
            "model": self._llm_config["model"],
            "messages": messages,
            "temperature": 0.8,
        }
        headers = {"Authorization": f"Bearer {self._llm_config['api_key']}"}
        url = self._llm_config["url"].rstrip("/") + "/chat/completions"

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(url, content=json.dumps(payload), headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return data["choices"][0]["message"]["content"].strip()

    def _trim_history(self) -> None:
        """Keep system prompt + last max_history non-system messages."""
        system = [m for m in self._history if m["role"] == "system"]
        non_system = [m for m in self._history if m["role"] != "system"]
        if len(non_system) > self._max_history:
            non_system = non_system[-self._max_history:]
        self._history = system + non_system
