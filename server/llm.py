"""Streaming chat client for the Ollama HTTP API."""
from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from server.config import LLMConfig

log = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


@dataclass(slots=True)
class LLMStats:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    first_token_s: float | None = None
    total_s: float = 0.0


class LLM:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._client = httpx.AsyncClient(
            base_url=cfg.host,
            timeout=httpx.Timeout(cfg.request_timeout_s, connect=3.0),
        )
        self.last_stats = LLMStats()

    async def aclose(self) -> None:
        await self._client.aclose()

    async def check(self) -> None:
        """Verify Ollama is reachable and the model is pulled."""
        try:
            r = await self._client.get("/api/tags")
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(
                f"Cannot reach Ollama at {self.cfg.host} ({e!r}). "
                "Start it with `ollama serve` (or launch the Ollama app)."
            ) from e
        names = {m["name"] for m in r.json().get("models", [])}
        want = self.cfg.model
        if want not in names and f"{want}:latest" not in names:
            raise LLMError(f"Model '{want}' not found. Run: ollama pull {want}")

    async def stream_chat(
        self, messages: list[dict[str, str]], num_predict: int | None = None
    ) -> AsyncIterator[str]:
        """Yield text pieces as they arrive. Closing/cancelling the generator
        closes the HTTP connection, which makes Ollama stop generating."""
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.cfg.keep_alive,
            "options": {
                "num_ctx": self.cfg.num_ctx,
                "num_predict": num_predict or self.cfg.num_predict,
                "temperature": self.cfg.temperature,
            },
        }
        stats = LLMStats()
        t0 = time.perf_counter()
        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            if resp.status_code != 200:
                body = (await resp.aread()).decode(errors="replace")
                raise LLMError(f"Ollama HTTP {resp.status_code}: {body[:300]}")
            async for line in resp.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise LLMError(chunk["error"])
                piece = chunk.get("message", {}).get("content", "")
                if piece:
                    if stats.first_token_s is None:
                        stats.first_token_s = time.perf_counter() - t0
                    yield piece
                if chunk.get("done"):
                    stats.prompt_tokens = chunk.get("prompt_eval_count", 0)
                    stats.completion_tokens = chunk.get("eval_count", 0)
                    break
        stats.total_s = time.perf_counter() - t0
        self.last_stats = stats

    async def warmup(self) -> None:
        """Dummy call: loads the model into RAM and pins it via keep_alive."""
        t0 = time.perf_counter()
        msgs = [{"role": "system", "content": "Reply briefly."},
                {"role": "user", "content": "Hi"}]
        async for _ in self.stream_chat(msgs, num_predict=4):
            pass
        log.info("LLM warmup done in %.2fs (%s)", time.perf_counter() - t0, self.cfg.model)
