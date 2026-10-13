"""Turn orchestration: audio -> STT -> streaming LLM -> sentence chunker -> TTS -> audio frames.

Models are created once and shared by all sessions. STT and TTS run in
single-thread executors so the asyncio loop never blocks and RAM stays flat.
LLM token generation, sentence chunking and TTS run as a pipeline: while the TTS
worker synthesizes sentence N, the LLM keeps generating sentence N+1, and the
browser plays sentence N-1.
"""
from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from server.chunker import SentenceChunker
from server.config import Settings
from server.llm import LLM, LLMError
from server.session import Session
from server.stt import STT
from server.tts import TTS
from server.utils import log_ram

log = logging.getLogger(__name__)


class Sink(Protocol):
    async def send_json(self, data: dict) -> None: ...
    async def send_bytes(self, data: bytes) -> None: ...


@dataclass
class Timings:
    """All values in milliseconds."""
    stt_ms: float = 0.0              # utterance received -> transcript ready
    llm_first_token_ms: float = 0.0  # transcript -> first LLM token
    first_sentence_ms: float = 0.0   # first token -> first sentence flushed
    first_audio_ms: float = 0.0      # first sentence -> first audio bytes sent
    ttfa_ms: float = 0.0             # utterance received -> first audio bytes sent
    total_ms: float = 0.0
    audio_s: float = 0.0             # seconds of speech synthesized
    sentences: int = 0
    llm_tokens: int = 0
    language: str = ""

    def as_dict(self) -> dict:
        return {k: (round(v, 1) if isinstance(v, float) else v) for k, v in asdict(self).items()}


class Pipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stt = STT(settings.stt)
        self.llm = LLM(settings.llm)
        self.tts = TTS(settings.tts)
        self._stt_pool = ThreadPoolExecutor(1, thread_name_prefix="stt")
        self._tts_pool = ThreadPoolExecutor(1, thread_name_prefix="tts")

    async def startup(self) -> None:
        """Load + warm up everything once. Order is chosen to keep peak RAM low."""
        loop = asyncio.get_running_loop()
        log_ram("startup (nothing loaded)")
        await self.llm.check()
        await self.llm.warmup()
        log_ram("after LLM warmup")
        await loop.run_in_executor(self._stt_pool, self.stt.warmup)
        log_ram("after STT warmup")
        await loop.run_in_executor(self._tts_pool, self.tts.warmup)
        log_ram("after TTS warmup")
        log.info("Pipeline ready.")

    async def shutdown(self) -> None:
        await self.llm.aclose()
        self._stt_pool.shutdown(wait=False, cancel_futures=True)
        self._tts_pool.shutdown(wait=False, cancel_futures=True)

    async def run_turn(
        self,
        session: Session,
        pcm: np.ndarray,
        sink: Sink,
        turn_id: int,
        t_start: float | None = None,
    ) -> Timings | None:
        """Process one utterance. Cancel the task to barge-in."""
        loop = asyncio.get_running_loop()
        t_start = t_start if t_start is not None else time.perf_counter()
        tm = Timings()
        emitted: list[str] = []
        began = False
        producer: asyncio.Task | None = None
        try:
            # 1) STT ----------------------------------------------------------
            stt = await loop.run_in_executor(self._stt_pool, self.stt.transcribe, pcm)
            t_stt = time.perf_counter()
            tm.stt_ms = (t_stt - t_start) * 1000
            tm.language = stt.language
            log.info("STT %.0fms [%s] %r", tm.stt_ms, stt.language, stt.text)
            await sink.send_json({"type": "transcript", "turn": turn_id,
                                  "text": stt.text, "language": stt.language})
            if not stt.text.strip():
                await sink.send_json({"type": "done", "turn": turn_id, "empty": True})
                return None

            # 2) LLM stream -> sentences (producer task) ------------------------
            session.begin_turn(turn_id, stt.text)
            began = True
            messages = session.build_messages(stt.language)
            queue: asyncio.Queue = asyncio.Queue()
            marks: dict[str, float] = {}

            async def produce() -> None:
                chunker = SentenceChunker(self.settings.chunker)
                try:
                    async for tok in self.llm.stream_chat(messages):
                        now = time.perf_counter()
                        marks.setdefault("first_token", now)
                        for sent in chunker.feed(tok):
                            marks.setdefault("first_sentence", time.perf_counter())
                            await queue.put(sent)
                    for sent in chunker.flush():
                        marks.setdefault("first_sentence", time.perf_counter())
                        await queue.put(sent)
                    await queue.put(None)
                except asyncio.CancelledError:
                    raise
                except Exception as e:  # noqa: BLE001
                    await queue.put(e)

            producer = asyncio.create_task(produce())

            # 3) TTS each sentence as it arrives, stream frames to the client ----
            idx = 0
            while True:
                item = await queue.get()
                if item is None:
                    break
                if isinstance(item, Exception):
                    raise item
                res = await loop.run_in_executor(self._tts_pool, self.tts.synthesize, item)
                if len(res.audio) == 0:
                    continue
                pcm16 = res.to_int16().tobytes()
                await sink.send_json({
                    "type": "audio", "turn": turn_id, "index": idx, "text": item,
                    "sample_rate": res.sample_rate, "samples": len(res.audio),
                    "engine": res.engine, "voice": res.voice, "lang": res.lang,
                })
                await sink.send_bytes(pcm16)
                now = time.perf_counter()
                if idx == 0:
                    tm.ttfa_ms = (now - t_start) * 1000
                    t_tok = marks.get("first_token", now)
                    t_sent = marks.get("first_sentence", now)
                    tm.llm_first_token_ms = (t_tok - t_stt) * 1000
                    tm.first_sentence_ms = (t_sent - t_tok) * 1000
                    tm.first_audio_ms = (now - t_sent) * 1000
                emitted.append(item)
                tm.audio_s += res.duration_s
                idx += 1

            tm.sentences = idx
            tm.total_ms = (time.perf_counter() - t_start) * 1000
            tm.llm_tokens = self.llm.last_stats.completion_tokens
            log.info("TURN %d | stt %.0f | 1st token %.0f | 1st sentence %.0f | 1st audio %.0f "
                     "| TTFA %.0f ms | total %.0f ms | %d sentences",
                     turn_id, tm.stt_ms, tm.llm_first_token_ms, tm.first_sentence_ms,
                     tm.first_audio_ms, tm.ttfa_ms, tm.total_ms, tm.sentences)
            await sink.send_json({"type": "done", "turn": turn_id,
                                  "text": " ".join(emitted), "timings": tm.as_dict()})
            return tm

        except asyncio.CancelledError:
            log.info("Turn %d cancelled (barge-in) after %d sentence(s)", turn_id, len(emitted))
            raise
        except LLMError as e:
            log.error("LLM error: %s", e)
            await sink.send_json({"type": "error", "message": str(e)})
        except Exception as e:  # noqa: BLE001
            log.exception("Turn %d failed", turn_id)
            await sink.send_json({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            if producer is not None and not producer.done():
                producer.cancel()
                await asyncio.gather(producer, return_exceptions=True)
            if began:
                session.end_turn(emitted)  # history = only what was actually sent
        return None
