"""Speech-to-text with faster-whisper (loaded once, reused)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from server.config import STTConfig

log = logging.getLogger(__name__)
SAMPLE_RATE = 16_000


@dataclass(slots=True)
class STTResult:
    text: str
    language: str
    language_probability: float
    audio_seconds: float
    elapsed_s: float


def to_float32(pcm: np.ndarray) -> np.ndarray:
    """int16 PCM -> float32 in [-1, 1]. float32 input is passed through."""
    if pcm.dtype == np.int16:
        return pcm.astype(np.float32) / 32768.0
    return np.ascontiguousarray(pcm, dtype=np.float32)


class STT:
    def __init__(self, cfg: STTConfig) -> None:
        self.cfg = cfg
        self._model = None  # faster_whisper.WhisperModel

    def load(self) -> None:
        if self._model is not None:
            return
        from faster_whisper import WhisperModel

        t0 = time.perf_counter()
        log.info("Loading faster-whisper '%s' (%s, %s)...",
                 self.cfg.model, self.cfg.device, self.cfg.compute_type)
        self._model = WhisperModel(
            self.cfg.model,
            device=self.cfg.device,
            compute_type=self.cfg.compute_type,
            cpu_threads=self.cfg.cpu_threads,
            num_workers=1,
        )
        log.info("STT loaded in %.1fs", time.perf_counter() - t0)

    def _run(self, audio: np.ndarray, language: str | None, vad: bool):
        assert self._model is not None
        return self._model.transcribe(
            audio,
            language=language,
            beam_size=self.cfg.beam_size,
            vad_filter=vad,
            vad_parameters={"min_silence_duration_ms": self.cfg.vad_min_silence_ms},
            condition_on_previous_text=False,
            temperature=0.0,
            without_timestamps=True,
        )

    def transcribe(self, pcm: np.ndarray, sample_rate: int = SAMPLE_RATE) -> STTResult:
        """Blocking call - run it in an executor from async code."""
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz audio, got {sample_rate}")
        self.load()
        audio = to_float32(pcm)
        t0 = time.perf_counter()

        segments, info = self._run(audio, self.cfg.language, self.cfg.vad_filter)

        # Restrict auto-detect to the configured languages (e.g. avoids
        # Urdu/Punjabi/etc. being picked for short Hindi utterances).
        allowed = self.cfg.languages
        if self.cfg.language is None and allowed and info.language not in allowed:
            probs = dict(getattr(info, "all_language_probs", None) or [])
            best = max(allowed, key=lambda lang: probs.get(lang, 0.0))
            log.debug("Detected '%s' not in %s -> forcing '%s'", info.language, allowed, best)
            segments, info = self._run(audio, best, self.cfg.vad_filter)

        text = " ".join(s.text.strip() for s in segments).strip()
        return STTResult(
            text=text,
            language=info.language,
            language_probability=float(info.language_probability),
            audio_seconds=len(audio) / SAMPLE_RATE,
            elapsed_s=time.perf_counter() - t0,
        )

    def warmup(self) -> None:
        """Dummy transcription so the first real request isn't slow.

        VAD is disabled here on purpose: with VAD on, silence is stripped and the
        encoder would never run.
        """
        self.load()
        rng = np.random.default_rng(0)
        noise = (rng.standard_normal(SAMPLE_RATE) * 0.001).astype(np.float32)
        t0 = time.perf_counter()
        segments, _ = self._run(noise, "en", vad=False)
        list(segments)
        log.info("STT warmup done in %.2fs", time.perf_counter() - t0)
