"""Text-to-speech: Kokoro-82M (primary, EN + HI) with Piper as optional fallback."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from server.config import ROOT, TTSConfig

log = logging.getLogger(__name__)


@dataclass(slots=True)
class TTSResult:
    audio: np.ndarray  # float32 mono in [-1, 1]
    sample_rate: int
    engine: str
    voice: str
    lang: str
    elapsed_s: float

    @property
    def duration_s(self) -> float:
        return len(self.audio) / self.sample_rate if self.sample_rate else 0.0

    def to_int16(self) -> np.ndarray:
        return (np.clip(self.audio, -1.0, 1.0) * 32767.0).astype(np.int16)


def pick_language(text: str, default: str = "en") -> str:
    """Choose the TTS language from the *text script* (Devanagari -> Hindi).

    Phase 5 refines this for mixed Hindi/English (Hinglish) sentences.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return default
    dev = sum(1 for c in letters if "\u0900" <= c <= "\u097f")
    return "hi" if dev / len(letters) >= 0.3 else "en"


class TTSEngine(Protocol):
    name: str

    def load(self) -> None: ...
    def synthesize(self, text: str, lang: str) -> TTSResult: ...


class KokoroEngine:
    name = "kokoro"
    SAMPLE_RATE = 24_000
    LANG_CODES = {"en": "a", "hi": "h"}  # Kokoro lang codes: a=US English, h=Hindi

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self._kmodel = None
        self._pipelines: dict[str, object] = {}

    def load(self) -> None:
        if self._kmodel is not None:
            return
        import torch
        from kokoro import KModel, KPipeline

        torch.set_num_threads(self.cfg.threads)
        t0 = time.perf_counter()
        log.info("Loading Kokoro-82M (%s)...", self.cfg.device)
        # One shared model; one lightweight G2P pipeline per language.
        self._kmodel = KModel(repo_id=self.cfg.kokoro_repo).to(self.cfg.device).eval()
        for lang in self.cfg.voices:
            code = self.LANG_CODES.get(lang)
            if code is None:
                log.warning("Kokoro: unsupported language '%s' in voices, skipping", lang)
                continue
            self._pipelines[lang] = KPipeline(
                lang_code=code, repo_id=self.cfg.kokoro_repo, model=self._kmodel
            )
        log.info("Kokoro loaded in %.1fs (langs: %s)", time.perf_counter() - t0,
                 list(self._pipelines))

    def synthesize(self, text: str, lang: str) -> TTSResult:
        self.load()
        if lang not in self._pipelines:
            lang = "en"
        voice = self.cfg.voices[lang]
        t0 = time.perf_counter()
        parts: list[np.ndarray] = []
        for result in self._pipelines[lang](text, voice=voice, speed=self.cfg.speed):
            if result.audio is not None:
                parts.append(result.audio.detach().cpu().numpy().astype(np.float32))
        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        return TTSResult(audio, self.SAMPLE_RATE, self.name, voice, lang,
                         time.perf_counter() - t0)


class PiperEngine:
    name = "piper"

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self._voices: dict[str, object] = {}
        self._dir = (ROOT / cfg.piper.models_dir).resolve()

    def load(self) -> None:
        import piper  # noqa: F401  (fail early if not installed)

    def _get_voice(self, lang: str):
        if lang not in self.cfg.piper.voices:
            lang = "en"
        if lang not in self._voices:
            from piper import PiperVoice

            name = self.cfg.piper.voices[lang]
            path = Path(self._dir) / f"{name}.onnx"
            if not path.exists():
                raise FileNotFoundError(
                    f"Piper voice missing: {path}. Run: python scripts/download_models.py --piper"
                )
            log.info("Loading Piper voice %s", name)
            self._voices[lang] = PiperVoice.load(str(path))
        return lang, self._voices[lang]

    def synthesize(self, text: str, lang: str) -> TTSResult:
        lang, voice = self._get_voice(lang)
        t0 = time.perf_counter()
        sr = int(voice.config.sample_rate)
        if hasattr(voice, "synthesize_stream_raw"):  # piper-tts 1.2.x
            raw = b"".join(voice.synthesize_stream_raw(text))
            pcm = np.frombuffer(raw, dtype=np.int16)
        else:  # piper-tts >= 1.3
            pcm = np.concatenate([c.audio_int16_array for c in voice.synthesize(text)])
        audio = pcm.astype(np.float32) / 32768.0
        return TTSResult(audio, sr, self.name, self.cfg.piper.voices[lang], lang,
                         time.perf_counter() - t0)


def _make_engine(name: str, cfg: TTSConfig) -> TTSEngine:
    return KokoroEngine(cfg) if name == "kokoro" else PiperEngine(cfg)


class TTS:
    """Primary engine with automatic fallback. Only one engine is loaded unless
    the primary fails."""

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self._active: TTSEngine | None = None

    def load(self) -> None:
        if self._active is not None:
            return
        try:
            engine = _make_engine(self.cfg.engine, self.cfg)
            engine.load()
            self._active = engine
        except Exception as e:  # noqa: BLE001
            if self.cfg.fallback == "none" or self.cfg.fallback == self.cfg.engine:
                raise
            log.error("Primary TTS '%s' failed to load (%r). Falling back to '%s'.",
                      self.cfg.engine, e, self.cfg.fallback)
            engine = _make_engine(self.cfg.fallback, self.cfg)
            engine.load()
            self._active = engine

    @property
    def engine_name(self) -> str:
        return self._active.name if self._active else "unloaded"

    def synthesize(self, text: str, lang: str | None = None) -> TTSResult:
        """Blocking call - run it in an executor from async code."""
        self.load()
        assert self._active is not None
        lang = lang or pick_language(text)
        return self._active.synthesize(text, lang)

    def warmup(self) -> None:
        self.load()
        t0 = time.perf_counter()
        for lang, text in (("en", "Hello there."), ("hi", "नमस्ते।")):
            if lang in self.cfg.voices:
                try:
                    self.synthesize(text, lang)
                except Exception as e:  # noqa: BLE001
                    log.warning("TTS warmup (%s) failed: %r", lang, e)
        log.info("TTS warmup done in %.2fs (engine=%s)", time.perf_counter() - t0,
                 self.engine_name)
