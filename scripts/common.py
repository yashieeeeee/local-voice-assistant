"""Shared helpers for the scripts/ folder (path setup, UTF-8, audio I/O)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from server.utils import setup_logging  # noqa: E402


def init(level: str = "INFO") -> None:
    """Force UTF-8 console output (Windows PowerShell + Devanagari) and set up logging."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    setup_logging(level)


def load_wav_16k_mono_int16(path: str | Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != 16_000:  # simple linear resample, fine for tests
        n_out = int(round(len(audio) * 16_000 / sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n_out), np.arange(len(audio)), audio)
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16)


def record_16k_mono_int16(seconds: float) -> np.ndarray:
    import sounddevice as sd

    print(f"Recording {seconds:.0f}s... speak now!")
    data = sd.rec(int(seconds * 16_000), samplerate=16_000, channels=1, dtype="int16")
    sd.wait()
    return data[:, 0].copy()


def play(audio: np.ndarray, sample_rate: int) -> None:
    import sounddevice as sd

    sd.play(audio, sample_rate)
    sd.wait()
