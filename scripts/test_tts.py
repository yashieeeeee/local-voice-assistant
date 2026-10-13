"""Standalone TTS test: synthesizes English / Hindi / Hinglish samples to out/*.wav.

  python scripts/test_tts.py
  python scripts/test_tts.py --play
  python scripts/test_tts.py --engine piper
"""
from __future__ import annotations

import argparse

import common
import soundfile as sf

from server.config import ROOT, get_settings
from server.tts import TTS
from server.utils import log_ram

SAMPLES = [
    ("en", "Hello! I'm your voice assistant. How can I help you today?"),
    ("hi", "नमस्ते! मैं आपकी आवाज़ सहायक हूँ। आज मैं आपकी कैसे मदद कर सकती हूँ?"),
    ("hinglish", "आज मौसम काफी अच्छा है, क्या आप walk पर चलेंगे?"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["kokoro", "piper"])
    ap.add_argument("--play", action="store_true", help="play audio through speakers")
    args = ap.parse_args()

    common.init()
    s = get_settings()
    if args.engine:
        s.tts.engine = args.engine
    out_dir = ROOT / "out"
    out_dir.mkdir(exist_ok=True)

    log_ram("before TTS load")
    tts = TTS(s.tts)
    tts.warmup()
    log_ram("after TTS load+warmup")

    for name, text in SAMPLES:
        lang = None if name == "hinglish" else name  # hinglish -> script-based pick
        r = tts.synthesize(text, lang)
        path = out_dir / f"tts_{name}_{r.engine}.wav"
        sf.write(path, r.audio, r.sample_rate)
        print(f"[{name}] engine={r.engine} voice={r.voice} lang={r.lang} "
              f"synth={r.elapsed_s:.2f}s audio={r.duration_s:.1f}s "
              f"RTF={r.elapsed_s / max(r.duration_s, 1e-6):.2f} -> {path.name}")
        if args.play:
            common.play(r.audio, r.sample_rate)


if __name__ == "__main__":
    main()
