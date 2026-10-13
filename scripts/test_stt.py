"""Standalone STT test.

  python scripts/test_stt.py --record 5        # speak into the mic
  python scripts/test_stt.py --wav data/samples/hello.wav
"""
from __future__ import annotations

import argparse

import common

from server.config import get_settings
from server.stt import STT
from server.utils import log_ram


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", help="path to a WAV file (any rate/channels)")
    ap.add_argument("--record", type=float, default=5.0, help="seconds to record if no --wav")
    args = ap.parse_args()

    common.init()
    s = get_settings()
    log_ram("before STT load")
    stt = STT(s.stt)
    stt.load()
    stt.warmup()
    log_ram("after STT load+warmup")

    pcm = common.load_wav_16k_mono_int16(args.wav) if args.wav else common.record_16k_mono_int16(args.record)
    for run in (1, 2):  # run 2 shows steady-state latency
        r = stt.transcribe(pcm)
        print(f"\n[run {run}] lang={r.language} (p={r.language_probability:.2f}) "
              f"audio={r.audio_seconds:.1f}s stt={r.elapsed_s:.2f}s "
              f"RTF={r.elapsed_s / max(r.audio_seconds, 1e-6):.2f}")
        print(f"[run {run}] text: {r.text!r}")


if __name__ == "__main__":
    main()
