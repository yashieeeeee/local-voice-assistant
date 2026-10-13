"""Latency benchmark: runs sample WAVs through the full STT->LLM->TTS pipeline.

  python scripts/bench.py                       # uses data/samples/*.wav (auto-generates some if empty)
  python scripts/bench.py --wavs my_recordings --repeat 3

Prints avg / p50 / p95 per stage and writes docs/latency_results.md.
For honest numbers, put a few REAL microphone recordings (WAV) in data/samples/.
"""
from __future__ import annotations

import argparse
import asyncio
import platform
import subprocess

import common
import numpy as np
import psutil
import soundfile as sf

from server.config import ROOT, get_settings
from server.pipeline import Pipeline
from server.session import Session

SYNTH_PROMPTS = {
    "en_1": "What is the capital of France and why is it famous?",
    "en_2": "Give me a quick tip for staying focused while studying.",
    "en_3": "Explain what a neural network is in simple words.",
    "hi_1": "भारत की राजधानी क्या है और वहाँ क्या खास है?",
    "hi_2": "मुझे पढ़ाई में ध्यान लगाने के लिए एक आसान सुझाव दो।",
}
STAGES = [
    ("stt_ms", "STT"), ("llm_first_token_ms", "STT -> first LLM token"),
    ("first_sentence_ms", "first token -> first sentence"),
    ("first_audio_ms", "first sentence -> first audio"),
    ("ttfa_ms", "TIME TO FIRST AUDIO"), ("total_ms", "turn total"),
]


class CollectSink:
    def __init__(self) -> None:
        self.audio_bytes = 0
        self.transcript = ""

    async def send_json(self, d: dict) -> None:
        if d.get("type") == "transcript":
            self.transcript = d["text"]

    async def send_bytes(self, b: bytes) -> None:
        self.audio_bytes += len(b)


def make_synthetic_samples(pipeline: Pipeline, out_dir) -> None:
    print("No WAVs found - synthesizing sample questions with the local TTS...")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in SYNTH_PROMPTS.items():
        r = pipeline.tts.synthesize(text)
        n = int(len(r.audio) * 16000 / r.sample_rate)
        a16 = np.interp(np.linspace(0, len(r.audio) - 1, n), np.arange(len(r.audio)), r.audio)
        sf.write(out_dir / f"synth_{name}.wav", a16.astype(np.float32), 16000)


def hardware() -> str:
    gpu = "none"
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip() or "none"
    except Exception:  # noqa: BLE001
        pass
    return (f"{platform.system()} {platform.release()} | CPU: {platform.processor()} "
            f"({psutil.cpu_count(logical=False)}C/{psutil.cpu_count()}T) | "
            f"RAM: {psutil.virtual_memory().total / 1e9:.1f} GB | GPU: {gpu}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wavs", default=str(ROOT / "data" / "samples"))
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "docs" / "latency_results.md"))
    args = ap.parse_args()

    common.init()
    s = get_settings()
    pipeline = Pipeline(s)
    await pipeline.startup()

    from pathlib import Path
    wav_dir = Path(args.wavs)
    wavs = sorted(wav_dir.glob("*.wav"))
    if not wavs:
        make_synthetic_samples(pipeline, wav_dir)
        wavs = sorted(wav_dir.glob("*.wav"))

    rows: list[dict] = []
    for rep in range(args.repeat):
        for wav in wavs:
            pcm = common.load_wav_16k_mono_int16(wav)
            sink = CollectSink()
            tm = await pipeline.run_turn(Session(s), pcm, sink, turn_id=1)
            if tm is None:
                print(f"  {wav.name}: no result (empty transcript?)")
                continue
            rows.append(tm.as_dict())
            print(f"[{rep + 1}/{args.repeat}] {wav.name:<22} TTFA {tm.ttfa_ms:6.0f} ms | "
                  f"total {tm.total_ms:6.0f} ms | {tm.language} | {sink.transcript[:50]!r}")
    await pipeline.shutdown()
    if not rows:
        raise SystemExit("No successful turns.")

    lines = [f"Hardware: {hardware()}",
             f"Profile: {s.profile} | STT {s.stt.model} ({s.stt.device}/{s.stt.compute_type}, beam {s.stt.beam_size})"
             f" | LLM {s.llm.model} (ctx {s.llm.num_ctx}, max {s.llm.num_predict} tok) | TTS {s.tts.engine} ({s.tts.device})",
             f"Samples: {len(rows)} turns ({len(wavs)} files x {args.repeat})", "",
             "| Stage | avg (ms) | p50 (ms) | p95 (ms) |", "|---|---:|---:|---:|"]
    for key, label in STAGES:
        v = np.array([r[key] for r in rows])
        lines.append(f"| {label} | {v.mean():.0f} | {np.percentile(v, 50):.0f} | {np.percentile(v, 95):.0f} |")
    report = "\n".join(lines)
    print("\n" + report)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(report + "\n", encoding="utf-8")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
