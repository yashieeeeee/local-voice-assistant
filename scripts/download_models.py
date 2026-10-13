"""Download every model the active profile needs (run once, needs internet).

  python scripts/download_models.py                  # active profile
  python scripts/download_models.py --profile low_memory
  python scripts/download_models.py --piper          # also fetch Piper fallback voices
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import common

import httpx


def dl_whisper(name: str) -> None:
    from faster_whisper.utils import download_model

    print(f"[STT] faster-whisper '{name}'...")
    path = download_model(name)
    print(f"[STT] ready: {path}")


def dl_ollama(host: str, model: str) -> None:
    print(f"[LLM] ollama pull {model}")
    try:
        with httpx.stream("POST", f"{host}/api/pull", json={"model": model, "stream": True},
                          timeout=None) as r:
            last = ""
            for line in r.iter_lines():
                if not line:
                    continue
                d = json.loads(line)
                if "error" in d:
                    raise RuntimeError(d["error"])
                status = d.get("status", "")
                if d.get("total") and d.get("completed"):
                    status += f" {d['completed'] / d['total'] * 100:5.1f}%"
                if status != last:
                    print(f"\r[LLM] {status:<60}", end="", flush=True)
                    last = status
            print()
    except httpx.ConnectError:
        sys.exit(f"[LLM] Cannot reach Ollama at {host}. Install it (winget install Ollama.Ollama) "
                 "and make sure it is running, then re-run.")


def dl_kokoro(repo: str, voices: list[str]) -> None:
    from huggingface_hub import snapshot_download

    print(f"[TTS] Kokoro '{repo}' voices={voices}")
    patterns = ["config.json", "kokoro-v1_0.pth", *[f"voices/{v}.pt" for v in voices]]
    path = snapshot_download(repo, allow_patterns=patterns)
    print(f"[TTS] ready: {path}")


def dl_piper(models_dir, voices: dict[str, str]) -> None:
    from huggingface_hub import hf_hub_download

    models_dir.mkdir(parents=True, exist_ok=True)
    for name in voices.values():
        lang_region, speaker, quality = name.split("-")  # e.g. en_US-lessac-medium
        lang = lang_region.split("_")[0]
        base = f"{lang}/{lang_region}/{speaker}/{quality}/{name}"
        for ext in (".onnx", ".onnx.json"):
            try:
                p = hf_hub_download("rhasspy/piper-voices", base + ext, local_dir=str(models_dir),
                                    local_dir_use_symlinks=False)
                # flatten into models_dir
                target = models_dir / f"{name}{ext}"
                if not target.exists():
                    os.replace(p, target)
            except Exception as e:  # noqa: BLE001
                print(f"[TTS] Piper voice '{name}{ext}' failed: {e!r} "
                      "(check the voice name in config.yaml)")
        print(f"[TTS] Piper voice ready: {name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", help="default | low_memory | gpu (overrides config.yaml)")
    ap.add_argument("--skip-stt", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--skip-tts", action="store_true")
    ap.add_argument("--piper", action="store_true", help="also download Piper fallback voices")
    args = ap.parse_args()

    common.init()
    if args.profile:
        os.environ["VA_PROFILE"] = args.profile  # must be set before get_settings()
    from server.config import ROOT, get_settings

    s = get_settings()
    print(f"Profile: {s.profile}\n")
    if not args.skip_stt:
        dl_whisper(s.stt.model)
    if not args.skip_llm:
        dl_ollama(s.llm.host, s.llm.model)
    if not args.skip_tts:
        dl_kokoro(s.tts.kokoro_repo, sorted(set(s.tts.voices.values())))
        if args.piper:
            dl_piper(ROOT / s.tts.piper.models_dir, s.tts.piper.voices)
    print("\nAll done.")


if __name__ == "__main__":
    main()
