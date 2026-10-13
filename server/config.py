"""Typed configuration: config.yaml + profile overlay + .env / env-var overrides."""
from __future__ import annotations

import copy
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger(__name__)


class AssistantConfig(BaseModel):
    system_prompt: str = "You are a helpful voice assistant. Keep replies short."


class STTConfig(BaseModel):
    model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    beam_size: int = 1
    vad_filter: bool = True
    vad_min_silence_ms: int = 500
    cpu_threads: int = 4
    languages: list[str] = Field(default_factory=lambda: ["en", "hi"])
    language: str | None = None


class LLMConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "qwen2.5:3b-instruct"
    num_ctx: int = 2048
    num_predict: int = 120
    temperature: float = 0.6
    keep_alive: str = "30m"
    request_timeout_s: float = 120.0


class PiperConfig(BaseModel):
    models_dir: str = "models/piper"
    voices: dict[str, str] = Field(
        default_factory=lambda: {"en": "en_US-lessac-medium", "hi": "hi_IN-pratham-medium"}
    )


class TTSConfig(BaseModel):
    engine: Literal["kokoro", "piper"] = "kokoro"
    fallback: Literal["piper", "none"] = "piper"
    device: str = "cpu"
    threads: int = 4
    speed: float = 1.0
    kokoro_repo: str = "hexgrad/Kokoro-82M"
    voices: dict[str, str] = Field(default_factory=lambda: {"en": "af_heart", "hi": "hf_alpha"})
    piper: PiperConfig = Field(default_factory=PiperConfig)


class SessionConfig(BaseModel):
    max_history_tokens: int = 900
    keep_last_turns: int = 6


class ChunkerConfig(BaseModel):
    min_chars_first: int = 20
    min_chars: int = 40
    max_chars: int = 220


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class Settings(BaseModel):
    profile: str = "default"
    assistant: AssistantConfig = Field(default_factory=AssistantConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)


def cuda_available() -> bool:
    """CUDA check that doesn't need torch (CTranslate2 ships with faster-whisper)."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (overlay or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _apply_env_overrides(raw: dict[str, Any]) -> None:
    """VA_STT__MODEL=base  ->  raw['stt']['model'] = 'base'."""
    for key, value in os.environ.items():
        if not key.startswith("VA_") or "__" not in key:
            continue
        path = key[3:].lower().split("__")
        node = raw
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = yaml.safe_load(value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(ROOT / ".env")
    cfg_path = Path(os.getenv("VA_CONFIG", ROOT / "config.yaml"))
    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    profiles: dict[str, Any] = raw.pop("profiles", {}) or {}
    requested = os.getenv("VA_PROFILE") or raw.get("profile", "auto")
    profile = ("gpu" if cuda_available() else "default") if requested == "auto" else requested
    if profile not in profiles and profile != "default":
        raise ValueError(f"Unknown profile '{profile}'. Available: {sorted(profiles)}")

    raw = _deep_merge(raw, profiles.get(profile, {}))
    raw["profile"] = profile
    _apply_env_overrides(raw)

    settings = Settings(**raw)
    log.info(
        "Profile=%s | STT=%s/%s/%s | LLM=%s | TTS=%s/%s",
        profile, settings.stt.model, settings.stt.device, settings.stt.compute_type,
        settings.llm.model, settings.tts.engine, settings.tts.device,
    )
    return settings


if __name__ == "__main__":
    from server.utils import setup_logging

    setup_logging()
    print(get_settings().model_dump_json(indent=2))
    print("CUDA available:", cuda_available())
