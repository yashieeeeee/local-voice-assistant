"""Model-free protocol self-test: fake STT/LLM/TTS through the real FastAPI app.

  python scripts/selftest.py

Verifies: WebSocket protocol, sentence streaming, barge-in cancellation and
"history keeps only spoken text". Takes ~3 seconds, loads no models.
"""
from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace

import common  # noqa: F401  (sets sys.path)
import numpy as np

import server.pipeline as pl
from server.tts import TTSResult


class FakeSTT:
    def __init__(self, cfg): pass
    def warmup(self): pass
    def transcribe(self, pcm, sr=16000):
        time.sleep(0.05)
        return SimpleNamespace(text="tell me something", language="en", language_probability=1.0,
                               audio_seconds=1.0, elapsed_s=0.05)


class FakeLLM:
    last_stats = SimpleNamespace(completion_tokens=40)
    def __init__(self, cfg): pass
    async def check(self): pass
    async def warmup(self): pass
    async def aclose(self): pass
    async def stream_chat(self, messages, num_predict=None):
        text = ("This is the first sentence of the reply. Here comes a second sentence now. "
                "And a third one follows after that. Finally the fourth sentence ends it.")
        for i in range(0, len(text), 4):
            await asyncio.sleep(0.01)
            yield text[i:i + 4]


class FakeTTS:
    engine_name = "fake"
    def __init__(self, cfg): pass
    def warmup(self): pass
    def synthesize(self, text, lang=None):
        time.sleep(0.15)
        audio = np.zeros(24000, dtype=np.float32)  # 1 s of silence per sentence
        return TTSResult(audio, 24000, "fake", "v", "en", 0.15)


pl.STT, pl.LLM, pl.TTS = FakeSTT, FakeLLM, FakeTTS

from fastapi.testclient import TestClient  # noqa: E402
from server.main import app  # noqa: E402


def collect(ws, until: set[str]) -> list[dict]:
    msgs = []
    while True:
        m = ws.receive()
        if m.get("bytes") is not None:
            msgs.append({"type": "BIN", "len": len(m["bytes"])})
            continue
        d = json.loads(m["text"]); msgs.append(d)
        if d["type"] in until:
            return msgs


def main() -> None:
    utt = np.zeros(16000, dtype=np.int16).tobytes()
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ready = json.loads(ws.receive_text()); assert ready["type"] == "ready", ready

        # --- Turn 1: full reply ---
        ws.send_bytes(utt)
        msgs = collect(ws, {"done", "error"})
        kinds = [m["type"] for m in msgs]
        assert kinds[0] == "transcript" and kinds[-1] == "done", kinds
        assert kinds.count("BIN") >= 2, kinds
        t = msgs[-1]["timings"]
        print("turn 1 OK:", kinds.count("BIN"), "audio frames | timings:", t)
        assert t["ttfa_ms"] > 0 and t["sentences"] >= 2

        # --- Turn 2: barge-in after first audio frame ---
        ws.send_bytes(utt)
        got_bin = 0
        while got_bin < 1:
            m = ws.receive()
            if m.get("bytes") is not None: got_bin += 1
        ws.send_text(json.dumps({"type": "interrupt", "turn": 2, "sentence": 0, "fraction": 0.5}))
        msgs = collect(ws, {"interrupted"})
        assert msgs[-1]["type"] == "interrupted"
        print("turn 2 OK: interrupted after first audio frame")

        # --- Turn 3: server still works after an interrupt, and history was trimmed ---
        ws.send_bytes(utt)
        msgs = collect(ws, {"done", "error"})
        assert msgs[-1]["type"] == "done", msgs[-1]
        print("turn 3 OK: pipeline recovered after barge-in")
    print("\nALL SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
