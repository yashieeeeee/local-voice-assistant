"""FastAPI app: serves the web UI and the /ws voice WebSocket.

Run:  python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect, WebSocketState

from server.config import ROOT, get_settings
from server.pipeline import Pipeline
from server.session import Session
from server.utils import log_ram, setup_logging

log = logging.getLogger("server")
MIN_SAMPLES = 16_000 // 5  # ignore utterances shorter than 200 ms


class WSSink:
    """Serializes sends from the turn task and the receive loop."""

    def __init__(self, ws: WebSocket) -> None:
        self.ws = ws
        self._lock = asyncio.Lock()

    def _open(self) -> bool:
        return self.ws.application_state == WebSocketState.CONNECTED

    async def send_json(self, data: dict) -> None:
        async with self._lock:
            if self._open():
                await self.ws.send_text(json.dumps(data, ensure_ascii=False))

    async def send_bytes(self, data: bytes) -> None:
        async with self._lock:
            if self._open():
                await self.ws.send_bytes(data)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    settings = get_settings()
    pipeline = Pipeline(settings)
    await pipeline.startup()
    log_ram("server ready")
    app.state.settings = settings
    app.state.pipeline = pipeline
    yield
    await pipeline.shutdown()


app = FastAPI(title="Local Voice Assistant", lifespan=lifespan)


async def _cancel(task: asyncio.Task | None) -> None:
    if task is not None and not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    pipeline: Pipeline = app.state.pipeline
    settings = app.state.settings
    session = Session(settings)
    sink = WSSink(ws)
    task: asyncio.Task | None = None
    turn_id = 0
    log.info("Client connected")
    await sink.send_json({
        "type": "ready", "profile": settings.profile, "stt": settings.stt.model,
        "llm": settings.llm.model, "tts": pipeline.tts.engine_name,
    })
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if msg.get("bytes") is not None:  # one complete utterance, int16 16 kHz mono
                t0 = time.perf_counter()
                await _cancel(task)  # safety: client should already have interrupted
                pcm = np.frombuffer(msg["bytes"], dtype=np.int16)
                if len(pcm) < MIN_SAMPLES:
                    await sink.send_json({"type": "done", "turn": turn_id, "empty": True})
                    continue
                turn_id += 1
                task = asyncio.create_task(pipeline.run_turn(session, pcm, sink, turn_id, t0))

            elif msg.get("text") is not None:
                data = json.loads(msg["text"])
                kind = data.get("type")
                if kind == "interrupt":
                    await _cancel(task)  # LLM stream + TTS stop here
                    session.apply_interrupt(
                        data.get("turn"), int(data.get("sentence", -1)),
                        float(data.get("fraction", 0.0)),
                    )
                    await sink.send_json({"type": "interrupted"})
                elif kind == "reset":
                    await _cancel(task)
                    session.reset()
                    await sink.send_json({"type": "reset_ok"})
    except WebSocketDisconnect:
        pass
    finally:
        await _cancel(task)
        log.info("Client disconnected")


# Static UI last so it doesn't shadow /ws
app.mount("/", StaticFiles(directory=str(ROOT / "web"), html=True), name="web")
