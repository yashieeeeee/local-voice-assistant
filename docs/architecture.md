# Architecture

Everything runs locally. The browser does voice-activity detection; the Python server does STT, LLM and TTS.

```mermaid
flowchart LR
    subgraph Browser
        MIC[Microphone] --> VAD["Silero VAD<br/>(vad-web, WASM)"]
        VAD -- "utterance PCM<br/>16 kHz int16" --> WSC[WebSocket client]
        WSC -- "audio frames" --> PLAY["AudioWorklet player<br/>(gapless queue)"]
        PLAY --> SPK[Speakers]
        VAD -. "speech start while playing<br/>= barge-in" .-> PLAY
    end

    WSC <== "ws://localhost:8000/ws" ==> WS[FastAPI WebSocket]

    subgraph Server["Python server (asyncio)"]
        WS --> STT["STT<br/>faster-whisper large-v3-turbo<br/>(GPU, thread executor)"]
        STT -- "transcript + language" --> LLM["LLM<br/>Ollama qwen2.5:3b<br/>(token stream)"]
        LLM -- tokens --> CH["Sentence chunker<br/>. ? ! । newline"]
        CH -- sentences --> TTS["TTS<br/>Kokoro-82M (EN/HI)<br/>(CPU, thread executor)"]
        TTS -- "PCM frames" --> WS
        SES[(Session history<br/>rolling window)] <--> LLM
    end
```

## Streaming pipeline (one turn)

```mermaid
sequenceDiagram
    participant B as Browser
    participant S as Server
    participant W as Whisper (GPU)
    participant L as Ollama LLM
    participant K as Kokoro TTS (CPU)
    B->>S: utterance (binary, int16 16 kHz)
    S->>W: transcribe (executor)
    W-->>S: text + language
    S-->>B: transcript (JSON, immediately)
    S->>L: chat stream (history + system prompt)
    loop tokens
        L-->>S: token
        Note over S: chunker flushes at sentence end
        S->>K: synthesize sentence N (executor)
        K-->>S: PCM
        S-->>B: audio header (JSON) + PCM (binary)
        Note over B,K: browser plays N-1 while server synthesizes N and LLM writes N+1
    end
    S-->>B: done + per-stage timings
```

## Barge-in

1. VAD fires `onSpeechRealStart` while audio is playing (or a turn is in flight).
2. The playback worklet is flushed instantly; it reports which sentence was playing and how far.
3. Client sends `{"type":"interrupt","turn":T,"sentence":i,"fraction":f}`.
4. Server cancels the turn task. Cancelling closes the HTTP stream to Ollama (generation stops) and drops queued TTS work.
5. `Session.apply_interrupt` rewrites the assistant message to **only what was spoken** (earlier sentences + a word-boundary cut of the current one). Nothing spoken -> the assistant message is dropped.
6. Server acks with `{"type":"interrupted"}`; the next utterance proceeds normally.

## Memory discipline (8 GB machine)

* One STT, one LLM (inside the Ollama process), one TTS model, all loaded once at startup and shared by every session.
* STT and TTS run in single-thread executors, so there is never more than one inference of each kind in flight.
* Only complete utterances are held in memory (not session audio); history is capped by turns *and* an estimated token budget, and trimmed turns collapse into a one-line "earlier topics" note.

## WebSocket protocol

| Direction | Message | Meaning |
|---|---|---|
| C -> S | binary | one utterance, int16 mono 16 kHz |
| C -> S | `{"type":"interrupt","turn","sentence","fraction"}` | barge-in with playback progress |
| C -> S | `{"type":"reset"}` | clear conversation |
| S -> C | `{"type":"ready", ...}` | models loaded |
| S -> C | `{"type":"transcript","text","language"}` | STT result |
| S -> C | `{"type":"audio","turn","index","text","sample_rate"}` + binary PCM | one synthesized sentence |
| S -> C | `{"type":"done","timings":{...}}` | turn complete + latency breakdown |
| S -> C | `{"type":"interrupted"}` | barge-in acknowledged |
| S -> C | `{"type":"error","message"}` | failure |
