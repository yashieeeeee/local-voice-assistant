# Local Voice AI Assistant

An end-to-end voice assistant built for the Trikon Software Labs AI Internship assignment. You talk, it transcribes your speech, generates a reply with a local language model, and speaks the reply back. English, Hindi and Hinglish are supported.

All speech recognition, language-model and speech-synthesis inference runs locally on your machine. No paid or hosted AI APIs are used.

```
Mic -> Silero VAD (browser) -> WebSocket -> faster-whisper -> Ollama LLM -> sentence chunker -> Kokoro TTS -> WebSocket -> browser playback
```

Architecture diagram and WebSocket protocol: [`docs/architecture.md`](docs/architecture.md) | Demo script: [`docs/demo_script.md`](docs/demo_script.md)

## Project status (please read)

This project is **under active testing**. Real-microphone voice testing is still in progress, and no latency figures have been published yet.

| Area | Status |
|---|---|
| STT, LLM and TTS each work individually (`scripts/test_*.py`) | Tested on the development machine |
| Full server run (Whisper small CPU/int8 + Ollama Qwen2.5 3B + Kokoro CPU) | Runs end to end on the development machine |
| Protocol, sentence streaming, barge-in cancellation, history trimming | Passes the simulated self-test (`scripts/selftest.py`, fake models) |
| Live voice use | **In progress.** Observed so far on an 8 GB machine: RAM allocation errors, slow or inaccurate transcription, occasional wrong LLM answers, and unexpected barge-in cancellations. Next step is re-testing on a machine with more RAM to separate hardware limits from bugs. |
| Hinglish output | Improved using mixed-script text (Hindi words in Devanagari, English words in Latin script) in `server/tts.py`. Quality depends on the LLM following the prompt. |
| GPU configuration (Whisper large-v3-turbo on CUDA) | **Optional and not validated end to end.** It worked in the standalone STT test, but the full pipeline ran out of memory on a 4 GB GPU / 8 GB RAM machine. |
| Latency benchmark (`scripts/bench.py`) | Implemented, but not yet run on the target hardware. The results table below is empty on purpose. |

## Models

### Current tested setup (CPU)

| Role | Model | Runs on |
|---|---|---|
| STT | `faster-whisper` **small**, `int8`, beam size 1, built-in VAD filter | CPU |
| LLM | **Qwen2.5 3B** through Ollama (exact tag: see `llm.model` in `config.yaml`) | Ollama |
| TTS | **Kokoro-82M** (English and Hindi voices) | CPU |
| VAD | **Silero VAD** (`@ricky0123/vad-web`, runs in the browser) | Browser |

Settings used for the LLM: `num_ctx 2048`, `num_predict 120`, `keep_alive 30m`. Models are loaded once at startup, warmed up, and shared by all sessions.

### Optional configurations (not the tested default)

* **Piper TTS fallback.** The code can fall back to Piper if Kokoro fails to load (`tts.fallback` in `config.yaml`). It is not installed by default and has not been tested here. Install `piper-tts` and run `python scripts/download_models.py --piper` to try it.
* **`low_memory` profile.** Intended for small machines: Whisper `base` and `qwen2.5:1.5b-instruct`. Switch with `$env:VA_PROFILE="low_memory"`. Not yet tested end to end.
* **GPU configuration.** For a machine with a capable NVIDIA GPU (more VRAM than the 4 GB machine this was developed on). Needs CUDA 12 cuBLAS/cuDNN DLLs on `PATH`. A lighter option is to move only STT to the GPU:
  ```powershell
  $env:VA_STT__MODEL="large-v3-turbo"
  $env:VA_STT__DEVICE="cuda"
  $env:VA_STT__COMPUTE_TYPE="float16"
  $env:VA_STT__CPU_THREADS="0"
  ```
  The built-in `gpu` profile (`$env:VA_PROFILE="gpu"`) is heavier: it also selects `qwen2.5:7b-instruct` and TTS on CUDA. Qwen 7B ran out of memory on the development machine, so treat it as untested.

## Hardware

* **Development machine:** Windows 11, 8 GB RAM. The tested setup above runs on CPU. Real voice testing on this machine is still in progress, and 8 GB has produced RAM allocation errors under live use.
* **Recommendation (not yet verified):** 16 GB RAM or more for a smoother experience. A GPU is optional.
* **Disk:** a few GB for models (Whisper, Kokoro weights, the Ollama model). If `C:` is short on space, point the Hugging Face cache to another drive: `setx HF_HOME D:\hf_cache`, then open a new terminal. Ollama has `OLLAMA_MODELS` for the same purpose.

## Internet requirements

* **Inference is fully local.** Transcription, the language model and speech synthesis never contact an external service while you use the assistant.
* **Setup needs internet** to download the models (Hugging Face, Ollama) and Python packages.
* **The browser's VAD runtime needs internet on its first page load.** `index.html` loads `onnxruntime-web` and `@ricky0123/vad-web` (including the Silero model) from the jsDelivr CDN. A browser may cache these for later visits, but this is not guaranteed. Bundling them locally is possible but not done yet.

## Installation (Windows 11, PowerShell, Python 3.11)

```powershell
git clone <your-repo-url>
cd voice-assistant

py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Ollama and the model tag

Install Ollama, then open a **new** terminal:

```powershell
winget install Ollama.Ollama
```

The model tag is defined in `config.yaml` under `llm.model`. Pull exactly that tag and verify it:

```powershell
Select-String -Path config.yaml -Pattern "^\s+model:"   # shows stt.model and llm.model
python scripts/download_models.py                       # pulls the llm.model tag, Whisper and Kokoro weights
ollama list                                             # the llm.model tag must appear here
```

If `ollama list` does not show the exact tag from `config.yaml`, either pull that tag (`ollama pull <tag>`) or change `llm.model` to a tag that is installed. A mismatch causes a "model not found" error at server startup.

## Run

```powershell
$env:PYTHONUTF8=1
python scripts/selftest.py        # optional: protocol test with fake models, takes a few seconds
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Wait for `Pipeline ready.` in the log. The first start loads and warms up every model, which takes a while on CPU. Then open **http://127.0.0.1:8000** in Chrome or Edge, click **Start conversation**, allow the microphone and talk.

**Use headphones.** With speakers, the microphone can pick up the assistant's own voice and trigger false interruptions.

## Features implemented in the code

* **Streaming pipeline.** LLM tokens are split into sentences (at `.`, `?`, `!`, `।` and newlines). Each sentence is synthesized as soon as it is ready, so speech can start before the full reply exists.
* **Voice activity detection** in the browser decides when an utterance ends.
* **Interruption (barge-in).** When the browser VAD detects you speaking during playback, it stops audio, tells the server, and the server cancels the in-flight LLM stream and TTS work. The conversation history keeps only the text that was actually played. This passes the simulated test; live behaviour is still being tuned (see status above).
* **Multilingual.** Whisper language detection is restricted to the languages in `stt.languages` (default English and Hindi). The TTS voice is chosen from the script of each sentence.
* **Conversation context.** Per-session history with a rolling window (last N turns plus a token budget). Trimmed turns are reduced to a short "earlier topics" note.
* **Per-turn latency timings** are logged by the server and shown in the UI.

## Benchmark

```powershell
python scripts/bench.py --repeat 3
```

Put a few real microphone recordings (WAV) in `data/samples/` for realistic results. If the folder is empty, the script synthesizes sample questions with the local TTS instead, which gives cleaner audio than real speech. The report is written to `docs/latency_results.md`.

### Observed latency

_Not measured yet. Run the benchmark on the machine you are reporting for and paste the table here, together with that machine's specification._

| Stage | avg (ms) | p95 (ms) |
|---|---:|---:|
| STT | | |
| STT to first LLM token | | |
| first token to first sentence | | |
| first sentence to first audio | | |
| **Time to first audio** | | |

Server timings start when the utterance arrives at the server. The browser's "Perceived" figure additionally includes the VAD silence window (`redemptionMs` in `web/app.js`).

## Tuning

Settings can be overridden without editing files, for example `$env:VA_STT__LANGUAGE="en"`.

| Problem | Things to try |
|---|---|
| Slow or inaccurate transcription | Compare Whisper `small` and `medium` on the same recording. For an English-only demo, `VA_STT__LANGUAGE="en"` skips language detection. `base` is faster but less accurate. |
| Long wait for the first LLM token | A smaller model, lower `llm.num_ctx`, a shorter system prompt. |
| Odd or inconsistent LLM answers | Lower `llm.temperature` (for example 0.3). First check the transcript: wrong answers often come from a wrong transcript. |
| Long wait for the first spoken sentence | Lower `chunker.min_chars_first`. |
| Assistant interrupts itself | Use headphones. In `web/app.js`, raise `positiveSpeechThreshold` and `minSpeechMs`. |
| Feels slow after you stop talking | Lower `redemptionMs` in `web/app.js`. |
| Out of memory | Close other applications, try the `low_memory` profile, check the Windows pagefile size. |

## Troubleshooting

* **"Cannot reach Ollama"**: start the Ollama app or run `ollama serve`.
* **"Model not found"**: the tag in `config.yaml` is not installed. Compare with `ollama list`.
* **CUDA / cublas / cudnn DLL errors**: only relevant to the GPU configuration. Refresh `PATH` in the current terminal or go back to the CPU settings.
* **Page reports that the VAD failed to load**: check internet access (first load) and hard-refresh. The page must be opened from `localhost` for microphone access.
* **Hindi sounds wrong**: Kokoro's Hindi voice needs Devanagari text. If the LLM answered in Latin script, the system prompt in `config.yaml` is the place to adjust.
* **`?` characters instead of Devanagari in the console**: `$env:PYTHONUTF8=1`.
* **`C:` drive full during downloads**: set `HF_HOME` (and `OLLAMA_MODELS`) to another drive.

## Project layout

```
server/   main.py (FastAPI + /ws)   pipeline.py (turn orchestration)
          stt.py   llm.py   tts.py   chunker.py   session.py   config.py   utils.py
web/      index.html   app.js   worklet.js   style.css
scripts/  download_models.py   bench.py   selftest.py   test_stt.py   test_llm.py   test_tts.py
docs/     architecture.md   demo_script.md   latency_results.md (generated by bench.py)
config.yaml   requirements.txt
```

## Design notes and limitations

* VAD runs in the browser, so no audio is streamed while you are silent.
* TTS is sentence by sentence, a compromise between naturalness and latency on CPU.
* History token counts are estimated with a character heuristic rather than a tokenizer, to avoid loading another model.
* One conversation per WebSocket connection. Models are shared and turns are serialized.
* A 3B model has limited accuracy and limited Hindi and Hinglish quality. A larger model needs more memory.