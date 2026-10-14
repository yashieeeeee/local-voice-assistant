# Local Voice AI Assistant

An end-to-end voice assistant built for the Trikon Software Labs AI Internship assignment. You talk, it transcribes your speech, generates a reply with a local language model, and speaks the reply back. English, Hindi and Hinglish are supported.

All speech recognition, language-model and speech-synthesis inference runs locally on your machine. No paid or hosted AI APIs are used.

```
Mic -> Silero VAD (browser) -> WebSocket -> faster-whisper -> Ollama LLM -> sentence chunker -> Kokoro TTS -> WebSocket -> browser playback
```

Architecture diagram and WebSocket protocol: [`docs/architecture.md`](docs/architecture.md) | Demo script: [`docs/demo_script.md`](docs/demo_script.md)

## Models (default setup, runs on CPU)

| Role | Model | Runs on |
|---|---|---|
| STT | `faster-whisper` **small**, `int8`, beam size 1, built-in VAD filter | CPU |
| LLM | **Qwen2.5 3B** through Ollama (exact tag: `llm.model` in `config.yaml`) | Ollama |
| TTS | **Kokoro-82M** (English and Hindi voices) | CPU |
| VAD | **Silero VAD** (`@ricky0123/vad-web`, runs in the browser) | Browser |

LLM settings: `num_ctx 2048`, `num_predict 120`, `keep_alive 30m`. Every model is loaded once at startup, warmed up, and shared.

## Status

| Area | Status |
|---|---|
| Full pipeline (default CPU setup) on the author's Windows 11 laptop | Works end to end, as reported by the author |
| Simulated self-test (`scripts/selftest.py`, fake models) | Passes |
| Testing on other machines | Requested; see "Testing guide" below |
| Hinglish | Mixed-script output (Hindi words in Devanagari, English words in Latin script) handled in `server/tts.py` |
| Latency numbers | **Not published yet.** The table in "Benchmark" is empty until `scripts/bench.py` is run |
| GPU configuration, `low_memory` profile, Piper fallback | Optional. Implemented in code but **not validated end to end** (see "Optional configurations") |

## Requirements

* Windows 11 (PowerShell). The code is plain Python and may work on Linux or macOS, but only Windows is documented here.
* Python 3.11, Git and Ollama.
* 8 GB RAM is tight for the default setup. 16 GB is recommended (not yet verified on other machines). No GPU is needed.
* A few GB of free disk space for models.
* Chrome or Edge, a microphone, and **headphones** (speakers let the mic hear the assistant and cause false interruptions).
* Internet for setup (downloads) and for the first page load (see "Internet requirements").

## Testing guide (step by step)

### Step 1: Install the prerequisites

Install Git, Python 3.11 and Ollama. If you have `winget`:

```powershell
winget install Git.Git
winget install Python.Python.3.11
winget install Ollama.Ollama
```

Then **close PowerShell and open a new window** so the new programs are on `PATH`.

### Step 2: Clone the project

```powershell
git clone https://github.com/yashieeeeee/local-voice-assistant.git
cd local-voice-assistant
```

### Step 3: Set up Python

Run these one by one:

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The prompt should now start with `(.venv)`. The torch install is large and takes a few minutes.

### Step 4: Download the AI models

Make sure Ollama is running (the Ollama app in the system tray, or `ollama serve` in another window). Check with `ollama list`; it should print a table without an error.

```powershell
$env:PYTHONUTF8=1
python scripts/download_models.py
```

This downloads Whisper, the Qwen2.5 3B model (the tag from `config.yaml`) and the Kokoro voices. It can take a while. Afterwards, confirm the LLM is installed:

```powershell
ollama list
```

The tag in the `llm.model` line of `config.yaml` must appear in that list. If it does not, run `ollama pull <that tag>`; otherwise the server stops at startup with "model not found".

### Step 5: Run the self-test

```powershell
python scripts/selftest.py
```

It uses fake models and takes a few seconds. It passes when the last line reads `ALL SELF-TESTS PASSED`. It checks the WebSocket protocol, sentence streaming, barge-in cancellation and history handling, not the real models.

### Step 6: Start the assistant

```powershell
python -m uvicorn server.main:app --host 127.0.0.1 --port 8000
```

Wait until the log says `Pipeline ready.` The first start loads and warms up every model and can take a while. **Keep this PowerShell window open** while testing.

### Step 7: Test it

1. Open http://127.0.0.1:8000 in Chrome or Edge.
2. Click **Start conversation** and allow microphone access.
3. Ask an English question, a Hindi question and a Hinglish question.
4. Ask a simple maths question, such as "What is 2 plus 3?"
5. Ask a follow-up that depends on the previous answer, to check conversation context.
6. Interrupt the assistant while it is speaking to test barge-in.

The panel on the right shows per-stage latency for each turn, in milliseconds.

### Step 8: Report back

Please share:

* Any errors (copy the text from the PowerShell window).
* Incorrect transcripts: what you said and what appeared on screen.
* Response delays: roughly how long until you heard the first word.
* Unexpected interruptions: the assistant stopping when you were not speaking.
* Your laptop specs. These commands print them:

```powershell
systeminfo | findstr /C:"Total Physical Memory" /C:"OS Name"
Get-CimInstance Win32_Processor | Select-Object Name
Get-CimInstance Win32_VideoController | Select-Object Name
```

## Internet requirements

* **Inference is fully local.** Transcription, the language model and speech synthesis never contact an external service while you use the assistant.
* **Setup needs internet** to download models (Hugging Face, Ollama) and Python packages.
* **The browser's VAD runtime needs internet on the first page load.** `index.html` loads `onnxruntime-web` and `@ricky0123/vad-web` (including the Silero model) from the jsDelivr CDN. A browser may cache these for later visits, but that is not guaranteed.

## Features

* **Streaming pipeline.** LLM tokens are split into sentences (at `.`, `?`, `!`, `।` and newlines). Each sentence is synthesized as soon as it is ready, so speech can start before the full reply exists.
* **Voice activity detection** in the browser decides when an utterance ends.
* **Interruption (barge-in).** When the browser VAD detects you speaking during playback, it stops audio and tells the server. The server cancels the in-flight LLM stream and TTS work, and the conversation history keeps only the text that was actually played.
* **Multilingual.** Whisper language detection is restricted to the languages in `stt.languages` (default English and Hindi). The TTS voice is chosen from the script of each sentence.
* **Conversation context.** Per-session history with a rolling window (last N turns plus a token budget). Trimmed turns are reduced to a short "earlier topics" note.
* **Per-turn latency timings** are logged by the server and shown in the UI.

## Benchmark

```powershell
python scripts/bench.py --repeat 3
```

Put a few real microphone recordings (WAV) in `data/samples/` for realistic results. If the folder is empty, the script synthesizes sample questions with the local TTS, which is cleaner audio than real speech. The report is written to `docs/latency_results.md`.

### Observed latency

_Not measured yet. Run the benchmark and paste the table here with the machine's specification._

| Stage | avg (ms) | p95 (ms) |
|---|---:|---:|
| STT | | |
| STT to first LLM token | | |
| first token to first sentence | | |
| first sentence to first audio | | |
| **Time to first audio** | | |

Server timings start when the utterance arrives at the server. The browser's "Perceived" figure additionally includes the VAD silence window (`redemptionMs` in `web/app.js`).

## Optional configurations (not validated end to end)

* **GPU.** Needs an NVIDIA GPU with enough VRAM and CUDA 12 cuBLAS/cuDNN DLLs on `PATH`. Whisper large-v3-turbo on a 4 GB GPU worked in the standalone STT test but ran out of memory in the full pipeline on the author's machine, so treat GPU use as experimental. To move only STT to the GPU:
  ```powershell
  $env:VA_STT__MODEL="large-v3-turbo"
  $env:VA_STT__DEVICE="cuda"
  $env:VA_STT__COMPUTE_TYPE="float16"
  $env:VA_STT__CPU_THREADS="0"
  ```
  The built-in `gpu` profile (`$env:VA_PROFILE="gpu"`) also selects `qwen2.5:7b-instruct` and TTS on CUDA. Qwen 7B ran out of memory on the author's machine.
* **`low_memory` profile** (Whisper `base`, `qwen2.5:1.5b-instruct`): `$env:VA_PROFILE="low_memory"`, then run `python scripts/download_models.py` once to fetch those models.
* **Piper TTS fallback.** The code can fall back to Piper if Kokoro fails to load (`tts.fallback` in `config.yaml`). It is not installed by default. Try it with `pip install piper-tts` and `python scripts/download_models.py --piper`.

Make sure `profile` in `config.yaml` is `default` for the CPU setup described above (`auto` selects the GPU profile whenever CUDA is detected).

## Tuning

Settings can be overridden without editing files, for example `$env:VA_STT__LANGUAGE="en"` (set it in the same PowerShell window before starting the server).

| Problem | Things to try |
|---|---|
| Slow or inaccurate transcription | Compare Whisper `small` and `medium` on the same recording. For an English-only demo, `VA_STT__LANGUAGE="en"` skips language detection. |
| Long wait for the first LLM token | A smaller model, lower `llm.num_ctx`, a shorter system prompt. |
| Odd or inconsistent answers | Lower `llm.temperature` (for example 0.3). Check the transcript first: wrong answers often come from a wrong transcript. |
| Long wait for the first spoken sentence | Lower `chunker.min_chars_first`. |
| Assistant interrupts itself | Use headphones. In `web/app.js`, raise `positiveSpeechThreshold` and `minSpeechMs`. |
| Feels slow after you stop talking | Lower `redemptionMs` in `web/app.js`. |
| Out of memory | Close other applications, try the `low_memory` profile, check the Windows pagefile size. |

## Troubleshooting

* **`py` or `git` not recognized**: close and reopen PowerShell after installing.
* **Activation is blocked**: run `Set-ExecutionPolicy -Scope Process Bypass` in the same window, then activate again.
* **"Cannot reach Ollama"**: start the Ollama app or run `ollama serve`.
* **"Model not found"**: the tag in `config.yaml` is not installed. Compare it with `ollama list`.
* **Page reports that the VAD failed to load**: check internet access and hard-refresh (Ctrl+F5). The page must be opened from `localhost` for microphone access.
* **No microphone prompt or no sound**: check the browser's site permissions and the Windows default input/output devices.
* **Hindi sounds wrong**: Kokoro's Hindi voice needs Devanagari text. If the LLM answered in Latin script, adjust the system prompt in `config.yaml`.
* **`?` characters instead of Devanagari in the console**: `$env:PYTHONUTF8=1`.
* **`C:` drive full during downloads**: set `HF_HOME` (and `OLLAMA_MODELS`) to another drive, then open a new terminal.
* **CUDA / cublas / cudnn errors**: only relevant to the GPU configuration; switch back to the default CPU settings.

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
* Whisper `small` can mis-transcribe Hindi, accented speech or noisy audio, and a 3B model can give wrong answers or weak Hindi/Hinglish. Larger models need more memory.