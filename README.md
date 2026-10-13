# Local Voice AI Assistant

STT (faster-whisper) -> LLM (Ollama) -> TTS (Kokoro / Piper), all open-weight, all local.

> Work in progress - Phase 1 (skeleton + standalone model tests). Full README arrives in Phase 6.

## Setup (Windows 11, PowerShell, Python 3.11)

```powershell
cd voice-assistant
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

winget install Ollama.Ollama      # then open a NEW terminal
python scripts/download_models.py
```

## Phase 1 checks

```powershell
python -m server.config           # prints resolved config + CUDA status
python scripts/test_stt.py --record 5
python scripts/test_llm.py
python scripts/test_tts.py --play
```
