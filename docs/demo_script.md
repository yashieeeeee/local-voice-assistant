# Demo video script (2–3 minutes)

Record with OBS or Win+G. Wear headphones (prevents the assistant hearing itself). Have the terminal visible on one side and the browser on the other.

**0:00 – 0:20 | Intro**
"This is a fully local voice assistant for the Trikon assignment. Faster-Whisper large-v3-turbo for speech-to-text, Qwen 2.5 3B through Ollama as the LLM, and Kokoro-82M for speech. No hosted AI APIs. Everything runs on my laptop: 8 GB RAM and an RTX 2050."
Show the terminal: server startup log with the RAM lines and "Pipeline ready."

**0:20 – 0:50 | Basic English turn**
Click *Start conversation*. Ask: "What is the capital of France, and why is it famous?"
Point at the latency table: "Time to first audio is about X seconds. The transcript appears immediately, and speech starts before the LLM has finished — that's sentence-level streaming."

**0:50 – 1:20 | Context**
Follow up: "And how many people live there?" → it answers about Paris without being told.
"History is a rolling window — last turns plus a token budget."

**1:20 – 1:50 | Hindi + Hinglish**
Ask in Hindi: "मुझे चाय बनाने का आसान तरीका बताओ।" → Hindi voice.
Then Hinglish: "Aaj mausam kaisa hai, walk pe jaana chahiye?" → show detected language tag and voice switch.

**1:50 – 2:20 | Barge-in**
Ask for a longer answer ("Tell me about the history of the internet"). Interrupt mid-sentence: "Stop — tell me a joke instead."
"Playback stops instantly, the LLM stream and TTS are cancelled server-side, and the history only keeps what I actually heard." Show server log line `Turn N cancelled (barge-in)`.

**2:20 – 2:45 | Wrap-up**
Show `docs/latency_results.md` (from `python scripts/bench.py`) and the architecture diagram.
"Models, hardware requirements and measured latency are in the README. Thanks for watching."
