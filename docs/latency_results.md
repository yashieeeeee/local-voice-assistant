Hardware: Windows 10 | CPU: Intel64 Family 6 Model 154 Stepping 3, GenuineIntel (12C/16T) | RAM: 8.3 GB | GPU: NVIDIA GeForce RTX 2050, 4096 MiB
Profile: default | STT small (cpu/int8, beam 1) | LLM qwen2.5:3b (ctx 2048, max 120 tok) | TTS kokoro (cpu)
Samples: 15 turns (5 files x 3)

| Stage | avg (ms) | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|
| STT | 7737 | 6696 | 11284 |
| STT -> first LLM token | 659 | 651 | 748 |
| first token -> first sentence | 720 | 562 | 1787 |
| first sentence -> first audio | 4215 | 4413 | 5698 |
| TIME TO FIRST AUDIO | 13331 | 13340 | 16699 |
| turn total | 13924 | 13387 | 18209 |
