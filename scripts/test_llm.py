"""Standalone LLM test: streams a reply from Ollama and prints latency stats.

  python scripts/test_llm.py
  python scripts/test_llm.py --prompt "भारत की राजधानी क्या है?"
"""
from __future__ import annotations

import argparse
import asyncio
import time

import common

from server.config import get_settings
from server.llm import LLM, LLMError

DEFAULT_PROMPTS = [
    "What is a voice assistant, in simple words?",
    "मुझे चाय बनाने का आसान तरीका बताओ।",
]


async def run(prompts: list[str]) -> None:
    s = get_settings()
    llm = LLM(s.llm)
    try:
        await llm.check()
        print(f"Warming up {s.llm.model} ...")
        await llm.warmup()
        for prompt in prompts:
            print(f"\nUSER: {prompt}\nBOT : ", end="", flush=True)
            msgs = [{"role": "system", "content": s.assistant.system_prompt.strip()},
                    {"role": "user", "content": prompt}]
            t0 = time.perf_counter()
            async for piece in llm.stream_chat(msgs):
                print(piece, end="", flush=True)
            st = llm.last_stats
            tps = st.completion_tokens / max(st.total_s - (st.first_token_s or 0), 1e-6)
            print(f"\n  -> first token {st.first_token_s:.2f}s | total {st.total_s:.2f}s | "
                  f"{st.completion_tokens} tokens (~{tps:.1f} tok/s) | prompt {st.prompt_tokens} tokens")
    except LLMError as e:
        raise SystemExit(f"LLM error: {e}")
    finally:
        await llm.aclose()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", help="single prompt (default: one English + one Hindi)")
    args = ap.parse_args()
    common.init()
    asyncio.run(run([args.prompt] if args.prompt else DEFAULT_PROMPTS))


if __name__ == "__main__":
    main()
