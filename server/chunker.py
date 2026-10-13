"""Sentence chunker: turns an LLM token stream into speakable sentences."""
from __future__ import annotations

import re

from server.config import ChunkerConfig

# Characters that end a sentence. "." only counts when followed by whitespace
# (so "3.5" and "e.g.x" don't split); the others count immediately.
_HARD_END = set("?!।\n")
_MD_CHARS = re.compile(r"[*_`#~>|]+")
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\uFE0F\u200d]+"
)
_BULLET = re.compile(r"^\s*(?:[-•]|\d+[.)])\s+", re.MULTILINE)


def clean_for_speech(text: str) -> str:
    """Strip markdown/emoji so TTS never reads out symbols."""
    text = _BULLET.sub("", text)
    text = _MD_CHARS.sub("", text)
    text = _EMOJI.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class SentenceChunker:
    """feed() tokens, get back complete sentences. Call flush() at stream end.

    The first sentence uses a lower minimum length so audio starts ASAP.
    """

    def __init__(self, cfg: ChunkerConfig) -> None:
        self.cfg = cfg
        self._buf = ""
        self._emitted = 0

    @property
    def _min_len(self) -> int:
        return self.cfg.min_chars_first if self._emitted == 0 else self.cfg.min_chars

    def _emit(self, raw: str, out: list[str]) -> None:
        text = clean_for_speech(raw)
        if any(c.isalnum() for c in text):  # skip empty / punctuation-only chunks
            out.append(text)
            self._emitted += 1

    def feed(self, token: str) -> list[str]:
        self._buf += token
        out: list[str] = []
        while True:
            cut = self._find_cut()
            if cut is None:
                break
            self._emit(self._buf[:cut], out)
            self._buf = self._buf[cut:]
        return out

    def flush(self) -> list[str]:
        out: list[str] = []
        self._emit(self._buf, out)
        self._buf = ""
        return out

    def _find_cut(self) -> int | None:
        buf = self._buf
        for i, ch in enumerate(buf):
            is_end = ch in _HARD_END or (
                ch == "."
                and i + 1 < len(buf)
                and buf[i + 1].isspace()
                and (i == 0 or buf[i - 1] != ".")
            )
            if is_end and len(clean_for_speech(buf[: i + 1])) >= self._min_len:
                return i + 1
        # No boundary but the buffer is getting long: split at a comma/space.
        if len(buf) > self.cfg.max_chars:
            window = buf[: self.cfg.max_chars]
            for sep in (", ", "; ", " "):
                j = window.rfind(sep)
                if j >= self._min_len:
                    return j + len(sep)
            return self.cfg.max_chars
        return None
