"""Per-connection conversation state with a rolling context window."""
from __future__ import annotations

import logging

from server.config import Settings

log = logging.getLogger(__name__)

LANG_NAMES = {"en": "English", "hi": "Hindi"}


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (no tokenizer needed): Latin ~4 chars/token,
    Devanagari/other scripts ~1.5 chars/token."""
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return int(ascii_chars / 4 + (len(text) - ascii_chars) / 1.5) + 4


class Session:
    def __init__(self, settings: Settings) -> None:
        self.cfg = settings.session
        self.system_prompt = settings.assistant.system_prompt.strip()
        self.history: list[dict[str, str]] = []
        self._dropped_topics: list[str] = []  # tiny rolling "summary" of trimmed turns
        self._last_assistant: dict[str, str] | None = None
        self._last_sentences: list[str] = []
        self.last_turn_id = 0

    # ---- turn lifecycle -------------------------------------------------
    def begin_turn(self, turn_id: int, user_text: str) -> None:
        self.last_turn_id = turn_id
        self._last_assistant = None
        self._last_sentences = []
        self.history.append({"role": "user", "content": user_text})
        self._trim()

    def end_turn(self, sentences: list[str]) -> None:
        """Commit what was actually sent to the client (called even on cancel)."""
        self._last_sentences = list(sentences)
        if sentences:
            entry = {"role": "assistant", "content": " ".join(sentences)}
            self.history.append(entry)
            self._last_assistant = entry

    def apply_interrupt(self, turn_id: int | None, sentence_idx: int, fraction: float) -> None:
        """Trim the last assistant message to what the user actually heard.

        sentence_idx = last sentence that started playing (-1 = none),
        fraction     = how much of that sentence had played.
        """
        if turn_id is not None and turn_id != self.last_turn_id:
            return
        entry = self._last_assistant
        if entry is None or not any(entry is m for m in self.history):
            return
        sents = self._last_sentences
        if sentence_idx < 0:
            spoken: list[str] = []
        else:
            sentence_idx = min(sentence_idx, len(sents) - 1)
            spoken = sents[:sentence_idx]
            partial = sents[sentence_idx]
            if fraction >= 0.97:
                spoken.append(partial)
            else:
                words = partial.split()
                keep = int(len(words) * max(0.0, min(fraction, 1.0)))
                if keep > 0:
                    spoken.append(" ".join(words[:keep]) + "…")
        if spoken:
            entry["content"] = " ".join(spoken)
        else:
            self.history = [m for m in self.history if m is not entry]
            self._last_assistant = None
        log.info("Interrupt applied; history keeps: %r", " ".join(spoken)[:80])

    def reset(self) -> None:
        self.history.clear()
        self._dropped_topics.clear()
        self._last_assistant = None
        self._last_sentences = []

    # ---- prompt building ------------------------------------------------
    def build_messages(self, lang: str | None = None) -> list[dict[str, str]]:
        system = self.system_prompt
        if lang in LANG_NAMES:
            system += f"\nThe user is currently speaking {LANG_NAMES[lang]}."
        if self._dropped_topics:
            system += "\nEarlier in this conversation the user asked about: " + "; ".join(
                self._dropped_topics
            ) + "."
        return [{"role": "system", "content": system}, *self.history]

    # ---- rolling window ---------------------------------------------------
    def _tokens(self) -> int:
        return sum(estimate_tokens(m["content"]) for m in self.history)

    def _user_turns(self) -> int:
        return sum(1 for m in self.history if m["role"] == "user")

    def _trim(self) -> None:
        """Drop the oldest turn(s) until within the turn and token budgets.
        The newest user message is always kept."""
        while len(self.history) > 1 and (
            self._user_turns() > self.cfg.keep_last_turns
            or self._tokens() > self.cfg.max_history_tokens
        ):
            first = self.history.pop(0)
            if first["role"] == "user":
                topic = first["content"].strip().replace("\n", " ")[:70]
                self._dropped_topics = (self._dropped_topics + [topic])[-5:]
            while self.history and self.history[0]["role"] != "user" and len(self.history) > 1:
                self.history.pop(0)
