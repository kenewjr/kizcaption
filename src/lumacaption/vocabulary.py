from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


STOP_WORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "pada", "adalah",
    "dengan", "saya", "kamu", "dia", "mereka", "kita", "kami", "juga", "sudah",
    "akan", "bisa", "ada", "tidak", "bukan", "lagi", "karena", "tapi", "hanya",
    "seperti", "dalam", "bagi", "oleh", "tentang", "atau", "saat", "bila", "jika",
    "maka", "agar", "supaya", "hingga", "sampai", "bahkan", "pun", "apa", "siapa",
    "bagaimana", "mengapa", "kenapa", "kapan", "dimana", "mana", "tersebut", "sama",
    "the", "and", "in", "to", "of", "a", "an", "is", "it", "you", "that", "he",
    "was", "for", "on", "are", "as", "with", "his", "they", "at", "be", "this",
    "have", "from", "or", "one", "had", "by", "word", "but", "not", "what", "all",
    "were", "we", "when", "your", "can", "said", "there", "use", "an", "each",
    "which", "she", "do", "how", "their", "if", "will", "up", "other", "about",
}


class VocabularyManager:
    """Manages vocabulary dictionary and adaptive self-learning for repeated speech terms."""

    def __init__(self, workspace_dir: Path | str) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.vocab_file = self.workspace_dir / "vocabulary.json"
        self.entries: list[dict[str, Any]] = []
        self._learned_terms: set[str] = set()
        self._all_terms_lower: set[str] = set()
        self._term_counter: Counter[str] = Counter()
        self._threshold = 3
        self.load()

    def load(self) -> None:
        if not self.vocab_file.is_file():
            bundled = Path(__file__).resolve().parent / "assets" / "vocabulary.json"
            if bundled.is_file():
                try:
                    data = json.loads(bundled.read_text(encoding="utf-8"))
                    self.entries = data.get("entries", [])
                except Exception:
                    self.entries = []
            else:
                self.entries = []
        else:
            try:
                data = json.loads(self.vocab_file.read_text(encoding="utf-8"))
                self.entries = data.get("entries", [])
            except Exception:
                self.entries = []

        self._all_terms_lower = {
            str(e.get("term", "")).strip().lower()
            for e in self.entries
            if e.get("term")
        }
        self._learned_terms = {
            str(e.get("term", "")).strip()
            for e in self.entries
            if e.get("package") == "learned" and e.get("term")
        }

    def save(self) -> None:
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "entries": self.entries,
        }
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix="vocab_", suffix=".tmp", dir=self.workspace_dir
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.vocab_file)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def observe(self, text: str) -> list[str]:
        if not text:
            return []

        words = re.findall(r"\b[A-Za-z0-9_]{3,25}\b", text)
        newly_learned = []

        for word in words:
            w_lower = word.lower()
            if w_lower in STOP_WORDS or w_lower.isdigit():
                continue

            clean_term = word.strip()
            if clean_term in self._learned_terms:
                weight = 2 if word[0].isupper() else 1
                self._term_counter[clean_term] += weight
                continue

            if w_lower in self._all_terms_lower:
                continue

            weight = 2 if word[0].isupper() else 1
            self._term_counter[clean_term] += weight

            if self._term_counter[clean_term] >= self._threshold:
                entry_id = "learned_" + re.sub(r"[^a-zA-Z0-9_]", "_", clean_term.lower())
                entry = {
                    "id": entry_id,
                    "package": "learned",
                    "term": clean_term,
                    "type": "custom",
                    "protect": False,
                    "stt_hint": True,
                    "enabled": True,
                    "notes": "Dipelajari otomatis dari ucapan berulang",
                }
                self.entries.append(entry)
                self._all_terms_lower.add(w_lower)
                self._learned_terms.add(clean_term)
                newly_learned.append(clean_term)

        if newly_learned:
            try:
                self.save()
            except Exception:
                pass

        return newly_learned

    def get_hotwords(
        self,
        user_hotwords: str = "",
        include_packages: list[str] | None = None,
        max_length: int = 250,
    ) -> str:
        tokens: list[str] = []
        seen: set[str] = set()

        if user_hotwords:
            for part in user_hotwords.split(","):
                w = part.strip()
                if w and w.lower() not in seen:
                    tokens.append(w)
                    seen.add(w.lower())

        # Prioritize high-frequency terms first, then alphabetical tie-breaker
        sorted_learned = sorted(
            self._learned_terms,
            key=lambda t: (-self._term_counter.get(t, 0), t.lower()),
        )
        for term in sorted_learned:
            if term.lower() not in seen:
                tokens.append(term)
                seen.add(term.lower())

        if include_packages:
            active_set = set(include_packages)
            for entry in self.entries:
                if (
                    entry.get("enabled", True)
                    and entry.get("stt_hint", False)
                    and entry.get("package") in active_set
                ):
                    term = str(entry.get("term", "")).strip()
                    if term and term.lower() not in seen:
                        tokens.append(term)
                        seen.add(term.lower())

        result: list[str] = []
        current_len = 0
        for token in tokens:
            added_len = len(token) + (2 if result else 0)
            if current_len + added_len > max_length:
                break
            result.append(token)
            current_len += added_len

        return ", ".join(result)

    @property
    def learned_count(self) -> int:
        return len(self._learned_terms)

    def reset_learned(self) -> int:
        removed = len(self._learned_terms)
        self.entries = [e for e in self.entries if e.get("package") != "learned"]
        self._learned_terms.clear()
        self._term_counter.clear()
        self._all_terms_lower = {
            str(e.get("term", "")).strip().lower()
            for e in self.entries
            if e.get("term")
        }
        self.save()
        return removed
