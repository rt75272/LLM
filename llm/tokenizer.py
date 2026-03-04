"""
Character-level tokenizer.

Builds a vocabulary from the unique characters in a corpus and provides
encode / decode methods.  The vocabulary is serialisable to / from JSON so
that trained tokenizers can be reused without re-fitting.
"""

import json


class CharTokenizer:
    """Map every unique character in a corpus to an integer index."""

    def __init__(self):
        self._char_to_idx: dict = {}
        self._idx_to_char: dict = {}

    # ------------------------------------------------------------------
    # Building the vocabulary
    # ------------------------------------------------------------------

    def fit(self, text: str) -> "CharTokenizer":
        """Derive the vocabulary from *text* and return self."""
        chars = sorted(set(text))
        self._char_to_idx = {c: i for i, c in enumerate(chars)}
        self._idx_to_char = {i: c for i, c in enumerate(chars)}
        return self

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self._char_to_idx)

    def encode(self, text: str) -> list:
        """Convert a string to a list of integer token ids."""
        try:
            return [self._char_to_idx[c] for c in text]
        except KeyError as exc:
            raise ValueError(
                f"Character {exc} not in vocabulary. "
                "Call fit() on a corpus that contains all required characters."
            ) from exc

    def decode(self, tokens: list) -> str:
        """Convert a list of integer token ids back to a string."""
        try:
            return "".join(self._idx_to_char[t] for t in tokens)
        except KeyError as exc:
            raise ValueError(f"Token id {exc} not in vocabulary.") from exc

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise the vocabulary to a JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"char_to_idx": self._char_to_idx}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        """Deserialise a vocabulary from a JSON file."""
        tok = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok._char_to_idx = data["char_to_idx"]
        tok._idx_to_char = {int(v): k for k, v in data["char_to_idx"].items()}
        return tok

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
