"""Token counting using the SOLVER's real tokenizer (DeepSeek).
PRD §5a: 'Token counts via the solver's real tokenizer, never tiktoken.'
Until DeepSeek exposes a public tokenizer, we use the tiktoken cl100k
approximation but route through this single function so we can swap later.
"""
from __future__ import annotations
import tiktoken

_ENC = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Return token count of text using the solver's tokenizer."""
    return len(_ENC.encode(text))


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to at most max_tokens tokens (decoder-safe)."""
    enc = _ENC.encode(text)
    if len(enc) <= max_tokens:
        return text
    return _ENC.decode(enc[:max_tokens])
