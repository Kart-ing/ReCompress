"""Pluggable compressor backend for ReZero (Act 2 ⇄ Act 1 integration).

ReZero's checkpoint builder compresses old turns. By default that uses the DeepSeek API
(engine.compressor.compress). This module lets the SAME call route to:

  - "deepseek"  : the original API teacher compressor (engine.compressor.compress)
  - "naive"     : word-truncation, no LLM (engine.compressor with use_llm=False)
  - "distilled" : OUR distilled Qwen2.5-1.5B (Act 1 v3) running on Modal
  - "bear"      : The Token Company blind-deletion SDK

Signature matches engine.compressor.compress exactly, so it's a drop-in:
    compress_backend(text, question, ratio, use_llm=True, exclude="", backend="deepseek")

The "distilled" path lazily imports the Act-1 Modal app from the repo root, so Act 2's
normal (deepseek/naive) runs need neither modal nor the src/ package on the path.
"""
from __future__ import annotations
import os
import sys

from engine.compressor import compress as _compress_deepseek
from engine.tokens import count_tokens

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# cached handles so we don't re-import / re-instantiate per checkpoint
_distilled_cls = None
_bear_fn = None


def _truncate(text: str, max_tokens: int) -> str:
    """Token-truncate using whatever tokenizer engine.tokens provides."""
    try:
        from engine.tokens import truncate_to_tokens as _t
        return _t(text, max_tokens)
    except Exception:
        words = text.split()
        while count_tokens(" ".join(words)) > max_tokens and words:
            words.pop()
        return " ".join(words)


def _get_distilled():
    """Lazy-load the Act-1 Modal Compressor class (from repo-root src/distill/infer.py)."""
    global _distilled_cls
    if _distilled_cls is None:
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        from src.distill.infer import Compressor
        _distilled_cls = Compressor
    return _distilled_cls


def _get_bear():
    global _bear_fn
    if _bear_fn is None:
        if _REPO_ROOT not in sys.path:
            sys.path.insert(0, _REPO_ROOT)
        from src.act1.bear import compress_bear
        _bear_fn = compress_bear
    return _bear_fn


def compress_backend(text: str, question: str, ratio: float, use_llm: bool = True,
                     exclude: str = "", backend: str = "deepseek") -> str:
    """Drop-in for engine.compressor.compress, with a backend selector."""
    if backend in ("deepseek", "naive"):
        # naive = use_llm forced False; deepseek = the original path (respects use_llm/exclude)
        return _compress_deepseek(text, question, ratio,
                                  use_llm=(use_llm and backend != "naive"), exclude=exclude)

    target = max(20, int(count_tokens(text) * ratio))

    if backend == "distilled":
        # ONE remote call per checkpoint (correctness over latency, per the PRD).
        Compressor = _get_distilled()
        out = Compressor().compress_batch.remote(
            [{"text": text, "question": question}], ratio
        )[0]
        return _truncate(out, target)

    if backend == "bear":
        out = _get_bear()(text, ratio)          # bear is query-blind by design
        return _truncate(out, target)

    raise ValueError(f"unknown compressor backend: {backend!r}")
