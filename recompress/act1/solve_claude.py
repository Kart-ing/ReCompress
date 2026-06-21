"""Cross-solver: a Claude solver, for the teacher/solver-circularity check.

The headline 5-bar uses a frozen DeepSeek solver — but the teacher that generates
the compressions is ALSO DeepSeek, so a reviewer can argue "ours" enjoys solver-affinity
that bear (never tuned to any solver) doesn't. Claude is independent of BOTH the DeepSeek
teacher and the Qwen student, so re-scoring with it is the cleanest test of whether the
win is real or a same-family artifact.

Same SYSTEM prompt + interface as recompress.act1.solve.solve — only the model differs.
Uses the cheapest Claude (claude-haiku-4-5); the solver reads a ~48-token context + question
and emits a short answer, so cost per call is negligible.
"""
from __future__ import annotations
import os
import time

import anthropic
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# IDENTICAL prompt to the DeepSeek solver — the only variable is the model family.
_SYSTEM = (
    "You are a strict QA agent. Answer the question using ONLY the provided context. "
    "Be concise: answer with the minimal span, no reasoning, no preamble."
)

_MODEL = "claude-sonnet-4-6"  # independent of DeepSeek (teacher) + Qwen (student); stronger judge than Haiku
_client = None


def _get():
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


def solve_claude(context: str, question: str, max_tokens: int = 128, max_attempts: int = 4) -> str:
    """Answer with Claude Haiku. Same contract as solve() but a different model family."""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            resp = _get().messages.create(
                model=_MODEL,
                max_tokens=max_tokens,
                system=_SYSTEM,
                messages=[{"role": "user",
                           "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}],
            )
            # concatenate any text blocks
            return "".join(b.text for b in resp.content if b.type == "text").strip()
        except (anthropic.APITimeoutError, anthropic.APIConnectionError, anthropic.RateLimitError) as e:
            last_exc = e
            time.sleep(min(2.0 * (attempt + 1), 12.0))
    raise last_exc
