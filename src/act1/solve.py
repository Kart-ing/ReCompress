"""Frozen solver: answers a question given compressed context.
PRD: frozen DeepSeek Flash, same for every bar. This is the quality judge.
"""
from __future__ import annotations
import time

from openai import APITimeoutError, APIConnectionError, RateLimitError
from src.act1.client import get_client
from src.config import CFG

_SYSTEM = (
    "You are a strict QA agent. Answer the question using ONLY the provided context. "
    "Be concise: answer with the minimal span, no reasoning, no preamble."
)

# The OpenAI client already retries (max_retries=5), but flaky conference wifi
# throws ConnectTimeout that can still exhaust those. One more bounded backoff
# layer here keeps a single timeout from ever killing a 50-instance run.
_TRANSIENT = (APITimeoutError, APIConnectionError, RateLimitError)


def solve(context: str, question: str, max_tokens: int = 128, max_attempts: int = 4) -> str:
    """Return the solver's answer string, retrying transient network/rate errors."""
    last_exc = None
    for attempt in range(max_attempts):
        try:
            resp = get_client().chat.completions.create(
                model=CFG.solver_model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except _TRANSIENT as e:
            last_exc = e
            time.sleep(min(2.0 * (attempt + 1), 12.0))
    raise last_exc
