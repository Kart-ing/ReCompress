"""compress_ours(): the single query-aware pass — drop off-topic passages + densify verbose prose.
"""
from __future__ import annotations
import time

from openai import APITimeoutError, APIConnectionError, RateLimitError
from src.act1.client import get_client
from src.config import CFG
from src.act1.tokens import count_tokens, truncate_to_tokens

_TRANSIENT = (APITimeoutError, APIConnectionError, RateLimitError)

_SYSTEM = (
    "You are a context compressor. Given a long context and a QUESTION, produce the MINIMAL "
    "compressed context that still lets a downstream QA agent answer the QUESTION correctly.\n\n"
    "Rules:\n"
    "1. DROP any passage irrelevant to the question (distractors).\n"
    "2. DENSIFY verbose-but-relevant prose into terse, information-dense sentences. Paraphrase freely.\n"
    "3. Preserve all facts needed to answer: entities, numbers, relations, multi-hop links.\n"
    "4. Do NOT answer the question. Do NOT add reasoning. Output ONLY compressed context.\n"
    "5. Target ~{ratio:.0%} of the input token count, never more."
)

# Distillation-grade prompt: same query-aware deletion + densification, but with a
# FLOOR (so the teacher doesn't collapse to a 2% bare-answer span) and a hard
# anti-answer rule. Used only for generating training data for the student — the
# API-eval `compress_ours` above is intentionally left unchanged (it already PASSES).
_SYSTEM_DISTILL = (
    "You are a context compressor. Given a long context and a QUESTION, produce a "
    "compressed context that lets a downstream QA agent answer the QUESTION correctly.\n\n"
    "Rules:\n"
    "1. DROP passages irrelevant to the question (distractors).\n"
    "2. DENSIFY verbose-but-relevant prose into terse, information-dense sentences. Paraphrase freely.\n"
    "3. Preserve ALL facts needed to answer: entities, numbers, relations, and every link in a multi-hop chain. "
    "When in doubt, KEEP a fact — under-compressing is far better than dropping the answer's evidence.\n"
    "4. CRITICAL: Do NOT answer the question. Do NOT output a bare entity or a one-line answer. "
    "Output CONTEXT (full sentences with their supporting facts), not the answer itself. "
    "Your output must read like a short passage someone could reason over, not an answer key.\n"
    "5. Aim for roughly {ratio:.0%} of the input tokens. Write at least 2-3 complete sentences; "
    "never return fewer than ~25 tokens.\n"
    "6. Output ONLY the compressed context."
)


def compress_ours(text: str, question: str, ratio: float = 0.3, max_attempts: int = 4,
                  distill: bool = False) -> str:
    """Query-aware compression. `distill=True` uses the floor/anti-answer prompt for
    generating student training data; default keeps the proven API-eval behavior."""
    target_tokens = max(1, int(count_tokens(text) * ratio))
    system = (_SYSTEM_DISTILL if distill else _SYSTEM).format(ratio=ratio)
    last_exc = None
    for attempt in range(max_attempts):
        try:
            resp = get_client().chat.completions.create(
                model=CFG.compressor_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{text}"},
                ],
                temperature=0.0,
            )
            out = resp.choices[0].message.content.strip()
            return truncate_to_tokens(out, target_tokens)
        except _TRANSIENT as e:
            last_exc = e
            time.sleep(min(2.0 * (attempt + 1), 12.0))
    raise last_exc
