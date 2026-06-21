"""Data loading: 50 seeded HotpotQA instances (multi-hop + distractors).
PRD §5a: '50 HotpotQA instances, seeded, fixed.'
Uses streaming to avoid downloading the full validation split.
"""
from __future__ import annotations
import random
from datasets import load_dataset
from src.config import CFG


def load_hotpotqa(n: int = CFG.n_instances, seed: int = CFG.seed) -> list[dict]:
    """Load n seeded HotpotQA distractor instances via streaming.
    Returns list of dicts: {id, question, answer, context (list of {title, text})}
    """
    # Stream enough rows to seed-shuffle and pick n deterministically.
    # HotpotQA validation has ~7400 rows; we pull a window then shuffle.
    pool_size = max(n * 20, 1000)  # pull a window, shuffle deterministically, take n
    ds = load_dataset(
        "hotpotqa/hotpot_qa", "distractor", split="validation",
        streaming=True,
    )
    rows = list(ds.take(pool_size))
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]

    out = []
    for ex in rows:
        ctx = []
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            ctx.append({"title": title, "text": " ".join(sents)})
        out.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "context": ctx,
        })
    return out


def context_to_text(instance: dict) -> str:
    """Flatten the context passages into one text blob (title + passage per block)."""
    blocks = [f"[{c['title']}]\n{c['text']}" for c in instance["context"]]
    return "\n\n".join(blocks)
