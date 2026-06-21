"""Multi-benchmark loaders — normalize several QA datasets to the same instance schema
the eval expects: {id, question, answer, context: [{title, text}, ...]}.

Lets us test the distilled compressor on benchmarks BEYOND the HotpotQA it was distilled
from, to show cross-dataset generalization (not just in-distribution performance).

Benchmarks:
  - hotpotqa   : multi-hop + distractors (in-distribution; same as training source)
  - 2wiki      : 2WikiMultiHopQA — multi-hop, different source (cross-dataset, in-domain)
  - musique    : MuSiQue — harder multi-hop (stress test)
  - squad       : SQuAD v2 — single-hop reading comprehension (different QA regime)
"""
from __future__ import annotations
import random
from datasets import load_dataset
from recompress.config import CFG


def _seeded_take(ds_iter, pool_size: int, n: int, seed: int) -> list:
    rows = list(ds_iter.take(pool_size))
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:n]


def load_hotpotqa(n: int = CFG.n_instances, seed: int = CFG.seed) -> list[dict]:
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    rows = _seeded_take(ds, max(n * 20, 1000), n, seed)
    out = []
    for ex in rows:
        ctx = [{"title": t, "text": " ".join(s)}
               for t, s in zip(ex["context"]["title"], ex["context"]["sentences"])]
        out.append({"id": ex["id"], "question": ex["question"], "answer": ex["answer"], "context": ctx})
    return out


def load_2wiki(n: int = CFG.n_instances, seed: int = CFG.seed) -> list[dict]:
    """2WikiMultiHopQA — context is {title: [...], content/sentences: [...]} like HotpotQA."""
    ds = load_dataset("scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35",
                      split="validation", streaming=True)
    rows = _seeded_take(ds, max(n * 20, 1000), n, seed)
    out = []
    for ex in rows:
        ctx_field = ex["context"]
        ctx = []
        # context is typically {"title": [...], "content": [[sent,...], ...]}
        titles = ctx_field.get("title", [])
        contents = ctx_field.get("content") or ctx_field.get("sentences") or []
        for t, c in zip(titles, contents):
            text = " ".join(c) if isinstance(c, list) else str(c)
            ctx.append({"title": t, "text": text})
        out.append({"id": str(ex.get("_id", ex.get("id"))), "question": ex["question"],
                    "answer": ex["answer"], "context": ctx})
    return out


def load_musique(n: int = CFG.n_instances, seed: int = CFG.seed) -> list[dict]:
    """MuSiQue — paragraphs: [{title, paragraph_text, is_supporting}, ...]."""
    ds = load_dataset("dgslibisey/MuSiQue", split="validation", streaming=True)
    rows = _seeded_take(ds, max(n * 20, 1000), n, seed)
    out = []
    for ex in rows:
        ctx = [{"title": p.get("title", ""), "text": p.get("paragraph_text", "")}
               for p in ex["paragraphs"]]
        out.append({"id": str(ex["id"]), "question": ex["question"],
                    "answer": ex["answer"], "context": ctx})
    return out


def load_squad(n: int = CFG.n_instances, seed: int = CFG.seed) -> list[dict]:
    """SQuAD v2 — single context passage per question (single-hop). Skips unanswerable."""
    ds = load_dataset("rajpurkar/squad_v2", split="validation", streaming=True)
    rows = _seeded_take(ds, max(n * 40, 2000), n * 3, seed)  # over-pull; many are unanswerable
    out = []
    for ex in rows:
        answers = ex["answers"]["text"]
        if not answers:
            continue  # skip unanswerable (no gold span to score)
        out.append({"id": ex["id"], "question": ex["question"], "answer": answers[0],
                    "context": [{"title": ex.get("title", ""), "text": ex["context"]}]})
        if len(out) >= n:
            break
    return out


def load_msmarco(n: int = CFG.n_instances, seed: int = CFG.seed) -> list[dict]:
    """MS MARCO v2.1 — LONG free-form answers (6-30 words), so QA-F1 is genuinely
    continuous (not the near-binary entity-span case of HotpotQA/2Wiki/SQuAD).
    10 passages per query (with distractors); skips 'No Answer Present.' rows."""
    ds = load_dataset("microsoft/ms_marco", "v2.1", split="validation", streaming=True)
    rows = _seeded_take(ds, max(n * 8, 800), n * 4, seed)  # over-pull; some have no answer
    out = []
    for ex in rows:
        answers = [a for a in ex.get("answers", []) if a and a.strip() and a != "No Answer Present."]
        if not answers:
            continue
        p = ex["passages"]
        ctx = [{"title": "", "text": t} for t in p.get("passage_text", [])]
        if not ctx:
            continue
        out.append({"id": str(ex["query_id"]), "question": ex["query"].strip(),
                    "answer": answers[0], "context": ctx})
        if len(out) >= n:
            break
    return out


LOADERS = {
    "hotpotqa": load_hotpotqa,
    "2wiki": load_2wiki,
    "musique": load_musique,
    "squad": load_squad,
    "msmarco": load_msmarco,
}


def load_benchmark(name: str, n: int = CFG.n_instances, seed: int = CFG.seed) -> list[dict]:
    if name not in LOADERS:
        raise ValueError(f"unknown benchmark {name!r}; choices: {list(LOADERS)}")
    return LOADERS[name](n=n, seed=seed)
