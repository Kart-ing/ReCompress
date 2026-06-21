"""Combined Act1+Act2 multi-turn benchmark.

Chains HotpotQA supporting documents into a multi-turn conversation (each doc = a turn),
then asks the question. Compares memory strategies × compressor backends:

  - Naive                : growing context (O(n) tokens) — quality ceiling, no compression
  - ReZero + deepseek    : flat ~300 tok, checkpoints compressed by DeepSeek API
  - ReZero + distilled   : flat ~300 tok, checkpoints compressed by OUR v3 (Modal, offline)
  - ReZero + bear        : flat ~300 tok, checkpoints compressed by TTC blind deletion

Metrics per instance: final-answer F1 (rougeL), and tokens in the final solver prompt
(flatness). Aggregated across N instances.

Run from repo root:  modal run rezero/experiments/combined_benchmark.py --n 30
"""
from __future__ import annotations
import os
import sys
import json

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ACTII = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_REPO, _ACTII):
    if p not in sys.path:
        sys.path.insert(0, p)

from recompress.distill.infer import app  # Act-1 Modal app (gives us the Compressor on the volume)


def _build_convo(sample, max_turns=6):
    docs = [f"{t}: {' '.join(s)}" for t, s in
            zip(sample["context"]["title"], sample["context"]["sentences"])]
    return docs[:max_turns]


@app.local_entrypoint()
def main(n: int = 30, max_turns: int = 6):
    from datasets import load_dataset
    from rouge_score import rouge_scorer
    from rezero.session import ReZeroSession
    from baselines.naive import NaiveSession
    from engine.deepseek import solve
    from engine.tokens import count_tokens

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    def f1(pred, gold): return scorer.score(gold, pred)["rougeL"].fmeasure

    # same seeded HotpotQA window as Act 1 for comparability
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    import random
    rows = list(ds.take(max(n * 20, 1000)))
    random.Random(42).shuffle(rows)
    rows = rows[:n]
    print(f"loaded {len(rows)} multi-turn instances ({max_turns} turns each)")

    backends = ["deepseek", "distilled", "bear"]
    agg = {"naive": {"f1": [], "tok": []}}
    for b in backends:
        agg[f"rezero_{b}"] = {"f1": [], "tok": []}

    for idx, s in enumerate(rows):
        q, gold = s["question"], s["answer"]
        docs = _build_convo(s, max_turns)

        # --- Naive (growing context) ---
        naive = NaiveSession(goal=q)
        for i, d in enumerate(docs):
            naive.add_turn(f"Document {i+1}: {d}", "Noted.")
        n_ans = solve(naive.prompt_for_solver(), q)
        agg["naive"]["f1"].append(f1(n_ans, gold))
        agg["naive"]["tok"].append(naive.token_count())

        # --- ReZero with each backend ---
        for b in backends:
            sess = ReZeroSession(goal=q, use_llm=True, backend=b)
            for i, d in enumerate(docs):
                sess.add_turn(f"Document {i+1}: {d}", "Noted.")
            ans = solve(sess.prompt_for_solver(), q)
            agg[f"rezero_{b}"]["f1"].append(f1(ans, gold))
            agg[f"rezero_{b}"]["tok"].append(sess.token_count())

        if (idx + 1) % 5 == 0:
            print(f"  [{idx+1}/{n}] done")

    # --- report ---
    def mean(xs): return sum(xs) / len(xs) if xs else 0.0
    print("\n" + "=" * 64)
    print("  COMBINED MULTI-TURN BENCHMARK (HotpotQA, %d turns, n=%d)" % (max_turns, n))
    print("=" * 64)
    print(f"  {'strategy':22s} {'final F1':>9s} {'ctx tokens':>11s}")
    results = {}
    for k, v in agg.items():
        mf, mt = mean(v["f1"]), mean(v["tok"])
        results[k] = {"mean_f1": mf, "mean_tokens": mt, "n": len(v["f1"])}
        print(f"  {k:22s} {mf:9.3f} {mt:11.0f}")

    out = "results/combined_benchmark.json"
    os.makedirs(os.path.join(_REPO, "results"), exist_ok=True)
    with open(os.path.join(_REPO, out), "w") as f:
        json.dump({"n": n, "max_turns": max_turns, "results": results}, f, indent=2)
    print(f"\nsaved {out}")
    print("\nKEY: Naive ctx grows with turns; ReZero stays flat. Does ReZero+distilled hold F1?")
