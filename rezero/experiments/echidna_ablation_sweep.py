"""Echidna ablation + horizon crossover sweep (reviewer #2 follow-up).

Two findings in one run:
  (A) ECHIDNA ABLATION: the LLM Echidna decides 'checkpoint' ~98% of turns (see
      data/echidna/echidna_train.jsonl) — it adds ~no decision value over the free
      rule-based trigger gated by the cooldown/min-history guardrails. We compare
      echidna_mode = "llm" vs "mock" (rule, 0 tokens) on F1 + total tokens.
  (B) CROSSOVER: we sweep conversation length T to find where flat-context RbD-Compress
      (with the cheap trigger) beats the growing naive agent on TOTAL tokens (ctx + overhead),
      vs both uncached and cached naive accounting.

Uses the distilled backend (our shipped compressor). Reports per (T, echidna_mode):
final F1, ctx->solver, overhead (DeepSeek tokens for trauma + Echidna + compressor),
and total. Naive reported as final(cached) and cumulative(uncached).

Run from repo root:  modal run rezero/experiments/echidna_ablation_sweep.py --turns 6,10,15,20
"""
from __future__ import annotations
import os, sys, json

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ACTII = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_REPO, _ACTII):
    if p not in sys.path:
        sys.path.insert(0, p)

from recompress.distill.infer import app


def _build_convo(sample, max_turns):
    docs = [f"{t}: {' '.join(s)}" for t, s in
            zip(sample["context"]["title"], sample["context"]["sentences"])]
    # pad by repeating docs if the sample has fewer than max_turns (long-horizon sweep)
    while len(docs) < max_turns:
        docs = docs + docs
    return docs[:max_turns]


@app.local_entrypoint()
def main(n: int = 20, turns: str = "6,10,15,20"):
    from datasets import load_dataset
    from rouge_score import rouge_scorer
    from rezero.session import ReZeroSession
    from baselines.naive import NaiveSession
    from engine.deepseek import solve, reset_overhead, get_overhead
    import random

    horizons = [int(t) for t in turns.split(",")]
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    def f1(pred, gold): return scorer.score(gold, pred)["rougeL"].fmeasure

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    rows = list(ds.take(max(n * 20, 1000)))
    random.Random(42).shuffle(rows)
    rows = rows[:n]
    print(f"sweep: horizons={horizons}, n={n}, echidna modes = llm vs mock(rule)")

    def mean(xs): return sum(xs) / len(xs) if xs else 0.0
    out = {"n": n, "horizons": horizons, "by_turns": {}}

    for T in horizons:
        rec = {"naive": {"f1": [], "ctx": [], "cum": []}}
        for mode in ("llm", "mock"):
            rec[f"rezero_{mode}"] = {"f1": [], "ctx": [], "overhead": [], "total": []}

        for s in rows:
            q, gold = s["question"], s["answer"]
            docs = _build_convo(s, T)

            naive = NaiveSession(goal=q); cum = 0
            for i, d in enumerate(docs):
                naive.add_turn(f"Document {i+1}: {d}", "Noted."); cum += naive.token_count()
            na = solve(naive.prompt_for_solver(), q)
            rec["naive"]["f1"].append(f1(na, gold))
            rec["naive"]["ctx"].append(naive.token_count())
            rec["naive"]["cum"].append(cum)

            for mode in ("llm", "mock"):
                reset_overhead()
                sess = ReZeroSession(goal=q, use_llm=True, backend="distilled", echidna_mode=mode)
                for i, d in enumerate(docs):
                    sess.add_turn(f"Document {i+1}: {d}", "Noted.")
                ov = get_overhead()["total"]
                ans = solve(sess.prompt_for_solver(), q)
                ctx = sess.token_count()
                rec[f"rezero_{mode}"]["f1"].append(f1(ans, gold))
                rec[f"rezero_{mode}"]["ctx"].append(ctx)
                rec[f"rezero_{mode}"]["overhead"].append(ov)
                rec[f"rezero_{mode}"]["total"].append(ctx + ov)

        block = {
            "naive_f1": mean(rec["naive"]["f1"]),
            "naive_ctx_cached": mean(rec["naive"]["ctx"]),
            "naive_cum_uncached": mean(rec["naive"]["cum"]),
        }
        for mode in ("llm", "mock"):
            v = rec[f"rezero_{mode}"]
            block[f"rezero_{mode}"] = {
                "f1": mean(v["f1"]), "ctx": mean(v["ctx"]),
                "overhead": mean(v["overhead"]), "total": mean(v["total"]),
            }
        out["by_turns"][str(T)] = block
        print(f"\n  T={T}: naive_cached={block['naive_ctx_cached']:.0f} "
              f"naive_uncached={block['naive_cum_uncached']:.0f} | "
              f"llm_total={block['rezero_llm']['total']:.0f} (F1 {block['rezero_llm']['f1']:.3f}) | "
              f"mock_total={block['rezero_mock']['total']:.0f} (F1 {block['rezero_mock']['f1']:.3f})")

    path = os.path.join(_REPO, "results/echidna_ablation_sweep.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 80)
    print("  ECHIDNA ABLATION + CROSSOVER SWEEP")
    print(f"  {'T':>3s} {'naive$':>8s} {'naive_unc':>9s} {'llm_tot':>8s} {'llmF1':>6s} {'mock_tot':>8s} {'mockF1':>6s}")
    for T in horizons:
        b = out["by_turns"][str(T)]
        print(f"  {T:3d} {b['naive_ctx_cached']:8.0f} {b['naive_cum_uncached']:9.0f} "
              f"{b['rezero_llm']['total']:8.0f} {b['rezero_llm']['f1']:6.3f} "
              f"{b['rezero_mock']['total']:8.0f} {b['rezero_mock']['f1']:6.3f}")
    print("\n  KEY: mock(rule) Echidna should ~match llm F1 at far lower total tokens (no Echidna LLM).")
    print("       crossover = smallest T where rezero_mock total < naive_uncached.")
    print(f"  saved results/echidna_ablation_sweep.json")
