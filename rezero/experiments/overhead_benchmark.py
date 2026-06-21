"""Honest multi-turn cost: context-to-solver PLUS compression overhead (reviewer #2).

The original combined_benchmark reports only tokens in the final solver prompt. But
RbD-Compress spends DeepSeek calls EVERY turn on the trauma extractor + Echidna (+ the
compressor for the deepseek backend). Those tokens are real cost. This run instruments
engine.deepseek's call() accumulator (reset per conversation) to report:

  - ctx_tokens   : final solver-prompt size (the flatness number, as before)
  - overhead_tok : total DeepSeek tokens spent on trauma/Echidna/compress across all turns
  - total_tokens : ctx + overhead  (the honest cost a budget-conscious reader cares about)

Also reports the naive baseline under TWO accounting models:
  - naive_uncached : full growing history resent every turn (sum of per-turn prompts)
  - naive_final    : just the final-turn prompt (what a KV/prefix-cached deployment pays)

Run from repo root:  modal run rezero/experiments/overhead_benchmark.py --n 20
"""
from __future__ import annotations
import os, sys, json

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ACTII = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_REPO, _ACTII):
    if p not in sys.path:
        sys.path.insert(0, p)

from recompress.distill.infer import app


def _build_convo(sample, max_turns=6):
    docs = [f"{t}: {' '.join(s)}" for t, s in
            zip(sample["context"]["title"], sample["context"]["sentences"])]
    return docs[:max_turns]


@app.local_entrypoint()
def main(n: int = 20, max_turns: int = 6):
    from datasets import load_dataset
    from rouge_score import rouge_scorer
    from rezero.session import ReZeroSession
    from baselines.naive import NaiveSession
    from engine.deepseek import solve, reset_overhead, get_overhead
    from engine.tokens import count_tokens
    import random

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    def f1(pred, gold): return scorer.score(gold, pred)["rougeL"].fmeasure

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    rows = list(ds.take(max(n * 20, 1000)))
    random.Random(42).shuffle(rows)
    rows = rows[:n]
    print(f"loaded {len(rows)} multi-turn instances ({max_turns} turns each)")

    backends = ["deepseek", "distilled", "bear"]
    agg = {"naive": {"f1": [], "ctx": [], "cum": []}}
    for b in backends:
        agg[f"rezero_{b}"] = {"f1": [], "ctx": [], "overhead": [], "total": []}

    for idx, s in enumerate(rows):
        q, gold = s["question"], s["answer"]
        docs = _build_convo(s, max_turns)

        # --- Naive: track both the final prompt AND the cumulative (uncached) cost ---
        naive = NaiveSession(goal=q)
        cum = 0
        for i, d in enumerate(docs):
            naive.add_turn(f"Document {i+1}: {d}", "Noted.")
            cum += naive.token_count()          # what an uncached agent resends THIS turn
        n_ans = solve(naive.prompt_for_solver(), q)
        agg["naive"]["f1"].append(f1(n_ans, gold))
        agg["naive"]["ctx"].append(naive.token_count())   # final-turn prompt (cached deployment)
        agg["naive"]["cum"].append(cum)                    # sum of per-turn prompts (uncached)

        # --- ReZero backends: reset the overhead counter per conversation ---
        for b in backends:
            reset_overhead()
            sess = ReZeroSession(goal=q, use_llm=True, backend=b)
            for i, d in enumerate(docs):
                sess.add_turn(f"Document {i+1}: {d}", "Noted.")
            ov = get_overhead()                  # trauma + Echidna (+ deepseek-compressor) tokens
            ans = solve(sess.prompt_for_solver(), q)
            ctx = sess.token_count()
            agg[f"rezero_{b}"]["f1"].append(f1(ans, gold))
            agg[f"rezero_{b}"]["ctx"].append(ctx)
            agg[f"rezero_{b}"]["overhead"].append(ov["total"])
            agg[f"rezero_{b}"]["total"].append(ctx + ov["total"])

        if (idx + 1) % 5 == 0:
            print(f"  [{idx+1}/{n}] done")

    def mean(xs): return sum(xs) / len(xs) if xs else 0.0
    results = {}
    print("\n" + "=" * 78)
    print("  HONEST MULTI-TURN COST (HotpotQA, %d turns, n=%d)" % (max_turns, n))
    print("=" * 78)
    print(f"  {'strategy':20s} {'F1':>6s} {'ctx→solver':>11s} {'overhead':>9s} {'TOTAL':>8s}")
    nv = agg["naive"]
    results["naive"] = {"mean_f1": mean(nv["f1"]), "ctx_final": mean(nv["ctx"]),
                        "cumulative_uncached": mean(nv["cum"]), "n": len(nv["f1"])}
    print(f"  {'naive (final/cached)':20s} {mean(nv['f1']):6.3f} {mean(nv['ctx']):11.0f} "
          f"{'--':>9s} {mean(nv['ctx']):8.0f}")
    print(f"  {'naive (uncached sum)':20s} {mean(nv['f1']):6.3f} {mean(nv['cum']):11.0f} "
          f"{'--':>9s} {mean(nv['cum']):8.0f}")
    for b in backends:
        v = agg[f"rezero_{b}"]
        results[f"rezero_{b}"] = {"mean_f1": mean(v["f1"]), "ctx_tokens": mean(v["ctx"]),
                                  "overhead_tokens": mean(v["overhead"]),
                                  "total_tokens": mean(v["total"]), "n": len(v["f1"])}
        print(f"  {'rezero_'+b:20s} {mean(v['f1']):6.3f} {mean(v['ctx']):11.0f} "
              f"{mean(v['overhead']):9.0f} {mean(v['total']):8.0f}")

    out = "results/overhead_benchmark.json"
    os.makedirs(os.path.join(_REPO, "results"), exist_ok=True)
    with open(os.path.join(_REPO, out), "w") as f:
        json.dump({"n": n, "max_turns": max_turns, "results": results}, f, indent=2)
    print(f"\nsaved {out}")
    print("\nKEY: 'overhead' = trauma+Echidna(+compressor) DeepSeek tokens per conversation.")
    print("     Compare rezero TOTAL vs naive-cached AND naive-uncached for the honest picture.")
