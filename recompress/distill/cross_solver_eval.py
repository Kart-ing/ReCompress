"""Cross-solver + answer-leakage + faithfulness audit (addresses reviewer #1, #2, #6).

For HotpotQA, n instances, ratio 0.3, this:
  - compresses each context with the distilled v3 model (Modal) AND with bear
  - scores BOTH compressions with TWO independent solvers: DeepSeek (the in-family judge,
    same family as the teacher) and Claude Sonnet (independent of teacher AND student)
  - SAVES the compressed text + both answers + gold, so we can:
      #1 cross-solver: does ours-beats-bear survive a non-DeepSeek solver?
      #2 answer-leakage: how often does the gold answer appear verbatim in ours' compression?
      #6 faithfulness: dump example (source-question-compression-answer) tuples to inspect

Run from repo root:  modal run recompress/distill/cross_solver_eval.py --n 50
"""
from __future__ import annotations
import os
import sys
import json
import re
import string

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from recompress.distill.infer import app, Compressor


def _norm(s):
    s = s.lower()
    s = "".join(c for c in s if c not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return [w for w in s.split() if w.strip()]


def _gold_in_compression(compressed: str, gold: str) -> bool:
    """Answer-leakage proxy: is the gold answer (normalized) a contiguous span of the
    compressed text? Catches the 'compressor wrote the answer down' failure mode."""
    c = _norm(compressed)
    g = _norm(gold)
    if not g:
        return False
    cs = " " + " ".join(c) + " "
    gs = " " + " ".join(g) + " "
    return gs in cs


@app.local_entrypoint()
def main(n: int = 50, ratio: float = 0.3, out: str = "results/cross_solver_audit.json"):
    from recompress.act1.data import load_hotpotqa, context_to_text
    from recompress.act1.bear import compress_bear
    from recompress.act1.solve import solve as solve_deepseek
    from recompress.act1.solve_claude import solve_claude
    from recompress.act1.metrics import qa_f1, bootstrap_ci
    from recompress.act1.tokens import count_tokens
    from recompress.config import CFG
    from concurrent.futures import ThreadPoolExecutor

    insts = load_hotpotqa(n=n)
    texts = [context_to_text(i) for i in insts]
    qs = [i["question"] for i in insts]
    golds = [i["answer"] for i in insts]
    print(f"loaded {n} HotpotQA instances")

    comp = Compressor()
    # ours (distilled v3) compressions — one batched remote call
    print("compressing with distilled v3...")
    ours_items = [{"text": texts[i], "question": qs[i]} for i in range(n)]
    ours_comp = comp.compress_batch.remote(ours_items, ratio)
    # bear compressions — local, rate-limited
    print("compressing with bear...")
    with ThreadPoolExecutor(max_workers=6) as p:
        bear_comp = list(p.map(lambda t: compress_bear(t, ratio), texts))

    def score_all(comp_list, solver):
        with ThreadPoolExecutor(max_workers=8) as p:
            preds = list(p.map(lambda i: solver(comp_list[i], qs[i]), range(n)))
        return preds, [qa_f1(preds[i], golds[i]) for i in range(n)]

    print("solving: ours x DeepSeek..."); ours_ds_pred, ours_ds_f1 = score_all(ours_comp, solve_deepseek)
    print("solving: bear x DeepSeek..."); bear_ds_pred, bear_ds_f1 = score_all(bear_comp, solve_deepseek)
    print("solving: ours x Claude...");   ours_cl_pred, ours_cl_f1 = score_all(ours_comp, solve_claude)
    print("solving: bear x Claude...");   bear_cl_pred, bear_cl_f1 = score_all(bear_comp, solve_claude)

    def summ(of1, bf1):
        deltas = [of1[i] - bf1[i] for i in range(n)]
        lo, hi = bootstrap_ci(deltas, n_iters=CFG.bootstrap_iters, seed=CFG.seed)
        return {"ours_mean_f1": sum(of1)/n, "bear_mean_f1": sum(bf1)/n,
                "delta": sum(deltas)/n, "ci_lo": lo, "ci_hi": hi,
                "excludes_zero": lo > 0 or hi < 0}

    # #2 answer-leakage on ours' compressions
    leak = [_gold_in_compression(ours_comp[i], golds[i]) for i in range(n)]
    leak_rate = sum(leak) / n

    results = {
        "n": n, "ratio": ratio,
        "cross_solver": {
            "deepseek": summ(ours_ds_f1, bear_ds_f1),   # in-family judge
            "claude_sonnet": summ(ours_cl_f1, bear_cl_f1),  # independent judge
        },
        "answer_leakage": {
            "ours_gold_verbatim_rate": leak_rate,
            "n_leaked": sum(leak),
            "note": "fraction of ours' compressions containing the gold answer as a contiguous normalized span",
        },
        # #6 faithfulness: keep everything per-instance for hand inspection
        "per_instance": [
            {"id": insts[i]["id"], "question": qs[i], "gold": golds[i],
             "ours_compressed": ours_comp[i], "ours_tok": count_tokens(ours_comp[i]),
             "ours_pred_deepseek": ours_ds_pred[i], "ours_f1_deepseek": ours_ds_f1[i],
             "ours_pred_claude": ours_cl_pred[i], "ours_f1_claude": ours_cl_f1[i],
             "bear_f1_deepseek": bear_ds_f1[i], "bear_f1_claude": bear_cl_f1[i],
             "gold_leaked_in_ours": leak[i]}
            for i in range(n)
        ],
    }

    os.makedirs(os.path.join(_REPO, "results"), exist_ok=True)
    with open(os.path.join(_REPO, out), "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 64)
    print("  CROSS-SOLVER (the circularity test): ours vs bear, two judges")
    for name, s in results["cross_solver"].items():
        v = "PASS" if s["excludes_zero"] and s["delta"] > 0 else "n.s."
        print(f"  {name:14s}: ours={s['ours_mean_f1']:.3f} bear={s['bear_mean_f1']:.3f} "
              f"Δ={s['delta']:+.3f} CI=({s['ci_lo']:+.3f},{s['ci_hi']:+.3f}) {v}")
    print(f"\n  ANSWER-LEAKAGE (#2): gold appears verbatim in {sum(leak)}/{n} = {leak_rate:.1%} of ours' compressions")
    print(f"\n  saved {out}")
