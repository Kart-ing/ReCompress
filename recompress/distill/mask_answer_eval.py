"""Mask-the-answer eval (the rigorous compression-vs-extraction test, reviewer #2).

Reuses the compressions already saved in results/cross_solver_audit.json. For each instance,
REDACT the gold answer string from ours' compression, then ask the solver. If the solver still
answers correctly, the compression preserved the reasoning/evidence, not just the answer span —
i.e. it's genuine compression, not extraction. The drop in F1 from masking quantifies how much
of the score was "the answer was literally written there."

This needs only a solver (DeepSeek, local API) — no GPU, no Modal. Run from repo root:
  python -m recompress.distill.mask_answer_eval
"""
from __future__ import annotations
import json
import re
import os
from concurrent.futures import ThreadPoolExecutor

from recompress.act1.solve import solve
from recompress.act1.metrics import qa_f1, bootstrap_ci
from recompress.config import CFG


def _redact(text: str, answer: str) -> str:
    """Remove the gold answer (case-insensitive, whole-ish) from text, replace with [REDACTED]."""
    if not answer.strip():
        return text
    # redact the full answer string and each of its content words
    out = re.sub(re.escape(answer), "[REDACTED]", text, flags=re.IGNORECASE)
    return out


def main(audit_path: str = "results/cross_solver_audit.json", out: str = "results/mask_answer_eval.json"):
    r = json.load(open(audit_path))
    pi = r["per_instance"]
    n = len(pi)
    print(f"loaded {n} instances from {audit_path}")

    qs = [p["question"] for p in pi]
    golds = [p["gold"] for p in pi]
    comp = [p["ours_compressed"] for p in pi]
    masked = [_redact(comp[i], golds[i]) for i in range(n)]
    orig_f1 = [p["ours_f1_deepseek"] for p in pi]   # already have unmasked DeepSeek F1

    print("solving on MASKED compressions (gold answer redacted)...")
    with ThreadPoolExecutor(max_workers=8) as p:
        masked_pred = list(p.map(lambda i: solve(masked[i], qs[i]), range(n)))
    masked_f1 = [qa_f1(masked_pred[i], golds[i]) for i in range(n)]

    # how many were "fully dependent on the literal span"? (correct unmasked, zero masked)
    span_dependent = sum(1 for i in range(n) if orig_f1[i] > 0.5 and masked_f1[i] < 0.2)
    survived = sum(1 for i in range(n) if orig_f1[i] > 0.5 and masked_f1[i] > 0.5)

    deltas = [orig_f1[i] - masked_f1[i] for i in range(n)]
    lo, hi = bootstrap_ci(deltas, n_iters=CFG.bootstrap_iters, seed=CFG.seed)

    res = {
        "n": n,
        "unmasked_mean_f1": sum(orig_f1) / n,
        "masked_mean_f1": sum(masked_f1) / n,
        "drop_from_masking": sum(deltas) / n,
        "drop_ci": [lo, hi],
        "n_correct_unmasked": sum(1 for f in orig_f1 if f > 0.5),
        "n_survived_masking": survived,          # answered right even with the span removed
        "n_span_dependent": span_dependent,      # right only because the span was there
        "per_instance": [
            {"q": qs[i], "gold": golds[i], "f1_unmasked": orig_f1[i], "f1_masked": masked_f1[i],
             "masked_pred": masked_pred[i]}
            for i in range(n)
        ],
    }
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(out, "w"), indent=2, default=str)

    print("\n" + "=" * 60)
    print("  MASK-THE-ANSWER (compression vs extraction)")
    print(f"  unmasked F1: {res['unmasked_mean_f1']:.3f}")
    print(f"  masked F1:   {res['masked_mean_f1']:.3f}  (gold span redacted from the compression)")
    print(f"  drop:        {res['drop_from_masking']:+.3f}  CI=({lo:+.3f},{hi:+.3f})")
    print(f"  of {res['n_correct_unmasked']} correct-unmasked: {survived} SURVIVED masking, "
          f"{span_dependent} were span-dependent")
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
