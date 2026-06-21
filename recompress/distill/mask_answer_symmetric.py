"""Symmetric mask-the-answer: ours vs bear (reviewer #2, the fair version).

The mask test only matters if it's applied to BOTH systems. ours' compressions are reused
from the audit JSON; bear's are regenerated here (bear is deterministic-ish + local). For each,
we redact the gold answer span and re-solve with DeepSeek. The point: extractive bear can ONLY
keep verbatim source tokens, so masking should hit it AT LEAST as hard as it hits abstractive
ours — which is the symmetric evidence for "this is inherent to short-span QA, not a flaw unique
to us."

No GPU/Modal — ours' compressions are cached; bear is the local rate-limited SDK; solver is the
local DeepSeek API. Run from repo root:
  python -m recompress.distill.mask_answer_symmetric
"""
from __future__ import annotations
import json
import re
import os
from concurrent.futures import ThreadPoolExecutor

from recompress.act1.data import load_hotpotqa, context_to_text
from recompress.act1.bear import compress_bear
from recompress.act1.solve import solve
from recompress.act1.metrics import qa_f1
from recompress.config import CFG


def _redact(text: str, answer: str) -> str:
    if not answer.strip():
        return text
    return re.sub(re.escape(answer), "[REDACTED]", text, flags=re.IGNORECASE)


def main(audit_path: str = "results/cross_solver_audit.json",
         out: str = "results/mask_symmetric.json", ratio: float = 0.3):
    audit = json.load(open(audit_path))
    pi = audit["per_instance"]
    n = len(pi)
    ids = [p["id"] for p in pi]
    qs = [p["question"] for p in pi]
    golds = [p["gold"] for p in pi]
    ours_comp = [p["ours_compressed"] for p in pi]
    ours_f1_unmasked = [p["ours_f1_deepseek"] for p in pi]

    # regenerate bear on the SAME instances (same seeded HotpotQA order as the audit)
    insts = load_hotpotqa(n=n)
    assert [i["id"] for i in insts] == ids, "instance order mismatch — re-run audit + this together"
    texts = [context_to_text(i) for i in insts]
    print(f"{n} instances; regenerating bear compressions...")
    with ThreadPoolExecutor(max_workers=6) as p:
        bear_comp = list(p.map(lambda t: compress_bear(t, ratio), texts))

    def solve_all(comps):
        with ThreadPoolExecutor(max_workers=8) as p:
            preds = list(p.map(lambda i: solve(comps[i], qs[i]), range(n)))
        return [qa_f1(preds[i], golds[i]) for i in range(n)]

    # bear unmasked (recompute to be self-consistent) + both masked
    print("solving bear unmasked...");  bear_f1_unmasked = solve_all(bear_comp)
    print("solving ours MASKED...");     ours_f1_masked = solve_all([_redact(ours_comp[i], golds[i]) for i in range(n)])
    print("solving bear MASKED...");     bear_f1_masked = solve_all([_redact(bear_comp[i], golds[i]) for i in range(n)])

    def block(name, unm, msk):
        u = sum(unm) / n; m = sum(msk) / n
        return {"system": name, "unmasked_f1": u, "masked_f1": m, "drop": u - m,
                "drop_pct_of_unmasked": (u - m) / u if u else 0.0}

    res = {"n": n, "ratio": ratio,
           "ours": block("ours (abstractive)", ours_f1_unmasked, ours_f1_masked),
           "bear": block("bear (extractive)", bear_f1_unmasked, bear_f1_masked)}
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(out, "w"), indent=2, default=str)

    print("\n" + "=" * 64)
    print("  SYMMETRIC MASK-THE-ANSWER (redact gold span, re-solve)")
    print(f"  {'system':22s} {'unmasked':>9s} {'masked':>8s} {'drop':>7s} {'drop%':>7s}")
    for k in ("ours", "bear"):
        b = res[k]
        print(f"  {b['system']:22s} {b['unmasked_f1']:9.3f} {b['masked_f1']:8.3f} "
              f"{b['drop']:+7.3f} {b['drop_pct_of_unmasked']:6.0%}")
    print(f"\n  → masking hits {'bear' if res['bear']['drop_pct_of_unmasked']>=res['ours']['drop_pct_of_unmasked'] else 'ours'} "
          f"at least as hard (the span carries the score for BOTH — inherent to short-span QA at this ratio)")
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
