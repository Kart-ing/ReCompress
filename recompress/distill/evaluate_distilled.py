"""Distilled-model 5-bar re-eval. Swaps the distilled 1.5B Qwen in as the "ours" bar.

Design: the distilled model lives on Modal (loaded once via a Modal class). We compute
ALL distilled compressions in batched remote calls (no per-instance cold start), then
do bear + solve + QA-F1 locally — reusing the same rate-limited bear client and the
solver from Act 1.

Bars @ matched token budget: none | bear | ours | bear→ours | ours→bear
Metric: QA-F1, paired, bootstrap 95% CI on each pairwise delta vs bear.

Run:  modal run recompress/distill/evaluate_distilled.py --n 50 --ratio 0.3
"""
from __future__ import annotations
import json
from pathlib import Path

import modal

from recompress.config import CFG
from recompress.act1.data import context_to_text
from recompress.act1.benchmarks import load_benchmark
from recompress.act1.tokens import count_tokens
from recompress.act1.bear import compress_bear
from recompress.act1.solve import solve
from recompress.act1.metrics import qa_f1, bootstrap_ci
from recompress.distill.infer import app as infer_app, Compressor


def _solve_and_score(instances, compressed_list):
    """Solve + QA-F1 for one bar's compressed contexts (parallel, error-tolerant)."""
    from concurrent.futures import ThreadPoolExecutor

    def one(idx):
        inst = instances[idx]
        comp = compressed_list[idx]
        try:
            ans = solve(comp, inst["question"])
            return {
                "id": inst["id"], "question": inst["question"], "gold": inst["answer"],
                "pred": ans, "f1": qa_f1(ans, inst["answer"]),
                "n_tokens_out": count_tokens(comp), "error": None,
            }
        except Exception as e:
            return {
                "id": inst["id"], "question": inst["question"], "gold": inst["answer"],
                "pred": "", "f1": 0.0, "n_tokens_out": count_tokens(comp),
                "error": f"{type(e).__name__}: {e}",
            }

    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(one, range(len(instances))))


def _bar_stats(name, per_instance, ratio):
    f1s = [r["f1"] for r in per_instance if not r["error"]]
    return {
        "bar": name, "ratio": ratio,
        "mean_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "n_ok": len(f1s),
        "n_err": sum(1 for r in per_instance if r["error"]),
        "per_instance": per_instance,
    }


def run_5bar(ratio: float, n: int, out_path: str | None, benchmark: str = "hotpotqa"):
    instances = load_benchmark(benchmark, n=n)
    full_texts = [context_to_text(i) for i in instances]
    questions = [i["question"] for i in instances]
    print(f"[{benchmark}] loaded {len(instances)} instances")

    comp = Compressor()

    # --- bear (local, rate-limited) for every instance ---
    print("computing bear compressions (rate-limited)...")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as pool:
        bear_comp = list(pool.map(lambda t: compress_bear(t, ratio), full_texts))

    # --- distilled "ours" for every instance: ONE batched remote call ---
    print("computing distilled 'ours' compressions (batched, model loaded once)...")
    ours_items = [{"text": full_texts[i], "question": questions[i]} for i in range(n)]
    ours_comp = comp.compress_batch.remote(ours_items, ratio)

    # --- bear→ours: distill the bear output (one batched remote call) ---
    print("computing bear→ours (distill the bear output)...")
    bo_items = [{"text": bear_comp[i], "question": questions[i]} for i in range(n)]
    bear_then_ours = comp.compress_batch.remote(bo_items, ratio)

    # --- ours→bear: bear-delete the distilled output (local, rate-limited) ---
    print("computing ours→bear (bear-delete the distilled output)...")
    with ThreadPoolExecutor(max_workers=6) as pool:
        ours_then_bear = list(pool.map(lambda c: compress_bear(c, ratio), ours_comp))

    compressed_by_bar = {
        "none": full_texts,
        "bear": bear_comp,
        "ours": ours_comp,
        "bear→ours": bear_then_ours,
        "ours→bear": ours_then_bear,
    }

    results = {"benchmark": benchmark, "ratio": ratio, "n": n, "bars": {}}
    for name, comp_list in compressed_by_bar.items():
        print(f"solving + scoring bar: {name}")
        per_inst = _solve_and_score(instances, comp_list)
        results["bars"][name] = _bar_stats(name, per_inst, ratio)

    # --- paired deltas vs bear ---
    bear_f1 = [r["f1"] for r in results["bars"]["bear"]["per_instance"] if not r["error"]]
    results["deltas_vs_bear"] = {}
    for name in compressed_by_bar:
        if name == "bear":
            continue
        bar_f1 = [r["f1"] for r in results["bars"][name]["per_instance"] if not r["error"]]
        m = min(len(bear_f1), len(bar_f1))
        deltas = [bar_f1[i] - bear_f1[i] for i in range(m)]
        lo, hi = bootstrap_ci(deltas, n_iters=CFG.bootstrap_iters, seed=CFG.seed)
        results["deltas_vs_bear"][name] = {
            "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
            "ci_lo": lo, "ci_hi": hi, "excludes_zero": lo > 0 or hi < 0, "n_paired": m,
        }

    d_ours = results["deltas_vs_bear"]["ours"]
    headline = d_ours["excludes_zero"] and d_ours["mean_delta"] > 0
    d_stack = results["deltas_vs_bear"].get("ours→bear")
    bonus = bool(d_stack and d_stack["excludes_zero"] and d_stack["mean_delta"] > 0)
    results["decision"] = {
        "headline_secured": headline, "stacking_bonus": bonus,
        "summary": f"[{benchmark}] DISTILLED HEADLINE: {'PASS' if headline else 'FAIL'} — distilled 1.5B vs bear at matched budget.",
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(results, indent=2, default=str))
    return results


def _print_result(res):
    print("\n" + "=" * 60)
    print(json.dumps(res["decision"], indent=2, ensure_ascii=False))
    print(f"\n[{res['benchmark']}] Per-bar mean QA-F1:")
    for name, b in res["bars"].items():
        print(f"  {name:12s} mean_f1={b['mean_f1']:.3f}  ok={b['n_ok']} err={b['n_err']}")
    print("\nDeltas vs bear (paired bootstrap 95% CI):")
    for name, d in res["deltas_vs_bear"].items():
        star = " ★" if d["excludes_zero"] and d["mean_delta"] > 0 else ""
        print(f"  {name:12s} Δ={d['mean_delta']:+.3f}  CI=({d['ci_lo']:+.3f}, {d['ci_hi']:+.3f})  excl0={d['excludes_zero']}{star}")


@infer_app.local_entrypoint()
def main(n: int = CFG.n_instances, ratio: float = 0.3, benchmark: str = "hotpotqa",
         out: str = ""):
    """benchmark: one of hotpotqa|2wiki|musique|squad, OR 'all' to run every benchmark
    in a single Modal session (model loaded once, reused across all)."""
    benches = ["hotpotqa", "2wiki", "musique", "squad"] if benchmark == "all" else [benchmark]
    summary = {}
    for bm in benches:
        out_path = out or f"results/5bar_distilled_{bm}.json"
        print(f"\n{'#'*60}\n# BENCHMARK: {bm}\n{'#'*60}")
        res = run_5bar(ratio=ratio, n=n, out_path=out_path, benchmark=bm)
        _print_result(res)
        d = res["deltas_vs_bear"]["ours"]
        summary[bm] = {
            "ours_f1": res["bars"]["ours"]["mean_f1"],
            "bear_f1": res["bars"]["bear"]["mean_f1"],
            "delta": d["mean_delta"], "ci": (d["ci_lo"], d["ci_hi"]),
            "pass": res["decision"]["headline_secured"],
        }
    if len(benches) > 1:
        print(f"\n{'='*60}\n  CROSS-BENCHMARK SUMMARY (distilled 1.5B vs bear)\n{'='*60}")
        for bm, s in summary.items():
            v = "✅ PASS" if s["pass"] else "❌ fail"
            print(f"  {bm:10s} ours={s['ours_f1']:.3f} bear={s['bear_f1']:.3f} "
                  f"Δ={s['delta']:+.3f} CI=({s['ci'][0]:+.3f},{s['ci'][1]:+.3f}) {v}")
