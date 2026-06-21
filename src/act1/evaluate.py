"""Act 1 eval harness: the 5-bar paired benchmark (PRD §5).
Bars @ matched token budget: none | bear | ours-only | bear→ours | ours→bear
Metric: QA-F1, paired, bootstrap 95% CI on each pairwise delta vs bear.

Runs instances in parallel (ThreadPoolExecutor), saves partial results after
each bar completes, and skips individual failures instead of crashing.
"""
from __future__ import annotations
import json
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from src.config import CFG
from src.act1.data import load_hotpotqa, context_to_text
from src.act1.tokens import count_tokens
from src.act1.compress import compress_ours
from src.act1.bear import compress_bear
from src.act1.solve import solve
from src.act1.metrics import qa_f1, bootstrap_ci


def bar_none(text: str, q: str, ratio: float) -> str:
    return text


def bar_ours(text: str, q: str, ratio: float) -> str:
    return compress_ours(text, q, ratio)


def bar_bear(text: str, q: str, ratio: float) -> str:
    return compress_bear(text, ratio)


def bar_bear_then_ours(text: str, q: str, ratio: float) -> str:
    after_bear = compress_bear(text, ratio)
    return compress_ours(after_bear, q, ratio)


def bar_ours_then_bear(text: str, q: str, ratio: float) -> str:
    after_ours = compress_ours(text, q, ratio)
    return compress_bear(after_ours, ratio)


BARS: dict[str, Callable[[str, str, float], str]] = {
    "none": bar_none,
    "bear": bar_bear,
    "ours": bar_ours,
    "bear→ours": bar_bear_then_ours,
    "ours→bear": bar_ours_then_bear,
}

MAX_WORKERS = 6


def _run_single(bar_fn, instance, ratio, idx, total):
    full_text = context_to_text(instance)
    try:
        compressed = bar_fn(full_text, instance["question"], ratio)
        answer = solve(compressed, instance["question"])
        f1 = qa_f1(answer, instance["answer"])
        return {
            "id": instance["id"],
            "question": instance["question"],
            "gold": instance["answer"],
            "pred": answer,
            "f1": f1,
            "n_tokens_in": count_tokens(full_text),
            "n_tokens_out": count_tokens(compressed),
            "error": None,
        }
    except Exception as e:
        return {
            "id": instance["id"],
            "question": instance["question"],
            "gold": instance["answer"],
            "pred": "",
            "f1": 0.0,
            "n_tokens_in": count_tokens(full_text),
            "n_tokens_out": 0,
            "error": f"{type(e).__name__}: {e}",
        }


def run_one_bar(bar_name: str, instances: list[dict], ratio: float, out_dir: str | None = None) -> dict:
    bar_fn = BARS[bar_name]
    per_instance = [None] * len(instances)
    n_done = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_run_single, bar_fn, inst, ratio, i, len(instances)): i
            for i, inst in enumerate(instances)
        }
        for fut in as_completed(futures):
            idx = futures[fut]
            per_instance[idx] = fut.result()
            n_done += 1
            if n_done % 5 == 0 or n_done == len(instances):
                n_ok = sum(1 for r in per_instance if r and not r["error"])
                n_err = sum(1 for r in per_instance if r and r["error"])
                print(f"  [{bar_name}] {n_done}/{len(instances)} done (ok={n_ok} err={n_err})")

                if out_dir:
                    partial = {
                        "bar": bar_name, "ratio": ratio,
                        "mean_f1": sum(r["f1"] for r in per_instance if r) / max(1, n_done),
                        "per_instance": per_instance,
                    }
                    Path(out_dir).mkdir(parents=True, exist_ok=True)
                    Path(out_dir, f"_partial_{bar_name}.json").write_text(
                        json.dumps(partial, indent=2, default=str))

    f1s = [r["f1"] for r in per_instance if r and not r["error"]]
    return {
        "bar": bar_name,
        "ratio": ratio,
        "mean_f1": sum(f1s) / len(f1s) if f1s else 0.0,
        "n_ok": len(f1s),
        "n_err": sum(1 for r in per_instance if r and r["error"]),
        "per_instance": per_instance,
    }


def run_5bar(ratio: float = 0.3, n: int = CFG.n_instances, out_path: str | None = None,
             out_dir: str | None = "eval/partial") -> dict:
    print(f"loading {n} hotpotqa instances...")
    instances = load_hotpotqa(n=n)
    print(f"loaded {len(instances)} instances")

    results = {"ratio": ratio, "n": n, "bars": {}}
    for bar_name in BARS:
        print(f"\n=== running bar: {bar_name} ===")
        results["bars"][bar_name] = run_one_bar(bar_name, instances, ratio, out_dir=out_dir)

    bear_f1 = [r["f1"] for r in results["bars"]["bear"]["per_instance"] if r and not r["error"]]
    results["deltas_vs_bear"] = {}
    for bar_name in BARS:
        if bar_name == "bear":
            continue
        bar_f1 = [r["f1"] for r in results["bars"][bar_name]["per_instance"] if r and not r["error"]]
        min_len = min(len(bear_f1), len(bar_f1))
        deltas = [bar_f1[i] - bear_f1[i] for i in range(min_len)]
        lo, hi = bootstrap_ci(deltas, n_iters=CFG.bootstrap_iters, seed=CFG.seed)
        results["deltas_vs_bear"][bar_name] = {
            "mean_delta": sum(deltas) / len(deltas) if deltas else 0.0,
            "ci_lo": lo, "ci_hi": hi,
            "excludes_zero": lo > 0 or hi < 0,
            "n_paired": min_len,
        }

    d_ours = results["deltas_vs_bear"]["ours"]
    headline_pass = d_ours["excludes_zero"] and d_ours["mean_delta"] > 0
    d_stack = results["deltas_vs_bear"].get("ours→bear")
    bonus = d_stack and d_stack["excludes_zero"] and d_stack["mean_delta"] > 0
    results["decision"] = {
        "headline_secured": headline_pass,
        "stacking_bonus": bool(bonus),
        "summary": (
            f"HEADLINE: {'PASS' if headline_pass else 'FAIL'} — ours beats bear at matched budget."
        ),
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--n", type=int, default=CFG.n_instances)
    p.add_argument("--out", default="eval/5bar_results.json")
    args = p.parse_args()
    res = run_5bar(ratio=args.ratio, n=args.n, out_path=args.out)
    print("\n" + "=" * 60)
    print(json.dumps(res["decision"], indent=2))
    print("\nDeltas vs bear (paired bootstrap 95% CI):")
    for bar_name, d in res["deltas_vs_bear"].items():
        star = " ★" if d["excludes_zero"] and d["mean_delta"] > 0 else ""
        print(f"  {bar_name:12s} Δ={d['mean_delta']:+.3f}  CI=({d['ci_lo']:+.3f}, {d['ci_hi']:+.3f})  excludes0={d['excludes_zero']}{star}")
