"""Latency benchmark: bear vs ours vs stacking.

Measures per-instance wall-clock latency of each compression strategy (NOT the solver
— this is about how long compression itself takes). Reports mean / median / p90 over
a set of HotpotQA instances, plus tokens-in / tokens-out so latency is interpretable
against compression amount.

Strategies measured:
  - bear        : blind deletion (TTC SDK, rate-limited)
  - ours        : query-aware API compressor (DeepSeek)
  - bear→ours   : bear then ours
  - ours→bear   : ours then bear

The distilled model's latency is measured separately on Modal (see
src/distill/latency_distilled.py) since it runs on the GPU.

Run: python -m src.act1.latency --n 20 --ratio 0.3
"""
from __future__ import annotations
import json
import time
import statistics
from pathlib import Path

from src.config import CFG
from src.act1.data import load_hotpotqa, context_to_text
from src.act1.tokens import count_tokens
from src.act1.compress import compress_ours
from src.act1.bear import compress_bear


def _timed(fn, *args):
    """Return (result, elapsed_seconds). Latency only — exceptions propagate to caller."""
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


STRATEGIES = {
    "bear": lambda text, q, ratio: _timed(compress_bear, text, ratio),
    "ours": lambda text, q, ratio: _timed(compress_ours, text, q, ratio),
}


def _bear_then_ours(text, q, ratio):
    t0 = time.perf_counter()
    mid = compress_bear(text, ratio)
    out = compress_ours(mid, q, ratio)
    return out, time.perf_counter() - t0


def _ours_then_bear(text, q, ratio):
    t0 = time.perf_counter()
    mid = compress_ours(text, q, ratio)
    out = compress_bear(mid, ratio)
    return out, time.perf_counter() - t0


STRATEGIES["bear→ours"] = _bear_then_ours
STRATEGIES["ours→bear"] = _ours_then_bear


def _summarize(latencies: list[float], tin: list[int], tout: list[int]) -> dict:
    return {
        "n": len(latencies),
        "mean_s": statistics.mean(latencies) if latencies else 0.0,
        "median_s": statistics.median(latencies) if latencies else 0.0,
        "p90_s": (statistics.quantiles(latencies, n=10)[8] if len(latencies) >= 10
                  else (max(latencies) if latencies else 0.0)),
        "min_s": min(latencies) if latencies else 0.0,
        "max_s": max(latencies) if latencies else 0.0,
        "avg_tokens_in": statistics.mean(tin) if tin else 0.0,
        "avg_tokens_out": statistics.mean(tout) if tout else 0.0,
    }


def run_latency(n: int = 20, ratio: float = 0.3, out_path: str | None = None) -> dict:
    instances = load_hotpotqa(n=n)
    texts = [context_to_text(i) for i in instances]
    print(f"measuring latency on {len(instances)} instances, ratio={ratio}\n")

    results = {"n": n, "ratio": ratio, "strategies": {}}
    for name, fn in STRATEGIES.items():
        lats, tin, tout, errs = [], [], [], 0
        for i, (inst, text) in enumerate(zip(instances, texts)):
            try:
                out, dt = fn(text, inst["question"], ratio)
                lats.append(dt)
                tin.append(count_tokens(text))
                tout.append(count_tokens(out))
            except Exception as e:
                errs += 1
                print(f"  [{name}] instance {i} error: {type(e).__name__}: {e}")
        summ = _summarize(lats, tin, tout)
        summ["n_err"] = errs
        results["strategies"][name] = summ
        print(f"  {name:12s} mean={summ['mean_s']:.2f}s median={summ['median_s']:.2f}s "
              f"p90={summ['p90_s']:.2f}s  (in≈{summ['avg_tokens_in']:.0f}→out≈{summ['avg_tokens_out']:.0f} tok)")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(results, indent=2, default=str))
    return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--out", default="eval/latency_api.json")
    args = p.parse_args()
    res = run_latency(n=args.n, ratio=args.ratio, out_path=args.out)
    print("\n" + "=" * 60)
    print("LATENCY SUMMARY (compression wall-clock, per instance):")
    for name, s in res["strategies"].items():
        print(f"  {name:12s} median={s['median_s']:.2f}s  mean={s['mean_s']:.2f}s  p90={s['p90_s']:.2f}s")
