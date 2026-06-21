"""Distilled-model latency benchmark (GPU side).

Measures per-instance wall-clock latency of the distilled 1.5B compressor on the H100
(model loaded once via the Compressor class), plus the two stacked variants that use it.
Pairs with src/act1/latency.py (API path) for the bear-vs-ours-vs-stacking comparison.

Run: modal run src/distill/latency_distilled.py --n 20 --ratio 0.3
"""
from __future__ import annotations
import json
import time
import statistics
from pathlib import Path

from recompress.config import CFG
from recompress.act1.data import load_hotpotqa, context_to_text
from recompress.act1.tokens import count_tokens
from recompress.act1.bear import compress_bear
from recompress.distill.infer import app as infer_app, Compressor


def _summarize(latencies, tin, tout):
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


def run(n: int, ratio: float, out_path: str | None):
    instances = load_hotpotqa(n=n)
    texts = [context_to_text(i) for i in instances]
    questions = [i["question"] for i in instances]
    comp = Compressor()
    print(f"measuring distilled latency on {len(instances)} instances, ratio={ratio}\n")

    # Distilled "ours": time each single compression on the GPU. The Compressor class
    # returns per-item timings so we exclude the one-time model load from the numbers.
    items = [{"text": texts[i], "question": questions[i]} for i in range(n)]
    per_item = comp.compress_batch_timed.remote(items, ratio)
    ours_lat = [p["latency_s"] for p in per_item]
    ours_out = [p["n_out"] for p in per_item]
    tin = [count_tokens(t) for t in texts]

    results = {"n": n, "ratio": ratio, "strategies": {}}
    results["strategies"]["ours_distilled"] = _summarize(ours_lat, tin, ours_out)

    # bear (local) for reference in the same run
    bear_lat, bear_out = [], []
    for t in texts:
        t0 = time.perf_counter()
        out = compress_bear(t, ratio)
        bear_lat.append(time.perf_counter() - t0)
        bear_out.append(count_tokens(out))
    results["strategies"]["bear"] = _summarize(bear_lat, tin, bear_out)

    for name, s in results["strategies"].items():
        print(f"  {name:16s} median={s['median_s']:.3f}s mean={s['mean_s']:.3f}s "
              f"p90={s['p90_s']:.3f}s  (in≈{s['avg_tokens_in']:.0f}→out≈{s['avg_tokens_out']:.0f} tok)")

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(results, indent=2, default=str))
    return results


@infer_app.local_entrypoint()
def main(n: int = 20, ratio: float = 0.3, out: str = "results/latency_distilled.json"):
    res = run(n=n, ratio=ratio, out_path=out)
    print("\n" + "=" * 60)
    print("DISTILLED LATENCY (GPU inference, per instance, model preloaded):")
    for name, s in res["strategies"].items():
        print(f"  {name:16s} median={s['median_s']:.3f}s  mean={s['mean_s']:.3f}s  p90={s['p90_s']:.3f}s")
