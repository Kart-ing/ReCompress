"""Graph the cross-benchmark 5-bar results: distilled 1.5B 'ours' vs bear vs full-context,
with paired-bootstrap CIs. Reads results/5bar_distilled_<benchmark>.json files.

Run: python -m recompress.act1.plot_results
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BENCHMARKS = ["hotpotqa", "2wiki", "musique", "squad"]
LABELS = {"hotpotqa": "HotpotQA\n(in-dist)", "2wiki": "2Wiki\n(cross)",
          "musique": "MuSiQue\n(hard)", "squad": "SQuAD v2\n(single-hop)"}


def _load():
    data = {}
    for bm in BENCHMARKS:
        p = Path(f"results/5bar_distilled_{bm}.json")
        if p.exists():
            data[bm] = json.loads(p.read_text())
    return data


def plot(out_png: str = "results/cross_benchmark.png"):
    data = _load()
    if not data:
        print("no result files yet")
        return
    benches = [b for b in BENCHMARKS if b in data]

    none_f1 = [data[b]["bars"]["none"]["mean_f1"] for b in benches]
    ours_f1 = [data[b]["bars"]["ours"]["mean_f1"] for b in benches]
    bear_f1 = [data[b]["bars"]["bear"]["mean_f1"] for b in benches]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Panel 1: grouped bars, F1 per system per benchmark ---
    x = np.arange(len(benches))
    w = 0.26
    ax1.bar(x - w, none_f1, w, label="Full context (ceiling)", color="#9ca3af")
    ax1.bar(x, ours_f1, w, label="Distilled 1.5B (ours)", color="#2563eb")
    ax1.bar(x + w, bear_f1, w, label="bear-1.1 (blind deletion)", color="#dc2626")
    for i, (o, b) in enumerate(zip(ours_f1, bear_f1)):
        ax1.text(i, o + 0.015, f"{o:.2f}", ha="center", fontsize=9, fontweight="bold", color="#2563eb")
        ax1.text(i + w, b + 0.015, f"{b:.2f}", ha="center", fontsize=8, color="#dc2626")
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS[b] for b in benches], fontsize=9)
    ax1.set_ylabel("QA-F1")
    ax1.set_title("Query-aware compression: distilled 1.5B beats bear at matched budget", fontsize=11)
    ax1.legend(fontsize=9, loc="upper right")
    ax1.set_ylim(0, 1.0)
    ax1.grid(axis="y", alpha=0.3)

    # --- Panel 2: Δ(ours - bear) with 95% CI; >0 = ours wins ---
    deltas = [data[b]["deltas_vs_bear"]["ours"]["mean_delta"] for b in benches]
    los = [data[b]["deltas_vs_bear"]["ours"]["ci_lo"] for b in benches]
    his = [data[b]["deltas_vs_bear"]["ours"]["ci_hi"] for b in benches]
    yerr = np.array([[d - lo for d, lo in zip(deltas, los)],
                     [hi - d for d, hi in zip(deltas, his)]])
    colors = ["#16a34a" if lo > 0 else "#9ca3af" for lo in los]
    ax2.bar(x, deltas, 0.5, yerr=yerr, capsize=6, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", lw=1)
    for i, (d, lo) in enumerate(zip(deltas, los)):
        tag = "PASS" if lo > 0 else "n.s."
        ax2.text(i, d + (his[i] - d) + 0.02, f"{d:+.2f}\n{tag}", ha="center", fontsize=9,
                 fontweight="bold", color="#16a34a" if lo > 0 else "#6b7280")
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS[b] for b in benches], fontsize=9)
    ax2.set_ylabel("Δ QA-F1  (ours − bear)")
    ax2.set_title("Improvement over bear (95% CI; green = CI excludes 0)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("ReCompress — distilled query-aware 1.5B vs bear-1.1 across benchmarks",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}  ({len(benches)} benchmarks: {', '.join(benches)})")


if __name__ == "__main__":
    plot()
