"""Per-instance F1 distribution (dot / strip plot) for a benchmark.

The mean F1 hides *concentration*: a model with mean 0.5 because it scores 0.9 on half and
0.1 on the other half is worse (for reliability) than one that scores a steady 0.4 everywhere.
This plots EVERY per-instance F1 so you can see whether scores are bimodal (0/1 split) or
concentrated — and compare bear vs our distilled variants directly.

Run: python -m src.act1.plot_distribution --benchmark 2wiki
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (label, results-json path) — edit/extend as needed
SERIES_2WIKI = [
    ("bear",            "results/5bar_distilled_2wiki.json",     "bear"),
    ("v3 (imitate)",    "results/5bar_distilled_2wiki.json",     "ours"),
    ("v4 (best-of-N)",  "results/5bar_answergrounded_2wiki.json","ours"),
    ("v5 (greedy+filt)","results/5bar_v5_2wiki.json",            "ours"),
]


def _f1s(path, bar):
    r = json.loads(Path(path).read_text())
    return [x["f1"] for x in r["bars"][bar]["per_instance"] if x and not x.get("error")]


def plot(series, benchmark: str, out_png: str):
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#dc2626", "#2563eb", "#f59e0b", "#10b981", "#8b5cf6"]
    rng = np.random.RandomState(42)

    labels, means, trims = [], [], []
    for i, (label, path, bar) in enumerate(series):
        try:
            fs = _f1s(path, bar)
        except FileNotFoundError:
            continue
        y = i
        # horizontal jitter so overlapping dots at 0.0/1.0 are visible
        jitter = rng.uniform(-0.18, 0.18, size=len(fs))
        ax.scatter(fs, np.full(len(fs), y) + jitter, alpha=0.55, s=45,
                   color=colors[i % len(colors)], edgecolors="none", zorder=2)
        m = statistics.mean(fs)
        sv = sorted(fs); n = len(sv); d = int(n * (1 - 0.34) / 2)
        tm = statistics.mean(sv[d:n - d]) if n - 2 * d > 0 else m
        # mean (solid) and trimmed-mean (dashed) markers
        ax.plot([m, m], [y - 0.35, y + 0.35], color="black", lw=2.5, zorder=3)
        ax.plot([tm, tm], [y - 0.28, y + 0.28], color="black", lw=1.5, ls=":", zorder=3)
        n0 = sum(1 for f in fs if f < 0.05); n1 = sum(1 for f in fs if f > 0.95)
        labels.append(f"{label}\nmean={m:.2f} mid33%={tm:.2f}\n{n0}×fail · {n1}×perfect")
        means.append(m); trims.append(tm)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("per-instance QA-F1  (each dot = one question)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_title(f"Score concentration on {benchmark} — bimodal? (─ mean, ┊ trimmed mid-33%)",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()
    fig.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"saved {out_png}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", default="2wiki")
    p.add_argument("--out", default="")
    args = p.parse_args()
    out = args.out or f"results/distribution_{args.benchmark}.png"
    series = SERIES_2WIKI if args.benchmark == "2wiki" else SERIES_2WIKI
    plot(series, args.benchmark, out)
