"""Comprehensive visualization suite for ReCompress.

Generates a gallery of figures into results/figures/:
  1. cross_benchmark_bars.png    — grouped bars: full/ours/bear F1 per benchmark
  2. delta_ci.png                — Δ(ours-bear) with 95% CI per benchmark (forest-style)
  3. dots_<bench>.png            — per-instance F1 dot/strip plots (concentration)
  4. tokens_vs_f1.png            — scatter: compression (tokens out) vs quality (F1)
  5. variant_compare.png         — v3 vs v4 vs v5 vs bear bars (the distillation-objective study)
  6. distill_trajectory.png      — v1→v3 headline F1 progression
  7. latency_bars.png            — compression wall-clock per strategy (API path)
  8. multiturn_tokens.png        — (if combined benchmark exists) flat vs growing context
  9. multiturn_f1.png            — (if combined benchmark exists) final-answer F1 per strategy

Run: python -m src.act1.plot_all
"""
from __future__ import annotations
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path("results/figures")
BENCHES = ["hotpotqa", "2wiki", "musique", "squad"]
NICE = {"hotpotqa": "HotpotQA", "2wiki": "2Wiki", "musique": "MuSiQue", "squad": "SQuAD v2",
        "msmarco": "MS MARCO"}
C = {"none": "#9ca3af", "ours": "#2563eb", "bear": "#dc2626", "v3": "#2563eb",
     "v4": "#f59e0b", "v5": "#10b981", "deepseek": "#8b5cf6", "distilled": "#2563eb"}


def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except FileNotFoundError:
        return None


def _f1s(r, bar):
    return [x["f1"] for x in r["bars"][bar]["per_instance"] if x and not x.get("error")]


def fig1_cross_benchmark_bars():
    data = {b: _load(f"results/5bar_distilled_{b}.json") for b in BENCHES}
    data = {b: r for b, r in data.items() if r}
    if not data:
        return
    benches = list(data)
    x = np.arange(len(benches)); w = 0.26
    fig, ax = plt.subplots(figsize=(11, 6))
    for off, bar, lbl in [(-w, "none", "Full context"), (0, "ours", "ReCompress 1.5B"), (w, "bear", "bear-1.1")]:
        vals = [data[b]["bars"][bar]["mean_f1"] for b in benches]
        ax.bar(x + off, vals, w, label=lbl, color=C[bar])
        for i, v in enumerate(vals):
            ax.text(i + off, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([NICE[b] for b in benches])
    ax.set_ylabel("QA-F1"); ax.set_ylim(0, 1.0); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Query-aware compression vs blind deletion — QA-F1 at matched budget", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "cross_benchmark_bars.png", dpi=150); plt.close(fig)


def fig2_delta_ci():
    data = {b: _load(f"results/5bar_distilled_{b}.json") for b in BENCHES}
    data = {b: r for b, r in data.items() if r}
    if not data:
        return
    benches = list(data)
    deltas = [data[b]["deltas_vs_bear"]["ours"]["mean_delta"] for b in benches]
    los = [data[b]["deltas_vs_bear"]["ours"]["ci_lo"] for b in benches]
    his = [data[b]["deltas_vs_bear"]["ours"]["ci_hi"] for b in benches]
    y = np.arange(len(benches))
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#16a34a" if lo > 0 else "#9ca3af" for lo in los]
    for i, (d, lo, hi) in enumerate(zip(deltas, los, his)):
        ax.plot([lo, hi], [i, i], color=colors[i], lw=3, zorder=2)
        ax.scatter([d], [i], color=colors[i], s=80, zorder=3)
        ax.text(hi + 0.01, i, f"{d:+.2f} {'sig' if lo>0 else 'n.s.'}", va="center", fontsize=9)
    ax.axvline(0, color="black", lw=1, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([NICE[b] for b in benches])
    ax.set_xlabel("Δ QA-F1 (ReCompress − bear), 95% CI"); ax.grid(axis="x", alpha=0.3)
    ax.set_title("Improvement over bear with confidence intervals (green = significant)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "delta_ci.png", dpi=150); plt.close(fig)


def fig3_dots():
    for b in BENCHES:
        r = _load(f"results/5bar_distilled_{b}.json")
        if not r:
            continue
        fig, ax = plt.subplots(figsize=(10, 4.5))
        rng = np.random.RandomState(42)
        for i, (bar, lbl) in enumerate([("bear", "bear"), ("ours", "ReCompress"), ("none", "full ctx")]):
            fs = _f1s(r, bar)
            ax.scatter(fs, np.full(len(fs), i) + rng.uniform(-0.16, 0.16, len(fs)),
                       alpha=0.55, s=42, color=C[bar], edgecolors="none")
            m = statistics.mean(fs)
            ax.plot([m, m], [i - 0.33, i + 0.33], color="black", lw=2.5)
            n0 = sum(1 for f in fs if f < 0.05); n1 = sum(1 for f in fs if f > 0.95)
            ax.text(1.07, i, f"mean {m:.2f}\n{n0}×0 · {n1}×1", va="center", fontsize=8)
        ax.set_yticks(range(3)); ax.set_yticklabels(["bear", "ReCompress", "full ctx"])
        ax.set_xlim(-0.05, 1.25); ax.set_xlabel("per-instance QA-F1 (each dot = one question)")
        ax.set_title(f"Score concentration on {NICE[b]} (— = mean)", fontweight="bold")
        ax.grid(axis="x", alpha=0.3); ax.invert_yaxis()
        fig.tight_layout(); fig.savefig(FIG / f"dots_{b}.png", dpi=150); plt.close(fig)


def fig4_tokens_vs_f1():
    fig, ax = plt.subplots(figsize=(10, 6))
    for b in BENCHES:
        r = _load(f"results/5bar_distilled_{b}.json")
        if not r:
            continue
        for bar, mark in [("ours", "o"), ("bear", "x")]:
            pis = [x for x in r["bars"][bar]["per_instance"] if x and not x.get("error")]
            toks = [x["n_tokens_out"] for x in pis]; f1 = [x["f1"] for x in pis]
            ax.scatter(toks, f1, alpha=0.4, s=30, marker=mark,
                       label=f"{NICE[b]} {'ReCompress' if bar=='ours' else 'bear'}",
                       color=C[bar])
    ax.set_xlabel("compressed tokens (output size)"); ax.set_ylabel("QA-F1")
    ax.set_title("Compression vs quality: tokens-out vs answer F1 (o=ReCompress, x=bear)", fontweight="bold")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "tokens_vs_f1.png", dpi=150); plt.close(fig)


def fig5_variant_compare():
    # v3 vs v4 vs v5 vs bear on the benchmarks we have all of (hotpotqa, 2wiki)
    rows = []
    for b in ["hotpotqa", "2wiki"]:
        v3 = _load(f"results/5bar_distilled_{b}.json")
        v4 = _load(f"results/5bar_answergrounded_{b}.json")
        v5 = _load(f"results/5bar_v5_{b}.json")
        if not (v3 and v4 and v5):
            continue
        rows.append((b, v3["bars"]["bear"]["mean_f1"], v3["bars"]["ours"]["mean_f1"],
                     v4["bars"]["ours"]["mean_f1"], v5["bars"]["ours"]["mean_f1"]))
    if not rows:
        return
    benches = [r[0] for r in rows]; x = np.arange(len(benches)); w = 0.2
    fig, ax = plt.subplots(figsize=(10, 6))
    series = [("bear", 1, "#dc2626"), ("v3 (imitate)", 2, "#2563eb"),
              ("v4 (best-of-N)", 3, "#f59e0b"), ("v5 (greedy+filter)", 4, "#10b981")]
    for j, (lbl, col_idx, color) in enumerate(series):
        vals = [r[col_idx] for r in rows]
        ax.bar(x + (j - 1.5) * w, vals, w, label=lbl, color=color)
        for i, v in enumerate(vals):
            ax.text(i + (j - 1.5) * w, v + 0.012, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels([NICE[b] for b in benches])
    ax.set_ylabel("QA-F1"); ax.set_ylim(0, 0.85); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Distillation objective study: teacher-imitation (v3) vs answer-grounded (v4/v5)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "variant_compare.png", dpi=150); plt.close(fig)


def fig6_distill_trajectory():
    # v1 wash -> v3 win, HotpotQA Δ vs bear
    v1 = _load("experiments/v1/eval/5bar_distilled_v1.json")
    v3 = _load("results/5bar_distilled_hotpotqa.json")
    if not v3:
        return
    pts = []
    if v1:
        pts.append(("v1\n261 ex, r16", v1["deltas_vs_bear"]["ours"]["mean_delta"]))
    pts.append(("v3\n5000 ex, r32", v3["deltas_vs_bear"]["ours"]["mean_delta"]))
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = [p[0] for p in pts]; vals = [p[1] for p in pts]
    ax.plot(range(len(pts)), vals, "o-", color="#2563eb", lw=2, ms=10)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:+.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(0, color="#dc2626", ls="--", label="bear (Δ=0)")
    ax.set_xticks(range(len(pts))); ax.set_xticklabels(labels)
    ax.set_ylabel("Δ QA-F1 vs bear (HotpotQA)"); ax.legend(); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Distillation trajectory: wash → significant win", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "distill_trajectory.png", dpi=150); plt.close(fig)


def fig7_latency():
    r = _load("results/latency_api.json")
    if not r:
        return
    strat = list(r["strategies"].keys())
    med = [r["strategies"][s]["median_s"] for s in strat]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(strat, med, color=["#dc2626", "#2563eb", "#8b5cf6", "#f59e0b"][:len(strat)])
    for i, v in enumerate(med):
        ax.text(i, v + 0.01, f"{v:.2f}s", ha="center", fontsize=9)
    ax.set_ylabel("median compression latency (s)"); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Compression latency per strategy (API path)", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "latency_bars.png", dpi=150); plt.close(fig)


def fig8_9_multiturn():
    r = _load("results/combined_benchmark.json")
    if not r:
        print("  (combined_benchmark.json not found — skipping multi-turn figs)")
        return
    res = r["results"]
    keys = list(res.keys())
    nice = {"naive": "Naive\n(growing)", "rezero_deepseek": "ReZero+\ndeepseek",
            "rezero_distilled": "ReZero+\ndistilled-v3", "rezero_bear": "ReZero+\nbear"}
    # fig 8: tokens (flatness)
    fig, ax = plt.subplots(figsize=(9, 5))
    toks = [res[k]["mean_tokens"] for k in keys]
    cols = ["#dc2626" if k == "naive" else "#2563eb" if "distilled" in k else "#9ca3af" for k in keys]
    ax.bar([nice.get(k, k) for k in keys], toks, color=cols)
    for i, v in enumerate(toks):
        ax.text(i, v + max(toks) * 0.01, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_ylabel("context tokens at final turn"); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Context size: Naive grows, ReZero stays flat", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "multiturn_tokens.png", dpi=150); plt.close(fig)
    # fig 9: F1
    fig, ax = plt.subplots(figsize=(9, 5))
    f1 = [res[k]["mean_f1"] for k in keys]
    ax.bar([nice.get(k, k) for k in keys], f1, color=cols)
    for i, v in enumerate(f1):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_ylabel("final-answer QA-F1"); ax.grid(axis="y", alpha=0.3)
    ax.set_title("Multi-turn answer quality per memory strategy", fontweight="bold")
    fig.tight_layout(); fig.savefig(FIG / "multiturn_f1.png", dpi=150); plt.close(fig)


def fig10_token_trajectory():
    """The line graph: context tokens per turn — Naive grows, ReZero stays flat."""
    r = _load("results/token_trajectory.json")
    if not r:
        print("  (token_trajectory.json not found — skipping line graph)")
        return
    traj = r["trajectory"]
    turns = list(range(1, r["turns"] + 1))
    nice = {"naive": "Naive (growing)", "rezero_deepseek": "ReZero + deepseek",
            "rezero_distilled": "ReZero + distilled-v3", "rezero_bear": "ReZero + bear"}
    col = {"naive": "#dc2626", "rezero_deepseek": "#8b5cf6",
           "rezero_distilled": "#2563eb", "rezero_bear": "#f59e0b"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for k, ys in traj.items():
        ax.plot(turns[:len(ys)], ys, "o-", label=nice.get(k, k), color=col.get(k, "#333"),
                lw=2.5 if "distilled" in k else 2, ms=6)
    ax.set_xlabel("conversation turn"); ax.set_ylabel("context tokens sent to solver")
    ax.set_title("Context growth per turn: Naive blows up, ReZero stays flat", fontweight="bold")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG / "token_trajectory_line.png", dpi=150); plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    made = []
    for fn in [fig1_cross_benchmark_bars, fig2_delta_ci, fig3_dots, fig4_tokens_vs_f1,
               fig5_variant_compare, fig6_distill_trajectory, fig7_latency, fig8_9_multiturn,
               fig10_token_trajectory]:
        try:
            fn()
            made.append(fn.__name__)
        except Exception as e:
            print(f"  ! {fn.__name__} failed: {type(e).__name__}: {e}")
    pngs = sorted(FIG.glob("*.png"))
    print(f"\ngenerated {len(pngs)} figures in {FIG}/:")
    for p in pngs:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
