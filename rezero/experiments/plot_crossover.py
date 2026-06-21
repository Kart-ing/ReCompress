"""Crossover + cheap-Echidna figures from the ablation sweep.

Reads results/echidna_ablation_sweep.json (whatever horizons are present) and writes:
  - results/figures/crossover.png       : total tokens vs turns (naive cached/uncached,
                                          rezero+LLM-Echidna, rezero+rule-Echidna)
  - results/figures/echidna_ablation.png: total tokens + F1, LLM vs rule Echidna, per horizon

Run from repo root:  python -m rezero.experiments.plot_crossover
"""
from __future__ import annotations
import os, sys, json

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = os.path.join(_REPO, "results/figures")
os.makedirs(FIG, exist_ok=True)


def main():
    d = json.load(open(os.path.join(_REPO, "results/echidna_ablation_sweep.json")))
    bt = d["by_turns"]
    T = sorted(int(k) for k in bt.keys())
    g = lambda t, *keys: _dig(bt[str(t)], keys)

    naive_cached = [g(t, "naive_ctx_cached") for t in T]
    naive_unc = [g(t, "naive_cum_uncached") for t in T]
    llm_tot = [g(t, "rezero_llm", "total") for t in T]
    mock_tot = [g(t, "rezero_mock", "total") for t in T]

    # --- crossover figure ---
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(T, naive_unc, "o-", color="#c1121f", label="Naive (uncached, resends history)")
    ax.plot(T, naive_cached, "s--", color="#e8896b", label="Naive (KV/prefix-cached, final prompt)")
    ax.plot(T, llm_tot, "^-", color="#7b7b7b", label="RbD-Compress + LLM Echidna (total)")
    ax.plot(T, mock_tot, "D-", color="#2a6f97", label="RbD-Compress + rule Echidna (total)", linewidth=2.2)
    ax.set_xlabel("conversation length (turns)")
    ax.set_ylabel("total tokens spent (ctx + compression overhead)")
    ax.set_title("Total-token crossover: cheap-Echidna RbD-Compress vs naive")
    ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/crossover.png", dpi=150); plt.close(fig)
    print("wrote crossover.png")

    # --- echidna ablation figure (tokens + F1) ---
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    x = range(len(T)); w = 0.38
    a1.bar([i - w/2 for i in x], llm_tot, w, label="LLM Echidna", color="#7b7b7b")
    a1.bar([i + w/2 for i in x], mock_tot, w, label="rule Echidna", color="#2a6f97")
    a1.set_xticks(list(x)); a1.set_xticklabels([f"T={t}" for t in T])
    a1.set_ylabel("total tokens / conversation"); a1.set_title("Cost: LLM vs rule Echidna")
    a1.legend(); a1.grid(axis="y", alpha=0.3)

    llm_f1 = [g(t, "rezero_llm", "f1") for t in T]
    mock_f1 = [g(t, "rezero_mock", "f1") for t in T]
    a2.plot(T, llm_f1, "^-", color="#7b7b7b", label="LLM Echidna")
    a2.plot(T, mock_f1, "D-", color="#2a6f97", label="rule Echidna")
    a2.set_xlabel("turns"); a2.set_ylabel("final-answer QA-F1")
    a2.set_title("Quality: ~unchanged"); a2.set_ylim(0.4, 0.65)
    a2.legend(); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{FIG}/echidna_ablation.png", dpi=150); plt.close(fig)
    print("wrote echidna_ablation.png")


def _dig(obj, keys):
    for k in keys:
        obj = obj[k]
    return obj


if __name__ == "__main__":
    main()
