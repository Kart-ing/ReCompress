"""Plot the cross-solver + mask-the-answer audits (the honest-evaluation figures).
Reads results/cross_solver_audit.json + results/mask_symmetric.json, writes two PNGs
into results/figures/ (and they get copied to media/ for the paper)."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = "results/figures"
os.makedirs(FIG, exist_ok=True)

cs = json.load(open("results/cross_solver_audit.json"))["cross_solver"]
ms = json.load(open("results/mask_symmetric.json"))

# --- Figure 1: cross-solver — ours vs bear under two independent judges ---
fig, ax = plt.subplots(figsize=(7, 4.2))
judges = ["DeepSeek\n(in-family)", "Claude Sonnet\n(independent)"]
ours = [cs["deepseek"]["ours_mean_f1"], cs["claude_sonnet"]["ours_mean_f1"]]
bear = [cs["deepseek"]["bear_mean_f1"], cs["claude_sonnet"]["bear_mean_f1"]]
x = range(len(judges)); w = 0.35
ax.bar([i - w/2 for i in x], ours, w, label="ReCompress", color="#2a6f97")
ax.bar([i + w/2 for i in x], bear, w, label="bear-2", color="#c1121f")
for i in x:
    d = ours[i] - bear[i]
    ax.text(i, max(ours[i], bear[i]) + 0.02, f"$\\Delta$=+{d:.3f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(list(x)); ax.set_xticklabels(judges)
ax.set_ylabel("Mean QA-F1 (HotpotQA, n=50)"); ax.set_ylim(0, 0.85)
ax.set_title("Cross-solver check: the +0.29 gap is invariant to the judge")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIG}/cross_solver_bars.png", dpi=150); plt.close(fig)
print("wrote cross_solver_bars.png")

# --- Figure 2: mask-the-answer — unmasked vs masked for ours and bear ---
fig, ax = plt.subplots(figsize=(7, 4.2))
systems = ["ReCompress\n(abstractive)", "bear-2\n(extractive)"]
unm = [ms["ours"]["unmasked_f1"], ms["bear"]["unmasked_f1"]]
msk = [ms["ours"]["masked_f1"], ms["bear"]["masked_f1"]]
x = range(len(systems)); w = 0.35
ax.bar([i - w/2 for i in x], unm, w, label="unmasked", color="#2a9d8f")
ax.bar([i + w/2 for i in x], msk, w, label="gold span redacted", color="#e76f51")
for i in x:
    drop = (unm[i] - msk[i]) / unm[i] * 100
    ax.text(i, unm[i] + 0.02, f"$-${drop:.0f}%", ha="center", fontsize=11, fontweight="bold")
ax.set_xticks(list(x)); ax.set_xticklabels(systems)
ax.set_ylabel("Mean QA-F1 (HotpotQA, n=50)"); ax.set_ylim(0, 0.85)
ax.set_title("Mask-the-answer: how much F1 is carried by the literal answer span")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(f"{FIG}/mask_answer_bars.png", dpi=150); plt.close(fig)
print("wrote mask_answer_bars.png")
