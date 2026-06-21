
---

## Cross-solver / answer-leakage / faithfulness audit (response to independent review)

An independent reviewer flagged three load-bearing risks. We ran a dedicated audit
(`recompress/distill/cross_solver_eval.py` → `results/cross_solver_audit.json`, HotpotQA n=50,
ratio 0.3, v3 model). Verdicts:

**#1 Teacher=solver circularity → DEFENDED.** Teacher and solver are both DeepSeek, so "ours"
could enjoy solver-affinity bear lacks. We re-scored with **Claude Sonnet** (independent of both
the DeepSeek teacher and the Qwen student):

| Solver | ours | bear | Δ | 95% CI |
|---|---|---|---|---|
| DeepSeek (in-family) | 0.737 | 0.452 | +0.285 | (+0.136,+0.437) PASS |
| Claude Sonnet (independent) | 0.587 | 0.299 | **+0.288** | (+0.149,+0.426) **PASS** |

The Δ is essentially identical under an independent judge (+0.288 vs +0.285); absolute scores
drop (Sonnet grades harder) but the *margin* — the claim — is invariant. The win is NOT a
same-family artifact. (HotpotQA only; not cross-solvered on the other benchmarks.)

**#2 Answer-leakage → CONFIRMED, reported.** Gold answer appears verbatim (normalized contiguous
span) in **33/50 = 66%** of ours' compressions. Material. Mitigations: most "leaks" are the
compressor correctly keeping the single supporting sentence (good selection, not cheating); and
leakage alone can't explain the +0.29 gap under an independent solver (bear leaks verbatim source
tokens too). But QA-F1 at ~3.5% ratio on short-span QA partly rewards near-extraction — stated,
not hidden.

**#6 Faithfulness → MEASURED.** **9/50 = 18%** wrong under BOTH judges. Concrete hallucination:
gold "1952" → ours compressed "Rebel Without a Cause (1955)" (wrong year, confident). Another
dropped the relevant passage entirely. This is the real cost of abstraction; bear (extractive)
can't invent a wrong fact. Example tuples saved in the audit JSON's per_instance.

**Other reviewer points fixed by reframing (no run needed):** dropped "matched budget" →
"beats bear at ~8× fewer tokens" (true, stronger); Act 2 leads on the token result (8.1× flat),
not the n=20 F1 delta (which is within noise, no CI); generalization headline now says
"significant on multi-hop-with-distractors; directional on dissimilar tasks" rather than
"generalizes zero-shot."
