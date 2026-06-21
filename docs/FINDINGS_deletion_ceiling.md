# The Deletion Ceiling: why extraction-based compression has a limit that rewriting does not

**Claim.** Compression methods that only *delete or select* tokens from the source (extractive:
bear-2, LLMLingua, LongLLMLingua, Selective-Context) have a hard ceiling on a class of
queries. No amount of cleverness in *which* tokens to keep can cross it. Compression that
*rewrites* (abstractive) is not bound by this ceiling. This is a structural property of the
hypothesis class, not a quality gap between particular models.

This reframes the whole ReCompress result: we don't merely *beat* bear empirically — we show
*why* a query-aware **rewriter** must exist, because the deletion family provably cannot reach
some answers.

---

## 1. The argument

Let a context `C` be a sequence of tokens. Define two compressor families at a token budget `B`:

- **Extractive** `E(C, q) ⊆ C` — the output is a *subsequence* of the original tokens (deletion
  / selection). bear, LLMLingua, and all token-dropping methods live here.
- **Abstractive** `A(C, q)` — the output is *any* string of ≤ B tokens (paraphrase, synthesis,
  re-expression). ReCompress lives here.

**Observation (extractive is a strict subset).** Every extractive output is achievable by an
abstractive compressor (it can choose to copy), but not vice versa. So `A`'s reachable output
set strictly contains `E`'s. The question is whether that extra reach ever *matters* for the
downstream answer.

**The ceiling.** Consider a query `q` whose answer requires a fact `f` that is **distributed
across many low-salience tokens** in `C` — i.e. no short *contiguous or selectable* subsequence
of `C` expresses `f` within budget `B`, but a *paraphrase* of `f` fits easily. For such `q`:

- Any extractive `E(C,q)` within budget `B` must omit some token(s) `f` depends on ⇒ the solver
  cannot recover `f` ⇒ bounded answer quality. **This holds for the *optimal* token selection**,
  so it is a ceiling on the whole extractive family, not a failure of a specific scorer.
- An abstractive `A(C,q)` can *write* `f` in a few tokens ⇒ the answer survives.

The gap is therefore not "bear picked the wrong tokens" — it is "the right answer is not a
subsequence of the input at this budget."

## 2. A constructed example (existence proof)

Context (the relevant facts are spread thin and phrased verbosely):

> "...The composer, who was born in the northern city, later relocated. The northern city in
> question had, since the prior century, served as the administrative seat of the surrounding
> province. That province's administrative seat was, by long-standing convention, also the place
> where the regional orchestra was headquartered..." (plus 9 distractor paragraphs)

Question: **"Where was the regional orchestra headquartered?"**
Gold answer: **the northern city** (call it *Trondheim*).

- **Extractive at a tight budget:** the chain "born in the northern city" → "northern city = the
  province's seat" → "province's seat = orchestra HQ" is spread across three verbose sentences
  totalling far more than budget `B`. Selecting any subsequence that fits `B` breaks at least one
  link; the solver loses the chain and cannot output *Trondheim*. The *best possible* token
  subset still can't carry a 3-hop coreference chain in `B` tokens.
- **Abstractive at the same budget:** "Trondheim is the regional orchestra's headquarters."
  — 7 tokens. The hop is *resolved* and *re-expressed*, not selected. Answer survives.

This is exactly the multi-hop-with-distractors regime, which is why our significant wins are on
HotpotQA and 2Wiki — those benchmarks are *built* from spread-out, coreference-chained facts.

## 3. Empirical confirmation — deletion never rescues rewriting

If deletion had headroom over rewriting, then **adding** bear after our rewrite should sometimes
help. It never does. Across all 4 benchmarks, `ours→bear` is **strictly worse** than `ours`:

| Benchmark | ours (abstractive) | ours→bear (then delete) | Δ from adding deletion |
|---|---|---|---|
| HotpotQA | 0.704 | 0.464 | **−0.240** |
| 2Wiki | 0.570 | 0.407 | **−0.163** |
| MuSiQue | 0.297 | 0.213 | **−0.084** |
| SQuAD v2 | 0.593 | 0.295 | **−0.298** |

Deletion applied to an already-dense rewrite can only *remove* signal the rewrite deliberately
placed — it has no mechanism to *add* any. This is the ceiling showing up from the other side:
once you are abstractive, going back to extractive is pure loss.

It is also why our earlier **bear-improver** idea was dropped: training an SLM to make
`bear(SLM(text))` better cannot exceed the standalone abstractive model, because the final
`bear(·)` step re-imposes the extractive ceiling. (See `experiments/EXPERIMENT_LOG.md`.)

## 4. Why this is the novel contribution

Prior work (LLMLingua, LongLLMLingua, LLMLingua-2, Selective-Context, RECOMP) compares
compression *methods* and reports ratios. To our knowledge none **characterizes a structural
ceiling of the extractive family** and shows abstractive compression is needed to cross it,
backed by (a) a reachability argument, (b) a constructed existence example, and (c) a 4-benchmark
`ours→bear`-never-helps confirmation.

The practical upshot for The Token Company: bear's blind-deletion design is excellent for the
regime it targets (fast, query-agnostic, reusable), but it has a *provable* blind spot on
multi-hop / distributed-fact queries — and the fix is not a better deletion heuristic, it is a
small **rewriter** that runs in front. That rewriter is what ReCompress distills into 1.5B.

---

*Evidence files: `eval/5bar_distilled_{hotpotqa,2wiki,musique,squad}.json` (the ours / ours→bear
columns), `experiments/EXPERIMENT_LOG.md` (the dropped bear-improver dominance result).*
