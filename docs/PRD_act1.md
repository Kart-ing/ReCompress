# Re:Zero — Query-Aware Compression Beyond Deletion — Our Entry

> **One line:** The Token Company's **bear-2** compresses prompts by **deleting** low-value tokens — explicitly *character-for-character, nothing summarized, paraphrased, or generated.* That's fast and lossless-by-design, but it's **blind in two ways deletion structurally cannot fix: it can't read your question, and it can't rewrite.** We build a **single query-aware SLM pass** that does exactly those two things — drops passages irrelevant to *this* question, and densifies verbose-but-relevant prose — and we prove it head-to-head **on their own model** on multi-hop + distractor tasks where blind deletion provably fails. Then the same engine becomes **Re:Zero**: a conversational memory that keeps you under the context window at turn 50 — a compressed *checkpoint* + a protected *trauma memory* + a small *delta* instead of ballooning history. **Headline: query-aware compression beats blind deletion at matched budget, on bear-2.**

**Event:** UC Berkeley AI Hackathon 2026 (June 20–21, 24h, team of 4)
**Track:** **The Token Company Compression Challenge** (1st: $2,000 + Claude Code 5× Max 6mo/person + MTS interview · 2nd: $1,000) — judged on **depth of research, ingenuity, creativity**
**Status:** DRAFT v6 — hardened after 3 independent adversarial reviews (win-odds, technical validity, prior-art). Read `Token Company Prompt.pdf`. **§13 = what the reviews killed and how we changed.**
**Author:** Kartikey + Claude

---

## 0. The thesis (reframed after review — read this first)

**Earlier drafts led with "we stack on bear and beat bear alone." Three reviews killed that as the *headline*.** Two reasons, both decisive:

1. **bear deletes character-for-character and generates nothing** (their own site, verbatim). So running bear *first* hands our rewriter shredded token-soup — wrong order. And "bear+ours beats bear" is likely true only because **ours does the work and bear rides along**, which inverts into the one thing we must never say to these judges: *"our layer makes your model redundant."*
2. **Compression-method stacking is already published.** Jha et al. (2024, arXiv:2407.08892) cascade a reranker → LongLLMLingua on HotpotQA; CompactPrompt stacks compressors and shows ratios compound; a recent survey lists "combining methods to further enhance compression ratios" as known future work. So "stacking" is **incremental, not novel.**

**The reframe that survives all three reviews:**

> **Pure deletion is blind to the query and to cross-passage redundancy. We add the one thing deletion provably can't: a single question-conditioned SLM pass that (a) drops passages irrelevant to *this* question and (b) densifies verbose-but-relevant prose. We benchmark it honestly, paired, on bear's own model, on the exact regime — multi-hop + distractors — where blind deletion fails.**

This is **additive and flattering, honestly**: we extend bear into the regime they *explicitly say they don't touch* ("nothing is paraphrased or generated"). We're not competing; we're covering their stated blind spot. That's a truer, harder-to-dismiss story than "stacked," and it's the one we lead with.

**`bear+ours` becomes a *tested hypothesis*, not the headline** — we run it, and present it **only if the data shows real compounding** (§5b). The pitch never depends on it.

---

## 1. What the brief says (and why this fits it)

From their prompt:
- **About Us:** custom context-compression models that **(1) reduce token costs ~50%, (2) improve downstream LLM performance, (3) enable more efficient AI.**
- **Challenge:** "Build your own compression solution that reduces the amount of information sent to an LLM while preserving the context needed for high-quality outputs." Can be a **system, model, algorithm, OR framework**, at the **model, application, or system level.**
- **Judging:** depth of research, ingenuity, creativity.

**Why query-aware-beyond-deletion fits:**
1. **It's on-brief** — "reduce information sent while preserving context needed for quality."
2. **It engages their tech honestly.** bear says it never paraphrases or reads the query; we do exactly that, and show the matched-budget head-to-head on their model. Depth of research = the honest paired benchmark on their own product.
3. **Creativity** is the Re:Zero conversational memory — checkpoint + protected facts + delta — reframing compression from "shrink one prompt" to "stay under the window across a conversation."

---

## 2. The contribution, precisely scoped (post-prior-art)

| Piece | What it does | Novelty (honest, post-review) | Role |
|---|---|---|---|
| **bear-2 (their model)** | Deletes low-value tokens, char-for-char. Fast, lossless-by-design, blind to query. | Theirs — the baseline we measure against. | Act 1 baseline bar |
| **Query-aware selection** | Reads the question; drops passages irrelevant to it (distractors blind deletion keeps). | **Exists** (LongLLMLingua is query-aware). Our angle: head-to-head *vs bear specifically*, on their model, in the distractor regime. | Act 1 "ours" |
| **Dense rewrite** | Rewrites verbose-but-relevant prose densely — the paraphrase bear refuses to do. | **Exists** (RECOMP abstractive, etc.). Prior art finds extractive often ≥ abstractive — so we **test, not assume**, and report honestly. | Act 1 "ours" |
| **Quality gate (frontier, offline)** | Detects when the answer no longer follows the compressed prompt; **re-runs compression at a softer ratio.** A floor — *not* a magic restorer. | Standard eval-time check. | Act 1 credibility |
| **Re:Zero multi-turn** | checkpoint + protected facts + delta instead of full history. | Summary+pinned-facts buffer (≈ MemGPT / ConversationSummaryBufferMemory). **The one fresh micro-claim:** protecting criticals lets us compress the summary *tighter at equal accuracy* (§3.5). | Act 2 — the demo |

**Where the real white space is (and it's thin but real):** a clean, paired, **head-to-head against a deletion-only compressor on its own model**, isolating *query-awareness* as the lever, in the *distractor* regime — most prior work compares against LLMLingua-2/truncation, not against a character-for-character deletion product. That specific comparison, done rigorously, is the defensible "depth of research" core.

---

## 3. Act 2 — the Re:Zero multi-turn system (the demo that can't fail live)

> **Why this is the demo, not Act 1:** its punchline — per-turn cost stays flat instead of growing — is **architecturally guaranteed true**, so it cannot blow up on stage in front of experts. Act 1's number is real but noisy on 24h evals; we lead the *demo* with the curve and present Act 1 as rigorous evidence.

### 3.1 The problem: quadratic re-payment
Resending full history every turn makes **per-turn cost grow linearly** and **cumulative cost grow quadratically (O(n²))**. Over 10 turns at +100 tok/turn you pay ~**5,500 cumulative tokens** for ~1,000 tokens of distinct information.

> **Pitch the cumulative number (~5,500), not the per-turn snapshot (1,000).** It's the honest figure and it's the actual spend.

### 3.2 The fix: checkpoint + protected facts + delta
Periodically compress history into a **checkpoint** (~150 tok: state, decisions, open questions). Each turn send only:
- **Checkpoint** (~150 tok, **fixed size — does not grow with conversation length**)
- **Trauma memory** (~50 tok — a *protected* layer: user's goal, names, hard constraints; never recompressed)
- **Delta** (~100 tok — what's new this turn)

**≈ 300 tok/turn instead of ~1000.**

### 3.3 Honest claim discipline (reviews forced these — do NOT overclaim on stage)
- ❌ "Nothing is lost." ✅ **"Nothing *critical* is lost."** A 150-tok checkpoint of 1000 tok has dropped ~850 tok by pigeonhole — say so. We **own the failure mode** (§3.4).
- ❌ Verifier "restores over-cuts" (circular — you'd need the original). ✅ **"A gate: detects the answer no longer follows, re-runs at a softer ratio."**
- ❌ "Re:Zero is novel." ✅ **"It's a summary + protected-facts buffer (like MemGPT) — our specific, measured claim is §3.5."**

### 3.4 The turn-15 probe (we ship the failure test ourselves)
Run a 15-turn conversation where **turn 14 references a detail introduced at turn 4 that the extractor did NOT flag.** Show Re:Zero vs naive-full-history on that question. This is exactly the probe an expert designs to break us — so we **bring it ourselves**, report when naive wins, and show how trauma-memory coverage + checkpoint depth trade off. Owning the limitation *is* the depth-of-research signal.

### 3.5 The one genuinely fresh micro-claim (measure it or concede it)
**Because criticals are protected in a separate buffer, the checkpoint can be compressed to a *tighter* ratio at *equal* task accuracy than a single undifferentiated summary.** This is the only part not clearly subsumed by MemGPT. **Deliverable: the ratio-at-equal-accuracy delta, protected-buffer vs single-summary.** If it holds, that's our ingenuity beat; if not, we concede novelty is packaging and lean on Act 1.

### 3.6 The caching objection (pre-baked — only the strong leg)
A sponsor judge **will** ask "doesn't prompt caching solve this?" Honest answer, leading with the leg that survives:
- ✅ **STRONGEST: caching cuts the *bill*, not *tokens-on-the-wire / context-window pressure*. You still blow the context window at turn 50. A checkpoint keeps you under it. Caching cannot extend the window; compression can. Different axis.**
- ⚠️ Weaker legs (mention briefly, don't lean): cache breaks on context shift (but a recomputed checkpoint also changes the prefix); short TTL (but ~1h covers many sessions). Under caching, the *cost* win shrinks to the read-multiplier delta — **so we don't lead with cost vs caching.**
- ✅ **We stack on caching** — checkpoint+trauma are stable and cache well; delta is cheap.

### 3.7 The Re:Zero analogy
Subaru carries a **compressed understanding forward** (checkpoint), **a few things he can't un-know** (trauma memory), and each loop adds only **what's new** (delta). A mnemonic for the architecture — *not* a claim of mechanism.

---

## 4. System overview

```
 ┌──────────── ACT 1: QUERY-AWARE vs BLIND DELETION (rigorous evidence) ─────────────┐
 │                                                                                    │
 │  long prompt + question ─►[ ONE query-aware SLM pass ]─► compressed                │
 │                              (drop off-topic passages + densify verbose prose)     │
 │                                                                            │        │
 │  baseline:  bear-2 alone (char-for-char deletion, blind to question)     ▼        │
 │                                                        [ SOLVER: frozen DeepSeek ]  │
 │  ALSO TESTED (present only if it compounds): bear+ours, ours+bear          │        │
 │                                                                            ▼        │
 │  [ QUALITY GATE (frontier, offline): answer still follows? else recompress softer ] │
 │                                                                                    │
 │  5-bar paired head-to-head @ matched budget:                                       │
 │      none │ bear │ ours-only │ bear→ours │ ours→bear                                │
 │  HEADLINE: "query-aware beats blind deletion at matched budget — on bear's model"  │
 │  logged in Arize, paired, bootstrap 95% CIs                                        │
 └────────────────────────────────────────────────────────────────────────────────────┘
                                     │  (same engine, now over a CONVERSATION)
 ┌──────────── ACT 2: RE:ZERO MULTI-TURN (the demo — punchline can't fail) ──────────┐
 │  history ─►[ checkpoint builder ]─► checkpoint(~150) ─┐                            │
 │  critical facts ─────────────────► trauma memory(~50) ─┼─► send ~300 vs ~1000      │
 │  this turn ──────────────────────► delta(~100) ────────┘                          │
 │  DEMO: token-per-turn graph — naive climbs O(n²), Re:Zero flat. + turn-15 probe.   │
 └────────────────────────────────────────────────────────────────────────────────────┘
```

### Models (best/cheapest — no track politics)
| Role | Model | Why |
|---|---|---|
| **Baseline (deletion)** | **bear-2** (The Token Company) | their model; "ours vs bear" is literal on their product |
| **Frozen solver** (answers from compressed context — the quality judge) | **DeepSeek V4 Flash** | cheap, fast, frozen for consistent measurement |
| **Our compressor** (query-aware select + dense rewrite — one pass) | **DeepSeek V4 Pro** | strong + cheap; the "intelligence" that reads the question |
| **Quality gate** (offline) | **frontier (Claude Sonnet; Opus for final stamp)** | strongest floor judge; **eval-time only, not in the shipped hot path** |

> **Thesis stays clean:** shipped compressor is SLM-only (DeepSeek). The frontier model is the **offline gate that proves accuracy holds**, not part of the product loop.

> **Economics, stated honestly (review K4):** an LLM compression call has real cost/latency vs bear's pennies/milliseconds. The win is positive **when the compressed context is reused across many downstream calls (amortize) or the downstream context is huge.** We will show the **net** token+latency+$ ledger for one end-to-end query, compression-call cost INCLUDED — not just prompt-tokens-saved. Judges will do this arithmetic; we do it first.

Budget: ~$20 DeepSeek + modest frontier-gate eval calls (aggressive caching). No GPU for the core.

---

## 5. The FLOOR — the validation experiment (run this FIRST, before building anything)

**Both judgment reviews said: run one cheap experiment before touching Re:Zero, the gate, or distillation. It tells you whether the headline is true.** ~2 hours, ~$2.

### 5a. The 5-bar paired test
- **Data:** 50 **HotpotQA** instances (multi-hop + distractor passages = exactly where query-awareness should beat blind deletion). Seeded, fixed.
- **Solver:** frozen DeepSeek Flash, same for every bar.
- **Bars @ matched token budget:** `none` · `bear` · `ours-only` · `bear→ours` · `ours→bear`.
- **Metric:** QA-F1, **paired** (every method on the same instances), **bootstrap 95% CI** on each pairwise delta vs `bear`.
- Token counts via the **solver's real tokenizer** (DeepSeek's), never tiktoken.

### 5b. The decision rule (this is the whole strategy)
- **PASS (headline secured):** CI for **(ours-only − bear)** excludes 0 in our favor → *query-aware beats blind deletion on their model.* **This alone wins the track.** Lead with it.
- **BONUS (only then do we say "stacked"):** CI for **(bear+ours − ours-only)** *also* excludes 0 → compounding is real; present `bear+ours`. Per prior art (Jha et al.) and bear's delete-only nature, **expect this to be small or zero** — if so, we **drop "stacked" entirely**, no exposed flank.
- **Order check:** report `bear→ours` vs `ours→bear`. Expect `ours→bear ≥ bear→ours` (bear-first shreds prose the rewriter needs). Running ours-first is the correct pipeline; we say so.

### 5c. The benchmark proper (after the 2h gate passes)
Extend to 2–3 LongBench QA subtasks (`hotpotqa`, `2wikimqa`, `multifieldqa`) for breadth. Same paired design, same Arize logging. **Headline framing:** *at matched budget, query-aware (ours) > blind deletion (bear); accuracy held vs full-context.* Keep truncation as a sanity floor we beat.

**This is the floor. If everything else fails, the 5-bar `ours vs bear` result on their own model is a complete, on-brief, sponsor-flattering submission.**

---

## 6. The CEILING — Re:Zero multi-turn (ingenuity + the live demo)

Build §3 on top of the §5 compressor:
- Scripted ~10–15-turn conversation (realistic multi-hop task).
- Two runs: **naive** (full history) vs **Re:Zero** (checkpoint + trauma + delta).
- **Demo artifact:** live token-per-turn graph (naive O(n²) vs Re:Zero flat) + side panel showing the checkpoint and protected facts.
- **Ship the turn-15 probe** (§3.4) — show where it holds and where naive wins.
- **Measure the §3.5 micro-claim** (tighter checkpoint at equal accuracy via the protected buffer) — our one fresh beat.

**Framing:** Act 2 is the *demo and the creativity score*; Act 1 is the *depth score*. Neither depends on the other.

---

## 7. The TROPHY — distill into a small model (stretch only)
Only if floor+ceiling locked. Brief doesn't require weights.
1. Best query-aware strategy → **(long → compressed)** pairs.
2. **LoRA fine-tune Qwen2.5-1.5B** (Unsloth, ~30 min on one A100 on Modal) to imitate query-aware+dense compression.
3. Re-eval on held-out tasks vs bear.
4. Closer: *"We distilled the query-aware layer into a 1.5B model — your product category — runs offline."*
⚠️ Cut the instant floor/ceiling is at risk. Data-gen is the long pole.

---

## 8. 24h build path

**Core is API-bound** (bear + DeepSeek), no GPU until the optional trophy.

### Priority order
1. **§5 VALIDATION EXPERIMENT (2h)** — proves or kills the headline before any real build.
2. **FLOOR** — full 5-bar benchmark + Arize. *Guarantees a sponsor submission.*
3. **CEILING** — Re:Zero curve + turn-15 probe + §3.5 measurement (creativity).
4. **TROPHY** — distill (bonus).

### Gate to "we have a submission" (~hour 8)
| # | Component | Effort |
|---|---|---|
| 0 | **§5 5-bar validation on 50 HotpotQA** — decide PASS/BONUS/order | **2h, do first** |
| 1 | bear-2 access wired + deletion output | 0.5–1h |
| 2 | Benchmark data + metric on uncompressed context (2 subtasks) | 1–2h |
| 3 | `compress_ours(text, question, ratio)` — query-aware select + dense rewrite, **ours-first** | 1–2h |
| 4 | `evaluate(method) → (acc, ratio)` — compress → solve → parse → score → real token count | 2–3h |
| 5 | Wire bars (none/bear/ours/bear→ours/ours→bear) into one harness | 1h |
| → | **FLOOR DONE: 5-bar `ours vs bear` on their model. Submission exists.** | |
| 6 | Offline gate pass (frontier) → quality-floor numbers + recompress-on-fail | 1h |
| 7 | Re:Zero loop + token-per-turn graph + turn-15 probe + §3.5 delta | 2–3h |
| 8 | (stretch) Distill Qwen-1.5B on Modal | 2–3h |

### Hour-by-hour (4 people)
- **H0–2:** keys (bear, DeepSeek; Modal only if trophy). **B+D run §5 validation immediately.** **C:** dashboard on mock data. **A:** Re:Zero loop on a mock compressor.
- **H2–8:** **B+D land the FLOOR** (5-bar, Arize). **← submission exists.** §5 result decides whether we say "stacked."
- **H8–14:** **A** swaps real compressor into Re:Zero → curve + turn-15 probe + §3.5; **D** runs offline gate.
- **H14–18:** lock results; paired-stats table + 5-bar chart + multi-turn curve + net-cost ledger; (stretch) start distillation.
- **H18–22:** polish demo (**C**), dry-run pitch, hit **workshop/booth (Floor 5, Tilden Room, 6 PM)** — show them `ours vs bear`, tune to their reaction. **Know your number before you walk on.**
- **H22–24:** submit + buffer.

### Cut order if behind
Trophy → Re:Zero multi-turn (ship single-prompt) → §3.5 micro-claim → fewer baselines (keep bear + truncation) → fewer subtasks. **The 5-bar `ours vs bear` floor is never cut.**

### 2-person fallback
**P1 "Compressor + eval"** (§5 → floor → Re:Zero); **P2 "bear wiring + baselines + gate + demo + pitch."** P2 on a mock `evaluate()` so neither blocks. Cut to: floor + short Re:Zero curve, one chart, no distillation. Lean on parallel coding agents for glue.

---

## 9. Team & parallelization (4 people)

| Stream | Owner | Deliverable |
|---|---|---|
| **B. Compressor + eval harness** | full-stack | benchmark + metrics, `compress_ours()` (ours-first), `evaluate()→(acc,ratio)`, token counting, Arize. *Critical path — owns §5 + FLOOR.* |
| **D. bear wiring + baselines + gate + pitch** | you | bear-2 integration, truncation baseline, offline frontier gate, net-cost ledger, pitch, booth |
| **A. Re:Zero engine** | AI/agents | checkpoint builder, trauma memory, delta, turn-15 probe, §3.5 measurement — *on B's compressor* |
| **C. Demo surface (HTML)** | 3D/creative | 5-bar headline chart, before/after text, token-per-turn flattening curve, checkpoint+trauma panel |

**Dependency rule:** B+D deliver §5 + FLOOR independently of A. A debugs Re:Zero on a mock compressor, swaps real at ~hour 8. **The `ours vs bear` headline never depends on multi-turn finishing.**

---

## 10. Top risks & fallbacks

| # | Risk | Likelihood | Fallback |
|---|---|---|---|
| 1 | **`ours` doesn't beat `bear` (the floor fails)** | Low-Med | Query-awareness on multi-hop+distractor data is where deletion is weakest — strong prior reason to expect a win (LongLLMLingua-style). If it ties: relax to "best ratio at full quality," report the honest curve, lead with the multi-turn demo. |
| 2 | **`bear+ours` is flat/negative in front of bear's authors** | **High** | **By design we don't headline it.** §5b decision rule: present only if CI excludes 0. If not, drop "stacked." No exposed flank — this is the whole point of v6. |
| 3 | **"Stacking is already published" (Jha et al., CompactPrompt)** | High (an expert may know it) | We **cite it ourselves** (§2, §13) and scope our claim to the *unstudied* slice: head-to-head vs a *delete-only product model* in the distractor regime, isolating query-awareness. Honesty about prior art *is* depth of research. |
| 4 | **"Caching solves multi-turn"** | High (they'll ask) | §3.6 — lead with context-window survival (the leg that holds); we stack on caching; we don't claim a big cost win vs caching. |
| 5 | **bear access flaky / rate-limited** | Med | Cache bear outputs (deterministic per input); pre-compute the benchmark offline; if access dies, fall back to a deletion baseline (LLMLingua-2) framed as "vs any blind-deletion compressor." |
| 6 | **Net economics don't close on a single call** | Med | State plainly (§4): win is under amortization / huge context. Show the honest ledger; don't claim per-call savings we don't have. |
| 7 | **Re:Zero rough at demo** | Med | It's the ceiling. Ship single-prompt floor; show the curve on a pre-recorded run if live is shaky (punchline still guaranteed). |
| 8 | **Distillation eats the clock** | Low (stretch) | Cut it. |

**Guiding principle:** **Validate (§5) → Floor → Ceiling → Trophy.** Know the `ours vs bear` number before the booth; add the Re:Zero curve on top; add the model only if free.

---

## 11. The 3-minute pitch (reframed)

1. **Hook (their model, honest):** "bear-2 deletes low-value tokens — character-for-character, nothing paraphrased or generated. Fast, lossless-by-design. But that means it's blind two ways it *can't* fix: it can't read your question, and it can't rewrite. So we built the one pass that does both."
2. **The floor (proof, on their model):** "On HotpotQA — multi-hop, full of distractor passages — at matched token budget, our query-aware pass **beats blind deletion**, because it drops the off-topic passages bear keeps and condenses the verbose ones it can't touch. Paired, with CIs, in Arize. We're not replacing bear — we're covering the regime they *say* they don't: paraphrase and query-awareness."
3. **The gate (credibility):** "And it's safe — an offline gate checks the answer still follows; if not, it recompresses softer. A real quality floor, not a magic restorer."
4. **The ingenuity (Re:Zero — the live demo):** "Same engine, now across a *conversation*. Instead of re-paying ~O(n²) for history, we keep a compressed **checkpoint**, a protected **trauma memory** of never-forget facts, and a small **delta**. Watch per-turn cost: naive climbs, ours stays flat — and here's the turn-15 probe proving we don't drop what matters. The key point caching can't touch: **we keep you under the context window at turn 50.** And we stack on caching."
5. **Close (honest depth):** "We took your model, found the two things deletion structurally can't do, proved query-awareness wins on your own product where blind deletion is weakest — and we'll show you the net-cost ledger and the prior art we built past. We measured it like we'd have to defend it to you. Because we did."

---

## 12. Open questions (decide / confirm)

- **Q1.** Confirm headline = **`ours` (query-aware) > `bear` (deletion) at matched budget** on multi-hop+distractor data. `bear+ours` is a tested hypothesis, shown only if §5b passes. (This is the v6 reframe — confirm you're good with demoting "stacked.")
- **Q2.** bear-2 access — confirmed API/weights + rate limits? (Affects pre-computing the benchmark offline.)
- **Q3.** Run the **§5 2-hour validation experiment first**, before any real build? (Both reviews: yes. Strongly recommend.)
- **Q4.** Gate model — Sonnet (cheap, for the loop) + Opus (final stamp), or Opus throughout?
- **Q5.** Re:Zero demo — synthetic scripted conversation, or a real multi-hop task across turns? (Real is more convincing if time allows.)
- **Q6.** Keep the distilled-SLM trophy, or drop to protect floor+ceiling?
- **Q7.** Booth/workshop **6 PM, Floor 5 Tilden Room** — go early, show `ours vs bear`, tune the pitch. (Strongly recommend — they're the judges.)
- **Q8.** Arize: keep for the $1k Arize track too, or focus purely on Token Company?
- **Q9 (research-paper angle).** Given the prior art (LongLLMLingua = query-aware; Jha et al./CompactPrompt = stacking; RECOMP = abstractive), a *publishable* novel claim is narrow. The honest candidates: (a) the §3.5 protected-buffer-enables-tighter-checkpoint result if it holds with stats; (b) a rigorous study of query-aware vs delete-only *on a production deletion model* in the distractor regime. **Decide:** pursue the paper only if (a) or (b) produces a clean, defensible delta — otherwise it's re-deriving known results. (See §13.)

---

## 13. What the reviews killed (and how v6 changed) — keep this; it's our honesty ledger

Three independent adversarial reviews (win-odds, technical validity, prior-art). What they found:

**KILLED — "bear → ours" pipeline order.** bear deletes char-for-char and generates nothing (their site), so bear-first shreds the prose our rewriter needs. → **v6 runs ours-FIRST; §5 reports both orders.**

**KILLED — "bear+ours beats bear" as the headline.** Likely true only because ours does the work; would read as "your model is redundant"; and stacking is already published (**Jha et al. 2024 arXiv:2407.08892** cascades reranker→LongLLMLingua on HotpotQA; **CompactPrompt** stacks compressors; a survey lists "combining methods" as known future work). → **v6 headline = `ours vs bear` (query-aware beats blind). `bear+ours` shown only if §5b CI excludes 0.**

**KILLED — verifier "restores over-cuts" (circular).** Can't restore deleted content without the original. → **v6: a gate that detects failure and recompresses softer. Not a restorer.**

**KILLED — "nothing is lost."** False by pigeonhole for a 150-tok checkpoint of 1000 tok. → **v6: "nothing *critical* is lost," and we ship the turn-15 probe (§3.4) that tests exactly this.**

**DEMOTED — Re:Zero novelty.** It's summary + pinned-facts buffer ≈ **MemGPT / ConversationSummaryBufferMemory**. → **v6: own that; isolate the one fresh micro-claim (§3.5, protected buffer → tighter checkpoint at equal accuracy) and *measure* it.**

**TIGHTENED — caching objection + economics.** Cost win vs caching is small; LLM-compression cost can swamp single-call savings. → **v6: lead with context-window survival (§3.6); show the net cost ledger including the compression call (§4); claim savings only under amortization/huge context.**

**Net odds from the reviews:** places ~55–65%; wins 1st ~30–40%, conditional on the §5 number landing — which is *why* we run §5 first and never let experts watch us discover a negative result live. Scores: Depth 8 / Ingenuity 8 / Creativity 6 — creativity (the thin axis) is carried by the Re:Zero demo. **The single highest-leverage move, agreed by both judgment reviews: demote "we beat bear," lead the demo with the multi-turn flattening curve whose punchline can't fail, and walk in already knowing your benchmark number.** v6 implements exactly that.
