# PRD — Act 2: Re:Zero Multi-Turn Compression

> **Owner:** Multi-turn stream (teammate). **Dependency:** Act 1 `compress_ours(text, question, ratio)` from the single-shot stream.
> **Event:** UC Berkeley AI Hackathon 2026, 24h. **Track:** The Token Company Compression Challenge.
> **One line:** Same query-aware compressor, now run across a *conversation* — keep per-turn cost flat instead of O(n²) by sending `checkpoint + trauma memory + delta` instead of full history.

---

## 1. Goal

Turn the Act 1 single-shot compressor into a **conversational memory** that keeps the agent under the context window at turn 50. The live-demo punchline — **per-turn token cost stays flat while naive full-history grows quadratically** — is *architecturally guaranteed true*, so it cannot blow up on stage. That's why Act 2 is the demo and Act 1 is the evidence.

**Non-goals (do NOT build these here):**
- The single-shot `compress_ours()` / 5-bar benchmark / bear wiring — that's Act 1.
- Distillation into a small model — separate trophy stream.
- Cost-beats-caching claims — we lead with *context-window survival*, not $ vs cache (§3.6 of main PRD).

---

## 2. The architecture (build exactly this)

Each turn, send to the solver **only** (~300 tok):

| Layer | Size | Grows with convo? | What it holds |
|---|---|---|---|
| **Checkpoint** | ~150 tok | **No — fixed** | Compressed running state: decisions made, open questions, current sub-goal. Rebuilt periodically from full history via the Act 1 compressor. |
| **Trauma memory** | ~50 tok | **No — protected, never recompressed** | Never-forget facts: user's goal, names, hard constraints, any fact flagged *critical*. Append-only; carved out *before* checkpoint compression. |
| **Delta** | ~100 tok | No (per-turn) | What's new *this* turn only. |

Contrast: **naive** resends full history → per-turn cost grows linearly, cumulative O(n²). Over 10 turns at +100 tok/turn ≈ **5,500 cumulative tokens** for ~1,000 of distinct info. Re:Zero sends ≈300/turn.

```
history ─►[ checkpoint builder (Act 1 compressor) ]─► checkpoint(~150) ─┐
critical facts ───────────────────────────────────► trauma memory(~50) ─┼─► send ~300 vs ~1000
this turn ────────────────────────────────────────► delta(~100) ────────┘
```

### 2.1 Checkpoint builder
- Input: full conversation history so far + current user goal (the "question" for query-awareness).
- Output: ≤150-token compressed checkpoint, produced by calling **`compress_ours(history, question=goal, ratio=...)`** from Act 1.
- **Rebuild cadence:** every `K` turns (start `K=3`; tune). Between rebuilds, checkpoint is frozen and only the delta carries new info.
- On rebuild, the prior checkpoint + trauma + recent deltas become the new "history" to compress — so the checkpoint is a *rolling* compression, not from raw transcript every time (keeps call cost bounded).

### 2.2 Trauma memory (the protected layer — this is our ingenuity beat)
- **Append-only, never recompressed.** Facts extracted by a small classifier/prompt over each user turn + each assistant answer.
- A fact is "critical" if: a name/entity the user introduced, a numeric constraint, an explicit goal, or a fact the user later references.
- Hard cap ~50 tok; if overflowing, drop lowest-confidence non-goal facts (never drop the user's stated goal).
- **Why it matters — §3.5 micro-claim:** because criticals live in a *separate protected buffer*, the checkpoint can be compressed to a *tighter* ratio at *equal* task accuracy than a single undifferentiated summary. **You must measure this** (§5).

### 2.3 Delta
- Raw (uncompressed) most-recent turn, truncated to ~100 tok. Cheap; no model call.

---

## 3. Interfaces you depend on from Act 1

> These are the contract. If Act 1's signature changes, flag it immediately.

```python
from act1 import compress_ours, count_tokens, solve

# Query-aware compressor (the single-shot engine)
compressed_text: str = compress_ours(
    text: str,              # the context/history to compress
    question: str,          # the user's goal / current question (query-conditioning)
    ratio: float,           # target compression ratio (0..1, fraction of tokens to KEEP)
) -> str

# Token count using the SOLVER's real tokenizer (DeepSeek) — never tiktoken
n: int = count_tokens(text: str) -> int

# Frozen solver — answers a question given context
answer: str = solve(context: str, question: str) -> str
```

**Mock-first:** build against a stub `compress_ours` that just truncates, so you're unblocked from H0. Swap the real one in at ~H8.

---

## 4. Deliverables (in priority order)

| # | Deliverable | Done when |
|---|---|---|
| **D1** | `rezero.py`: `ReZeroSession` class with `.add_turn(user, assistant)`, `.prompt_for_solver()` → `checkpoint+trauma+delta` string, `.token_count()` → int. Works on mock compressor. | A 10-turn scripted convo runs end-to-end; token count stays ~flat. |
| **D2** | `naive.py`: same interface but resends full history. | Paired run produces the O(n²) curve. |
| **D3** | **Token-per-turn graph** (the demo artifact): naive vs Re:Zero, turns 1–15, cumulative + per-turn. HTML/PNG, embeddable in the demo page. | Two curves visible: naive climbing, Re:Zero flat. |
| **D4** | **Turn-15 probe (§3.4):** a scripted conversation where turn 14 references a detail from turn 4 that the extractor did *not* flag as critical. Run both naive and Re:Zero on it. Report QA-F1 for both. | We *ship the failure test ourselves*. Expect naive to win this one — owning that is the depth-of-research signal. Show how trauma-memory coverage + checkpoint depth trade off. |
| **D5** | **§3.5 micro-claim measurement:** two Re:Zero variants — (a) protected-buffer + checkpoint, (b) single undifferentiated summary at the same total budget. Sweep checkpoint ratio; plot **task accuracy vs ratio** for both. | A chart showing (or refuting) that the protected buffer lets the checkpoint go tighter at equal accuracy. If it doesn't hold, we concede novelty-is-packaging and lean on Act 1 — report honestly. |
| **D6** | Live demo panel (HTML, teammate C may own visual): shows checkpoint text, trauma facts, delta, and the flattening curve updating turn-by-turn. | Runs on the scripted convo; can accept a live question. |

---

## 5. Experiments & honesty discipline (from main PRD §3.3 — do NOT overclaim on stage)

- ❌ "Nothing is lost." → ✅ **"Nothing *critical* is lost."** A 150-tok checkpoint of 1000 tok dropped ~850 by pigeonhole — say so.
- ❌ Verifier "restores over-cuts." → ✅ No restorer. The gate (Act 1, offline) detects and recompresses softer; you just consume that signal if provided.
- ❌ "Re:Zero is novel." → ✅ **"It's a summary + protected-facts buffer, like MemGPT. Our specific measured claim is §3.5."**
- **Caching objection (§3.6) — pre-baked, lead with the strong leg:** caching cuts the *bill*, not *tokens-on-the-wire / context-window pressure*. You still blow the window at turn 50. A checkpoint keeps you under it. **Caching cannot extend the window; compression can.** We stack *on* caching (checkpoint+trauma are stable, cache well; delta is cheap).

### 5.1 §3.5 measurement spec (the one fresh beat — measure it or concede it)
- **Task:** a 10–15-turn multi-hop conversation with a ground-truth final answer (use a HotpotQA-style chain or adapt Act 1's data).
- **Variant A (ours):** trauma(~50) + checkpoint(ratio r) + delta(~100).
- **Variant B (control):** single summary of full history at total budget = 50 + checkpoint_budget(r) + 100 (same token spend).
- **Sweep** `r ∈ {0.10, 0.15, 0.20, 0.25, 0.30}`.
- **Metric:** QA-F1 on the final-turn question, averaged over ≥5 seeded conversations.
- **PASS:** at equal QA-F1, variant A tolerates a *lower* `r` than B (tighter checkpoint). Report the `r` delta + a paired CI. **This is our ingenuity score.**

---

## 6. Demo scenario (script the conversation — make it realistic)

A multi-hop task across turns, e.g. *"Help me find which of these two companies had a founder who also founded a non-profit, and compare their latest funding rounds."* Across 12–15 turns the user:
- introduces the two companies + a constraint (→ trauma),
- asks clarifying sub-questions (→ delta, rolling checkpoint),
- at turn 14 references a detail from turn 4 (→ the probe).

Keep the scripted transcript in `demo/scripted_convo.jsonl` so both the curve and the live demo replay from the same source.

---

## 7. Timeline (your hours, parallel to Act 1)

| Hour | You |
|---|---|
| H0–2 | `ReZeroSession` + `naive.py` on **mock** `compress_ours` (truncation). Token-count curve visible on mock. |
| H2–6 | Turn-15 probe + §3.5 harness on mock. Scripted conversation locked. |
| H6–8 | Swap real `compress_ours` in. Verify curve still flat, numbers real. |
| H8–12 | Run §3.5 sweep + probe on real compressor. Lock results. |
| H12–14 | Hand artifacts (curve PNG, checkpoint/trauma panel, §3.5 chart) to demo owner (C). |
| H14–18 | Dry-run live demo; buffer for Act 1 integration surprises. |

**If behind:** drop §3.5 (keep curve + probe); then drop live demo (show pre-recorded curve — punchline still holds); never drop the flattening curve itself.

---

## 8. File layout (propose to team)

```
src/
├── act1/                  # single-shot (not yours)
│   ├── compress.py        # compress_ours()
│   ├── solve.py           # frozen solver
│   └── tokens.py          # count_tokens()
├── rezero/
│   ├── session.py         # ReZeroSession (D1)
│   ├── naive.py           # full-history baseline (D2)
│   ├── trauma.py          # critical-fact extractor (D2.2)
│   └── checkpoint.py      # rolling checkpoint builder (D2.1)
├── experiments/
│   ├── turn15_probe.py    # D4
│   └── microclaim_3_5.py  # D5
└── demo/
    ├── scripted_convo.jsonl
    └── panel.html         # D6 (with C)
```

---

## 9. Definition of done

1. `ReZeroSession` runs a 15-turn scripted convo on the **real** Act 1 compressor, token count flat at ~300/turn.
2. Token-per-turn curve (naive O(n²) vs Re:Zero flat) rendered as an embeddable artifact.
3. Turn-15 probe results reported (honestly — including if naive wins).
4. §3.5 chart delivered: accuracy vs ratio, protected-buffer vs single-summary. Pass or fail, stated plainly.
5. Live demo panel shows checkpoint + trauma + delta updating per turn.

**The submission never depends on multi-turn finishing** — but if you land all five, Act 2 is the demo that carries the creativity score.
