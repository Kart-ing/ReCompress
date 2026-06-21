# PRD — Integrating ReCompress (Act 1) into ReZero (Act 2)

**Status:** draft for review
**Goal:** make the distilled query-aware compressor (v3, Act 1) the engine that powers the
multi-turn memory system (ReZero, Act 2), then benchmark the combined system on a custom
multi-turn task. One unified story: *"ReCompress (the distilled 1.5B) powers ReZero (flat-context
multi-turn)."*

---

## 1. Background — what each half is (verified from the code)

**Act 1 (ReCompress):** a distilled Qwen2.5-1.5B + LoRA that does **query-aware single-shot
compression** (drop distractors + densify). Beats bear ~50% on multi-hop QA. Runs on Modal
(`src/distill/infer.py::Compressor.compress_batch`). Adapter = v3 on `/vol/adapter`.

**Act 2 (ReZero):** a **multi-turn conversation memory** system. Instead of letting context
grow O(n²) with every turn, it keeps a fixed ~300-token budget:
- **trauma** (≤50 tok): persistent must-never-forget facts
- **checkpoint** (≤150 tok): compressed summary of older turns
- **delta** (≤100 tok): recent raw turns
An "Echidna" policy decides when to checkpoint/revert. Core: `ACTII/rezero/session.py`.

**The seam:** ReZero builds checkpoints by calling `ACTII/engine/compressor.py::compress(text,
question, ratio)` — which today calls **DeepSeek API**. Used in `rezero/checkpoint.py` and
`experiments/microclaim.py`. **This is the single integration point.**

**Act 2 already has baselines:** `NaiveSession` (O(n²) growing context) and `baselines/
token_company.py` (the real TTC `with_compression` SDK). So bear is already wired as a comparator.

---

## 2. What we're building

### 2a. A pluggable compressor backend for Act 2
Replace the hard-coded DeepSeek `compress()` with a backend selector:

| Backend | What | Source |
|---|---|---|
| `deepseek` (current) | API teacher compression | `engine/deepseek.py` |
| **`distilled`** (NEW) | our v3 model on Modal | `src/distill/infer.py::Compressor` |
| `bear` | TTC blind deletion | `thetokencompany` SDK |

Implementation: a thin `engine/compressor_backend.py` with `compress(text, question, ratio,
backend=...)`. ReZero's checkpoint builder reads the backend from session config. **No change to
ReZero's memory logic** — only *what* does the compressing.

Caveat to resolve: ReZero calls `compress()` **synchronously, one checkpoint at a time**. Our
v3 lives on Modal (remote). Options: (a) call `Compressor().compress_batch.remote([one])` per
checkpoint (simple, ~3-5s latency/checkpoint), or (b) keep a warm Modal container for the session.
**Decision: (a) for the benchmark** (correctness over latency); note the latency honestly.

### 2b. A custom multi-turn benchmark
There's no standard "multi-turn compression" benchmark — we build one. Two candidate designs:

**Design A — Multi-turn HotpotQA (recommended):** chain several HotpotQA/2Wiki instances into
one "conversation" (each turn adds a passage + a sub-question), then at the end ask a question
that requires a fact from an *early* turn. Measures: does the memory system retain the early
fact after many turns? This directly tests ReZero's value (flat context that doesn't forget).

**Design B — Synthetic goal-tracking:** a scripted multi-turn task (Act 2 already has
`demo/scripted_convo.jsonl`, 15 turns) where the answer depends on facts scattered across turns.

We use **A** for rigor (real QA, gold answers, F1), optionally show **B** for the demo.

### 2c. The comparison matrix
Memory strategy × compressor backend, measured on the multi-turn benchmark:

| | tokens@turn-N (flatness) | final-answer QA-F1 | cost/latency |
|---|---|---|---|
| Naive (growing) | grows O(n) | (baseline quality ceiling) | grows |
| ReZero + deepseek | flat ~300 | ? | API $ |
| **ReZero + distilled-v3** | flat ~300 | ? | offline/cheap |
| ReZero + bear | flat ~300 | ? | API $ |

**Headline we're testing:** *ReZero keeps context flat (~300 tok) while a Naive agent blows up,
AND ReZero+our-distilled-v3 retains answer quality competitively — for free, offline.*

---

## 3. Metrics
- **Flatness:** tokens in `prompt_for_solver()` at each turn (the O(n²)-vs-flat curve — the demo visual).
- **Quality:** QA-F1 of the final answer (frozen DeepSeek solver, same as Act 1).
- **Retention:** F1 specifically on questions whose evidence appeared in early (pre-checkpoint) turns.
- **Cost proxy:** total tokens sent to the solver across the conversation (Naive = sum of growing
  prompts; ReZero = ~300 × turns).

## 4. Scope / non-goals
- **In:** backend toggle, v3-via-Modal wiring, Design-A benchmark, the matrix above, the flatness curve.
- **Out (for now):** repo restructure to merge Act1+Act2 cleanly (explicitly deferred per direction —
  "we'll restructure later"); keeping a warm Modal container; tuning ReZero's caps.
- **Risk:** v3 was trained on single-shot HotpotQA compression; checkpoint summarization is a
  *slightly* different distribution (conversation turns, not passages). It may underperform the
  DeepSeek backend on checkpoints — **that's a finding either way**, and the standalone Act-1
  result stands regardless.

## 5. Build order
1. `engine/compressor_backend.py` — backend selector (deepseek | distilled | bear).
2. Wire `rezero/checkpoint.py` + `ReZeroSession` to accept `backend=`.
3. Local smoke: run a 3-turn ReZero session with `backend="deepseek"` (no Modal) to confirm
   the seam works, then `backend="distilled"` (1 Modal call) on one session.
4. `experiments/multiturn_benchmark.py` — Design A: build N multi-turn convos, run the matrix.
5. Run it (detached on Modal for the distilled rows). Render the flatness curve + F1 table.
6. Commit. (Repo restructure = later.)

## 6. Success criteria
- ReZero+distilled-v3 runs end-to-end and produces the flatness curve + a final-answer F1.
- We can state, with numbers: tokens-flat (ReZero) vs tokens-growing (Naive), and how
  ReZero+v3's answer quality compares to ReZero+deepseek and ReZero+bear.
- Honest verdict even if v3 underperforms on checkpoints (scopes where the distilled model helps).
