# ReCompress — Build Progress

**Last updated:** H2 of 24h hackathon
**Track:** The Token Company Compression Challenge (UC Berkeley AI Hackathon 2026)

---

## What we're building

**bear-1.1** (The Token Company) compresses prompts by deleting tokens character-for-character — blind to the query, can't rewrite. **ReCompress** adds the two things deletion can't:
1. **Query-aware selection** — reads the question, drops irrelevant passages
2. **Dense rewrite** — densifies verbose-but-relevant prose

Then **distills** the query-aware compressor into a **1.5B model (Qwen2.5-1.5B-Instruct)** via LoRA on Modal (H100) — so it runs offline, cheap, same product category as bear. **This is the main submission.**

---

## Architecture (3 API calls → then distill into weights)

| Role | Model | Status |
|---|---|---|
| Compressor (teacher) | DeepSeek V4 Pro (API) | ✅ wired |
| Solver (frozen judge) | DeepSeek V4 Flash (API) | ✅ wired |
| Baseline (blind deletion) | bear-1.1 (TheTokenCompany SDK) | ✅ wired |
| Student (distilled) | Qwen2.5-1.5B-Instruct + LoRA (Modal H100) | ⏳ scaffolded, not trained |

---

## Smoke test result (1 HotpotQA instance, ratio=0.3)

| Bar | Output tokens | Answer | QA-F1 |
|---|---|---|---|
| bear (blind deletion) | 544 | Fels Institute Government | **0.00** |
| ours (query-aware) | 25 | Mossad | **1.00** |

✅ Pipeline works end-to-end. Query-awareness wins on the exact regime PRD predicts (multi-hop + distractors).

---

## Code built (all in `src/`)

### Act 1 — single-shot (Kartikey)
| File | What | Status |
|---|---|---|
| `src/config.py` | env-driven config | ✅ |
| `src/act1/client.py` | shared OpenAI client (120s timeout, 5 retries) | ✅ |
| `src/act1/tokens.py` | tokenizer-aware counting + truncation | ✅ |
| `src/act1/compress.py` | `compress_ours(text, question, ratio)` — query-aware | ✅ |
| `src/act1/bear.py` | `compress_bear(text, ratio)` — blind deletion via TTC SDK | ✅ |
| `src/act1/solve.py` | `solve(context, question)` — frozen DeepSeek solver | ✅ |
| `src/act1/metrics.py` | QA-F1 + paired bootstrap 95% CIs | ✅ |
| `src/act1/data.py` | HotpotQA streaming loader (50 seeded instances) | ✅ |
| `src/act1/evaluate.py` | 5-bar paired benchmark + §5b decision rule | ✅ |
| `src/act1/smoke.py` | 1-instance smoke test | ✅ passed |

### Distillation — the main submission path
| File | What | Status |
|---|---|---|
| `src/distill/gen_data.py` | teacher data gen: 1000 (text,q,compressed) pairs via DeepSeek | ✅ written, not run |
| `src/distill/train.py` | Modal app: Unsloth LoRA fine-tune Qwen2.5-1.5B on H100 | ✅ written, not validated |
| `src/distill/infer.py` | Modal deployed inference: `compress_distilled.remote()` | ✅ written, not tested |
| `src/distill/evaluate_distilled.py` | 5-bar re-eval with distilled model as "ours" | ✅ written by subagent |

### Act 2 — multi-turn (teammate)
| File | What | Status |
|---|---|---|
| `prd-multiturn.md` | PRD for Re:Zero multi-turn stream | ✅ written, pushed |

---

## §5 gate — RESOLVED & PASSED ✅ (clean 50/50, 0 errors)

Timeout + rate-limit issues fixed; full 50-instance 5-bar benchmark now runs end-to-end with **zero errors on every bar**.

**Final headline (50 HotpotQA instances, ratio=0.3, paired bootstrap 95% CI vs bear):**

| Bar | mean QA-F1 | Δ vs bear | 95% CI | excludes 0 |
|---|---|---|---|---|
| none (full ctx) | 0.877 | +0.425 | (+0.285, +0.571) | ✅ ★ |
| **ours (query-aware)** | **0.847** | **+0.395** | **(+0.259, +0.538)** | **✅ ★** |
| bear (blind del.) | 0.452 | — | — | — |
| bear→ours | 0.576 | +0.123 | (+0.027, +0.233) | ✅ ★ |
| ours→bear | 0.486 | +0.034 | (−0.084, +0.163) | ✗ |

**HEADLINE: PASS** — ours beats bear by **+0.395 F1** at matched budget, CI excludes zero. `ours` (0.847) nearly matches full-context `none` (0.877) at ~2-3% of the tokens. `bear→ours` stacking also significantly beats bear (+0.123). `ours→bear` is a wash (expected — bear-deleting our already-dense output mangles it). Results: `eval/5bar_results.json`.

**Fixes applied:**
- `src/act1/client.py` — shared client, 120s timeout + 5 retries ✅
- `src/act1/evaluate.py` — ThreadPoolExecutor (6 workers) + partial saves + per-instance try/except ✅ (was already done)
- `src/act1/bear.py` — **process-wide token-bucket rate limiter (55 req/min) + bounded retry** ✅ (root cause of the stacked-bar failures: TTC endpoint caps at 60 req/min, blown by parallel workers across 3 bear-using bars)
- `src/act1/solve.py` + `compress.py` — bounded retry on transient `APITimeoutError`/`APIConnectionError`/`RateLimitError` (flaky conference wifi hardening) ✅

---

## Build order (revised — distillation is main path, not stretch)

| Step | What | Status | ETA |
|---|---|---|---|
| 1 | **§5 5-bar validation** (50 instances, proves headline) | ✅ **DONE — PASS** (+0.395 F1, CI excl. 0) | — |
| 2 | **Generate distillation data** (300 pairs via DeepSeek API) | ✅ **DONE** — 300/300, 0 err, `data/distill/train.jsonl` | — |
| 3 | **Validate Modal training app** (syntax + config check) | ✅ **DONE — GO** (auth OK; fixed unsatisfiable deps + trl API breakages) | — |
| 4 | **LoRA train Qwen2.5-1.5B on Modal H100** (3 epochs, ~30 min) | ⏳ READY TO FIRE (`modal run src/distill/train.py --data data/distill/train.jsonl`) | ~45 min |
| 5 | **Re-run 5-bar with distilled model as "ours"** | blocked on step 4 | 20 min after |
| 6 | Re:Zero multi-turn (teammate, on real compressor) | parallel | their stream |
| 7 | Demo + pitch | after results | H18-22 |

### ⚠️ Data-quality caveat for distillation (Workstream 2 flagged)
The teacher compresses to **~2.4% ratio** (not the 30% target — `compress_ours` drops 8/10 distractor passages then densifies hard). More concerning: **~3-5% of pairs are effectively just the answer** (e.g. Q:"father of Jeanie Buss?" → `Jerry Buss`), violating the prompt's "do NOT answer" rule. Risk: the distilled student may learn to *answer* instead of *compress*, leaking answers / breaking the "compress, don't solve" contract. **Decision needed before/with training** — see options in chat. Training runs fine either way; this is about what behavior gets distilled.

### Modal fixes applied by Workstream 3 (so step 4 won't fail)
- `train.py`: removed unsatisfiable `trl==0.11.4`/`peft==0.13.2` pins (conflict with `unsloth`), added `unsloth_zoo`; `SFTTrainer(tokenizer=)` → `processing_class=`; `SFTConfig(max_seq_length=)` → `max_length=`.
- `infer.py`: same dep fix + `.add_local_python_source("src")` (modal ≥1.0 no longer auto-mounts; needed for `src.*` imports at runtime).
- ⚠️ `unsloth` left unpinned — GPU image build (pip resolution on CUDA) is the one thing untested; pin once a known-good version is confirmed on the build box for a reproducible demo.

### The real submission number
**Step 5** is the number we walk into the booth with: *our distilled 1.5B query-aware model vs bear-1.1, both small, both cheap, head-to-head on HotpotQA.* If ours beats bear at matched budget — that's the win.

### Fallback
If distillation eats the clock or the 1.5B can't learn it: the API-based 5-bar floor (step 1) still exists as a complete submission — weaker on economics but defensible.

---

## Key files for reference
- `token-company-prd.md` — main PRD (296 lines)
- `prd-multiturn.md` — Act 2 PRD for teammate
- `.env.example` — env var template (keys in `.env`, gitignored)
- `requirements.txt` — deps including `the-token-company`, `modal`, `unsloth`
