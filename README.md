# ReCompress

**Context compression that *rewrites* instead of *deletes* — distilled into a 1.5B model, then extended to keep multi-turn conversations flat.**

ReCompress is one research project in **two acts**, both attacking the same problem — *sending fewer tokens to an LLM without losing the answer* — from two angles:

| | **Act 1 — Single-shot compression** | **Act 2 — Re:Zero multi-turn memory** |
|---|---|---|
| Question | Can a small model compress *one* prompt better than blind deletion? | Can it keep a *long conversation* from growing O(n²)? |
| Method | Query-aware **rewrite** (drop distractors + densify), distilled into **Qwen2.5-1.5B + LoRA** | A flat ~300-token memory (protected facts + compressed checkpoint + recent delta), whose checkpoints are compressed **by the Act 1 model** |
| Headline | Beats [The Token Company](https://thetokencompany.com)'s **bear-1.1** by **+0.252 F1 on HotpotQA (+56%, CI excludes 0)**, generalizes zero-shot to 2Wiki (+46%) | Context stays **flat (~184 tok) while a naive agent grows to 1482 tok over 12 turns (8.1× less)** — at *higher* answer quality, with our distilled model as the engine |

**The two acts are one system.** Act 1 builds a query-aware compressor and distills it into a small offline model. Act 2 makes that model the memory engine of a multi-turn agent — and the distilled compressor beats *both* the DeepSeek API and bear as the Act 2 backend. The thesis across both: **query-aware rewriting beats deletion, and a 1.5B can carry it — for single prompts and for whole conversations.**

*Built in 24h for the **Token Company Compression Challenge**, UC Berkeley AI Hackathon 2026. This README is also the research writeup — it preserves the full failure→success trajectory across both acts, not just the wins.*

---

## 1. Act 1 — Headline results (single-shot compression)

Distilled **Qwen2.5-1.5B + LoRA** ("ours") vs **bear-1.1** ("bear"), at a **matched token budget** (ratio = 0.3), 50 seeded instances per benchmark, frozen DeepSeek solver, QA-F1, paired bootstrap 95% CI on the per-instance delta vs bear.

| Benchmark | ours (F1) | bear (F1) | Δ vs bear | 95% CI | rel. | verdict |
|---|---:|---:|---:|:--:|---:|:--:|
| **HotpotQA** (in-domain) | **0.704** | 0.452 | **+0.252** | (+0.103, +0.396) | **+56%** | ✅ **PASS** |
| **2Wiki** (cross-dataset) | **0.570** | 0.391 | **+0.180** | (+0.030, +0.340) | **+46%** | ✅ **PASS** |
| MuSiQue (cross-dataset) | 0.297 | 0.186 | +0.111 | (−0.027, +0.240) | +60% | ◐ n.s. |
| SQuAD v2 (cross-dataset, single-hop) | 0.593 | 0.471 | +0.123 | (−0.026, +0.269) | +26% | ◐ n.s. |

- **PASS** = the 95% CI on the paired delta excludes zero (ours is significantly better than bear).
- **n.s.** = "not significant": ours is *directionally* better on all four benchmarks (positive delta everywhere), but on MuSiQue and SQuAD the CI just includes zero at n=50, so we don't claim significance.
- The student was trained **only on HotpotQA-derived teacher data.** 2Wiki, MuSiQue and SQuAD are zero-shot transfer — the 2Wiki PASS is the strongest evidence the model learned *query-aware compression as a skill*, not HotpotQA trivia.

**What the upper bound looks like.** The API teacher it distills from (DeepSeek, query-aware) beats bear by **+0.395 F1** on HotpotQA (`ours`=0.847 vs bear=0.452, CI (+0.259, +0.538)) — and at ~3.5% of the tokens it nearly matches the **full-context ceiling** (`none`=0.877). The 1.5B student recovers about **64%** of that frontier-model margin (0.252 / 0.395) while running on a single small open model.

> On HotpotQA the budgets are: full context ≈ 1,364 tok → bear keeps ≈ 409 tok (30%) → **ours rewrites down to ≈ 48 tok (3.5%)** and is still **more accurate** than bear at 8× the tokens. Query-aware rewriting doesn't just compress harder; it compresses *better*, because it throws away the distractor passages bear faithfully preserves.

---

## 1b. Act 2 — Re:Zero multi-turn memory (the compressor, applied to conversations)

Single-shot compression shrinks *one* prompt. But a long **conversation** has the opposite problem: context grows every turn, so cost scales O(n²). **Re:Zero** (Act 2, in `ACTII/`) is a multi-turn memory that holds a **fixed ~300-token budget** — protected facts ("trauma") + a compressed checkpoint of older turns + the recent raw delta — so context stays flat no matter how long the chat runs. The catch: *something* has to compress the checkpoints. **We use the Act 1 distilled model.**

Combined benchmark (multi-turn HotpotQA, 6 turns, n=20; final-answer QA-F1 + context size at the last turn):

| Strategy | final F1 | context tokens |
|---|---:|---:|
| Naive (full growing history) | 0.485 | 846 |
| Re:Zero + DeepSeek API | 0.455 | 198 |
| **Re:Zero + distilled-v3 (ours)** | **0.501** | **174** |
| Re:Zero + bear | 0.472 | 257 |

**Re:Zero powered by our distilled compressor wins on *both* axes** — the highest answer quality (0.501, beating even the full-history naive agent) at the *fewest* tokens (174). And per-turn, the gap compounds:

> Over 12 turns, a **naive agent grows to 1,482 context tokens** while **Re:Zero stays flat at ~184 — 8.1× less, and diverging.** (`ACTII/results/token_trajectory.json`, figure `eval/figures/token_trajectory_line.png`.)

This is the unifying result: the same 1.5B query-aware compressor that beats bear on single prompts (Act 1) also beats DeepSeek *and* bear as a conversational-memory engine (Act 2) — offline, at a fraction of the tokens.

---

## 2. The idea

Retrieval-augmented prompts are mostly noise: 10 passages retrieved, 2 relevant, 8 distractors. A QA model then has to find the needle. Two ways to shrink that prompt:

| | **bear-1.1 (deletion)** | **ReCompress (rewrite)** |
|---|---|---|
| Operation | Deletes low-value tokens, char-for-char | Reads the question, **drops** off-topic passages, **rewrites** the rest densely |
| Sees the question? | **No** (query-agnostic) | **Yes** (query-aware) |
| Can paraphrase? | **No** (extractive) | **Yes** (abstractive) |
| Failure mode | Keeps distractors; truncates mid-passage; preserves the *wrong* facts faithfully | Can hallucinate / drop a fact if it mis-reads the question |

bear's design is honest about its blind spots ("nothing is paraphrased or generated"). ReCompress fills exactly that gap. It is **not** a competitor to bear — it's the abstractive, query-conditioned regime bear explicitly doesn't touch.

The novelty for *this track* is the comparison, not the components. Query-aware compression (LongLLMLingua) and abstractive compression (RECOMP) both exist in the literature. What's underexplored is a clean, **paired, matched-budget, head-to-head against a deletion-only product on its own model**, isolating *query-awareness + rewriting* as the lever, in the *distractor* regime — and then showing a **1.5B student** can carry most of that win offline.

---

## 3. Architecture

Three API roles do the research; then the query-aware behavior is **distilled into weights** so the production path is one small model.

```
                        ┌──────────────────────────────────────────────┐
   RESEARCH (API)       │                                              │
                        │   TEACHER  = DeepSeek (query-aware)          │  the quality ceiling
   context + question ──┼──▶  reads Q, drops distractors, densifies   │  (compress_ours)
                        │            │                                  │
                        │            ▼   (text, question, compressed)  │
                        │      1000s of training pairs ────────────────┼──┐
                        └──────────────────────────────────────────────┘  │
                                                                           │  DISTILL
   BASELINE (API/SDK)   bear-1.1  ── blind char-for-char deletion         │  (LoRA, H100)
                                                                           ▼
                        ┌──────────────────────────────────────────────────────────┐
   PRODUCTION (weights) │  STUDENT = Qwen2.5-1.5B-Instruct + LoRA (4-bit)           │
   context + question ──┼──▶  one offline pass, no API, ~cents to run             │  ← the submission
                        └──────────────────────────────────────────────────────────┘
                                            │ compressed context
                                            ▼
   JUDGE (frozen)       SOLVER = frozen DeepSeek  ──▶  answer  ──▶  QA-F1 vs gold
```

| Role | Model | Notes |
|---|---|---|
| Teacher (compressor) | DeepSeek (API) | query-aware; generates distillation data; also the standalone upper-bound bar |
| Baseline | bear-1.1 (TheTokenCompany SDK) | blind deletion; the system we measure against |
| **Student (the submission)** | **Qwen2.5-1.5B-Instruct + LoRA, 4-bit** | distilled on Modal H100; runs offline |
| Solver (judge) | DeepSeek (API), **frozen** | identical across all bars — it only ever sees the compressed context, never the original |

Holding the solver fixed is what makes the comparison fair: every bar is judged by the same downstream reader, so any F1 difference is attributable to the **compressor**, not the solver.

---

## 4. Results in depth — the 5-bar methodology

We don't report a single number; we run a **5-bar paired benchmark** so the result is interpretable and the stacking question is answered explicitly.

| Bar | What it feeds the solver |
|---|---|
| `none` | full original context (the accuracy **ceiling**, ~100% of tokens) |
| `bear` | bear-deleted context (the **baseline**) |
| `ours` | our query-aware compressed context |
| `bear→ours` | run bear first, then our model on its output |
| `ours→bear` | run our model first, then bear on its output |

All bars are truncated to the **same target token budget** and judged by the **same frozen solver**. Deltas are computed **paired** (per-instance `ours − bear`) with a 1000-iteration bootstrap 95% CI (`recompress/act1/metrics.py`).

### 4.1 The distilled student, full 5-bar, per benchmark

| Bar | HotpotQA | 2Wiki | MuSiQue | SQuAD v2 |
|---|---:|---:|---:|---:|
| `none` (ceiling) | 0.877 | 0.573 | 0.520 | 0.903 |
| `bear` (baseline) | 0.452 | 0.391 | 0.186 | 0.471 |
| **`ours`** | **0.704** ✅ | **0.570** ✅ | 0.297 ◐ | 0.593 ◐ |
| `bear→ours` (stack) | 0.323 | 0.371 | 0.060 | 0.282 |
| `ours→bear` (stack) | 0.464 | 0.407 | 0.213 | 0.295 |

### 4.2 What's significant vs directional — stated honestly

- **The headline holds where it's hardest.** On the two genuinely multi-hop, distractor-heavy benchmarks (HotpotQA, 2Wiki), ours beats bear with the CI excluding zero. These are the regime the whole thesis predicts deletion fails in.
- **MuSiQue and SQuAD are directional, not proven.** Ours is ahead on both (+60% and +26% relative) but the CI includes zero at n=50. We **do not** claim a win there. MuSiQue is brutal for *everything* (the full-context ceiling is only 0.520; even DeepSeek with the whole document barely clears 50% F1), so a small-model compressor having a noisy edge there is unsurprising. SQuAD is single-hop — there are few distractors to drop, so query-awareness has less to exploit.
- **Stacking does not work — and on easy data it actively hurts.** This is the most important honest finding:
  - `ours→bear` and `bear→ours` are a **wash** vs bear on HotpotQA and 2Wiki (CI includes 0).
  - On **SQuAD**, `ours→bear` is **significantly *worse*** than bear (Δ=−0.176, CI (−0.309, −0.031), excludes 0), and `bear→ours` is significantly worse on both **MuSiQue** (Δ=−0.126, excl. 0) and **SQuAD** (Δ=−0.189, excl. 0).
  - **Why:** our model's output is already a dense ~3.5%-ratio rewrite. Running bear's character-for-character deletion *on top of that* truncates a paragraph that has no slack left — it mangles a finished product. And running bear *first* hands our model shredded token-soup to rewrite. The two operations have incompatible contracts; composing them destroys evidence rather than compounding savings.
  - **Takeaway:** ReCompress is a *replacement* for bear in the query-aware regime, not a *layer* on top of it. (See §8 for the future-work idea this motivates.)

---

## 5. The distillation story (the research contribution)

The win in §1 was the **third** distillation attempt. The first two failed in instructive, different ways. Preserving that trajectory is the point — it's a textbook small-model-overfitting case study.

| Version | Data | LoRA | Epochs | Regularization | Result vs bear (HotpotQA) | Verdict |
|---|---:|---:|---:|---|---|---|
| **v1** | 261 train / 29 eval | r=16, α=32 | 3 | dropout 0.05 | Δ=**+0.063**, CI (−0.098, +0.220) | ❌ **WASH** (CI incl. 0) |
| **v2** | 2,500 | r=64, α=128 | 6 | dropout 0.05 | — (never finished cleanly) | 🔴 **OVERFIT** |
| **v3** | 5,000 | r=32, α=64 | 3 | dropout 0.1, weight_decay 0.01, **load_best_model_at_end** | Δ=**+0.252**, CI (+0.103, +0.396) | ✅ **PASS** |

### v1 — under-data wash
261 examples was simply not enough signal. The eval loss was **flat after epoch 1** (1.760 → 1.744 → 1.742) — not overfitting, but a *data-quantity ceiling*: the model had extracted everything 261 examples carry. It recovered only ~16% of the teacher's margin over bear. The teacher data was also pathologically over-compressed (~2.4% of input; ~3% of pairs were a bare answer, violating "don't answer the question"), so the student partly learned an over-aggressive, answer-leaking policy.

### v2 — overfit (and a logistics lesson)
We pushed three levers at once: more data (2,500), a much bigger adapter (**r=64** = 73.9M trainable params, 4.6% of the model), and more epochs (6). The bigger adapter had the capacity to **memorize**, and it did — a textbook divergence:

| Epoch | train_loss | eval_loss | |
|---|---:|---:|---|
| 1 | ~1.5 | 1.662 | |
| 2 | ~1.3 | **1.654** | ← best eval (the sweet spot) |
| 3 | ~1.1 | 1.685 | ⚠️ eval rising |
| 4 | ~1.0 | 1.762 | 🔴 overfit |
| 5 | ~0.95 | 1.861 | 🔴 badly overfit |

Train loss fell monotonically while eval loss bottomed at **epoch 2** then rose every epoch after — classic memorization. (Diagnosis: the overfitting is driven by the **rank increase**, since v1 at r=16 did *not* overfit at 3 epochs.) The run was also killed by a laptop network drop at step 770/846 — `modal run` is foreground and dies with the local connection; the fix is `modal run --detach` for long jobs.

### v3 — the win (textbook anti-overfitting)
v3 applied every standard anti-overfit lever at once, informed by the v2 curve:
1. **Lower rank** (r=64 → **r=32**) — halve trainable capacity.
2. **Fewer epochs** (6 → **3**) — eval bottomed near epoch 2.
3. **Early stopping** — `load_best_model_at_end=True` on `eval_loss` commits the best-eval checkpoint, not the last (overfit) one.
4. **More data + regularization** — 2,500 → **5,000** pairs, dropout 0.05 → **0.1**, **weight_decay 0.01**.

Result — **not overfitting**, eval loss fell every epoch:

```
epoch 1: eval_loss = 1.6787
epoch 2: eval_loss = 1.6655      (4,500 train / 500 eval, 213 steps, r=32)
epoch 3: eval_loss ≈ 1.664       OVERFITTING SIGNAL: no
```

…and downstream, this is the model that beats bear (§1). The lesson generalizes: **a 1.5B can learn query-aware multi-hop compression, but only with a regularized, early-stopped, adequately-data'd recipe** — not by simply throwing capacity and epochs at it.

> **Honest caveat that survives even v3:** with the floor/anti-answer teacher prompt, the teacher *still* compressed to ~3.8% of input on average (target was ~30%). The student learned that aggressive policy and it works — but "matched budget" here means matched to bear's ~30% cap via truncation, while ours typically lands far under it. We win on accuracy at a budget we rarely spend.

---

## 6. How it's cheap

The whole point of distillation is to move the quality from a frontier API into a small, ownable model.

- **Student:** Qwen2.5-1.5B-Instruct, **4-bit (Unsloth)**, **LoRA** (r=32 → ~37M trainable params, ~2.3% of the model). Runs on one GPU; inference is a single offline generation per prompt — **no API call, ~cents to run**.
- **Training:** one **H100** on **Modal**, ~**12–15 min** per run (5,000 examples × 3 epochs ≈ 213 steps at effective batch 64). 4-bit + LoRA means the optimizer state is tiny.
- **Total compute spend: ≈ $10** across all three distillation generations (the failed launches in §7 errored at startup and cost ≈ $0).
- **Data generation:** ~5,000 teacher pairs via the DeepSeek API, fanned out across 8 workers, resumable (each pair flushed to JSONL as it lands).

Compared to the API teacher (a per-call frontier-model charge on every compression, forever), the distilled student is a **fixed ~$10 one-time cost** then near-free at inference.

---

## 7. Engineering reality (reproducibility notes)

It took **7+ debugging iterations** to get the first clean H100 train. Documented because "it just worked" hides the real cost of the Unsloth + trl + Modal stack, and each is a concrete reproducibility lesson.

| # | Stage | Error | Fix |
|---|---|---|---|
| 0 | image build | `trl==0.11.4` / `peft==0.13.2` pins unsatisfiable with modern `unsloth` | drop the pins; add `unsloth_zoo`; let unsloth resolve a self-consistent stack |
| 1 | data load | `FileNotFoundError` — passed a local path to `train.remote()` | read the JSONL **locally** in the entrypoint, ship parsed rows as the arg |
| 2 | trainer init | `Unsloth: You must specify a formatting_func` | Unsloth's patched SFTTrainer doesn't auto-apply the chat template from a `messages` column |
| 3 | trainer probe | `dict object has no element 0` | Unsloth probes the func with a **single example**, not a batch |
| 4 | trainer probe | `formatting_func should return a list` | misleading error masking a brittle path — **abandon `formatting_func` entirely** |
| ✅ | — | **SUCCESS** | **pre-render the ChatML `text` column locally** (validated against the real Qwen template), ship plain text, train with `dataset_text_field="text"` and no `formatting_func` |
| 5 | inference | `ModuleNotFoundError: tiktoken` | `recompress.act1.tokens` imports tiktoken at runtime — add it to the infer image + `.add_local_python_source("recompress")` |
| 6 | eval (OOM) | 53 GB alloc during eval at the epoch boundary | `prediction_loss_only=True` (skip casting full logits `[bs, seq, vocab=151936]` to fp32) + `expandable_segments:True` |
| 7 | speed/arch | `packing=True` needs flash-attn (broken here); ~150 cold model loads in eval | revert to plain batching (eff. batch 64); restructure inference to a Modal **class** (`@enter` loads adapter once) + **batched** remote call |

**The robust pattern:** pre-render training text locally so the GPU job only tokenizes a plain `text` field. This decouples your data format from whatever Unsloth/trl version resolves on the build box — the chat-template path is version-brittle.

---

## 8. Honest limitations

We'd rather state clearly what we did **not** prove.

1. **bear has real, structural advantages we don't beat.** bear is **deletion**, so it's verbatim (no hallucination risk — it can only keep your real tokens), **query-agnostic** (compress a document *once* and reuse the result across many different questions; we must recompress per question), and **needs no training**. Our model can mis-read a question and drop the answer's evidence. For a compress-once-serve-many cache, or where verbatim fidelity is required, bear is the right tool.
2. **Latency is not a win for us.** At the API level the teacher and bear are essentially tied (≈1.08s vs ≈1.05s mean per call — both dominated by a network round-trip; see `results/latency_api.json`). bear's structural speed edge (deletion is not autoregressive generation, and it amortizes over reuse) is real but is **not** demonstrated as a wall-clock win in our measurements. We claim an **accuracy/token** win, not a speed win.
3. **Two of four benchmarks are not significant** (MuSiQue, SQuAD) — see §4.2. Positive everywhere, proven on two.
4. **Stacking fails, sometimes significantly** (§4.2) — ReCompress replaces bear in this regime; it does not layer on it.
5. **We did not beat published SOTA** (LLMLingua / LLMLingua-2 — multi-year Microsoft Research efforts). We beat **bear-1.1, the challenge sponsor's product**, which is the relevant comparison for this track. Our distinction vs LLMLingua is methodological: they **delete** tokens (extractive); we **rewrite** them (abstractive + query-aware).
6. **n=50 per benchmark, ratio=0.3 only.** A larger n and a ratio sweep would tighten the CIs and map where the win holds; out of scope for 24h.
7. **The student over-compresses** (~3.8% vs the ~30% target). It wins anyway, but the budget knob isn't well-calibrated yet.

---

## 9. Future work — the "bear-improver" model

The stacking failure (§4.2) points directly at the *original* thesis and the most interesting next step.

`ours→bear` fails because bear deletes our already-dense output and mangles it. But flip the objective: instead of training a student to *replace* bear, train a small model whose job is to **make bear's *output* better** — a query-aware post-processor that takes bear's deleted token-soup and repairs/re-densifies it for the question. That's a *different* training objective (input = bear output, target = a good compression of it), and it would be **additive and flattering** to bear rather than competitive — it extends bear into the query-aware regime instead of supplanting it. Not built yet; it's the clean next experiment.

Other directions: a ratio sweep + larger n to map the significance frontier; and calibrating the budget knob so the student hits a requested ratio. The **Act 2 multi-turn** application (`docs/PRD_act2.md`, implemented in `rezero/`) is already built and integrated — ReZero's checkpoint compression runs on our distilled model and beats both DeepSeek and bear as the backend (see `results/combined_benchmark.json`). A third experiment (Act 3, `ACTIII/`) is in progress.

---

## 10. Repo layout & how to reproduce

```
ReCompress/
├── recompress/                       # ACT 1 — query-aware compression + distillation (run from repo root)
│   ├── config.py                     # env-driven config (models, seeds, n, bootstrap iters)
│   ├── act1/                         # single-shot compression + the 5-bar benchmark
│   │   ├── compress.py               # compress_ours(): the query-aware teacher pass (+ distill prompt)
│   │   ├── bear.py                   # compress_bear(): blind deletion via TTC SDK (rate-limited)
│   │   ├── solve.py                  # solve(): the frozen DeepSeek judge
│   │   ├── metrics.py                # QA-F1 + paired bootstrap 95% CI
│   │   ├── data.py / benchmarks.py   # HotpotQA / 2Wiki / MuSiQue / SQuAD loaders (seeded)
│   │   ├── evaluate.py               # API-teacher 5-bar runner
│   │   ├── latency.py                # per-strategy latency benchmark
│   │   └── plot_results.py / plot_all.py / plot_distribution.py   # figures
│   └── distill/                      # the main submission path
│       ├── gen_data.py               # teacher data generation (parallel, resumable JSONL)
│       ├── train.py                  # Modal H100 app: Unsloth LoRA fine-tune Qwen2.5-1.5B
│       ├── infer.py                  # Modal class: load adapter once (@enter) + batched compress
│       └── evaluate_distilled.py     # 5-bar re-eval with the distilled model as "ours"
├── rezero/                           # ACT 2 — ReZero multi-turn memory (run FROM this dir)
│   ├── rezero/                       # session, trauma, checkpoint, echidna, context_builder
│   ├── engine/                       # compressor, deepseek, tokens, compressor_backend (Act1⇄Act2 seam)
│   ├── baselines/                    # naive (growing-context) baseline + TTC probes
│   ├── experiments/                  # combined_benchmark, token_trajectory, integration_smoke (modal run)
│   ├── tests/                        # 34 unit tests (pytest)
│   ├── demo/                         # panel/token-curve HTML + scripted_convo.jsonl
│   └── docs/                         # Act 2 STEP_* build notes
├── results/                          # all eval outputs (the numbers in this README)
│   ├── 5bar_results.json             # API teacher 5-bar (the +0.395 upper bound)        [LFS]
│   ├── 5bar_distilled_{hotpotqa,2wiki,musique,squad}.json   # distilled cross-benchmark  [LFS]
│   ├── combined_benchmark.json / token_trajectory.json      # Act 2 multi-turn results
│   ├── latency_api.json
│   └── figures/                      # all .png visuals (cross_benchmark, dots, multiturn …) [LFS]
├── ACTIII/                           # ACT 3 — third experiment (in progress; not yet restructured)
├── data/distill/                     # teacher training data (JSONL)                       [LFS]
├── artifacts/                        # distilled LoRA adapters (adapter, adapter_v3)        [LFS]
├── experiments/                      # archived v1/v3 runs: datasets, loss curves, Modal logs
├── logs/                             # raw Modal run logs
├── docs/                             # PRD_act1.md, PRD_act2.md, PRD_integration.md,
│                                     #   EXPERIMENT_LOG.md, FINDINGS_deletion_ceiling.md,
│                                     #   DEVPOST.md, PROGRESS.md
├── README.md  LICENSE  requirements.txt  .env.example
```

### Reproduce the pipeline

```bash
# 0. setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in DEEPSEEK / BEAR_API_KEY / Modal creds

# --- ACT 1 (run from repo root) ---

# 1. (optional) the API-teacher 5-bar floor — proves the +0.395 upper bound
python -m recompress.act1.evaluate                # -> results/5bar_results.json

# 2. generate teacher training data (parallel, resumable)
python -m recompress.distill.gen_data --n 5000 --out data/distill/train_v3.jsonl

# 3. LoRA fine-tune Qwen2.5-1.5B on a Modal H100 (~12-15 min; use --detach for long runs)
modal run --detach recompress/distill/train.py --data data/distill/train_v3.jsonl

# 4. the headline: re-run the 5-bar with the distilled model as "ours", all benchmarks
modal run recompress/distill/evaluate_distilled.py --benchmark all
#   -> results/5bar_distilled_{hotpotqa,2wiki,musique,squad}.json

# --- ACT 2 (ReZero multi-turn) — run FROM the rezero/ directory ---
cd rezero
python -m pytest tests/ -q                         # 34 unit tests (use_llm=False, no API key)
modal run experiments/combined_benchmark.py --n 30 # Act1⇄Act2: -> results/combined_benchmark.json
```

---

### TL;DR

bear-1.1 deletes tokens blind to your question. ReCompress reads the question, drops the distractors, rewrites the rest — then distills that into a **1.5B model that runs offline for ~cents** and still **beats bear at a matched token budget** (+56% F1 on HotpotQA, +46% on a dataset it never trained on, both with CIs excluding zero). It took three distillation attempts — a data-starved wash, a capacity-driven overfit, then a regularized, early-stopped win — and we report all three, including the two benchmarks where the edge is real but not yet significant, and the fact that stacking on bear doesn't work. **What we proved: a small open model can carry most of a frontier compressor's query-aware edge, offline, for ~$10.**
