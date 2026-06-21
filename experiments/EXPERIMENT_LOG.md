# ReCompress — Experiment Log (for the research writeup)

This log preserves the **full trajectory, including failures**. The point at which
things failed — and *why* — is part of the research contribution, not noise to hide.

---

## Headline numbers so far

| System | mean QA-F1 | Δ vs bear | 95% CI (paired bootstrap) | Beats bear? |
|---|---|---|---|---|
| Full context (`none`) | 0.877 | +0.425 | (+0.285, +0.571) | ✅ ceiling |
| **API teacher (DeepSeek, query-aware)** | **0.847** | **+0.395** | **(+0.259, +0.538)** | ✅ **decisive** |
| bear-1.1 (blind deletion) | 0.452 | — | — | baseline |
| **Distilled 1.5B (v1, LoRA)** | **0.515** | **+0.063** | **(−0.098, +0.220)** | ❌ **wash (CI incl. 0)** |

All @ ratio=0.3, 50 seeded HotpotQA-distractor instances, frozen DeepSeek solver, QA-F1.

---

## The central finding (v1): distillation lost most of the teacher's edge

The API teacher beats bear by **+0.395 F1** (decisive). The distilled 1.5B student
beats bear by only **+0.063 F1 with a CI that includes zero** — i.e. **on par with
bear, not better.** The student recovered ~16% of the teacher's margin over bear
(0.063 / 0.395). This is the core negative result to explain.

### Why the student fell short — three compounding causes
1. **Capacity gap.** Teacher = DeepSeek (large). Student = Qwen2.5-1.5B. A 1.5B model
   distilled from a frontier model on a hard multi-hop selection+rewrite task keeps
   only a fraction of the capability.
2. **Tiny dataset.** Only **261 train / 29 eval** examples (300 generated, 10 dropped
   as answer-leaks, 90/10 split). The eval loss **plateaued by epoch 2** (see below),
   indicating the model had extracted what little signal 261 examples carry — a
   data-size ceiling, not an optimization failure.
3. **Teacher data was pathologically over-compressed.** The teacher compressed to
   **~2.4% of input tokens** (target was ~30%), and **~3.3% of pairs (10/300) were
   the bare answer** (e.g. Q:"father of Jeanie Buss?" → `Jerry Buss`), violating the
   "do not answer" instruction. The student thus learned an over-aggressive policy
   that drops the very facts the solver needs. This is visible in the bar results:
   distilled `bear→ours` (0.317) is *worse* than bear alone (0.452) — compressing an
   already-deleted context destroys answer evidence.

### v1 training loss curve (overfitting check — clean)
```
epoch 1.00: train_loss=1.8171  eval_loss=1.7595  gap=-0.0576
epoch 2.00: train_loss=1.7437  eval_loss=1.7438  gap=+0.0001
epoch 3.00: train_loss=1.7355  eval_loss=1.7418  gap=+0.0063
OVERFITTING: no (eval_loss fell every epoch; train/eval gap ~0)
```
**Interpretation:** NOT overfitting — but the eval-loss curve is nearly flat after
epoch 1 (1.760 → 1.744 → 1.742). More epochs would not help; the ceiling is data
quantity + quality, not training duration. This motivated the v2 changes.

### v1 config
- Base: `Qwen/Qwen2.5-1.5B-Instruct`, 4-bit (Unsloth), LoRA r=16, α=32, dropout=0.05
- target_modules: q,k,v,o,gate,up,down proj
- 3 epochs, lr=2e-4, cosine, warmup 0.05, batch 2 × grad-accum 8 (eff. 16), adamw_8bit
- Teacher prompt: original `_SYSTEM` ("MINIMAL", "Target ~30%, never more") — the
  "MINIMAL" framing + a cap-only token target produced the 2.4% collapse.

### v1 artifacts (archived under `experiments/v1/`)
- `data/train_v1_raw_300.jsonl` — 300 teacher pairs as generated
- `data/train_v1_clean_290.jsonl` — after dropping 10 answer-leak pairs (the trained set)
- `eval/5bar_distilled_v1.json` — the wash result above
- `eval/5bar_api_baseline.json` — the API teacher PASS (+0.395)
- `logs/modal_train_v1.log`, `logs/modal_eval_v1.log` — full Modal stdout
- `loss_curve_v1.txt` — the loss curve

---

## Engineering failure trajectory (Modal / Unsloth) — reproducibility notes

Five consecutive H100 launches failed before the first successful train, each on a
distinct, real integration issue. Documented because "it just worked" hides the
actual cost of the unsloth+trl+Modal stack, and each is a concrete reproducibility
lesson. (Each failure cost only seconds of H100 — they errored at/near startup — so
dashboard cost stayed ≈ $0 until the first real run.)

| # | Stage | Error | Root cause | Fix |
|---|---|---|---|---|
| 0 | image build | `ResolutionImpossible` (would-be) | `trl==0.11.4` + `peft==0.13.2` pins conflict with modern `unsloth` (needs trl≥0.18.2, peft≥0.18) | drop pins; add `unsloth_zoo`; let unsloth resolve the stack |
| 1 | train data load | `FileNotFoundError: data/distill/train_clean.jsonl` | passed a **local path** to `train.remote()`; the container can't see local files | read JSONL locally in the entrypoint, ship parsed rows as the arg |
| 2 | trainer init | `Unsloth: You must specify a formatting_func` | Unsloth's patched SFTTrainer does NOT auto-apply chat template from a `messages` column (vanilla trl does) | add a `formatting_func` |
| 3 | trainer probe | `UndefinedError: dict object has no element 0` | Unsloth probes the func with a **single example** (`next(iter(dataset))`), not a batch; func indexed it as a batch | handle single-example shape |
| 4 | trainer probe | `ValueError: formatting_func should return a list of processed strings` | Unsloth requires the func to return a **list** even for the single-example probe; misleading error masked a brittle path | **abandon `formatting_func` entirely** |
| ✅ | — | SUCCESS | — | **pre-render the ChatML `text` column LOCALLY** (validated against the real Qwen template), ship plain text, train with `dataset_text_field="text"` and no `formatting_func` |

Secondary, found during the distilled **eval** (separate `-infer` app):
| # | Stage | Error | Fix |
|---|---|---|---|
| 5 | inference | `ModuleNotFoundError: No module named 'tiktoken'` | `src.act1.tokens` imports tiktoken at runtime; add `tiktoken` to the infer image |
| (arch) | inference | would-be ~150 cold model loads (one per instance per bar) | restructure to a Modal **class** (`@enter` loads adapter once) + **batched** `compress_batch` remote call |

**Lesson for the paper's reproducibility section:** the unsloth+trl chat-template
path is version-brittle. Pre-rendering the training text locally (so the GPU job only
tokenizes a plain `text` field) is the robust pattern — it decouples the data format
from whatever Unsloth/trl version resolves on the build box.

---

## v2 plan (in progress) — "do both 1 and 3": more+better data AND stronger hyperparams

Changes relative to v1, targeting the three failure causes:
1. **Better teacher prompt** (`_SYSTEM_DISTILL` in `src/act1/compress.py`): adds a
   FLOOR ("≥2-3 sentences, never <~25 tokens"), a hard anti-answer rule ("output
   CONTEXT not the answer"), and "when in doubt KEEP a fact". Aims to fix the 2.4%
   collapse + answer-leak. The original API-eval prompt is left unchanged (it PASSES).
2. **More data:** regenerate ~1000 pairs (vs 300), with an inline `<8 token` skip so
   leaks never enter the set.
3. **Stronger training:** LoRA r=16→64, epochs 3→(4-5). (Loss plateaued in v1, so the
   epoch bump is secondary; the data quantity/quality + capacity via higher rank are
   the real levers.)

Expected outcome is uncertain — documenting either way. If v2 still doesn't beat bear,
that itself is a finding: **a 1.5B may be under-capacity for query-aware multi-hop
compression**, and the honest product story is "distilled model runs offline/cheap at
~bear parity, while the API teacher is the quality ceiling."

---

## v2 result (r=64, 6 epochs, 2500 pairs) — OVERFIT. Run died at 91%, but the loss curve is the finding.

The v2 training run was killed by a **local network drop at step 770/846 (epoch ~5.5)** —
a reminder that `modal run` (foreground) dies with the local connection; the fix is
`modal run --detach` (server-side, survives drops). BUT the run logged 5 epochs of
eval before dying, and the curve is unambiguous:

| Epoch | train_loss (approx) | eval_loss | verdict |
|---|---|---|---|
| 1 | ~1.5 | 1.662 | |
| 2 | ~1.3 | **1.654** | ✅ best eval (sweet spot) |
| 3 | ~1.1 | 1.685 | ⚠️ eval rising |
| 4 | ~1.0 | 1.762 | 🔴 overfitting |
| 5 | ~0.95 | 1.861 | 🔴 badly overfit |

**Finding: r=64 + 6 epochs OVERFITS the 2500-example set.** Classic divergence —
train_loss fell monotonically (1.5→0.95) while eval_loss bottomed at **epoch 2** then
**rose every subsequent epoch**. The bigger adapter (r=64 = 73.9M trainable params,
4.57% of the model) had enough capacity to memorize. Contrast v1 (r=16, 261 ex) which
did NOT overfit at 3 epochs — so the overfitting is driven by the **rank increase**,
not just epochs.

**Lessons applied to v3:**
1. **Rank too high.** r=64 → r=32 (halve trainable params).
2. **Too many epochs.** 6 → 3 (eval bottomed ~epoch 2).
3. **No early-stopping.** Added `load_best_model_at_end` (commit best-eval ckpt, not last).
4. **More data + regularization** (the standard anti-overfit levers): 2500 → 5000 pairs,
   lora_dropout 0.05 → 0.1, weight_decay 0.01.

### Engineering notes from v2 attempts
- **`modal run` dies on local network drop.** Use `--detach` for long jobs — the H100
  job then runs fully server-side and survives the laptop disconnecting/sleeping.
- **`packing=True` is NOT viable on this image.** It requires `flash_attention_2`
  (xformers is broken in the resolved stack → PyTorch-attn fallback). Without it,
  packing flattened examples into a 61,405-token sequence that exceeded max_length
  (4096) and mismatched the labels → `cross_entropy` shape error. Reverted to plain
  batching (batch 32 → 64 via per_device_batch + grad_accum) for the speedup instead.
- **Speed levers that DO work (CUDA/H100, no backend change):** bigger batch (fewer
  steps), fewer epochs, shorter max_length (limited here — contexts are ~1600 median,
  so can't cap below ~3072 without truncating the relevant tail). A different *backend*
  (TPU/ROCm/MLX) or *smaller GPU* would be SLOWER for a 1.5B LoRA — the job is
  step-count-bound, not raw-FLOP-bound, and H100+CUDA is already the fastest path.

## v3 plan (running) — full anti-overfit + speed
- Data: **5000 pairs** (v2's 2500 + 2500 new, via `--skip 2500`)
- LoRA: **r=32** (was 64), **dropout=0.1** (was 0.05), weight_decay=0.01
- **3 epochs** (was 6), **load_best_model_at_end** on eval_loss
- Batch **eff. 64** (per_device 32 × accum 2) → ~211 steps → ~12-15 min on H100, detached

(v3 numbers appended once the run completes.)

---

## Bear-improver experiment — DROPPED (dominated by standalone). A useful negative result.

**Original thesis (Act-1 v0):** don't replace bear — train an SLM *adapter* that rewrites
text so bear's *blind deletion* yields a BETTER final answer. I.e. optimize `bear(SLM(text))`,
not `SLM(text)` alone. Code: `src/distill/gen_data_bearimprover.py` (deletion-robust teacher
prompt + a survival filter: run bear on each rewrite, keep only rewrites whose answer survives
bear with post-bear F1 ≥ 0.5). Smoke test (6 instances) worked: **83% keep-rate, avg post-bear
F1 of survivors = 0.933** — DeepSeek *can* write bear-survivable text.

**Why we dropped it before spending the full training compute — the dominance argument:**
A pre-processor in front of bear is *strictly more expensive and slower* than bear alone
(it adds SLM generation). So it can only justify itself on **final quality**. But our own
5-bar numbers show that if you have the SLM at all, using it **standalone** beats routing it
through bear, on every benchmark:

| Benchmark | bear alone | standalone ours (v3) | ours→bear |
|---|---|---|---|
| HotpotQA | 0.452 | **0.704** | 0.464 |
| 2Wiki | 0.390 | **0.570** | 0.407 |
| MuSiQue | 0.186 | **0.297** | 0.213 |
| SQuAD v2 | 0.471 | **0.593** | 0.295 |

`standalone ours` ≫ `ours→bear` everywhere. Bear's deletion can only *remove* information
the SLM deliberately placed — so the best the bear-improver could do is climb toward, but
not exceed, the standalone number, while costing an extra bear pass. It is dominated on all
three axes (latency, cost, quality) unless you are **contractually forced to keep bear** in
the pipeline.

**The finding (and why it strengthens the main result):** *You cannot fix blind deletion by
pre-processing. Query-aware rewriting must REPLACE deletion, not augment it.* This is exactly
why the standalone query-aware model is the right architecture — and why bear's deletion-only
design has a ceiling that no front-end can lift. The bear-improver script is retained as the
evidence for this claim; it was not trained at scale (no compute spent beyond the smoke test).

---

## v4 — Answer-grounded distillation (best-of-N). NEGATIVE result: did NOT beat v3.

**Idea:** instead of distilling the teacher's text (v3), distill compressions selected by
*downstream answer success* — generate 4 candidates per example (1 greedy + 3 at temp 0.7),
run the frozen solver on each, keep the one with highest answer-F1 (best-of-4). Data: 4363
survivors (87.3% keep-rate), avg answer-F1 of kept = 0.934. Trained with the SAME config as
v3 (r=32, 3 epochs, batch 64, dropout 0.1) so the only variable is the data/objective.

**Result — v4 lost to v3 on both significant benchmarks:**

| Benchmark | bear | v3 (teacher-imitate) | v4 (answer-grounded best-of-4) | v4 − v3 |
|---|---|---|---|---|
| HotpotQA | 0.452 | **0.704** (sig) | 0.659 (sig) | −0.045 |
| 2Wiki | 0.390 | **0.570** (sig) | 0.520 (**n.s.**) | −0.050 |

v4 still beats bear, but is ~0.05 F1 *worse* than v3 and loses significance on 2Wiki.
Per the "merge only if it wins" rule, **v4 was not merged; v3 remains the model.**

**Why (hypotheses):**
1. **Judge-selection bias (most likely).** Best-of-N picks the candidate the *frozen solver*
   scores highest — which selects for that solver's quirks, not generalizable compression
   quality. The student then distills judge-pleasing artifacts. A mild reward-hacking effect.
2. **Less + easier data.** The survival filter dropped ~13% of examples — disproportionately
   the *hard* ones the model most needs to learn from.
3. **Greedy teacher was already strong.** Temp-0.7 diversity added more noise than signal;
   best-of-N had little headroom to improve the target, only room to add variance.

**The finding itself:** *answer-grounded best-of-N selection against a frozen judge does not
beat plain teacher-imitation for compression distillation* — a non-obvious result worth
recording (most teams wouldn't test it). Motivates v5: keep answer-grounding's data-quality
benefit but REMOVE the best-of-N judge-selection (greedy-only + drop teacher-failures).

---

## v5 — Answer-grounded, GREEDY-only + answerable-filter (no best-of-N). NEGATIVE, decisive.

v4's loss might have been best-of-N judge-selection bias. v5 tests that: greedy compression
only (temp 0, ONE candidate), but still drop examples where the teacher's greedy output fails
the solver (keep answerable-only). Removes best-of-N entirely. Data: 4175 pairs, avg answer-F1
0.924. Same v3 config.

**Result — v5 reproduces v4's loss; it was NOT a best-of-N artifact:**

| Benchmark | bear | v3 (imitate) | v4 (best-of-N) | v5 (greedy+filter) |
|---|---|---|---|---|
| HotpotQA (in-dist) | 0.452 | **0.704** sig | 0.659 sig | 0.664 sig |
| 2Wiki (OOD) mean | 0.390 | **0.570** sig | 0.520 **n.s.** | 0.512 **n.s.** |
| 2Wiki (OOD) trimmed-mid33% | 0.196 | **0.694** | 0.556 | 0.534 |

**Decisive finding:** *answer-grounded distillation (filtering training data by frozen-solver
success) does NOT beat plain teacher-imitation — on either selection scheme (best-of-N v4 or
greedy v5).* Worse, on out-of-distribution 2Wiki BOTH answer-grounded variants drop to
non-significant, while v3 stays significant. The gap widens on the trimmed (typical-case) mean.

**Why (best current explanation):** the answerable-only filter (a) removes hard examples the
model needs, and (b) selects compressions tuned to the *frozen solver's in-distribution
behavior*, which overfits the judge and costs out-of-distribution generalization. Teacher-
imitation trains on the teacher's full output distribution (including imperfect compressions),
which generalizes better.

**Methodological note:** on in-distribution HotpotQA the gap looked small (−0.04); only the
OOD 2Wiki eval exposed the real failure (loss of significance). In-distribution flattered the
answer-grounded models — a reminder to judge generalization on OOD data, not the training source.

**Decision:** v3 remains the submission model. v4+v5 are a documented negative result.
The novelty contribution is #3 (the deletion ceiling), not answer-grounding.
