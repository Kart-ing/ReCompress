# ReCompress — Research Paper Source Material

> **Purpose:** the single compiled source of truth for writing the research paper. Every number,
> method, run, finding, and citation, pulled from the code / logs / results / docs / Modal runs.
> Compiled by fanning out reader agents over the whole repo. *This is raw material, not prose —
> the paper is written FROM this.*
>
> Project: **ReCompress** — query-aware context compression distilled to 1.5B (Act 1) + Re:Zero
> flat-context multi-turn memory (Act 2). Built for the Token Company Compression Challenge,
> UC Berkeley AI Hackathon 2026.
>
> **Status:** ✅ **COMPILED** from 4 reader agents (numbers / runs / docs / code), all stitched. Cross-flags reconciled in §10. Ready to write the paper from.

---

## 0. Abstract draft (refine for the paper)

> **NOTE:** the canonical, peer-reviewed-and-revised version is `templateArxiv.tex` (the
> arXiv paper). This file is the raw compilation source; the abstract below is kept in sync
> with the final claims (no "matched budget" — it's "8.5× fewer tokens"; generalization is
> scoped; multi-turn overhead is reported).

Context compression for LLMs is dominated by *extractive* methods that delete or select tokens (e.g. The Token Company's bear-2, LLMLingua). We argue and show that extraction has a **structural ceiling**: when an answer depends on facts spread across many individually-low-salience tokens, no in-budget subsequence preserves it, but a paraphrase does — so *abstractive, query-aware rewriting* is necessary, not merely better. We distill a query-aware compressor (DeepSeek teacher → **Qwen2.5-1.5B + LoRA**) and benchmark it head-to-head against bear-2 under the **same compression instruction** with a paired 5-bar design and bootstrap 95% CIs across four QA datasets. Both receive a ratio-0.3 instruction but realize very different token counts — bear keeps ~409 tokens, our rewriter ~48 (3.5%) — so the result is the stronger claim that ours is **more accurate while emitting ~8.5× fewer tokens**: **+0.252 F1 on HotpotQA (+56%, CI excludes 0)** and **+0.180 on 2WikiMultihopQA (+46%, near-in-distribution)**, with MuSiQue/SQuAD positive-but-not-significant at n=50 (we make the narrower claim). An independent-solver re-score (Claude Sonnet) confirms the gap is not a teacher↔solver artifact (+0.288), and an answer-masking audit shows much of the margin is *span selection* (retaining the answer-bearing span at a 3.5% budget where deletion truncates it), reported honestly. We then apply the same compressor as the memory engine of a multi-turn agent (Re:Zero), keeping the **context sent to the solver flat (~184 tokens) where a naive agent grows to 1,482 over 12 turns (8.1×)** at no measurable accuracy loss. We report the full trajectory including negative results: stacking with deletion is a wash-to-worse, answer-grounded distillation underperforms teacher-imitation, on abstractive free-form QA (MS MARCO) the method ties bear, and — counting per-turn compression overhead — the multi-turn system is not cheaper in total tokens until the per-turn LLM trigger (which was ~98% "checkpoint") is replaced with a free rule, after which it beats an uncached growing-history agent from ~6 turns (never a cached one). Total compute: ~$10.

**Contributions:** (1) the deletion-ceiling argument (reachability + constructed example + empirical confirmation); (2) a rigorous matched-budget paired-CI head-to-head vs a deletion *product* on its own model; (3) a 1.5B distilled query-aware compressor that beats it and generalizes cross-dataset; (4) the same compressor extended to flat-context multi-turn memory; (5) documented negative results (stacking, answer-grounding, abstractive-QA scope).

---

## 1. Thesis & research questions
_Source: docs agent._

**Unifying problem (README):** *"sending fewer tokens to an LLM without losing the answer."* One project, two acts, same problem from two angles.

**Act 1 thesis (the deletion-ceiling reframe — elevated to THE novelty).** bear-2 deletes tokens char-for-char, query-blind, can't rewrite. ReCompress adds "the one thing deletion provably can't: a single question-conditioned SLM pass that (a) drops passages irrelevant to *this* question and (b) densifies verbose-but-relevant prose." Deepest claim (FINDINGS): *"Compression methods that only delete or select tokens (extractive: bear-2, LLMLingua, LongLLMLingua, Selective-Context) have a hard ceiling on a class of queries. No amount of cleverness in which tokens to keep can cross it. Compression that rewrites (abstractive) is not bound by this ceiling — a structural property of the hypothesis class, not a quality gap between models."*

**Act 2 thesis:** same compressor, applied across a conversation — keep per-turn cost flat instead of O(n²) by sending checkpoint + trauma + delta instead of full history.

**Research questions (README two-act table):** Act 1 — *"Can a small model compress one prompt better than blind deletion?"* Act 2 — *"Can it keep a long conversation from growing O(n²)?"*

**Cross-act claim:** *"query-aware rewriting beats deletion, and a 1.5B can carry it — for single prompts and for whole conversations."*

**⚠️ Scope boundary (the honest limit, EXPERIMENT_LOG):** *"ReCompress's win is specific to multi-hop, distractor-heavy, entity-answer QA — exactly the regime bear's blind deletion cedes. It is NOT a universal compression win; on abstractive free-form QA it matches bear."*

## 2. Related work & positioning
_Source: docs agent._

**Methods named + differentiation:**
- **LLMLingua / LongLLMLingua / LLMLingua-2** (MSR). LongLLMLingua is *already query-aware*; LLMLingua-2 *already distills from a frontier model into a small classifier* — named as **the closest prior work**. **Our differentiation: they DELETE/classify tokens (extractive); we REWRITE (abstractive) + distill into a generative 1.5B.**
- **RECOMP** (abstractive compression). "Prior art finds extractive often ≥ abstractive — so we test, not assume."
- **Selective-Context** — extractive family.
- **Stacking is already published** (preempts the "novelty" claim): Jha et al. 2024 (arXiv:2407.08892, reranker→LongLLMLingua on HotpotQA); CompactPrompt (stacks compressors, ratios compound). → "stacking is incremental, not novel" — *this is why stacking was demoted from the headline.*
- **MemGPT / ConversationSummaryBufferMemory** — Act 2 prior art; Re:Zero is "a summary + protected-facts buffer (like MemGPT)," novelty demoted to one micro-claim (§3.5).
- **Prompt caching / compress-once-serve-many** — the axis where bear's reuse advantage is conceded.

**The differentiation, lift-able:**
1. **Extractive vs abstractive** is the core axis ("they delete tokens; we rewrite them").
2. **The structural-ceiling claim differentiates vs ALL prior art:** *"none [of LLMLingua/LongLLMLingua/LLMLingua-2/Selective-Context/RECOMP] characterizes a structural ceiling of the extractive family and shows abstractive compression is needed to cross it, backed by (a) a reachability argument, (b) a constructed existence example, (c) a 4-benchmark ours→bear-never-helps confirmation."*
3. **The white space:** *"a clean, paired, head-to-head against a deletion-only compressor on its own model, isolating query-awareness as the lever, in the distractor regime"* — prior work compares vs LLMLingua-2/truncation, not vs a char-for-char deletion *product*.
4. **Distill-to-1.5B** = the practical contribution.
5. **Framing vs sponsor:** "additive and flattering — we extend bear into the regime they say they don't touch ('nothing is paraphrased or generated')."

## 3. Methods — as implemented
_Source: code agent (verbatim from source). Three acts: `recompress/` (Act1), `rezero/` (Act2), `ACTIII/` (Act3)._

**Global config** (`recompress/config.py`): `compressor_model = solver_model = "deepseek-chat"` (the "V4 Pro/Flash" labels are aspirational comments — both are literally deepseek-chat); `seed=42`, `n_instances=50`, `bootstrap_iters=1000`. Client: `OpenAI(timeout=120, max_retries=5)`. Tokenizer: `tiktoken cl100k_base` (stand-in for DeepSeek's; Act 3 uses a separate `words*1.3` heuristic).

### 3.1 The 5-bar paired matched-budget benchmark (`recompress/act1/evaluate.py`)
Bars (all fed the SAME frozen solver): `none` (full ctx ceiling) | `bear` | `ours` | `bear→ours` | `ours→bear`, all truncated to matched `ratio` (0.3). `bear→ours = compress_ours(compress_bear(text))`; `ours→bear = compress_bear(compress_ours(text))`. ThreadPoolExecutor(6), partial saves, per-instance errors→f1=0. Decision rule: PASS = ours delta CI excludes 0 AND mean>0.

### 3.2 Metric — QA-F1 + paired bootstrap CI (`recompress/act1/metrics.py`) — EXACT
**Normalize:** lowercase → strip `string.punctuation` → remove `\b(a|an|the)\b` → whitespace split.
**qa_f1:** SQuAD-style token-overlap; both-empty→1.0; multiset `Counter(p)&Counter(g)`; `prec=n_common/len(p)`, `rec=n_common/len(g)`, `F1=2·p·r/(p+r)`.
**bootstrap_ci(deltas, n_iters=1000, seed=42):** `rng=np.random.default_rng(42)`; 1000× resample-with-replacement (same n), record mean; `lo,hi = percentile([2.5, 97.5])`. Deltas = per-instance paired `(bar−bear)`, errored dropped. `excludes_zero = lo>0 or hi<0`.

### 3.3 Datasets / seeding (`recompress/act1/data.py`, `benchmarks.py`)
Pattern: stream `pool_size` via HF `streaming=True` → `random.Random(42).shuffle` → take n. Schema `{id, question, answer, context:[{title,text}]}`; `context_to_text` = `"[{title}]\n{text}"` joined `\n\n`. `load_hotpotqa(n)` is **prefix-stable** (exploited by gen_data `--skip` to extend datasets).
| BM | HF dataset / config / split | notes |
|---|---|---|
| hotpotqa | `hotpotqa/hotpot_qa` distractor / validation | pool max(n·20,1000) |
| 2wiki | `scholarly-shadows-syndicate/2wikimultihopqa_with_q_gpt35` / validation | |
| musique | `dgslibisey/MuSiQue` / validation | paragraphs→{title,paragraph_text} |
| squad | `rajpurkar/squad_v2` / validation | skips unanswerable, single passage |
| msmarco | `microsoft/ms_marco` v2.1 / validation | skips "No Answer Present.", 10 passages, continuous-F1 |

### 3.4 Distillation setup + EXACT hyperparameters (`recompress/distill/train.py`)
Base: `Qwen/Qwen2.5-1.5B-Instruct`, 4-bit, Unsloth, H100. Adapter base = `unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit`.
**LoRA (Hearth/v3, confirmed vs saved adapter_config.json):** `r=32, target_modules=[q,k,v,o,gate,up,down]_proj, lora_alpha=64, lora_dropout=0.1, bias="none", use_gradient_checkpointing="unsloth", random_state=42`.
**SFTConfig:** `num_train_epochs=3, per_device_train_batch_size=32, gradient_accumulation_steps=2 (eff. 64), learning_rate=2e-4, max_length=3072(entrypoint)/4096(fn-default), weight_decay=0.01, dataset_text_field="text", logging_steps=5, save_strategy=eval_strategy="epoch", per_device_eval_batch_size=4, prediction_loss_only=True, eval_accumulation_steps=1, load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False, warmup_ratio=0.05, lr_scheduler_type="cosine", optim="adamw_8bit", seed=42`. NO packing (flash-attn broken). **Pre-rendered text approach:** entrypoint reads JSONL, applies real Qwen chat template locally (`apply_chat_template(..., tokenize=False)`), ships `text_rendered` rows → container never runs the template (sidesteps Unsloth formatting_func probe). 90/10 split, `processing_class=tokenizer`.
⚠️ `train()` fn-defaults (`lora_r=16, max_seq_len=4096`) differ from CLI entrypoint (`lora_r=32, max_seq_len=3072`); shipped v3 used r=32 (saved adapter confirms).

### 3.5 Prompts — VERBATIM
**Teacher `_SYSTEM` (`compress.py`, API-eval):** "You are a context compressor… produce the MINIMAL compressed context… 1.DROP distractors 2.DENSIFY/paraphrase 3.Preserve entities/numbers/relations/multi-hop 4.Do NOT answer 5.Target ~{ratio}%, never more."
**Distill `_SYSTEM_DISTILL` (distill=True, fixes v1 collapse):** adds FLOOR ("≥2-3 sentences, never <~25 tokens") + hard anti-answer ("Do NOT output a bare entity… Output CONTEXT… not an answer key") + "when in doubt KEEP a fact."
**Training SFT prompt (`train.py`):** = `_SYSTEM` minus rule 5. User="QUESTION:\n{q}\n\nCONTEXT:\n{text}", assistant=teacher's compressed.
**Solver `_SYSTEM` (`solve.py`):** "You are a strict QA agent. Answer using ONLY the provided context. Be concise: minimal span, no reasoning, no preamble." max_tokens=128, temp=0.
**Answer-grounded `_SYSTEM` (`gen_data_answergrounded.py`):** same instruction (novelty=selection not prompt).
**Bear-improver `_SYSTEM_BEAR`:** deletion-robust ("state facts PLAINLY and EARLY, verbatim entities… build in REDUNDANCY… name BOTH ends of each hop").

### 3.5b Answer-grounded selection rule
Best-of-N (n=4: temps `[0.0]+[0.7]*3`); per candidate generate→truncate→solve→qa_f1; sort desc, keep top; write if `best_f1≥0.5 AND n_out≥8`. v5/Oracle-Lite = greedy-only (1 cand). Plain distill filter: keep if `n_out≥8` (drops answer-leaks). Bear-improver filter: keep if `qa_f1(solve(bear(rewrite),q),gold)≥0.5`.

### 3.6 Inference (`recompress/distill/infer.py`)
Modal class `Compressor` (`@app.cls gpu=H100 scaledown_window=300`): `@enter` loads adapter ONCE (`from_pretrained("/vol/adapter", max_seq_length=4096, load_in_4bit=True)` + `for_inference`). Generation: `max_new_tokens=512, do_sample=False` (greedy, NO temperature), decode new tokens, truncate to `ratio`. `compress_batch` / `compress_batch_timed` methods.
**Bear (`bear.py`):** `TheTokenCompany().compress(text)`, query-ignored, truncated to target. Token-bucket limiter `_RPM=55` (under TTC 60/min), retry max 6.

### 3.6b Act 2 — Re:Zero memory architecture (`rezero/`)
**Caps:** `TRAUMA_CAP=50, CHECKPOINT_CAP=150, DELTA_CAP=100, TOTAL_CAP=300`; `CHECKPOINT_COOLDOWN=2, MIN_HISTORY_FOR_CHECKPOINT=400`; trauma `PINNED_CAP=15/BUFFER_CAP=30`; checkpoint `MAX_STACK_SIZE=10`; echidna `TOKEN_THRESHOLD=800, TURN_CADENCE=5`.
**Per-turn assembly** (`ContextBuilder`): `[TRAUMA]\n{}\n\n[CHECKPOINT]\n{}\n\n[DELTA]\n{}`; if total>300, shave checkpoint 5 words at a time (trauma+delta protected).
- **Trauma:** pinned=goal(≤15) + buffer=accumulating facts(≤30); LLM returns JSON `{update,buffer}`, never duplicates.
- **Checkpoint:** on decision, compress full history via backend at `effective_ratio=min(ratio, 150/history_tokens)` (default ratio 0.20), `exclude=trauma`, hard-cap 150. Stack: push/current/revert_to(id)/FIFO@10.
- **Delta:** last raw user msg, cap 100.
- **Echidna policy:** `checkpoint|revert|pass`; mock: checkpoint if history>800 OR turns_since≥5; LLM ("Witch of Greed") returns `{action,revert_to,reason,urgency}`. Session overrides checkpoint→pass if cooldown<2 or history<400.
**Backend selector (`compressor_backend.py`):** `deepseek` (API rewrite, temp0) | `naive` (word-truncate) | `distilled` (lazy-import Act-1 Modal Compressor, 1 remote call/checkpoint) | `bear` (TTC, query-blind). Naive baseline grows O(n): `[GOAL]\n{}\n\n[HISTORY]\n{full}`.

### 3.6c Multi-turn benchmark construction (`rezero/experiments/`)
**combined_benchmark.py:** same seeded HotpotQA window; each supporting doc = 1 turn (`add_turn("Document {i}: {d}", "Noted.")`); after turns, `solve(prompt_for_solver(), q)`. Compares naive vs rezero_{deepseek,distilled,bear}. Metric = **rougeL F1** (`rouge_score`, stemmer) + final-prompt tokens.
**token_trajectory.py:** ONE long convo chaining docs across instances; records `token_count()` at each turn → the flat-vs-growing curve.

### 3.7 Act 3 — the deletion-vs-random control study (`ACTIII/`, reconstructed from code — NO docs exist)
**What it is:** a control study testing whether bear/TTC's "intelligent" extractive deletion beats **RANDOM word-dropping** at matched budget — the empirical companion to the deletion-ceiling argument.
**`coin_toss` baseline (`baselines/coin_toss.py`):** the null control — keep each word independently with prob `(1−removal_rate)` (`random.Random(seed)`, floor `words[:5]`), then solve. Query-blind, structure-blind random deletion — the dumbest extractive compressor.
**Runner (`experiments/runner.py`):** builds the same 6-turn HotpotQA convo; compares at matched budget: Naive vs **Token Company** at aggressiveness {0.1,0.3,0.5,0.7,0.9} (`with_compression` SDK wrapper) vs **Coin Toss** at removal {0.1…0.7}. Metric = rougeL F1. **Decisive output:** for each TC aggressiveness, find the closest-F1 random-removal rate → *"TC aggr=X (F1 …) ≈ Coin Toss r=Y"*. **Hypothesis:** if bear's blind deletion ≈ random word-dropping at matched budget, extractive compression's value is illusory on this QA regime → motivates query-aware rewriting (Acts 1-2) as the family that breaks the ceiling.
⚠️ Act 3 infra differs: `count_tokens = int(len(text.split())*1.3)` (word heuristic, not tiktoken); uses UNshuffled dataset prefix (`list(dataset)[:n]`), unlike Act1/2's seeded shuffle. Runner defaults `--n 20 --workers 10 --seed 42`. **Status: code + tests exist, no results JSON found yet (not run, or results not saved).**

## 4. Experiment trajectory (the named runs)
_Source: docs agent (EXPERIMENT_LOG is source-of-truth). Runs agent will add per-run loss curves / durations / Modal IDs._

| Codename | aka | config | outcome |
|---|---|---|---|
| **Spark** | v1 | 261 train/29 eval, r=16/α=32, dropout 0.05, 3 ep, lr 2e-4 | ❌ **wash** +0.063 CI(−0.098,+0.220); recovered ~16% of teacher margin |
| **Bonfire** | v2 | 2,500 pairs, r=64/α=128 (73.9M, 4.57%), 6 ep, dropout 0.05 | 🔴 **overfit** — eval bottomed epoch 2 (1.654) then rose to 1.861@ep5; driven by *rank* not epochs; also died at step 770/846 (network drop → use --detach) |
| **Hearth** ⭐ | v3 | **5,000 pairs, r=32/α=64, 3 ep, dropout 0.1, weight_decay 0.01, load_best_model_at_end, eff. batch 64 (~213 steps)** | ✅ **SHIPPED** — HotpotQA +0.252 CI(0.103,0.396); 2Wiki +0.180 CI(0.030,0.340); eval 1.6787→1.6655→1.664, not overfit |
| **Oracle** | v4 | answer-grounded best-of-4 (1 greedy+3 temp0.7), keep max answer-F1; 4363 survivors (87.3%); else same as v3 | ❌ **lost** — HotpotQA 0.659 vs 0.704; **2Wiki 0.520 → n.s.**; judge-selection bias |
| **Oracle-Lite** | v5 | answer-grounded greedy (temp0, 1 cand) + answerable filter; 4175 pairs; same v3 config | ❌ **lost, decisively** — reproduces v4 (not a best-of-N artifact); 2Wiki trimmed gap *widens* (v3 0.694 vs v5 0.534) |
| **Bear-Booster** | — | train so `bear(SLM(x))` improves; deletion-robust prompt + post-bear-F1≥0.5 filter; smoke: 83% keep, 0.933 post-bear F1 | ❌ **dropped (dominated)** — standalone ours ≫ ours→bear on all 4; dominated on latency+cost+quality |

**Narrative rationale per run** (the spine):
- **Spark wash → 3 compounding causes:** capacity gap (1.5B from frontier), tiny data (eval loss flat after epoch 1 = data-size ceiling), pathologically over-compressed teacher data (~2.4%, 10/300 pairs were the bare answer). → motivated more+better data.
- **Bonfire overfit → "do both 1 and 3" pushed 3 levers at once;** the rank increase (r=64) was the overfit driver. → motivated the regularized recipe.
- **Hearth = full anti-overfit playbook** (more data + lower rank + dropout + weight_decay + early-stop). The shipped model. Lesson: "a 1.5B can learn query-aware multi-hop compression, but only with a regularized, early-stopped, adequately-data'd recipe."
- **Oracle/Oracle-Lite:** tested whether distilling on *downstream-answer-success* (not teacher text) beats imitation. It doesn't, and hurts OOD generalization — judge-selection overfits the frozen solver. **The OOD 2Wiki eval exposed the failure that in-distribution HotpotQA hid** (−0.04 looked small; 2Wiki dropped to n.s.).
- **Bear-Booster:** the *original* thesis (improve bear, don't replace it). Dropped via the dominance argument → became the empirical anchor for the deletion ceiling: "you cannot fix blind deletion by pre-processing."

## 5. Results — all numbers
_Source: numbers agent (read all 15 five-bar JSONs + combined + trajectory + latency + per-instance arrays). All 5-bar runs: ratio=0.30, n=50, n_err=0 everywhere. Significance = paired bootstrap 95% CI of (bar−bear) excludes 0._

### 5.1 API teacher upper bound — HotpotQA (`results/5bar_results.json`)
Decision: **headline PASS**, stacking_bonus=false. Only file with `n_tokens_in` (avg 1363.64).

| Bar | mean_f1 | median | trim33% | n=0 | n=1 | avg_tok_out |
|---|---|---|---|---|---|---|
| none | 0.877397 | 1.0 | 1.0 | 2 | 37 | 1363.64 |
| bear | 0.452143 | 0.571 | 0.367 | 24 | 17 | 408.66 |
| **ours (teacher)** | **0.847437** | 1.0 | 1.0 | 3 | 34 | **29.14** |
| bear→ours | 0.575540 | 0.775 | 0.710 | 17 | 20 | 28.24 |
| ours→bear | 0.485825 | 0.571 | 0.472 | 20 | 15 | 8.42 |

Deltas vs bear: none +0.4253 (CI 0.285,0.571 ★) · **ours +0.3953 (CI 0.259,0.538 ★)** · bear→ours +0.1234 (CI 0.027,0.233 ★) · ours→bear +0.0337 (CI −0.084,0.163 n.s.).
**Rel lift ours = +87.4%; ours = 96.6% of oracle; 14.0× fewer output tokens than bear.**

### 5.2 Distilled (Hearth/v3) cross-benchmark — mean F1 by bar
| Benchmark | none | bear | **ours (v3)** | bear→ours | ours→bear | headline |
|---|---|---|---|---|---|---|
| HotpotQA | 0.877397 | 0.452143 | **0.704181** | 0.322655 | 0.463857 | **PASS** |
| 2wiki | 0.572667 | 0.390476 | **0.570000** | 0.370921 | 0.406667 | **PASS** |
| musique | 0.519649 | 0.185852 | **0.297201** | 0.060238 | 0.213308 | FAIL |
| squad | 0.903147 | 0.470762 | **0.593332** | 0.282159 | 0.294897 | FAIL |

Deltas vs bear (ours bar): **HotpotQA +0.2520 (CI 0.103,0.396 ★)** · **2wiki +0.1795 (CI 0.030,0.340 ★)** · musique +0.1113 (CI −0.027,0.240 n.s.) · squad +0.1226 (CI −0.026,0.269 n.s.).
Stacking is never a significant gain; on musique/squad bear→ours and ours→bear are significantly **negative**.

Rel lift / oracle-recovery / compression (v3): HotpotQA +55.7% / 80.3% / 8.6× · 2wiki +46.0% / **99.5%** / 5.5× · musique +59.9% / 57.2% / 11.7× · squad +26.0% / 65.7% / 1.3×.

**Bimodality (per-instance F1), v3 ours bar:** HotpotQA 13×F1=0, 31×F1=1 (vs bear 24/17) · 2wiki 20×0, 26×1 (vs bear 29/17) · musique 26×0, 8×1 · squad 15×0, 22×1. The QA benchmarks are strongly bimodal (mass at 0 and 1); MS MARCO is NOT (see 5.6). v3 wins by converting bear's failures into successes (e.g. HotpotQA: 13 fails vs bear's 24).

### 5.3 Variant study — ours-bar F1 across v1/v3/v4/v5 (same frozen none/bear fixtures)
**HotpotQA** (none=0.8774, bear=0.4521 fixed):
| Variant | ours F1 | rel-lift | CI | sig |
|---|---|---|---|---|
| v1 (Spark) | 0.515035 | +13.9% | (−0.098,0.220) | n.s. FAIL |
| **v3 (Hearth) ⭐** | **0.704181** | +55.7% | (0.103,0.396) | ★ PASS |
| v4 (Oracle, best-of-N) | 0.659333 | +45.8% | (0.050,0.359) | ★ PASS |
| v5 (Oracle-Lite, greedy) | 0.664333 | +46.9% | (0.051,0.371) | ★ PASS |
| teacher (ceiling) | 0.847437 | +87.4% | (0.259,0.538) | ★ |

Ranking (HotpotQA ours-F1): **teacher 0.847 > v3 0.704 > v5 0.664 > v4 0.659 > v1 0.515.**

**2wiki** (none=0.5727, bear=0.3905 fixed): **v3 0.570 (CI 0.030,0.340 ★ PASS) — the ONLY variant that passes.** v4 0.520 (CI −0.014,0.273 n.s. FAIL); v5 0.512 (CI −0.031,0.260 n.s. FAIL). ⚠️ *Contradicts the HotpotQA story where v4/v5 also pass — answer-grounding loses significance out-of-distribution.*

**squad/musique:** only v3 evaluated (both n.s., see 5.2).

**v1 "the wash" full 5-bar** (`5bar_distilled_results.json` ≡ v1): ours 0.515 (CI −0.098,0.220 n.s.), headline FAIL.

### 5.4 Act 2 — combined multi-turn benchmark (`combined_benchmark.json`, n=20, 6 turns)
| Strategy | mean_f1 | mean_tokens | tokens vs naive |
|---|---|---|---|
| naive (growing) | 0.484940 | 845.5 | 1.00× |
| rezero_deepseek | 0.454634 | 198.4 | 4.26× fewer |
| **rezero_distilled** | **0.501052** | **174.5** | **4.85× fewer** |
| rezero_bear | 0.472369 | 256.6 | 3.29× fewer |

**rezero_distilled is the ONLY strategy that beats naive on F1 (0.501 > 0.485) AND uses fewest tokens.** F1 rank: distilled > naive > bear > deepseek. Token rank (fewest): distilled < deepseek < bear < naive.

**Token trajectory** (`token_trajectory.json`, 12 turns): naive grows 81→**1482** (18.3×); rezero_distilled stays flat 94–193, ends **184**. **Final ratio naive/rezero = 8.05×.** ⚠️ only naive + rezero_distilled trajectories present (no deepseek/bear).

### 5.5 Latency (`latency_api.json`, n=20, seconds)
| Strategy | mean_s | median_s | p90_s | avg_tok_out |
|---|---|---|---|---|
| bear | 1.047 | 1.081 | 1.249 | 429.65 |
| ours | 1.080 | 1.036 | 1.329 | 26.75 |
| bear→ours | 1.250 | 1.248 | 1.433 | 26.05 |
| ours→bear | 1.342 | 1.322 | 1.665 | 8.0 |

ours ≈ bear wall-clock (+3.1%) but **16.1× fewer output tokens** (26.75 vs 429.65). Stacking is slower. (No naive row; this is API-path latency, not distilled-GPU.)

### 5.6 MS MARCO — continuous-F1 scoping (`5bar_v5_msmarco.json`, v5)
Low-ceiling abstractive task (oracle none only **0.254**). Decision: headline FAIL.
| Bar | mean_f1 | n=0 | n=1 | avg_tok_out |
|---|---|---|---|---|
| none | 0.253964 | 15 | 1 | 725.42 |
| bear | 0.201839 | 21 | 1 | 217.1 |
| ours | 0.204175 | 20 | 2 | 39.46 |
| ours→bear | 0.128381 | 24 | 1 | 11.44 |

**ours ≈ bear (Δ=+0.0023, +1.2%, n.s.)** — query-aware compression **ties** bear on abstractive QA. Distribution is **continuous, NOT bimodal** (mass in (0,0.3), few exact-1s) — confirms the entity-span benchmarks are near-binary by nature. ours uses 5.5× fewer tokens, reaches 80.4% of the (weak) oracle.

### 5.X ⚠️ Cross-cutting flags (must reconcile in the paper)
1. **Duplicate files (identical data, don't double-count):** `5bar_results.json` = `v1/eval/5bar_api_baseline.json`; `5bar_distilled_results.json` = `v1/eval/5bar_distilled_v1.json`; `results/5bar_distilled_v3.json` = `experiments/v3/eval/5bar_distilled_v3.json`.
2. **`5bar_distilled_results.json` is actually v1 (the wash, ours=0.515 FAIL)**, NOT the v3 headline — no `benchmark` field. Easy to mis-cite.
3. **`5bar_distilled_hotpotqa.json` vs `_v3.json`:** same model + identical ours bar, stacked bars differ ~1e-4–2e-3 (re-run noise). Pick one canonical.
4. **2wiki variant inconsistency:** v3 PASSES, v4/v5 FAIL — answer-grounding loses OOD significance. Caveat needed.
5. **Missing data:** token_trajectory has only 2/4 strategies; latency has no naive row. If the paper needs deepseek/bear trajectories or distilled-GPU latency, they're absent (would need a run).
6. **Stacking never helps** in any of the 15 files (stacking_bonus=false everywhere); several significantly negative.
7. `smoke_5bar.json` = n=1 sanity (the "Mossad" instance), degenerate CIs.

## 6. Key findings
_Source: docs agent (numbers cross-verified by numbers agent in §5)._

**Positive:**
1. **Headline:** distilled Qwen2.5-1.5B (Hearth) beats bear while emitting ~8.5× fewer tokens (same ratio-0.3 instruction; ours ~48 tok vs bear ~409) — HotpotQA **+0.252 (+56%), CI (0.103,0.396)**. Confirmed under an independent solver (Claude Sonnet, +0.288). Much of the margin is span-selection (mask-the-answer audit), reported honestly.
2. **Cross-dataset transfer (strongest evidence, scoped):** trained ONLY on HotpotQA-derived data, yet 2Wiki **+0.180 (+46%), CI (0.030,0.340)** — but 2Wiki is near-in-distribution; the dissimilar OOD sets (MuSiQue, SQuAD) are positive-but-n.s. at n=50. The accurate claim: significant on multi-hop-with-distractors, directional elsewhere.
3. **Near-ceiling at tiny budget:** API teacher 0.847 ≈ full-ctx 0.877 at ~2-3% tokens; full 1364 → bear 409 (30%) → ours ≈48 (3.5%), still more accurate than bear at 8× the tokens.
4. **Student recovers 64% of frontier margin** (v3: 0.252/0.395). [Spark recovered only 16% — see provenance note.]
5. **The deletion ceiling (conceptual contribution):** reachability argument (extractive ⊊ abstractive reachable set) + constructed example (3-hop coreference "Trondheim" chain no in-budget subsequence carries but a 7-token paraphrase does) + empirical (ours→bear strictly worse on all 4: HotpotQA −0.240, 2Wiki −0.163, MuSiQue −0.084, SQuAD −0.298). "once you are abstractive, going back to extractive is pure loss."
6. **The compressor composes (Act 2):** Re:Zero+distilled wins both axes (0.501 F1 / 174 tok, beating naive 0.485/846, deepseek 0.455/198, bear 0.472/257); 12-turn flatness 1482 vs 184 (8.1×).

**Negative (each a contribution):**
7. **Stacking fails — sometimes significantly:** wash on HotpotQA/2Wiki; SQuAD ours→bear sig-worse (−0.176); MuSiQue/SQuAD bear→ours sig-worse. "replacement, not a layer — incompatible contracts."
8. **Answer-grounding loses twice** (Oracle/Oracle-Lite) — judge-selection overfits, hurts OOD.
9. **MS MARCO scoping:** abstractive QA → ours ≈ bear (+0.002, n.s.). F1 genuinely continuous there (28/50 partial) vs near-binary entity benchmarks. *Scopes and strengthens* the headline.
10. **Overfitting lessons:** data-quantity ceiling (Spark) vs capacity-driven overfit (Bonfire, rank-driven).

## 7. Engineering reality & reproducibility
_Source: runs agent (23 Modal app IDs mapped to logs; loss curves + wall-clocks cross-validated against EXPERIMENT_LOG)._
Shared setup: Modal, **1× H100 80GB**, Unsloth+TRL+PEFT, base `unsloth/qwen2.5-1.5b-instruct-unsloth-bnb-4bit`. xformers broken in every image (benign PyTorch-attn fallback).

### 7.1 Training runs (with Modal app IDs, loss curves, wall-clock)
| Codename | Modal app | config | eval_loss/epoch | final train_loss | overfit | steps / wall-clock | outcome |
|---|---|---|---|---|---|---|---|
| Spark/v1 | ap-94qRcHMauMNmvif5QZxaE1 | 261/29, 3ep, r=16, eff16, 18.46M (1.18%) | 1.7595→1.7438→1.7418 | 1.8178 | ✅ no | 51 steps / **139s** | ✅ trained (eval FAIL) |
| Bonfire/v2 | ap-wl7zrQM9LUeU1oJlRwltHR | 2250/250, 6ep, r=64, eff16, 73.86M (4.57%) | 1.6622→**1.6540**→1.6852→1.7617→1.8611 | ~0.95 | 🔴 **YES** | died ~770/846 | ❌ network drop + overfit finding |
| v2b/c/d | ap-6PJ…/ap-obC…/ap-xI5… | r=64 variants | — | — | — | died early | ❌ network drops / RemoteError |
| v3-first | ap-fIeFG2UEp8eOzIsmoOGirk | 5000pairs, r=32, eff64, 36.93M (2.34%) | crashed in 1st eval | — | — | crashed eval 3/16 | ❌ **CUDA OOM** (53GB fp32 logit cast) |
| v3b | ap-rmuVjgUyLk8rnXmukEuctZ | same | died ~step7/213 | — | — | — | ❌ network drop |
| **Hearth/v3c** ⭐ | **ap-ZPhQ1HfcHjVpofk8IlfiI4** | 4500/500, 3ep, r=32, dropout 0.1, wd 0.01, load_best, eff64 | **1.6787→1.6655→1.6638** | 1.6875 | ✅ no | 213 / **1458s (~24min)** | ✅ **SHIPPED** |
| Oracle/v4 | ap-q8Rtvb4belGcxd3Kon2puA | 3927/436 (best-of-4), same v3 cfg | 1.6876→1.6748→1.6732 | 1.6920 | ✅ no | 186 / 1195s | ✅ trained (lost to Hearth) |
| Oracle-Lite/v5 | ap-TUPapYFcmS7PVveb844sYr | 3758/417 (greedy), same | 1.6874→1.6749→1.6734 | 1.6887 | ✅ no | 177 / 1194s | ✅ trained (lost) |
| v3-retrain | ap-FHTyIx7D4zXEALZMbVENp5 | identical to Hearth | 1.6787→1.6655→1.6638 | 1.6875 | ✅ no | 213 / 1370s | ✅ **reproduces Hearth exactly** → adapter_v3_clean |

### 7.2 Eval runs (Modal app IDs)
API teacher (local, no Modal): HotpotQA ours 0.847 Δ+0.395. · v1 eval ap-dQGb07c3WcpVhyzY3AMlBu (ours 0.515 FAIL). · v3 eval ap-s0rdATa8kQEVPVmTD09EMe (ours 0.704 PASS). · v3 all-4-benchmarks ap-v4dRiQgiFCECEksX2YURRC. · v4 hotpot ap-U8vdNirMw04Wqg0vdgIqLm (0.659) / 2wiki ap-n3RjFNGjHmx4LzY2GEVUW6 (0.520 FAIL). · v5 hotpot ap-uROC3xJQbxLKuqzQOx1JeT (0.664) / 2wiki ap-Ev5v9y5zwJFpyUQIXfbWF5 (0.512 FAIL) / msmarco ap-A1UakPfQNIGhTdf4AXthOE (0.204 tie). · Act2 combined ap-aeuxOkX06gRrivsVOydFm1; trajectory ap-oonSAzFNceeeINzqhQL5F9; integration smoke ap-KMTqNa6ibgZd1NOuhFhKmp; restructure test ap-ZY2FyklDjPhk0uMHXISM95.

### 7.3 Data-gen runs
v1 [0:300]→300 kept (2.23% ratio, 8 workers) then 10 leaks dropped→261/29. v2 [0:1000]+[1000:2500]→2500 (3.5%). v3 [2500:5000]→2500 (total 5000). v4 [0:5000] best-of-4→**4363 kept (87.3%)**, avg ans-F1 0.934, **96 workers**. v5 [0:5000] greedy→**4175 kept (83.5%)**, avg ans-F1 0.924, 96 workers. Bear-Booster: smoke only (6 ex, 83% keep, 0.933 post-bear F1), dropped before full gen. All gens err=0, zero rate-limit errors.

### 7.4 Ordered failure→fix catalog (~9 distinct modes — the engineering-reality timeline)
**Phase A (5 startup failures before first clean train, each ≈$0):**
0. image build: `trl==0.11.4`+`peft==0.13.2` unsatisfiable w/ unsloth → drop pins + `unsloth_zoo` (resolved: trl 0.24, peft 0.19.1, torch 2.10-2.12).
1. `FileNotFoundError: train_clean.jsonl` (local path to `.remote()`) → read JSONL locally, ship parsed rows.
2. `Unsloth: You must specify a formatting_func` (no auto chat-template from messages col) → add func.
3. `UndefinedError: dict object has no element 0` (Unsloth probes with single example, not batch) → handle single-example.
4. `ValueError: formatting_func should return a list` → **abandon formatting_func; pre-render ChatML `text` column locally** → first clean train (Spark).
**Phase B (infer image):** 5. `ModuleNotFoundError: tiktoken` → add to infer image. 6. ~150 cold model loads → Modal **class with `@enter`** (load once) + batched `compress_batch`.
**Phase C (scale-up):** 7. `packing=True` needs flash-attn (broken) → flattened to 61,405-tok seq, label mismatch, CE crash → revert to plain batching (speed via eff-64 batch). 8. **CUDA OOM at eval** (53GB fp32 upcast of `[bs,seq,vocab=151936]` logits at batch 32) → `prediction_loss_only=True` + smaller eval batch + `expandable_segments:True`. 9. **Network drops kill non-detached `modal run`** (`Deadline exceeded`→`nodename nor servname`→`SSL handshake >60s`; killed v2/v2b/v2c/v3b) → **`modal run --detach`** (server-side, survives laptop disconnect).

### 7.5 Concurrency / throughput
MAX_WORKERS **8 → 96** (coded in all 3 gen scripts; comment: "DeepSeek allows ~2500 concurrent; 96 keeps ThreadPoolExecutor efficient, async needed beyond ~128"). v1-v3 gens ran at 8; v4/v5 at 96 — both 5000 instances, err=0 (bump is safe).
**⚠️ CORRECTION (runs agent caught my error):** the "~17-20 req/s ceiling" I wrote earlier is **NOT in any log** — those were *token counts* in the latency table, not req/s. **No standalone 2500-concurrent throughput benchmark exists in the archive.** Do NOT cite a req/s number; either re-run a timed sweep or frame qualitatively (8→96 workers, headroom to ~2500). Only concrete timing is latency_api.json (~1.0-1.3s/request single-threaded).

### 7.6 Cost / time
**Total ≈ $10** (DEVPOST says $10-15) across all shipped gens+training; **failed launches errored at startup ≈ $0**. Per-run training wall-clock (H100): v1 139s; Hearth/v3c **1458s (~24min)**; v4 1195s; v5 1194s; v3-retrain 1370s. **Shipped-model training ~20-25 min on one H100** (EXPERIMENT_LOG's "~12-15 min" pre-estimate was optimistic; eff-64 + periodic eval lands ~24min). Inference: ~$10 one-time then ~free offline per prompt vs API per-call forever. Images cached after first build.

## 8. Honest limitations
_Source: docs agent (README §8)._
1. **bear's real structural advantages:** verbatim (no hallucination), query-agnostic (compress-once-reuse; we recompress per Q), no training needed.
2. **Latency NOT a win:** ours ≈ bear (~1.08 vs ~1.05s, both network-bound). "We claim an accuracy/token win, not a speed win."
3. **2 of 4 benchmarks n.s.** (MuSiQue, SQuAD) — directional only at n=50. "Proven on two."
4. **Stacking fails, sometimes significantly** — replacement not layer.
5. **No LLMLingua head-to-head — did NOT beat published SOTA.** "We beat bear-2, the sponsor's product, the relevant comparison for this track." Distinction is methodological (delete vs rewrite).
6. **Lab setting:** n=50/benchmark, ratio=0.3 only. Larger n + ratio sweep out of 24h scope.
7. **Over-compression:** student lands ~3.8% vs ~30% target; budget knob uncalibrated. "We win on accuracy at a budget we rarely spend."
8. **Act 2 honesty:** "nothing *critical* lost" (not "nothing lost") — 150-tok checkpoint of 1000 tok drops ~850 by pigeonhole. Re:Zero ≈ MemGPT; only the §3.5 micro-claim is fresh.

## 9. Future work
_Source: docs agent._
1. **The "bear-improver"** (flip Bear-Booster): post-processor that repairs bear's *output* (input=bear output, target=good compression) — additive/flattering, not built.
2. **Direct LLMLingua-2 comparison** (closest prior work).
3. **Ratio sweep + larger n** to map the significance frontier (push MuSiQue/SQuAD to sig).
4. **Calibrate the budget knob** to hit a requested ratio.
5. **Live Re:Zero demo** (long convo flat in real time).
6. Act 3 (`ACTIII/`) in progress.

## 10. ⚠️ CRITICAL — provenance & inconsistency reconciliation (READ BEFORE WRITING)
_All three reader agents independently flagged the same hazard: the same-looking number appears with different values across files because it comes from **different runs**. The paper MUST disambiguate run provenance on every number._

**The golden rule:** lead with **Hearth/v3 (the shipped distilled student)** as the headline; cite the **API teacher** explicitly as the *upper bound*; the EXPERIMENT_LOG top tables + PROGRESS.md report the **API teacher** and **v1/Spark** — do NOT mistake those for the headline.

| Number | API teacher | v1 / Spark | **v3 / Hearth (HEADLINE)** |
|---|---|---|---|
| HotpotQA ours F1 | 0.847 | 0.515 | **0.704** |
| Δ vs bear (HotpotQA) | +0.395 (CI .259,.538) | +0.063 (CI −.098,.220, n.s.) | **+0.252 (CI .103,.396) ★** |
| margin recovered | (the 100% reference) | ~16% | **~64%** |
| `bear→ours` HotpotQA | 0.576 (sig +0.123!) | 0.317 | 0.323 (n.s.) |

**Specific reconciliations:**
1. **`5bar_distilled_results.json` = v1 (the wash), NOT v3.** No `benchmark` field. ours=0.515 FAIL.
2. **`bear→ours` differs ~0.25 across docs because PROGRESS reports the API-teacher run (0.576, significant) while README reports the distilled-student run (0.323, n.s.).** This is the most confusing discrepancy — label run provenance on EVERY stacking number.
3. **Duplicate files (don't double-count):** `5bar_results.json`=`v1/eval/5bar_api_baseline.json`; `5bar_distilled_results.json`=`v1/eval/5bar_distilled_v1.json`; `results/5bar_distilled_v3.json`=`experiments/v3/eval/5bar_distilled_v3.json`. `5bar_distilled_hotpotqa.json`≈`_v3.json` (ours identical, stacked bars differ ~1e-3 re-run noise — cite one).
4. **Duplicate logs:** `logs/modal_train_v3.log` = the OOM-FAILED first v3 (ap-fIeFG2...), NOT the success; success = `modal_train_v3c.log` = `experiments/v3/logs/train_v3.log` (ap-ZPhQ...). v3 vs v3-retrain produce identical loss (reproduction, cite one + note reproducibility).
5. **Compression ratio quoted 3 ways (all real, different runs/prompts):** ~2.4% (v1 teacher data, "MINIMAL" prompt) / ~3.5% (≈48/1364 tok, v3) / ~3.8% (the floor/anti-answer prompt avg). State which.
6. **2wiki variant inconsistency (a genuine finding, not an error):** v3 PASSES 2wiki, v4/v5 FAIL — answer-grounding loses OOD significance. Keep as a result.
7. **NO req/s measurement exists** — don't cite 17-20 req/s (see §7.5).
8. **Training time:** "~12-15 min" appears in docs as a pre-estimate; ACTUAL shipped run = ~24 min (§7.6). Use the measured number.
9. **Act 3 has no results JSON yet** — code+tests only; either run it or present as designed-not-yet-run.

## 11. Appendix — pointers
- Per-instance F1 distributions: in §5.2 (bimodality table) + the result JSONs' `per_instance` arrays.
- All 23 Modal app IDs: §7.1-7.2.
- 13 figures: `results/figures/*.png` (cross_benchmark_bars, delta_ci, dots_{4 benchmarks}, tokens_vs_f1, variant_compare, distill_trajectory, latency_bars, multiturn_{f1,tokens}, token_trajectory_line) + `results/distribution_2wiki.png`.
- Saved adapters (verified intact): `artifacts/adapter_v3_clean/` (v3/Hearth), `artifacts/adapter_v5/` (v5).
- Prior-art cites to chase: LLMLingua / LongLLMLingua / LLMLingua-2 (MSR), RECOMP, Selective-Context, MemGPT, Jha et al. 2024 (arXiv:2407.08892), CompactPrompt.
