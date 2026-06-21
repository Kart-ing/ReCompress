# ReCompress — Devpost submission

> Paste each section into the matching Devpost field. Tagline + "Built With" + links are at the bottom.

---

## Project name
**ReCompress: a distilled 1.5B query-aware compressor that beats blind token deletion**

## Tagline (one line — Devpost "tagline" field)
A tiny offline model that reads your question, rewrites the context to keep only what matters, and beats The Token Company's bear-1.1 by ~50% on multi-hop QA at the same token budget.

---

## Inspiration

The Token Company's **bear-1.1** compresses prompts by deleting low-value tokens — *character-for-character, blind to the query*. That's brilliant for speed and reuse, but it leaves two things on the table that deletion fundamentally **can't** do:

1. **Query-aware selection** — if you know the question, you can drop entire irrelevant passages, not just stray tokens.
2. **Dense rewriting** — verbose-but-relevant prose can be *paraphrased* into something far shorter; deletion can only cut, never compress an idea.

We wanted to measure exactly how much those two levers are worth — and then prove you don't need a giant model to get them, by **distilling the capability into a 1.5B model that runs offline for cents.**

## What it does

ReCompress is a **query-aware context compressor**. Given a long context and a question, it:
- **drops** passages irrelevant to the question (distractors), and
- **densifies** the relevant ones into terse, information-packed sentences,

producing a much smaller context that a downstream LLM can still answer from. We then **distilled** this behavior from a frontier teacher (DeepSeek) into **Qwen2.5-1.5B + LoRA**, so the compressor is small, offline, and cheap — the same product category as bear, but query-aware.

**Headline result** — distilled 1.5B vs bear-1.1, at *matched token budget*, QA-F1 with paired bootstrap 95% CIs, frozen DeepSeek solver as judge:

| Benchmark | full context | **ReCompress (1.5B)** | bear-1.1 | Δ vs bear | significant? |
|---|---|---|---|---|---|
| **HotpotQA** | 0.877 | **0.704** | 0.452 | **+0.252 (+56%)** | ✅ yes (CI excludes 0) |
| **2WikiMultiHop** | 0.573 | **0.570** | 0.390 | **+0.180 (+46%)** | ✅ yes — *and never trained on it* |
| MuSiQue | 0.520 | 0.297 | 0.186 | +0.111 (+60%) | directional (n.s. at n=50) |
| SQuAD v2 | 0.903 | 0.593 | 0.471 | +0.123 (+26%) | directional (n.s. at n=50) |

The win is **statistically significant on both multi-hop-with-distractor benchmarks** — exactly the regime where query-awareness should matter most. On 2Wiki the 1.5B nearly matches *full context* (0.570 vs 0.573) at a fraction of the tokens — and **it was only ever trained on HotpotQA-derived data**, so that's genuine cross-dataset generalization, not memorization.

## How we built it

A 3-API-call pipeline that we then **distill into weights**:

```
                        ┌─────────────────────────────────────┐
  context + question ──>│  Teacher: DeepSeek (query-aware      │
                        │  compress: drop distractors + densify)│
                        └───────────────┬─────────────────────┘
                                        │ 5,000 (text, q, compressed) pairs
                                        ▼
                        ┌─────────────────────────────────────┐
                        │  Distill: LoRA fine-tune             │
                        │  Qwen2.5-1.5B on Modal H100          │
                        └───────────────┬─────────────────────┘
                                        ▼
   ReCompress-1.5B (offline)  ──>  Frozen DeepSeek solver  ──>  QA-F1
   baseline: bear-1.1 (TheTokenCompany SDK), same budget
```

- **Teacher data:** DeepSeek generates 5,000 query-aware compressions of HotpotQA contexts (parallelized, filtered for answer-leaks).
- **Student:** Qwen2.5-1.5B-Instruct, 4-bit + LoRA (r=32), trained with Unsloth on a Modal H100.
- **Evaluation:** a **5-bar paired benchmark** — `none | bear | ours | bear→ours | ours→bear` — at matched token budget, scored by a frozen solver with **paired bootstrap 95% CIs** on every delta vs bear, across **4 QA benchmarks**.
- **Cost:** the whole project (data gen + multiple training runs + 4-benchmark eval) was **~$10 of compute** — LoRA on a 4-bit 1.5B trains in ~15 min on one H100.

## Challenges we ran into

- **Distillation didn't work on the first try — twice.** v1 (261 examples, LoRA r=16) was a statistical *wash* vs bear (Δ=+0.063, CI included 0). v2 (2,500 examples, r=64, 6 epochs) **overfit hard** — eval loss bottomed at epoch 2 then climbed every epoch after. Only v3 (5,000 examples, r=32, 3 epochs, dropout 0.1, weight decay, early-stopping on best eval) crossed the line into a significant win. We kept the full failure trajectory as part of the research record.
- **The Modal + Unsloth + trl stack is version-brittle.** We hit ~7 distinct runtime failures before the first clean train: an unsatisfiable dependency pin, a container that couldn't see the local data file, Unsloth's `formatting_func` contract, an eval-time CUDA OOM from full-vocab logit casting, and a `packing` incompatibility (needs flash-attn-2). Each is documented as a reproducibility note.
- **A network drop killed a training run at 91%** — fixed by running detached so the H100 job survives local disconnects.
- **We tested — and dropped — a second idea.** Our original thesis was to train the SLM to make *bear's* output better (`bear(SLM(text))`). Our own data proved it's **dominated**: the standalone model beats `ours→bear` on every benchmark, and a pre-processor is strictly slower + costlier than bear alone. We stopped before spending the full compute and recorded it as a negative result: *you can't fix blind deletion by pre-processing — query-aware rewriting must replace it.*

## Accomplishments we're proud of

- A **1.5B model that beats the challenge sponsor's own product** at matched budget, with statistical significance, that **generalizes to a benchmark it never trained on**.
- **Research-grade rigor for a 24h hackathon**: paired bootstrap CIs on every claim, 4 benchmarks, honestly-labeled non-significant results, owned failure modes, and a documented dropped idea.
- It runs **offline and cheap** — no per-token API fees, ~$10 to build the whole thing.

## What we learned

- **Query-aware *rewriting* beats blind *deletion*** at matched budget — most decisively exactly where there are distractors to drop (multi-hop QA).
- **Stacking with bear hurts.** Running bear's deletion on top of our dense rewrite mangles it (significantly *worse* than bear alone on SQuAD). Rewriting and deletion are different contracts; you pick one.
- **Small models can learn this** — but only with enough clean data and proper regularization; the v1→v3 arc is a textbook case of fixing a data-starved, then over-parameterized, model.
- **Honesty is a feature.** Reporting what we *didn't* prove (2 of 4 benchmarks n.s., no latency win, no SOTA comparison) makes the parts we *did* prove credible.

## What's next

- **Multi-turn compression (Act 2):** apply query-aware compression turn-by-turn in a long conversation so context stays flat instead of growing O(n²).
- **Direct comparison to LLMLingua-2** — the closest prior work. Our differentiator: LLMLingua *deletes/classifies* tokens (extractive); we *rewrite* them (abstractive) and distilled it into a generative 1.5B, benchmarked head-to-head against bear.
- **Scale the teacher data and ratios** beyond the 5,000 examples / 0.3 ratio / n=50 lab setting to push the directional wins (MuSiQue, SQuAD) into significance.

---

## "Built With"
`python` · `qwen2.5-1.5b` · `lora` · `unsloth` · `modal` · `h100` · `deepseek-api` · `the-token-company-sdk` · `huggingface-datasets` · `hotpotqa` · `2wikimultihop` · `musique` · `squad` · `matplotlib`

## Links
- **Code:** https://github.com/Kart-ing/ReCompress
- **Trained adapters + full experiment archive (incl. failures):** in the repo (Git LFS)

## One-line differentiator (for the pitch + Q&A — keep this ready)
> "LLMLingua **deletes** tokens; we **rewrite** them — and distilled that into a 1.5B that beats The Token Company's own bear head-to-head at matched budget, with confidence intervals, across four benchmarks."
