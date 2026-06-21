# Why the multi-turn overhead is huge — and does it amortize at scale?

> Written for Parth, who's going to try to fix this. TL;DR: the overhead is real,
> it comes almost entirely from **trauma + Echidna running an LLM call every turn on
> re-read inputs**, it's roughly **constant per turn** (so it does NOT blow up with
> conversation length), and because naive context grows *linearly* there is a real
> **crossover horizon** past which RbD-Compress wins on total tokens — we just measured
> below it. The fix is to make the per-turn machinery cheaper or less frequent.

## 1. The measurement that started this

From `results/overhead_benchmark.json` (HotpotQA, 6 turns, n=20). "overhead" = total
DeepSeek tokens spent on trauma + Echidna (+ the compressor, for the deepseek backend)
across the whole conversation; "ctx→solver" = the final solver prompt size:

| Strategy | F1 | ctx→solver | overhead | **total** |
|---|---|---|---|---|
| Naive (final / KV-cached) | 0.485 | 846 | — | **846** |
| Naive (uncached sum) | 0.485 | 2,960 | — | **2,960** |
| RbD-Compress + DeepSeek | 0.533 | 197 | 6,861 | **7,058** |
| **RbD-Compress + distilled** | **0.542** | **175** | 5,397 | **5,571** |
| RbD-Compress + bear | 0.411 | 255 | 5,386 | **5,641** |

The 8.1× flatness number we report is **ctx→solver only** (175 vs a growing naive context).
Once overhead is counted, RbD-Compress costs **~5,571 total tokens vs naive's 2,960** at 6
turns — i.e. it is **~1.9× *more* expensive**, not cheaper. That's the honest result now in
the paper (§ "honest total-token accounting").

## 2. Where the overhead actually comes from

Per `rezero/rezero/session.py::add_turn()`, **every turn** fires these LLM calls
(all through `engine/deepseek.py::call()`):

1. **Trauma extractor — TWICE per turn.** `self.trauma_extractor.update(user)` and again
   `update(assistant)` (unless the assistant message is ≤3 words). Each call
   (`rezero/rezero/trauma.py`) sends `system + "Current buffer: {buffer}\nNew message: {message}"`
   with `max_tokens=200`.
2. **Echidna — ONCE per turn.** `self.echidna.decide(...)` (`rezero/rezero/echidna.py`)
   sends `system + trauma + checkpoint_summary + last-4-history-messages` with `max_tokens=200`.
3. **Checkpoint compressor — only when Echidna says "checkpoint".** For the deepseek backend
   this is another `call()`; for distilled/bear it's a Modal/SDK call (not counted in the
   DeepSeek accumulator, which is why those two backends show ~the same overhead).

**The arithmetic that proves trauma+Echidna are the bulk:** distilled overhead (5,397) ≈ bear
overhead (5,386). Those two backends route their *compressor* through Modal/SDK, not DeepSeek,
so their measured overhead is **trauma + Echidna alone ≈ 5,390 tok/convo (~900/turn)**. The
deepseek backend is higher (6,861) precisely because its compressor *also* goes through
`call()` — so the **DeepSeek checkpoint compressor adds ~1,470 tok/convo** on top.

Breakdown per 6-turn conversation (approx):

| Component | calls/turn | tokens/convo | share |
|---|---|---|---|
| Trauma extractor | 2 | ~3,500 | ~63% |
| Echidna | 1 | ~1,900 | ~34% |
| Compressor (deepseek backend only) | ~0.3 (gated) | ~1,470 | (extra) |

> The single biggest lever is **trauma running twice every turn** with a full system prompt
> each time. That alone is ~3,500 of the ~5,400 tokens.

### Why each call is bigger than it looks
- `max_tokens=200` is the *output* cap, but the **system prompts are sent every call** and
  count as input tokens. A ~150-token system prompt × 3 calls/turn × 6 turns = ~2,700 input
  tokens just in repeated instructions, before any content.
- Trauma re-sends the **growing buffer** each call; Echidna re-sends trauma + checkpoint
  summary + recent history each turn. These grow slowly but they're not free.
- `MIN_MAX_TOKENS=1024` in `deepseek.py` floors the request — if the API bills on requested
  max in any tier, that inflates cost further (worth checking).

## 3. Does it get better at larger sizes? (the important question)

**Yes — but not because overhead shrinks. Because naive grows and overhead doesn't.**

Two regimes to separate:

- **Overhead per turn is ~constant (~900 tok/turn).** Trauma + Echidna read trauma (capped
  ~50 tok), a checkpoint summary (bounded), and only the **last 4** history messages — none of
  which grow unboundedly. So total overhead is roughly **linear in turns**: `overhead ≈ 900·T`.
- **Naive context grows ~linearly per turn, quadratically in cumulative cost.** Final-prompt
  naive ≈ `~140·T` tokens (from `token_trajectory.json`: 81→1,482 over 12 turns). The
  *uncached cumulative* naive cost is the sum, ≈ `70·T²`.

### The crossover (rough, order-of-magnitude — Parth should re-derive with real fits)

RbD-Compress total ≈ `175 (flat ctx) + 900·T` (overhead).
Naive depends entirely on **caching**:

- **vs uncached cumulative naive (~70·T²):** crossover when `70·T² > 175 + 900·T`
  → `T ≳ 15` turns. So past ~15 turns, RbD-Compress beats *uncached* naive on total tokens.
  At our T=6 we were well below it (that's why we lost).
- **vs KV/prefix-cached naive (~140·T final prompt):** `140·T` vs `175 + 900·T` —
  the cached naive line has a *smaller* slope (140 < 900), so **RbD-Compress never catches a
  cached naive deployment on total tokens.** Caching is the killer; deletion-based systems get
  it for free, we don't, because our per-turn LLM calls can't be prefix-cached the same way.

**So the honest answer to "does it get better at scale?":**
- Against an **uncached** agent: yes, crossover ~15 turns, then RbD-Compress pulls ahead and
  the gap widens (naive is quadratic, we're linear).
- Against a **cached** agent: no, not on total tokens — our per-turn overhead slope is too high.
- **Where it always wins regardless of horizon:** when the binding constraint is the **context
  window** (naive eventually overflows; we never do), or when the **solver is far more expensive
  than the compressor** (e.g. solver = GPT-4-class, trauma/Echidna = a tiny cheap model). In our
  benchmark everything is DeepSeek, so that asymmetry doesn't help us.

## 4. What Parth could try (ordered by expected impact)

1. **Stop running trauma twice per turn.** Update on the user message only, or batch
   user+assistant into one call. Expected: ~halve the trauma share → ~−1,700 tok/convo.
2. **Make Echidna a classifier, not an LLM call.** Echidna's job is a 3-way decision
   (checkpoint / revert / pass). A cheap heuristic or a tiny fine-tuned classifier (or even
   the existing token-threshold mock) removes ~1,900 tok/convo entirely. This is the cleanest
   structural fix — the paper already lists "replace Echidna's LLM call with a fine-tuned
   classifier" as future work.
3. **Trim the system prompts.** Three ~150-token system prompts re-sent every turn is most of
   the fixed cost. Shorten them, or move stable instructions into a cached prefix.
4. **Run trauma/Echidna on the distilled 1.5B (or a smaller model), not DeepSeek.** This is the
   "solver ≫ compressor" asymmetry — if the per-turn machinery runs on a cheap local model and
   only the *answer* uses an expensive solver, the total-cost story flips. Arguably the highest-
   leverage change: it makes the overhead "free" in the regime that matters.
5. **Gate trauma updates.** Don't call the extractor on turns that obviously add no new entities
   (e.g. short acknowledgements) — cheap skip already exists for ≤3-word assistant turns; extend
   the heuristic to user turns.
6. **Check `MIN_MAX_TOKENS=1024`** — if it inflates billed tokens, lower it for the
   trauma/Echidna calls (they only need ~200).

## 5. How to reproduce / verify a fix

```bash
# instrumented run (the accumulator lives in engine/deepseek.py: reset_overhead/get_overhead)
cd rezero && modal run experiments/overhead_benchmark.py --n 20 --max-turns 6
# look at results/overhead_benchmark.json -> per-strategy ctx_tokens / overhead_tokens / total_tokens
```

To actually demonstrate the crossover claim in §3, run the same benchmark at
`--max-turns 6,10,15,20` and plot total tokens vs turns for naive(uncached), naive(cached),
and rezero_distilled. If the ~15-turn crossover holds, that's a *publishable* "amortizes at
long horizons" figure — exactly the future-work the paper points at.

> Bottom line for Parth: the overhead isn't a bug, it's three LLM calls per turn. It doesn't
> explode with length (it's linear), and it amortizes against an *uncached* agent past ~15
> turns — but it will never beat a *cached* agent on raw tokens. The real win is to run the
> per-turn machinery on a cheap model so the overhead stops competing with the solver's budget.
