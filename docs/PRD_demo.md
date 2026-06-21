# PRD — ReCompress Interactive Demo (research-paper frontend)

**Goal:** A polished, interactive web demo that lets judges *explore the real results* of the
ReCompress paper — making three findings tangible: the cross-solver-robust win, the multi-turn
crossover, and our honest self-critique. Static (no backend), bound to the real result JSONs.

**Audience:** Hackathon judges (The Token Company / UC Berkeley), Devpost visitors, the README "live demo" link.

**Non-goals:** No live model inference (no Modal/DeepSeek calls, no API keys in the frontend).
No user-supplied text in v1. Everything replays *real saved data* — never fabricated.

---

## Stack & deployment
- **Vite + React + TypeScript**, charts via **Recharts** (clean, animatable, small).
- Lives in `demo/` in the repo. Build output deploys to **GitHub Pages** (or Vercel — Parth's call).
- Data: the real result JSONs copied into `demo/src/data/` at build time (no fetch from disk at runtime). Source files:
  - `cross_solver_audit.json` — 5-bar toggle, leakage, per-instance text
  - `mask_symmetric.json` — honesty/mask panel
  - `echidna_ablation_sweep.json` — crossover slider
  - `5bar_distilled_{hotpotqa,2wiki,musique,squad}.json` — benchmark bars
- **Single-page, scroll-driven story** with a sticky top nav (Headline → Cross-solver → Crossover → Honesty). Design-forward: custom palette, smooth transitions, readable on a projector.

---

## Section 1 — Hero / headline
- One-line thesis: *"Query-aware rewriting beats deletion — at ~8.5× fewer tokens. A 1.5B model, audited against itself."*
- Three animated stat counters (count up on load): **+56% F1** (HotpotQA), **8.5× fewer tokens** (48 vs 409), **~$10** total compute.
- Subtle: "Built in 24h for The Token Company Compression Challenge." CTA scroll cue.

## Section 2 — The 5-bar benchmark, with cross-solver toggle  *(interactive #1)*
- Grouped bar chart: per benchmark (HotpotQA / 2Wiki / MuSiQue / SQuAD), bars for `none` / `bear` / `ours`.
- **Toggle: "Judge: DeepSeek ↔ Claude Sonnet."** Flipping it re-animates `ours`/`bear` to the cross-solver numbers (HotpotQA only has both; others show "DeepSeek only" gracefully).
  - DeepSeek: ours 0.737 / bear 0.452 (Δ +0.285). Claude: ours 0.587 / bear 0.299 (Δ +0.288).
  - Caption updates live: *"The +0.29 gap survives an independent judge — not a teacher↔solver artifact."*
- Significance badges: HotpotQA & 2Wiki = ✅ CI excludes 0; MuSiQue & SQuAD = ◐ n.s. (honest).

## Section 3 — The multi-turn crossover  *(interactive #2, the showpiece)*
- Line chart, x = conversation length (6/10/15/20 turns), y = total tokens (ctx + overhead).
  Four lines: naive-uncached, naive-cached, RbD+LLM-Echidna, **RbD+rule-Echidna** (highlighted).
- **Slider: drag the horizon (6→20).** A vertical marker sweeps; a live readout shows
  "At T=N, RbD-Compress (rule) = X tok vs uncached naive Y tok → **Z× cheaper**."
  (T=6 → 1.4×, T=20 → 4.2×.)
- Annotation that appears when the slider passes ~11: *"LLM-Echidna gets so costly it's overtaken by the naive agent — the LLM trigger was counterproductive."*
- Honest caveat callout: *"Against a KV/prefix-cached agent we never win on raw tokens — caching is the equalizer."*

## Section 4 — The honesty panel  *(interactive #3 — the differentiator)*
Three small interactive reveals, side by side or stacked:
1. **Mask-the-answer:** show a real compression; a **"Redact gold span"** button blanks the
   answer and re-animates the F1 bar from 0.737 → 0.256 (ours, −65%) vs bear 0.452 → 0.314 (−31%).
   Caption: *"Much of the win is span-selection, not reasoning — we measured it on ourselves."*
2. **Answer-leakage meter:** a dial/bar showing 66% verbatim-gold, split into ~60% legitimate
   (answer in a real supporting sentence) vs ~6% bare dump. Hover for the split definition.
3. **"Echidna is 98% checkpoint":** a compact viz (938/954) with one line — *"Our own expensive
   trigger was making no real decision; we replaced it with a free rule (2.6× cheaper)."*

## Section 5 (optional, stretch) — Per-instance explorer
- A searchable list of the 50 HotpotQA instances: click one → shows question, gold answer,
  ours' compressed text + token count, ours vs bear F1 (both solvers), and a "gold leaked?" flag.
- Lets judges read *real* compressions. Pulls straight from `cross_solver_audit.json` per_instance.

---

## Footer
- Links: GitHub repo, the paper PDF, Devpost. "Cite this" (from CITATION.cff). Authors.
- "All numbers replayed from real evaluation runs in `/results` — nothing synthetic."

---

## Build phases
1. Scaffold Vite+React+TS, palette/typography, copy JSON data in, sticky-nav scroll shell.
2. Section 2 (5-bar + cross-solver toggle) + Section 3 (crossover slider) — the two charts.
3. Section 4 (honesty panel) + hero counters.
4. (Stretch) Section 5 explorer. Polish, animations, mobile/projector check, README link.

## Open questions for review
- GitHub Pages vs Vercel for deploy? (affects base path config)
- Include the stretch per-instance explorer in v1, or ship the 4 core sections first?
- House style: dark or light theme? (suggest dark, terminal-ish, with one accent color)
