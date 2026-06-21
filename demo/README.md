# ReCompress — interactive demo

A static (no-backend) frontend for the ReCompress paper. Every number is replayed from the
real evaluation JSONs in `../results/` (copied into `src/data/`). Five sections:

1. **Hero** — animated headline stats
2. **Benchmarks** — 5-bar chart with a DeepSeek ↔ Claude Sonnet **cross-solver toggle**
3. **Crossover** — multi-turn total-token **slider** (drag conversation length 6→20 turns)
4. **Honesty** — interactive mask-the-answer, answer-leakage split, Echidna-98% reveal
5. **Explorer** — search + read the 50 real HotpotQA compressions

## Run locally
```bash
cd demo
npm install
npm run dev      # http://localhost:5173/ReCompress/
```

## Build + deploy
```bash
npm run build    # outputs to demo/dist/
```

- **GitHub Pages:** the `base` in `vite.config.ts` is `/ReCompress/` (matches
  `https://<user>.github.io/ReCompress/`). Publish `demo/dist/` (e.g. via a Pages action or the
  `gh-pages` branch).
- **Vercel / Netlify (served at root):** change `base` in `vite.config.ts` to `"/"` and deploy
  the `demo/` directory (build command `npm run build`, output `dist`).

## Updating the data
The JSONs in `src/data/` are copies of `../results/*.json`. If results change, re-copy:
```bash
cp ../results/{cross_solver_audit,mask_symmetric,echidna_ablation_sweep}.json src/data/
for b in hotpotqa 2wiki musique squad; do cp ../results/5bar_distilled_$b.json src/data/5bar_$b.json; done
```
