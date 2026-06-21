// Single source of truth for the demo's data — all REAL evaluation results from /results.
import crossSolver from "./cross_solver_audit.json";
import maskSym from "./mask_symmetric.json";
import sweep from "./echidna_ablation_sweep.json";
import hotpotqa from "./5bar_hotpotqa.json";
import twowiki from "./5bar_2wiki.json";
import musique from "./5bar_musique.json";
import squad from "./5bar_squad.json";

export type Bench = {
  key: string;
  label: string;
  tag: string;
  none: number;
  bear: number;
  ours: number;
  delta: number;
  ciLo: number;
  ciHi: number;
  significant: boolean;
};

function bench(raw: any, key: string, label: string, tag: string): Bench {
  const b = raw.bars;
  const d = raw.deltas_vs_bear.ours;
  return {
    key, label, tag,
    none: b.none.mean_f1,
    bear: b.bear.mean_f1,
    ours: b.ours.mean_f1,
    delta: d.mean_delta,
    ciLo: d.ci_lo,
    ciHi: d.ci_hi,
    significant: d.excludes_zero,
  };
}

export const BENCHES: Bench[] = [
  bench(hotpotqa, "hotpotqa", "HotpotQA", "in-domain"),
  bench(twowiki, "2wiki", "2WikiMultiHop", "near-in-dist"),
  bench(musique, "musique", "MuSiQue", "OOD"),
  bench(squad, "squad", "SQuAD v2", "OOD (single-hop)"),
];

// Cross-solver (HotpotQA): ours vs bear under two independent judges
export const CROSS_SOLVER = {
  deepseek: {
    ours: crossSolver.cross_solver.deepseek.ours_mean_f1,
    bear: crossSolver.cross_solver.deepseek.bear_mean_f1,
    delta: crossSolver.cross_solver.deepseek.delta,
    ciLo: crossSolver.cross_solver.deepseek.ci_lo,
    ciHi: crossSolver.cross_solver.deepseek.ci_hi,
  },
  claude: {
    ours: crossSolver.cross_solver.claude_sonnet.ours_mean_f1,
    bear: crossSolver.cross_solver.claude_sonnet.bear_mean_f1,
    delta: crossSolver.cross_solver.claude_sonnet.delta,
    ciLo: crossSolver.cross_solver.claude_sonnet.ci_lo,
    ciHi: crossSolver.cross_solver.claude_sonnet.ci_hi,
  },
};

export const LEAKAGE = {
  rate: crossSolver.answer_leakage.ours_gold_verbatim_rate, // 0.66
  nLeaked: crossSolver.answer_leakage.n_leaked,             // 33
  total: crossSolver.per_instance.length,                  // 50
};

export const MASK = {
  ours: {
    unmasked: maskSym.ours.unmasked_f1,
    masked: maskSym.ours.masked_f1,
    dropPct: maskSym.ours.drop_pct_of_unmasked,
  },
  bear: {
    unmasked: maskSym.bear.unmasked_f1,
    masked: maskSym.bear.masked_f1,
    dropPct: maskSym.bear.drop_pct_of_unmasked,
  },
};

// Crossover sweep: total tokens vs conversation length
export type SweepPoint = {
  turns: number;
  naiveCached: number;
  naiveUncached: number;
  llm: number;
  mock: number;
  mockF1: number;
  llmF1: number;
};
export const SWEEP: SweepPoint[] = (sweep.horizons as number[])
  .map((t) => {
    const b = (sweep.by_turns as any)[String(t)];
    return {
      turns: t,
      naiveCached: Math.round(b.naive_ctx_cached),
      naiveUncached: Math.round(b.naive_cum_uncached),
      llm: Math.round(b.rezero_llm.total),
      mock: Math.round(b.rezero_mock.total),
      mockF1: b.rezero_mock.f1,
      llmF1: b.rezero_llm.f1,
    };
  })
  .sort((a, b) => a.turns - b.turns);

// Per-instance examples (real compressions) for the explorer
export type Instance = {
  id: string;
  question: string;
  gold: string;
  compressed: string;
  tok: number;
  oursF1ds: number;
  oursF1cl: number;
  bearF1ds: number;
  leaked: boolean;
};
export const INSTANCES: Instance[] = crossSolver.per_instance.map((p: any) => ({
  id: p.id,
  question: p.question,
  gold: p.gold,
  compressed: p.ours_compressed,
  tok: p.ours_tok,
  oursF1ds: p.ours_f1_deepseek,
  oursF1cl: p.ours_f1_claude,
  bearF1ds: p.bear_f1_deepseek,
  leaked: p.gold_leaked_in_ours,
}));

// Headline budget facts (from the paper)
export const BUDGET = { full: 1364, bear: 409, ours: 48, ratio: 8.5 };
