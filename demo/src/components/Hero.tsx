import { useEffect, useState } from "react";
import { BUDGET } from "../data";

function useCountUp(target: number, decimals = 0, ms = 1100) {
  const [v, setV] = useState(0);
  useEffect(() => {
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / ms);
      const eased = 1 - Math.pow(1 - t, 3);
      setV(target * eased);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [target, ms]);
  return v.toFixed(decimals);
}

export default function Hero() {
  const f1 = useCountUp(56, 0);
  const tok = useCountUp(BUDGET.ratio, 1);
  const cost = useCountUp(10, 0);
  return (
    <section className="hero" id="top">
      <h1>
        Query-aware <span className="hl">rewriting</span> beats deletion —
        at <span className="hl">8.5× fewer tokens</span>.
      </h1>
      <p className="sub">
        A 1.5B model (Qwen2.5-1.5B + LoRA) distilled to compress context by reading the
        question and rewriting — then audited against itself. Every number below is replayed
        from real evaluation runs.
      </p>
      <div className="stats">
        <div className="stat">
          <div className="num">+{f1}%</div>
          <div className="lbl">F1 over bear-1.1 on HotpotQA (CI excludes 0)</div>
        </div>
        <div className="stat">
          <div className="num">{tok}×</div>
          <div className="lbl">fewer tokens than deletion ({BUDGET.ours} vs {BUDGET.bear})</div>
        </div>
        <div className="stat">
          <div className="num">~${cost}</div>
          <div className="lbl">total compute to train</div>
        </div>
      </div>
      <p className="ctx">Built in 24h · The Token Company Compression Challenge · UC Berkeley AI Hackathon 2026</p>
    </section>
  );
}
