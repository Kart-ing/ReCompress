import { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Legend,
} from "recharts";
import { SWEEP } from "../data";

export default function Crossover() {
  const [idx, setIdx] = useState(SWEEP.length - 1); // start at the most dramatic (T=20)
  const pt = SWEEP[idx];
  const ratio = pt.naiveUncached / pt.mock;
  const llmOvertaken = pt.llm > pt.naiveUncached;

  return (
    <section id="crossover">
      <div className="kicker">Interactive · the showpiece</div>
      <h2>Multi-turn: where flat context wins on total tokens</h2>
      <p className="lead">
        Total tokens spent (context sent to the solver <i>plus</i> per-turn compression overhead)
        as a conversation grows. We replaced the LLM checkpoint-trigger — which decided
        "checkpoint" 98% of the time — with a free rule. Drag the horizon and watch the lines cross.
      </p>

      <div className="panel">
        <ResponsiveContainer width="100%" height={360}>
          <LineChart
            data={SWEEP}
            margin={{ top: 14, right: 18, left: 4, bottom: 4 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#283041" />
            <XAxis dataKey="turns" tick={{ fill: "#8b93a7", fontSize: 12 }}
              label={{ value: "conversation length (turns)", position: "insideBottom", offset: -2, fill: "#8b93a7", fontSize: 12 }} />
            <YAxis tick={{ fill: "#8b93a7", fontSize: 12 }}
              label={{ value: "total tokens", angle: -90, position: "insideLeft", fill: "#8b93a7", fontSize: 12 }} />
            <Tooltip
              contentStyle={{ background: "#161b27", border: "1px solid #283041", borderRadius: 8 }}
              labelFormatter={(t) => `T = ${t} turns`}
              formatter={(v: number, n: string) => [v.toLocaleString(), n]}
            />
            <Legend wrapperStyle={{ fontSize: 12, fontFamily: "var(--mono)" }} />
            <ReferenceLine x={pt.turns} stroke="#ffb454" strokeDasharray="4 4" />
            <Line type="monotone" dataKey="naiveUncached" name="naive (uncached)" stroke="#ff5d5d" strokeWidth={2} dot={{ r: 3 }} />
            <Line type="monotone" dataKey="naiveCached" name="naive (KV-cached)" stroke="#e8896b" strokeWidth={2} strokeDasharray="5 4" dot={{ r: 3 }} />
            <Line type="monotone" dataKey="llm" name="RbD + LLM Echidna" stroke="#8b93a7" strokeWidth={2} dot={{ r: 3 }} />
            <Line type="monotone" dataKey="mock" name="RbD + rule Echidna" stroke="#4aa3ff" strokeWidth={3.2} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>

        <input
          type="range"
          min={0}
          max={SWEEP.length - 1}
          step={1}
          value={idx}
          onChange={(e) => setIdx(Number(e.target.value))}
        />
        <div className="slider-readout">
          <span>At <b>T = {pt.turns}</b> turns:</span>
          <span>RbD-Compress (rule) = <b>{pt.mock.toLocaleString()}</b> tok</span>
          <span>uncached naive = <b style={{ color: "var(--bear)" }}>{pt.naiveUncached.toLocaleString()}</b> tok</span>
          <span className="big">→ {ratio.toFixed(1)}× cheaper</span>
        </div>

        {llmOvertaken && (
          <div className="caption warn">
            Note: the <b>LLM-Echidna</b> version ({pt.llm.toLocaleString()} tok) is now
            <b> more expensive than the naive agent</b> — the LLM trigger was counterproductive,
            not just wasteful.
          </div>
        )}
        <div className="caption">
          Honest caveat: against a <b>KV/prefix-cached</b> naive deployment (orange dashed) we
          never win on raw tokens — caching is the equalizer. The real wins are
          context-window-bound conversations and expensive-solver setups.
        </div>
      </div>
    </section>
  );
}
