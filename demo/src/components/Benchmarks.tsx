import { useState } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LabelList,
} from "recharts";
import { BENCHES, CROSS_SOLVER } from "../data";

type Judge = "deepseek" | "claude";

export default function Benchmarks() {
  const [judge, setJudge] = useState<Judge>("deepseek");

  // HotpotQA row swaps to the cross-solver numbers when the judge flips;
  // the other benchmarks were only scored with DeepSeek (shown honestly).
  const cs = CROSS_SOLVER[judge];
  const rows = BENCHES.map((b) => {
    if (b.key === "hotpotqa") {
      return { ...b, bear: cs.bear, ours: cs.ours, delta: cs.delta };
    }
    return b;
  });

  return (
    <section id="benchmarks">
      <div className="kicker">Interactive · the headline</div>
      <h2>The win survives an independent judge</h2>
      <p className="lead">
        Distilled 1.5B (<b style={{ color: "var(--ours)" }}>ours</b>) vs{" "}
        <b style={{ color: "var(--bear)" }}>bear-2</b> vs full context, QA-F1, 50 instances each.
        The natural objection: our teacher and solver are both DeepSeek. So flip the judge to a
        model independent of both — Claude Sonnet — and watch the gap.
      </p>

      <div className="panel">
        <div className="toggle-row">
          <span style={{ fontFamily: "var(--mono)", fontSize: 13, color: "var(--muted)" }}>
            Solver / judge:
          </span>
          <div className="seg">
            <button className={judge === "deepseek" ? "active" : ""} onClick={() => setJudge("deepseek")}>
              DeepSeek (in-family)
            </button>
            <button className={judge === "claude" ? "active" : ""} onClick={() => setJudge("claude")}>
              Claude Sonnet (independent)
            </button>
          </div>
        </div>

        <ResponsiveContainer width="100%" height={340}>
          <BarChart data={rows} margin={{ top: 24, right: 10, left: -10, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#283041" vertical={false} />
            <XAxis dataKey="label" tick={{ fill: "#8b93a7", fontSize: 12 }} />
            <YAxis domain={[0, 1]} tick={{ fill: "#8b93a7", fontSize: 12 }} />
            <Tooltip
              contentStyle={{ background: "#161b27", border: "1px solid #283041", borderRadius: 8 }}
              labelStyle={{ color: "#e6e9ef" }}
              formatter={(v: number, n: string) => [v.toFixed(3), n]}
            />
            <Bar dataKey="none" name="full ctx" fill="#6b7280" radius={[4, 4, 0, 0]} isAnimationActive />
            <Bar dataKey="bear" name="bear-2" fill="#ff5d5d" radius={[4, 4, 0, 0]} isAnimationActive />
            <Bar dataKey="ours" name="ReCompress" fill="#4aa3ff" radius={[4, 4, 0, 0]} isAnimationActive>
              <LabelList dataKey="ours" position="top" formatter={(v: number) => v.toFixed(2)}
                style={{ fill: "#4aa3ff", fontSize: 11, fontFamily: "var(--mono)" }} />
            </Bar>
          </BarChart>
        </ResponsiveContainer>

        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 8 }}>
          {BENCHES.map((b) => (
            <span key={b.key} className={`badge ${b.significant ? "sig" : "ns"}`}>
              {b.label}: {b.significant ? "✓ CI excl. 0" : "◐ n.s."}
            </span>
          ))}
        </div>

        <div className="caption good">
          {judge === "deepseek" ? (
            <>HotpotQA: ours <b>{cs.ours.toFixed(3)}</b> vs bear <b>{cs.bear.toFixed(3)}</b> —
              Δ <b>+{cs.delta.toFixed(3)}</b>. Now flip the judge →</>
          ) : (
            <>Under an <b>independent</b> solver the gap is essentially unchanged: ours{" "}
              <b>{cs.ours.toFixed(3)}</b> vs bear <b>{cs.bear.toFixed(3)}</b>, Δ{" "}
              <b>+{cs.delta.toFixed(3)}</b> (vs +0.285 in-family). The win is not a teacher↔solver artifact.</>
          )}
        </div>
      </div>
      <p className="lead" style={{ marginTop: 16, fontSize: 14 }}>
        Significant on the multi-hop, distractor-heavy benchmarks (HotpotQA, 2Wiki). MuSiQue and
        SQuAD are positive but not significant at n=50 — we claim only what the data shows.
      </p>
    </section>
  );
}
