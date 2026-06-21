import { useState } from "react";
import { MASK, LEAKAGE, INSTANCES } from "../data";

// pick a real leaked example with a clear answer-in-context for the mask demo
const EX = INSTANCES.find((i) => i.leaked && i.tok > 25 && i.oursF1ds > 0.5) ?? INSTANCES[0];

function FBar({ label, from, to, masked, color }: {
  label: string; from: number; to: number; masked: boolean; color: string;
}) {
  const v = masked ? to : from;
  const dropPct = Math.round((1 - to / from) * 100);
  return (
    <div className="fbar-wrap">
      <div className="fbar-label">{label}</div>
      <div className="fbar-track">
        <div className="fbar-fill" style={{ width: `${v * 100}%`, background: color }} />
      </div>
      <div className="fbar-label">
        F1 = <b style={{ color: "var(--text)" }}>{v.toFixed(3)}</b>
        {masked && <span style={{ color: "var(--warn)" }}> (−{dropPct}%)</span>}
      </div>
    </div>
  );
}

export default function Honesty() {
  const [masked, setMasked] = useState(false);

  // redact gold (case-insensitive) from the real compressed example
  const re = new RegExp(EX.gold.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "ig");
  const parts = EX.compressed.split(re);

  const legitPct = 60; // ~answer-in-context (from manual split in the paper)
  const dumpPct = 6;   // ~bare-answer dump

  return (
    <section id="honesty">
      <div className="kicker">Interactive · we audited ourselves</div>
      <h2>The honesty panel</h2>
      <p className="lead">
        The strongest thing about this paper is what we measured <i>against ourselves</i>.
        Three reveals — interact with each.
      </p>

      {/* 1. Mask the answer */}
      <div className="panel" style={{ marginBottom: 18 }}>
        <h3 style={{ margin: "0 0 8px" }}>1 · Mask-the-answer: is it compression, or extraction?</h3>
        <p style={{ color: "var(--muted)", fontSize: 14, marginTop: 0 }}>
          A real ReCompress output. Redact the gold answer span and re-solve — if F1 collapses,
          the score was carried by the literal answer being present, not by reasoning.
        </p>
        <div style={{ fontFamily: "var(--mono)", fontSize: 13, color: "var(--muted)" }}>
          Q: {EX.question}<br />gold: <b style={{ color: "var(--accent)" }}>{EX.gold}</b> · {EX.tok} tok
        </div>
        <div className="compressed">
          {parts.map((p, i) => (
            <span key={i}>
              {p}
              {i < parts.length - 1 && (
                masked
                  ? <span className="redacted">{"█".repeat(EX.gold.length)}</span>
                  : <b style={{ color: "var(--accent)" }}>{EX.gold}</b>
              )}
            </span>
          ))}
        </div>
        <button className={`action ${masked ? "on" : ""}`} style={{ marginTop: 14 }}
          onClick={() => setMasked((m) => !m)}>
          {masked ? "↺ Restore gold span" : "✂ Redact gold span"}
        </button>
        <div className="mask-grid" style={{ marginTop: 16 }}>
          <FBar label="ReCompress (abstractive)" from={MASK.ours.unmasked} to={MASK.ours.masked} masked={masked} color="var(--ours)" />
          <FBar label="bear-1.1 (extractive)" from={MASK.bear.unmasked} to={MASK.bear.masked} masked={masked} color="var(--bear)" />
        </div>
        <div className="caption warn">
          Masking drops <b>our</b> F1 by {Math.round(MASK.ours.dropPct * 100)}% vs bear's{" "}
          {Math.round(MASK.bear.dropPct * 100)}%. A large share of the win is <b>span selection</b>
          {" "}— keeping the answer-bearing span at a 3.5% budget where deletion truncates it — not
          "better reasoning." We report this rather than hide it.
        </div>
      </div>

      {/* 2. Leakage meter */}
      <div className="panel" style={{ marginBottom: 18 }}>
        <h3 style={{ margin: "0 0 8px" }}>2 · Answer-leakage, split honestly</h3>
        <p style={{ color: "var(--muted)", fontSize: 14, marginTop: 0 }}>
          The gold answer appears verbatim in <b>{LEAKAGE.nLeaked}/{LEAKAGE.total}
          {" "}({Math.round(LEAKAGE.rate * 100)}%)</b> of our compressions — but most of that is legitimate.
        </p>
        <div className="meter">
          <div className="legit" style={{ width: `${legitPct}%` }} />
          <div className="dump" style={{ width: `${dumpPct}%` }} />
          <div style={{ flex: 1, background: "var(--panel-2)" }} />
        </div>
        <div className="meter-legend">
          <span><span className="dot" style={{ background: "var(--good)" }} />~{legitPct}% answer in a real supporting sentence (good selection)</span>
          <span><span className="dot" style={{ background: "var(--warn)" }} />~{dumpPct}% bare-answer dump (the real problem)</span>
        </div>
      </div>

      {/* 3. Echidna reveal */}
      <div className="panel">
        <h3 style={{ margin: "0 0 8px" }}>3 · Our own expensive component was useless</h3>
        <p style={{ color: "var(--muted)", fontSize: 14, marginTop: 0 }}>
          The "Echidna" LLM checkpoint-trigger we built for multi-turn memory turned out to decide
          <b> "checkpoint" on 938 of 954 turns (98.3%)</b> — no real decision.
        </p>
        <div className="meter" title="938/954 decisions were 'checkpoint'">
          <div className="dump" style={{ width: "98.3%" }} />
          <div style={{ flex: 1, background: "var(--good)" }} />
        </div>
        <div className="meter-legend">
          <span><span className="dot" style={{ background: "var(--warn)" }} />98.3% "checkpoint"</span>
          <span><span className="dot" style={{ background: "var(--good)" }} />1.7% "pass"</span>
        </div>
        <div className="caption good">
          We replaced it with a free rule — <b>2.6× cheaper</b>, same answer quality. Finding your
          own dead weight and cutting it is the result, not a footnote.
        </div>
      </div>
    </section>
  );
}
