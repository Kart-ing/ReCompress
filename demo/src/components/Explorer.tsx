import { useMemo, useState } from "react";
import { INSTANCES } from "../data";

export default function Explorer() {
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);

  const filtered = useMemo(() => {
    const needle = q.toLowerCase().trim();
    return INSTANCES.map((it, i) => ({ it, i }))
      .filter(({ it }) => !needle || it.question.toLowerCase().includes(needle) || it.gold.toLowerCase().includes(needle));
  }, [q]);

  const cur = INSTANCES[sel];

  return (
    <section id="explorer">
      <div className="kicker">Interactive · the raw data</div>
      <h2>Read the real compressions</h2>
      <p className="lead">
        All 50 HotpotQA instances from the cross-solver audit. Pick one to see the actual
        compressed text our 1.5B produced, its token count, and how each solver scored it. Nothing
        synthetic.
      </p>

      <div className="panel">
        <div className="explorer">
          <div>
            <input
              type="text"
              placeholder="search question or answer…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              style={{
                width: "100%", padding: "9px 12px", marginBottom: 10,
                background: "var(--bg-soft)", border: "1px solid var(--border)",
                borderRadius: 9, color: "var(--text)", fontFamily: "var(--mono)", fontSize: 13,
              }}
            />
            <div className="inst-list">
              {filtered.map(({ it, i }) => (
                <div key={it.id} className={`inst-item ${i === sel ? "active" : ""}`} onClick={() => setSel(i)}>
                  {it.question.length > 70 ? it.question.slice(0, 70) + "…" : it.question}
                </div>
              ))}
              {filtered.length === 0 && (
                <div className="inst-item" style={{ color: "var(--muted)" }}>no matches</div>
              )}
            </div>
          </div>

          <div className="inst-detail">
            <div className="kv"><span className="k">question</span><span>{cur.question}</span></div>
            <div className="kv"><span className="k">gold</span><span style={{ color: "var(--accent)" }}>{cur.gold}</span></div>
            <div className="kv">
              <span className="k">ours · {cur.tok} tok</span>
              <span>
                <span className={`pill ${cur.oursF1ds > 0.5 ? "ok" : "no"}`}>DeepSeek F1 {cur.oursF1ds.toFixed(2)}</span>{" "}
                <span className={`pill ${cur.oursF1cl > 0.5 ? "ok" : "no"}`}>Claude F1 {cur.oursF1cl.toFixed(2)}</span>{" "}
                <span className="pill">bear F1 {cur.bearF1ds.toFixed(2)}</span>{" "}
                <span className={`pill ${cur.leaked ? "no" : "ok"}`}>{cur.leaked ? "gold present" : "no verbatim gold"}</span>
              </span>
            </div>
            <div className="compressed">{cur.compressed}</div>
          </div>
        </div>
      </div>
    </section>
  );
}
