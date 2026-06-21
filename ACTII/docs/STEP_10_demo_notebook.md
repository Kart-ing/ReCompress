# Step 10 — Token Curve Graph, Demo Panel & Optional Jupyter Notebook

> **Goal:** Generate the visual demo artifacts — the token curve HTML chart and the live checkpoint inspector panel. Optionally shift into Jupyter for interactive exploration once the core engine is complete.

---

## What to build

`experiments/plot_curve.py`, `demo/panel.html`, `experiments/generate_panel.py`

---

## experiments/plot_curve.py

```python
"""
Generates demo/token_curve.html — the core demo punchline chart.
Run: python experiments/plot_curve.py
Open: demo/token_curve.html
"""
import json, sys
from pathlib import Path
sys.path.insert(0, ".")

from rezero.session import ReZeroSession
from baselines.naive import NaiveSession

convo = [
    json.loads(l)
    for l in Path("demo/scripted_convo.jsonl").read_text().splitlines()
    if l.strip()
]

goal  = convo[0].get("goal", "Research task")
rbd   = ReZeroSession(goal=goal, use_llm=True)
naive = NaiveSession(goal=goal)

turns, rbd_counts, naive_counts = [], [], []
cum_rbd, cum_naive = [], []
r_cum, n_cum = 0, 0

for i, turn in enumerate(convo):
    rbd.add_turn(turn["user"],   turn["assistant"])
    naive.add_turn(turn["user"], turn["assistant"])
    r = rbd.token_count()
    n = naive.token_count()
    r_cum += r
    n_cum += n
    turns.append(i + 1)
    rbd_counts.append(r)
    naive_counts.append(n)
    cum_rbd.append(r_cum)
    cum_naive.append(n_cum)

html = f"""<!DOCTYPE html>
<html>
<head>
  <title>RbD-Compress Token Curve</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    body {{ font-family: sans-serif; padding: 2rem; background: #fff; }}
    h2   {{ color: #2C2C7A; }}
    .charts {{ display: flex; gap: 2rem; flex-wrap: wrap; }}
    canvas {{ max-width: 600px; }}
  </style>
</head>
<body>
  <h2>RbD-Compress vs Naive — Token Cost</h2>
  <div class="charts">
    <div><h3>Per-turn token cost</h3><canvas id="per_turn"></canvas></div>
    <div><h3>Cumulative token cost</h3><canvas id="cumulative"></canvas></div>
  </div>
  <script>
    const labels = {turns};
    const rbd    = {rbd_counts};
    const naive  = {naive_counts};
    const cum_r  = {cum_rbd};
    const cum_n  = {cum_naive};

    new Chart(document.getElementById('per_turn'), {{
      type: 'line',
      data: {{ labels, datasets: [
        {{ label: 'Naive (full history)', data: naive, borderColor: '#E24B4A', tension: 0.3, fill: false, pointRadius: 4 }},
        {{ label: 'RbD-Compress (~300 flat)', data: rbd, borderColor: '#1D9E75', tension: 0.3, fill: false, pointRadius: 4 }},
      ]}},
      options: {{ plugins: {{ legend: {{ position: 'top' }} }},
        scales: {{ y: {{ beginAtZero: true }}, x: {{ title: {{ display: true, text: 'Turn' }} }} }} }}
    }});

    new Chart(document.getElementById('cumulative'), {{
      type: 'line',
      data: {{ labels, datasets: [
        {{ label: 'Naive cumulative', data: cum_n, borderColor: '#E24B4A', tension: 0.3, fill: false, pointRadius: 4 }},
        {{ label: 'RbD cumulative',   data: cum_r, borderColor: '#1D9E75', tension: 0.3, fill: false, pointRadius: 4 }},
      ]}},
      options: {{ plugins: {{ legend: {{ position: 'top' }} }},
        scales: {{ y: {{ beginAtZero: true }}, x: {{ title: {{ display: true, text: 'Turn' }} }} }} }}
    }});
  </script>
</body>
</html>"""

Path("demo").mkdir(exist_ok=True)
Path("demo/token_curve.html").write_text(html)
print("Saved: demo/token_curve.html")
```

---

## demo/panel.html

Save this file as-is. `generate_panel.py` injects real data to produce `panel_live.html`.

```html
<!DOCTYPE html>
<html>
<head>
  <title>RbD-Compress Live Panel</title>
  <style>
    body { font-family: monospace; padding: 1.5rem; background: #0f0f1a; color: #c2c0b6; }
    h1   { color: #9F97EC; font-family: sans-serif; }
    .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; margin-bottom: 1rem; }
    .box  { background: #1a1a2e; border-radius: 8px; padding: 1rem; border: 1px solid #333; }
    .box h3 { margin: 0 0 0.5rem; font-size: 0.75rem; text-transform: uppercase; }
    .trauma-box     h3 { color: #F0997B; }
    .checkpoint-box h3 { color: #5DCAA5; }
    .delta-box      h3 { color: #AFA9EC; }
    .box pre { margin: 0; white-space: pre-wrap; font-size: 0.8rem; color: #d0cfc5; }
    .token-bar { background: #1a1a2e; border-radius: 8px; padding: 1rem; border: 1px solid #333; margin-bottom: 1rem; }
    .token-bar h3 { color: #EF9F27; margin: 0 0 0.5rem; font-size: 0.75rem; text-transform: uppercase; }
    .bar-wrap  { background: #333; border-radius: 4px; height: 20px; position: relative; }
    .bar-fill  { background: #1D9E75; height: 100%; border-radius: 4px; transition: width 0.3s; }
    .bar-label { position: absolute; right: 8px; top: 2px; font-size: 0.75rem; color: #fff; }
    .controls  { margin-bottom: 1rem; }
    button { background: #3A3A8A; color: #fff; border: none; border-radius: 6px;
             padding: 0.5rem 1rem; cursor: pointer; font-size: 0.85rem; margin-right: 0.5rem; }
    button:hover { background: #534AB7; }
    #turn-label { color: #EF9F27; font-size: 0.85rem; margin-left: 0.5rem; }
  </style>
</head>
<body>
  <h1>RbD-Compress — Live Panel</h1>
  <div class="controls">
    <button onclick="prevTurn()">← Prev</button>
    <button onclick="nextTurn()">Next →</button>
    <button onclick="resetTurns()">Reset</button>
    <span id="turn-label">Turn 0 / 0</span>
  </div>
  <div class="token-bar">
    <h3>Token budget used (≤300)</h3>
    <div class="bar-wrap">
      <div class="bar-fill" id="bar-fill" style="width:0%"></div>
      <span class="bar-label" id="bar-label">0 / 300</span>
    </div>
  </div>
  <div class="grid">
    <div class="box trauma-box">
      <h3>Trauma Memory (≤50 tok)</h3>
      <pre id="trauma-content">—</pre>
    </div>
    <div class="box checkpoint-box">
      <h3>Checkpoint (≤150 tok)</h3>
      <pre id="checkpoint-content">—</pre>
    </div>
    <div class="box delta-box">
      <h3>Delta — current turn (≤100 tok)</h3>
      <pre id="delta-content">—</pre>
    </div>
  </div>
  <script>
    // DATA_PLACEHOLDER replaced by generate_panel.py
    // Defaults to [] so panel.html is safe to open before generate_panel.py runs
    const DATA = DATA_PLACEHOLDER || [];
    let currentTurn = -1;

    function render(turn) {
      if (!DATA.length) {
        document.getElementById("turn-label").textContent = "No data — run generate_panel.py first";
        return;
      }
      if (turn < 0 || turn >= DATA.length) return;
      currentTurn = turn;
      const d = DATA[turn];
      document.getElementById("trauma-content").textContent     = d.trauma     || "—";
      document.getElementById("checkpoint-content").textContent = d.checkpoint || "—";
      document.getElementById("delta-content").textContent      = d.delta      || "—";
      const pct = Math.min(100, Math.round((d.tokens / 300) * 100));
      document.getElementById("bar-fill").style.width = pct + "%";
      document.getElementById("bar-label").textContent = d.tokens + " / 300";
      document.getElementById("turn-label").textContent = `Turn ${turn + 1} / ${DATA.length}`;
    }

    function nextTurn()  { render(currentTurn + 1); }
    function prevTurn()  { render(currentTurn - 1); }
    function resetTurns(){ render(0); }
    render(0);
  </script>
</body>
</html>
```

---

## experiments/generate_panel.py

```python
"""
Generates demo/panel_live.html with real turn-by-turn data injected.
Run: python experiments/generate_panel.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, ".")

from rezero.session import ReZeroSession

convo = [
    json.loads(l)
    for l in Path("demo/scripted_convo.jsonl").read_text().splitlines()
    if l.strip()
]

goal = convo[0].get("goal", "Research task")
s    = ReZeroSession(goal=goal, use_llm=True)

data = []
for turn in convo:
    s.add_turn(turn["user"], turn["assistant"])
    prompt = s.prompt_for_solver()
    trauma_text = checkpoint_text = delta_text = ""
    for part in prompt.split("\n\n"):
        if part.startswith("[TRAUMA]"):
            trauma_text = part.replace("[TRAUMA]", "").strip()
        elif part.startswith("[CHECKPOINT]"):
            checkpoint_text = part.replace("[CHECKPOINT]", "").strip()
        elif part.startswith("[DELTA]"):
            delta_text = part.replace("[DELTA]", "").strip()
    data.append({
        "trauma":     trauma_text,
        "checkpoint": checkpoint_text,
        "delta":      delta_text,
        "tokens":     s.token_count(),
    })

Path("demo/panel_data.json").write_text(json.dumps(data, indent=2))
panel_live = Path("demo/panel.html").read_text().replace("DATA_PLACEHOLDER", json.dumps(data))
Path("demo/panel_live.html").write_text(panel_live)
print(f"Saved: demo/panel_live.html ({len(data)} turns, max tokens: {max(d['tokens'] for d in data)})")
```

---

## Run everything

```bash
python experiments/plot_curve.py      # → demo/token_curve.html
python experiments/generate_panel.py  # → demo/panel_live.html
```

---

## Optional: Jupyter Notebook

> Start only after Steps 1–7 all pass.

```bash
pip install jupyter plotly --break-system-packages
mkdir -p notebooks
jupyter notebook
```

Create `notebooks/analysis.ipynb`:

**Cell 1 — Setup**
```python
import sys; sys.path.insert(0, "..")
from rezero.session import ReZeroSession
from baselines.naive import NaiveSession
from engine.deepseek import solve
import json
from pathlib import Path
import plotly.graph_objects as go
```

**Cell 2 — Token curve**
```python
convo = [json.loads(l) for l in Path("../demo/scripted_convo.jsonl").read_text().splitlines() if l.strip()]
goal  = convo[0].get("goal", "task")
rbd   = ReZeroSession(goal=goal, use_llm=True)
naive = NaiveSession(goal=goal)
rbd_c, naive_c = [], []
for turn in convo:
    rbd.add_turn(turn["user"], turn["assistant"])
    naive.add_turn(turn["user"], turn["assistant"])
    rbd_c.append(rbd.token_count())
    naive_c.append(naive.token_count())
fig = go.Figure()
fig.add_trace(go.Scatter(y=naive_c, name="Naive",        line=dict(color="red")))
fig.add_trace(go.Scatter(y=rbd_c,   name="RbD-Compress", line=dict(color="green")))
fig.update_layout(title="Token cost per turn", xaxis_title="Turn", yaxis_title="Tokens")
fig.show()
```

**Cell 3 — §3.5 results (paste from results/microclaim.csv)**
```python
ratios    = [0.10, 0.15, 0.20, 0.25, 0.30]
variant_a = [0.0, 0.0, 0.0, 0.0, 0.0]  # ← replace with real values
variant_b = [0.0, 0.0, 0.0, 0.0, 0.0]  # ← replace with real values
fig = go.Figure()
fig.add_trace(go.Scatter(x=ratios, y=variant_a, name="Variant A (RbD)",         line=dict(color="green")))
fig.add_trace(go.Scatter(x=ratios, y=variant_b, name="Variant B (single summary)", line=dict(color="orange")))
fig.update_layout(title="§3.5: QA-F1 vs compression ratio", xaxis_title="r", yaxis_title="QA-F1")
fig.show()
```

**Cell 4 — Live checkpoint inspector**
```python
s = ReZeroSession(goal="Inspect me", use_llm=True)
for turn in convo[:5]:
    s.add_turn(turn["user"], turn["assistant"])
    prompt = s.prompt_for_solver()
    print(f"--- Turn {s.turn_count} ({s.token_count()} tokens) ---")
    print(prompt)
    print(f"Checkpoints: {s.list_checkpoints()}\n")
```

**Cell 5 — Trauma evolution**
```python
s = ReZeroSession(goal="Watch trauma grow", use_llm=True)
for i, turn in enumerate(convo):
    s.add_turn(turn["user"], turn["assistant"])
    print(f"Turn {i+1:<4} {s.trauma_extractor.get()}")
```

---

## Done when

- `demo/token_curve.html` opens showing two diverging curves
- `demo/panel_live.html` shows trauma/checkpoint/delta updating per turn
- Opening `demo/panel.html` directly shows graceful "No data" message
