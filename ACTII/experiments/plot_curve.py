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
