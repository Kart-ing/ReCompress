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
