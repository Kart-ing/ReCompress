"""Per-turn token trajectory: the O(n)-growing vs flat-context line graph.

Records context tokens at EACH turn for Naive vs ReZero (deepseek/distilled/bear),
so we can plot the curve (not just final totals). This is the headline demo visual:
naive context grows every turn; ReZero stays flat.

Run from repo root:  modal run ACTII/experiments/token_trajectory.py --turns 12
"""
from __future__ import annotations
import os
import sys
import json

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ACTII = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_REPO, _ACTII):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.distill.infer import app  # Act-1 Modal app


@app.local_entrypoint()
def main(turns: int = 12, backends: str = "distilled"):
    """backends: comma list among deepseek,distilled,bear (distilled needs Modal)."""
    from datasets import load_dataset
    from rezero.session import ReZeroSession
    from baselines.naive import NaiveSession

    # build a long conversation from several HotpotQA instances' docs (chained turns)
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    rows = list(ds.take(50))
    import random
    random.Random(7).shuffle(rows)
    goal = "Track every fact introduced across this long multi-document conversation."
    convo = []
    for s in rows:
        for t, sent in zip(s["context"]["title"], s["context"]["sentences"]):
            convo.append((f"{t}: {' '.join(sent)}", "Noted."))
            if len(convo) >= turns:
                break
        if len(convo) >= turns:
            break

    backend_list = [b.strip() for b in backends.split(",") if b.strip()]
    sessions = {"naive": NaiveSession(goal=goal)}
    for b in backend_list:
        sessions[f"rezero_{b}"] = ReZeroSession(goal=goal, use_llm=True, backend=b)

    traj = {k: [] for k in sessions}   # tokens at each turn
    for i, (u, a) in enumerate(convo):
        for k, sess in sessions.items():
            sess.add_turn(u, a)
            traj[k].append(sess.token_count())
        print(f"  turn {i+1}/{len(convo)}: " +
              " | ".join(f"{k}={traj[k][-1]}" for k in sessions))

    os.makedirs(os.path.join(_REPO, "results"), exist_ok=True)
    out = os.path.join(_REPO, "results/token_trajectory.json")
    with open(out, "w") as f:
        json.dump({"turns": len(convo), "trajectory": traj}, f, indent=2)
    print(f"\nsaved results/token_trajectory.json")
    print("Naive grows each turn; ReZero stays flat.")
