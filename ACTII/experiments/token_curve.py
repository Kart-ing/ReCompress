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
rbd   = ReZeroSession(goal=goal, use_llm=False)
naive = NaiveSession(goal=goal)

Path("results").mkdir(exist_ok=True)
print("turn,rbd_tokens,naive_tokens,cumulative_rbd,cumulative_naive")
cum_rbd, cum_naive = 0, 0

for i, turn in enumerate(convo):
    rbd.add_turn(turn["user"],   turn["assistant"])
    naive.add_turn(turn["user"], turn["assistant"])
    r = rbd.token_count()
    n = naive.token_count()
    cum_rbd   += r
    cum_naive += n
    print(f"{i+1},{r},{n},{cum_rbd},{cum_naive}")
