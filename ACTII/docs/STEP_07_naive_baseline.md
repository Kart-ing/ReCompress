# Step 7 — Naive Baseline & Token Curve

> **Goal:** Build `baselines/naive.py` — same interface as `ReZeroSession` but resends full history every turn. Generate the O(n²) vs flat token curve that is the core demo punchline.

---

## What to build

`baselines/naive.py` and `experiments/token_curve.py`

---

## baselines/naive.py

```python
from engine.tokens import count_tokens

class NaiveSession:
    """
    Baseline: resends full conversation history every turn.
    Token cost grows linearly per turn → O(n²) cumulative.
    Same interface as ReZeroSession so they can be swapped in experiments.
    """
    def __init__(self, goal: str):
        self.goal = goal
        self.history: list[dict] = []

    def add_turn(self, user: str, assistant: str) -> None:
        self.history.append({"role": "user",      "content": user})
        self.history.append({"role": "assistant", "content": assistant})

    def prompt_for_solver(self) -> str:
        full = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.history)
        return f"[GOAL]\n{self.goal}\n\n[HISTORY]\n{full}"

    def token_count(self) -> int:
        return count_tokens(self.prompt_for_solver())
```

---

## experiments/token_curve.py

```python
"""
Prints a CSV of per-turn token counts for RbD-Compress vs Naive.
Run: python experiments/token_curve.py
Pipe to file: python experiments/token_curve.py > results/token_curve.csv
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
```

---

## Tests — `tests/test_naive.py`

```python
from baselines.naive import NaiveSession

LONG_USER = "This is a detailed question about Tech Corp funding rounds and investor relations in Silicon Valley"
LONG_ASST = "Tech Corp has raised several hundred million dollars across multiple rounds from top-tier investors"
N_TURNS = 20  # change freely — tests adapt automatically

def test_naive_grows_with_turns():
    n = NaiveSession(goal="Research Tech Corp")
    counts = []
    for i in range(N_TURNS):
        n.add_turn(f"Turn {i}: {LONG_USER}", f"Turn {i}: {LONG_ASST}")
        counts.append(n.token_count())
    assert counts[-1] > counts[0]
    assert counts[-1] > counts[0] * 1.5, \
        f"Naive grew too slowly: first={counts[0]}, last={counts[-1]}"

def test_naive_vs_rbd_after_many_turns():
    from rezero.session import ReZeroSession
    rbd   = ReZeroSession(goal="Research Tech Corp funding")
    naive = NaiveSession(goal="Research Tech Corp funding")
    for i in range(N_TURNS):
        rbd.add_turn(f"Turn {i}: {LONG_USER}", f"Turn {i}: {LONG_ASST}")
        naive.add_turn(f"Turn {i}: {LONG_USER}", f"Turn {i}: {LONG_ASST}")
    assert naive.token_count() > rbd.token_count(), \
        f"Naive ({naive.token_count()}) should exceed RbD ({rbd.token_count()}) after {N_TURNS} turns"

def test_naive_has_full_history_in_prompt():
    n = NaiveSession(goal="Test")
    n.add_turn("First message about Alice", "First reply about Alice founding Tech Corp")
    n.add_turn("Second message about Bob",  "Second reply about Bob joining later")
    prompt = n.prompt_for_solver()
    assert "First message"  in prompt
    assert "Second message" in prompt

def test_naive_interface_matches_rbd():
    from rezero.session import ReZeroSession
    rbd   = ReZeroSession(goal="test")
    naive = NaiveSession(goal="test")
    rbd.add_turn("q", "a")
    naive.add_turn("q", "a")
    assert isinstance(rbd.prompt_for_solver(),   str)
    assert isinstance(naive.prompt_for_solver(), str)
    assert isinstance(rbd.token_count(),         int)
    assert isinstance(naive.token_count(),       int)

def test_cumulative_naive_exceeds_rbd_2_5x():
    from rezero.session import ReZeroSession
    rbd   = ReZeroSession(goal="Research Tech Corp funding")
    naive = NaiveSession(goal="Research Tech Corp funding")
    cum_rbd, cum_naive = 0, 0
    for i in range(N_TURNS):
        rbd.add_turn(f"Turn {i}: {LONG_USER}", f"Turn {i}: {LONG_ASST}")
        naive.add_turn(f"Turn {i}: {LONG_USER}", f"Turn {i}: {LONG_ASST}")
        cum_rbd   += rbd.token_count()
        cum_naive += naive.token_count()
    assert cum_naive > cum_rbd * 2.5, \
        f"Cumulative naive ({cum_naive}) should be >2.5x RbD ({cum_rbd}) after {N_TURNS} turns"
```

---

## Run tests

```bash
mkdir -p results
pytest tests/test_naive.py -v
```

## Run token curve

```bash
python experiments/token_curve.py
```

`naive_tokens` should grow each row. `rbd_tokens` should stay flat under 300.

---

## Done when

- All 5 tests pass
- `python experiments/token_curve.py` prints one row per turn with naive clearly growing
- `results/` directory created automatically
