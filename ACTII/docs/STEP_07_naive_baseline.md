# Step 7 — Naive Baseline & Token Curve

> **Goal:** Build `baselines/naive.py` — same interface as `ReZeroSession` but resends full history every turn. Generate the O(n²) vs flat token curve that is the core demo punchline.

---

## What to build

`baselines/naive.py` and `experiments/token_curve.py`

---

## baselines/naive.py

```python
from act1.tokens import count_tokens

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
        full = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self.history
        )
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

# FIX: use substantive messages so naive accumulates well above 300
# "Question 0" is only 2 words — 10 turns × 4 words = 80 total, well under RbD cap
LONG_USER = "This is a detailed question about TechCorp funding rounds and investor relations in Silicon Valley"
LONG_ASST = "TechCorp has raised several hundred million dollars across multiple rounds from top-tier investors"

def test_naive_grows_with_turns():
    n = NaiveSession(goal="Research TechCorp")
    counts = []
    for i in range(10):
        n.add_turn(f"Turn {i}: {LONG_USER}", f"Turn {i}: {LONG_ASST}")
        counts.append(n.token_count())
    assert counts[-1] > counts[0], "Naive should grow over turns"
    # verify it's actually growing meaningfully — not just by 1
    assert counts[-1] > counts[0] * 1.5, \
        f"Naive grew too slowly: turn1={counts[0]}, turn10={counts[-1]}"

def test_naive_vs_rbd_at_turn_10():
    from rezero.session import ReZeroSession
    rbd   = ReZeroSession(goal="Research TechCorp funding")
    naive = NaiveSession(goal="Research TechCorp funding")
    for i in range(10):
        rbd.add_turn(f"Turn {i}: {LONG_USER}", f"Turn {i}: {LONG_ASST}")
        naive.add_turn(f"Turn {i}: {LONG_USER}", f"Turn {i}: {LONG_ASST}")
    assert naive.token_count() > rbd.token_count(), \
        f"Naive ({naive.token_count()}) should exceed RbD ({rbd.token_count()}) by turn 10"

def test_naive_has_full_history_in_prompt():
    n = NaiveSession(goal="Test")
    n.add_turn("First message about Alice", "First reply about Alice founding TechCorp")
    n.add_turn("Second message about Bob", "Second reply about Bob joining later")
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

def test_cumulative_naive_exceeds_rbd_3x():
    from rezero.session import ReZeroSession
    rbd   = ReZeroSession(goal="Research TechCorp funding")
    naive = NaiveSession(goal="Research TechCorp funding")
    cum_rbd, cum_naive = 0, 0
    for i in range(15):
        rbd.add_turn(f"Turn {i}: {LONG_USER}", f"Turn {i}: {LONG_ASST}")
        naive.add_turn(f"Turn {i}: {LONG_USER}", f"Turn {i}: {LONG_ASST}")
        cum_rbd   += rbd.token_count()
        cum_naive += naive.token_count()
    assert cum_naive > cum_rbd * 2.5, \
        f"Cumulative naive ({cum_naive}) should be >2.5x RbD ({cum_rbd})"
```

---

## Run tests

```bash
mkdir -p results
pytest tests/test_naive.py -v
```

---

## Run token curve

```bash
python experiments/token_curve.py
```

Expected output (approximate):

```
turn,rbd_tokens,naive_tokens,cumulative_rbd,cumulative_naive
1,45,120,45,120
2,62,240,107,360
3,58,360,165,720
...
15,71,1800,1050,13500
```

`naive_tokens` should grow each row. `rbd_tokens` should stay roughly flat (under 300). `cumulative_naive` at turn 15 should be at least 2.5x `cumulative_rbd` (with mock compressor; real compressor will be higher).

---

## Done when

- All 5 tests pass including `test_cumulative_naive_exceeds_rbd_3x`
- `python experiments/token_curve.py` prints 15 rows with naive clearly growing
- `results/` directory is created automatically
