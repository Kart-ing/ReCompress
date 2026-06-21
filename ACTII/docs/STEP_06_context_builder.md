# Step 6 — Context Builder & Full Pipeline Integration

> **Goal:** Build `rezero/context_builder.py`. Wire all components into `ReZeroSession`. Verify token count stays flat at ≤300 regardless of conversation length.

---

## What to build

`rezero/context_builder.py` — then update `rezero/session.py`

---

## rezero/context_builder.py

```python
from engine.tokens import count_tokens

TRAUMA_CAP     = 50
CHECKPOINT_CAP = 150
DELTA_CAP      = 100
TOTAL_CAP      = 300

class ContextBuilder:
    def build(self, trauma: str, checkpoint: str, delta: str) -> str:
        """
        Assemble final prompt. Enforces all hard caps.
        Trim order if total exceeds 300: checkpoint first, then trauma.
        """
        trauma     = self._enforce(trauma,     TRAUMA_CAP)
        checkpoint = self._enforce(checkpoint, CHECKPOINT_CAP)
        delta      = self._enforce(delta,      DELTA_CAP)

        prompt = self._assemble(trauma, checkpoint, delta)

        while count_tokens(prompt) > TOTAL_CAP:
            words = checkpoint.split()
            if not words:
                break
            checkpoint = " ".join(words[:-5])
            prompt = self._assemble(trauma, checkpoint, delta)

        return prompt

    def _assemble(self, trauma: str, checkpoint: str, delta: str) -> str:
        return f"[TRAUMA]\n{trauma}\n\n[CHECKPOINT]\n{checkpoint}\n\n[DELTA]\n{delta}"

    def _enforce(self, text: str, cap: int) -> str:
        words = text.split()
        while count_tokens(" ".join(words)) > cap and words:
            words.pop()
        return " ".join(words)
```

---

## Update rezero/session.py

### 1. Add import

```python
from rezero.context_builder import ContextBuilder
```

### 2. Add to `__init__`

```python
self.context_builder = ContextBuilder()
```

### 3. Replace `prompt_for_solver`

```python
def prompt_for_solver(self) -> str:
    return self.context_builder.build(
        trauma     = self.trauma_extractor.get(),
        checkpoint = self._get_checkpoint(),
        delta      = self._get_delta(),
    )
```

### 4. Remove from session.py

Delete `_assemble` — `ContextBuilder` owns it now.
Keep `_enforce` and `_get_delta` — still used internally.

---

## Tests — `tests/test_budget.py`

```python
import json
from pathlib import Path
from rezero.session import ReZeroSession
from engine.tokens import count_tokens

def _load_convo():
    candidates = [
        Path("demo/scripted_convo.jsonl"),
        Path("../demo/scripted_convo.jsonl"),
        Path(__file__).parent.parent / "demo" / "scripted_convo.jsonl",
    ]
    for p in candidates:
        if p.exists():
            return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    raise FileNotFoundError("scripted_convo.jsonl not found")

def test_token_count_never_exceeds_300():
    convo = _load_convo()
    s = ReZeroSession(goal=convo[0].get("goal", "Research task"), use_llm=False)
    counts = []
    for turn in convo:
        s.add_turn(turn["user"], turn["assistant"])
        counts.append(s.token_count())
    assert all(c <= 300 for c in counts), f"Exceeded 300: {list(enumerate(counts))}"

def test_token_count_is_flat():
    convo = _load_convo()
    s = ReZeroSession(goal=convo[0].get("goal", "Research task"), use_llm=False)
    counts = []
    for turn in convo:
        s.add_turn(turn["user"], turn["assistant"])
        counts.append(s.token_count())
    n = len(counts)
    quarter = max(1, n // 4)
    early_avg = sum(counts[:quarter]) / quarter
    late_avg  = sum(counts[-quarter:]) / quarter
    assert abs(late_avg - early_avg) < 80, \
        f"Not flat — early: {early_avg:.0f}, late: {late_avg:.0f}"

def test_trauma_section_always_present():
    convo = _load_convo()
    s = ReZeroSession(goal=convo[0].get("goal", "Research task"), use_llm=False)
    for turn in convo:
        s.add_turn(turn["user"], turn["assistant"])
    prompt = s.prompt_for_solver()
    trauma_section = prompt.split("[CHECKPOINT]")[0].replace("[TRAUMA]", "").strip()
    assert len(trauma_section) > 0

def test_all_three_sections_present_every_turn():
    convo = _load_convo()
    s = ReZeroSession(goal=convo[0].get("goal", "Research task"), use_llm=False)
    for turn in convo:
        s.add_turn(turn["user"], turn["assistant"])
        prompt = s.prompt_for_solver()
        assert "[TRAUMA]"     in prompt
        assert "[CHECKPOINT]" in prompt
        assert "[DELTA]"      in prompt

def test_delta_always_contains_latest_turn():
    convo = _load_convo()
    s = ReZeroSession(goal=convo[0].get("goal", "Research task"), use_llm=False)
    for turn in convo[:-1]:
        s.add_turn(turn["user"], turn["assistant"])
    last = convo[-1]
    s.add_turn(last["user"], last["assistant"])
    prompt = s.prompt_for_solver()
    delta_section = prompt.split("[DELTA]")[1].strip()
    last_words = last["user"].split()[:3]
    assert any(w in delta_section for w in last_words)

def test_context_builder_owns_assembly():
    s = ReZeroSession(goal="Test", use_llm=False)
    assert not hasattr(s, "_assemble"), \
        "_assemble should be in ContextBuilder, not session"
```

---

## Run tests

```bash
pytest tests/test_budget.py -v
```

## Full pipeline smoke test

```bash
python - <<'EOF'
import json
from pathlib import Path
from rezero.session import ReZeroSession

convo = [json.loads(l) for l in Path("demo/scripted_convo.jsonl").read_text().splitlines() if l.strip()]
s = ReZeroSession(goal=convo[0]["goal"], use_llm=False)
print(f"{'Turn':<6} {'Tokens':<8} {'CPs'}")
for i, turn in enumerate(convo):
    s.add_turn(turn["user"], turn["assistant"])
    print(f"{i+1:<6} {s.token_count():<8} {s.list_checkpoints()}")
EOF
```

---

## Done when

- All 6 tests pass
- Smoke test: token column stays ≤300 for every row
- `session.py` has no `_assemble` method
