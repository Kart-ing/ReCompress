# Step 5 — Echidna Trigger

> **Goal:** Build `rezero/echidna.py`. Echidna reads trauma memory first, then decides: `checkpoint`, `revert`, or `pass`. Mock uses token threshold + cadence. Real version calls `engine/deepseek.py`. Wire into `ReZeroSession.add_turn()`.

---

## What to build

`rezero/echidna.py` — then update `rezero/session.py`

---

## rezero/echidna.py

```python
import json
from dataclasses import dataclass
from engine.tokens import count_tokens

TOKEN_THRESHOLD = 800   # total history tokens before forcing checkpoint
TURN_CADENCE    = 5     # force checkpoint every N turns if no other trigger

@dataclass
class EchidnaDecision:
    action: str           # "checkpoint" | "revert" | "pass"
    revert_to: int | None
    reason: str
    urgency: str          # "low" | "medium" | "high"


class Echidna:
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm

    def decide(
        self,
        history: list[dict],
        trauma: str,
        checkpoint_summary: str,
        turns_since_checkpoint: int,
        available_checkpoints: list[int],
    ) -> EchidnaDecision:
        if self.use_llm:
            return self._llm_decide(
                history, trauma, checkpoint_summary,
                turns_since_checkpoint, available_checkpoints
            )
        return self._mock_decide(history, turns_since_checkpoint)

    # ── MOCK ────────────────────────────────────────────────────────────────
    def _mock_decide(self, history: list[dict], turns_since_checkpoint: int) -> EchidnaDecision:
        total_tokens = count_tokens(" ".join(m["content"] for m in history))
        if total_tokens > TOKEN_THRESHOLD:
            return EchidnaDecision("checkpoint", None, "token threshold exceeded", "high")
        if turns_since_checkpoint >= TURN_CADENCE:
            return EchidnaDecision("checkpoint", None, f"cadence: every {TURN_CADENCE} turns", "medium")
        return EchidnaDecision("pass", None, "within budget", "low")

    # ── REAL (DeepSeek) ──────────────────────────────────────────────────────
    def _llm_decide(
        self,
        history: list[dict],
        trauma: str,
        checkpoint_summary: str,
        turns_since_checkpoint: int,
        available_checkpoints: list[int],
    ) -> EchidnaDecision:
        from engine.deepseek import call

        system = """You are Echidna, the Witch of Greed. You observe conversations with
perfect clarity and decide when knowledge must be crystallized.
You receive: the current turn, trauma memory, and checkpoint summary.
Return JSON only:
{"action": "checkpoint"|"revert"|"pass",
 "revert_to": <id>|null, "reason": "...", "urgency": "low"|"medium"|"high"}
CHECKPOINT when: topic shifts, a reasoning hop is resolved, token budget near limit.
REVERT when: contradiction detected, reasoning chain has collapsed.
PASS otherwise. Always read trauma memory before deciding."""

        recent = history[-4:] if len(history) >= 4 else history
        recent_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)
        prompt = f"""Trauma memory: {trauma}
Checkpoint summary: {checkpoint_summary}
Turns since last checkpoint: {turns_since_checkpoint}
Available checkpoint IDs for revert: {available_checkpoints}
Recent conversation:
{recent_text}"""

        raw = call(system, prompt, max_tokens=80)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return EchidnaDecision("pass", None, "parse error — defaulting to pass", "low")

        return EchidnaDecision(
            action    = parsed.get("action",    "pass"),
            revert_to = parsed.get("revert_to"),
            reason    = parsed.get("reason",    ""),
            urgency   = parsed.get("urgency",   "low"),
        )
```

---

## Update rezero/session.py

### 1. Add import

```python
from rezero.echidna import Echidna
```

### 2. Add to `__init__`

```python
self.echidna = Echidna(use_llm=use_llm)
```

### 3. Replace `add_turn` entirely

```python
def add_turn(self, user: str, assistant: str) -> None:
    # 1. update trauma first — Echidna reads it next
    self.trauma_extractor.update(user)
    self.trauma_extractor.update(assistant)

    # 2. append to history
    self.history.append({"role": "user",      "content": user})
    self.history.append({"role": "assistant", "content": assistant})
    self.turn_count += 1

    # 3. Echidna reads trauma and decides
    decision = self.echidna.decide(
        history                = self.history,
        trauma                 = self.trauma_extractor.get(),
        checkpoint_summary     = self.checkpoint_stack.summary(),
        turns_since_checkpoint = self.turns_since_checkpoint,
        available_checkpoints  = self.checkpoint_stack.list_ids(),
    )

    if decision.action == "checkpoint":
        new_cp = self.checkpoint_builder.build(
            self.history, self.trauma_extractor.get()
        )
        self.checkpoint_stack.push(new_cp, self.turn_count, reason=decision.reason)
        self.turns_since_checkpoint = 0
    elif decision.action == "revert" and decision.revert_to is not None:
        self.checkpoint_stack.revert_to(decision.revert_to)
        self.turns_since_checkpoint = 0
    else:
        self.turns_since_checkpoint += 1
```

### 4. Update `_get_checkpoint` — pure read, no side effects

```python
def _get_checkpoint(self) -> str:
    return self.checkpoint_stack.current()
```

---

## Tests — `tests/test_echidna.py`

```python
from rezero.echidna import Echidna

def test_mock_pass_on_short_history():
    e = Echidna(use_llm=False)
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    d = e.decide(history, "Alice", "no checkpoint", 1, [])
    assert d.action == "pass"

def test_mock_checkpoint_on_token_threshold():
    e = Echidna(use_llm=False)
    history = [{"role": "user", "content": "word " * 900}]
    d = e.decide(history, "Alice", "cp", 1, [])
    assert d.action == "checkpoint"
    assert d.urgency == "high"

def test_mock_checkpoint_on_cadence():
    e = Echidna(use_llm=False)
    history = [{"role": "user", "content": "short"}, {"role": "assistant", "content": "reply"}]
    d = e.decide(history, "Alice", "cp", turns_since_checkpoint=5, available_checkpoints=[])
    assert d.action == "checkpoint"
    assert d.urgency == "medium"

def test_decision_fields_valid():
    e = Echidna(use_llm=False)
    d = e.decide([], "", "", 0, [])
    assert d.action  in ("checkpoint", "revert", "pass")
    assert d.urgency in ("low", "medium", "high")
    assert isinstance(d.reason, str)

def test_echidna_wired_into_session():
    from rezero.session import ReZeroSession
    s = ReZeroSession(goal="Test Alice at Tech Corp", use_llm=False)
    for i in range(6):
        s.add_turn(f"Question {i} about Tech Corp", f"Answer {i} about Tech Corp")
    assert len(s.checkpoint_stack.list_ids()) >= 1

def test_get_checkpoint_is_pure_read():
    from rezero.session import ReZeroSession
    s = ReZeroSession(goal="Test", use_llm=False)
    s.add_turn("hello", "hi")
    before = s.checkpoint_stack.list_ids()[:]
    _ = s._get_checkpoint()
    assert s.checkpoint_stack.list_ids() == before
```

---

## Run tests

```bash
pytest tests/test_echidna.py -v
```

---

## Done when

- All 6 tests pass
- `_get_checkpoint` has zero side effects
- After 5+ turns, at least one checkpoint exists in the stack
