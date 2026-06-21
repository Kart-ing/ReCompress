# Step 5 — Echidna Trigger

> **Goal:** Build `rezero/echidna.py`. Echidna reads trauma memory first, then returns a JSON decision: `checkpoint`, `revert`, or `pass`. Mock uses token threshold + turn cadence. Real version: DeepSeek Flash call. Wire into `ReZeroSession.add_turn()` and fully replace the Step 4 scheduled checkpoint logic.

---

## What to build

`rezero/echidna.py` — then update `rezero/session.py`

---

## rezero/echidna.py

```python
import json
from dataclasses import dataclass
from act1.tokens import count_tokens

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
        full_text    = " ".join(m["content"] for m in history)
        total_tokens = count_tokens(full_text)

        if total_tokens > TOKEN_THRESHOLD:
            return EchidnaDecision("checkpoint", None, "token threshold exceeded", "high")
        if turns_since_checkpoint >= TURN_CADENCE:
            return EchidnaDecision("checkpoint", None, f"cadence: every {TURN_CADENCE} turns", "medium")
        return EchidnaDecision("pass", None, "within budget", "low")

    # ── REAL (DeepSeek Flash) ────────────────────────────────────────────────
    def _llm_decide(
        self,
        history: list[dict],
        trauma: str,
        checkpoint_summary: str,
        turns_since_checkpoint: int,
        available_checkpoints: list[int],
    ) -> EchidnaDecision:
        from act1.solve import _deepseek_call

        system = """You are Echidna, the Witch of Greed. You observe conversations with
perfect clarity and decide when knowledge must be crystallized.
You receive: the current turn, trauma memory, and checkpoint summary.
Return JSON only:
{"action": "checkpoint"|"revert"|"pass",
 "revert_to": <id>|null, "reason": "...", "urgency": "low"|"medium"|"high"}
CHECKPOINT when: topic shifts, a reasoning hop is resolved, token budget near limit.
REVERT when: contradiction detected, reasoning chain has collapsed.
PASS otherwise. Always read trauma memory before deciding."""

        recent      = history[-4:] if len(history) >= 4 else history
        recent_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in recent)

        prompt = f"""Trauma memory: {trauma}
Checkpoint summary: {checkpoint_summary}
Turns since last checkpoint: {turns_since_checkpoint}
Available checkpoint IDs for revert: {available_checkpoints}
Recent conversation:
{recent_text}"""

        raw = _deepseek_call(system, prompt, max_tokens=80, fast=True)
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

### 1. Add import at top

```python
from rezero.echidna import Echidna
```

### 2. Add to `__init__`

```python
self.echidna = Echidna(use_llm=use_llm)
```

Also remove `self.K = 3` — Echidna replaces the fixed cadence entirely.

### 3. Replace `add_turn` — full method (replaces Step 3 version entirely)

```python
def add_turn(self, user: str, assistant: str) -> None:
    # 1. update trauma first — before Echidna reads it
    self.trauma_extractor.update(user)
    self.trauma_extractor.update(assistant)

    # 2. append to history
    self.history.append({"role": "user",      "content": user})
    self.history.append({"role": "assistant", "content": assistant})
    self.turn_count += 1

    # 3. Echidna reads trauma, decides action
    decision = self.echidna.decide(
        history                = self.history,
        trauma                 = self.trauma_extractor.get(),
        checkpoint_summary     = self.stack.summary() if self.stack else "none",
        turns_since_checkpoint = self.turns_since_checkpoint,
        available_checkpoints  = self.stack.list_ids() if self.stack else [],
    )

    if decision.action == "checkpoint":
        new_cp = self.checkpoint_builder.build(
            self.history, self.trauma_extractor.get()
        )
        self.stack.push(new_cp, self.turn_count, reason=decision.reason)
        self.turns_since_checkpoint = 0
    elif decision.action == "revert" and decision.revert_to is not None:
        self.stack.revert_to(decision.revert_to)
        self.turns_since_checkpoint = 0
    else:
        self.turns_since_checkpoint += 1
```

### 4. Replace `_get_checkpoint` — full method (replaces Step 4 version entirely)

> **FIX:** Step 4 had a scheduled fallback in `_get_checkpoint`. Echidna now fully owns checkpoint timing. Remove the `if turns_since_checkpoint >= K` block. `_get_checkpoint` is now just a stack read:

```python
def _get_checkpoint(self) -> str:
    # FIX: Echidna drives all checkpoint creation via add_turn.
    # This method is now a pure read — no side effects, no scheduling.
    if self.stack is None:
        return self.checkpoint  # Step 3 fallback before stack is wired
    return self.stack.current()
```

---

## Tests — `tests/test_echidna.py`

```python
from rezero.echidna import Echidna

def test_mock_pass_on_short_history():
    e = Echidna(use_llm=False)
    history = [
        {"role": "user",      "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    d = e.decide(history, "Alice", "no checkpoint", 1, [])
    assert d.action == "pass"

def test_mock_checkpoint_on_token_threshold():
    e = Echidna(use_llm=False)
    long_content = "word " * 900
    history = [{"role": "user", "content": long_content}]
    d = e.decide(history, "Alice", "cp", 1, [])
    assert d.action == "checkpoint"
    assert d.urgency == "high"

def test_mock_checkpoint_on_cadence():
    e = Echidna(use_llm=False)
    history = [
        {"role": "user",      "content": "short message"},
        {"role": "assistant", "content": "short reply"},
    ]
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
    s = ReZeroSession(goal="Test Alice at TechCorp", use_llm=False)
    # run enough turns to trigger cadence (TURN_CADENCE=5)
    for i in range(6):
        s.add_turn(
            f"Question {i} about TechCorp funding",
            f"Answer {i} about TechCorp investors"
        )
    # Echidna should have triggered at least one checkpoint by turn 5
    assert len(s.stack.list_ids()) >= 1

def test_get_checkpoint_is_pure_read():
    # FIX: verify _get_checkpoint has no side effects after Step 5
    from rezero.session import ReZeroSession
    s = ReZeroSession(goal="Test", use_llm=False)
    s.add_turn("hello", "hi")
    before = s.stack.list_ids()[:]
    _ = s._get_checkpoint()
    after = s.stack.list_ids()
    # calling _get_checkpoint must not push a new checkpoint
    assert before == after
```

---

## Run tests

```bash
pytest tests/test_echidna.py -v
```

---

## Done when

- All 6 tests pass
- `_get_checkpoint` has zero side effects — verified by `test_get_checkpoint_is_pure_read`
- After 5+ turns, `s.stack.list_ids()` is non-empty
- Step 4's `self.K` and scheduled logic are fully removed from session
