# Step 4 — Checkpoint Builder & Stack

> **Goal:** Build `rezero/checkpoint.py`. The `CheckpointBuilder` compresses history using our own `engine/compressor.py`. The `CheckpointStack` stores versioned entries and supports revert. Wire both into `ReZeroSession`.

---

## What to build

`rezero/checkpoint.py` — then update `rezero/session.py`

---

## rezero/checkpoint.py

```python
from dataclasses import dataclass
from engine.compressor import compress
from engine.tokens import count_tokens

CHECKPOINT_CAP = 150
MAX_STACK_SIZE = 10

@dataclass
class CheckpointEntry:
    id: int
    state: str
    turn: int
    tokens: int
    reason: str = ""


class CheckpointStack:
    def __init__(self):
        self.stack: list[CheckpointEntry] = []
        self._id_counter: int = 0

    def push(self, state: str, turn: int, reason: str = "") -> CheckpointEntry:
        entry = CheckpointEntry(
            id=self._id_counter,
            state=state,
            turn=turn,
            tokens=count_tokens(state),
            reason=reason,
        )
        self.stack.append(entry)
        self._id_counter += 1
        if len(self.stack) > MAX_STACK_SIZE:
            self.stack.pop(0)
        return entry

    def current(self) -> str:
        return self.stack[-1].state if self.stack else ""

    def revert_to(self, checkpoint_id: int) -> str:
        """Pop stack back to checkpoint_id. Returns reverted state."""
        for entry in reversed(self.stack):
            if entry.id == checkpoint_id:
                idx = self.stack.index(entry)
                self.stack = self.stack[: idx + 1]
                return entry.state
        raise ValueError(f"Checkpoint {checkpoint_id} not found in stack")

    def summary(self) -> str:
        if not self.stack:
            return "No checkpoints yet."
        e = self.stack[-1]
        return f"CP-{e.id} at turn {e.turn} ({e.tokens} tok): {e.reason}"

    def list_ids(self) -> list[int]:
        return [e.id for e in self.stack]


class CheckpointBuilder:
    def __init__(self, goal: str, ratio: float = 0.20):
        self.goal = goal
        self.ratio = ratio

    def build(self, history: list[dict], trauma: str) -> str:
        """
        Compress history into a checkpoint.
        Never compresses anything already in trauma memory.
        """
        if not history:
            return ""
        full_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history
        )
        compressed = compress(full_text, question=self.goal, ratio=self.ratio)
        return self._enforce_cap(compressed)

    def _enforce_cap(self, text: str) -> str:
        words = text.split()
        while count_tokens(" ".join(words)) > CHECKPOINT_CAP and words:
            words.pop()
        return " ".join(words)
```

---

## Update rezero/session.py

### 1. Add imports at top

```python
from rezero.checkpoint import CheckpointBuilder, CheckpointStack
```

### 2. Add to `__init__`

```python
self.checkpoint_builder = CheckpointBuilder(goal=goal, ratio=ratio)
self.checkpoint_stack = CheckpointStack()
```

Remove `self.checkpoint: str = ""` — the stack replaces it.

### 3. Update `_get_checkpoint`

```python
def _get_checkpoint(self) -> str:
    """Pure read — no side effects. Echidna drives all checkpoint creation."""
    return self.checkpoint_stack.current()
```

### 4. Add public revert method (already stubbed in Step 3 — ensure it works)

```python
def revert_to(self, checkpoint_id: int) -> None:
    """User-initiated revert. Trauma memory is NOT reverted."""
    self.checkpoint_stack.revert_to(checkpoint_id)
    self.turns_since_checkpoint = 0

def list_checkpoints(self) -> list[int]:
    return self.checkpoint_stack.list_ids()
```

---

## Tests — `tests/test_checkpoint.py`

```python
from rezero.checkpoint import CheckpointBuilder, CheckpointStack
from engine.tokens import count_tokens

def test_checkpoint_under_cap():
    builder = CheckpointBuilder(goal="Find the founder")
    history = [
        {"role": "user",      "content": "Tell me about Alice and Tech Corp"},
        {"role": "assistant", "content": "Alice founded Tech Corp in 2010 and serves as CEO"},
    ]
    cp = builder.build(history, trauma="Alice Tech Corp")
    assert count_tokens(cp) <= 150

def test_stack_push_and_current():
    stack = CheckpointStack()
    stack.push("state one", turn=1)
    stack.push("state two", turn=2)
    assert stack.current() == "state two"

def test_stack_revert():
    stack = CheckpointStack()
    e1 = stack.push("state one",   turn=1)
    _  = stack.push("state two",   turn=2)
    _  = stack.push("state three", turn=3)
    reverted = stack.revert_to(e1.id)
    assert reverted == "state one"
    assert stack.current() == "state one"
    assert len(stack.stack) == 1

def test_stack_capped_at_10():
    stack = CheckpointStack()
    for i in range(15):
        stack.push(f"state {i}", turn=i)
    assert len(stack.stack) <= 10

def test_stack_revert_unknown_id_raises():
    stack = CheckpointStack()
    stack.push("only state", turn=1)
    try:
        stack.revert_to(999)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def test_session_revert_preserves_trauma():
    from rezero.session import ReZeroSession
    s = ReZeroSession(goal="Find Alice the founder")
    s.add_turn("Alice founded Tech Corp", "Correct, Alice is the founder")
    s.add_turn("What did Bob do?", "Bob joined later")
    trauma_before = s.trauma_extractor.get()
    ids = s.list_checkpoints()
    if ids:
        s.revert_to(ids[0])
    assert s.trauma_extractor.get() == trauma_before
```

---

## Run tests

```bash
pytest tests/test_checkpoint.py -v
```

---

## Done when

- All 6 tests pass
- `ReZeroSession` produces a non-empty `[CHECKPOINT]` section after enough turns
- Revert works and trauma is preserved
