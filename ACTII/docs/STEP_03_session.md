# Step 3 — ReZeroSession Skeleton

> **Goal:** Build the main `ReZeroSession` class in `rezero/session.py`. At this step it holds state and produces a valid `[TRAUMA][CHECKPOINT][DELTA]` prompt. No real compression yet — checkpoint is empty. Full interface testable immediately.

---

## What to build

`rezero/session.py`

---

## Code

```python
from engine.tokens import count_tokens
from rezero.trauma import TraumaExtractor

TRAUMA_CAP     = 50
CHECKPOINT_CAP = 150
DELTA_CAP      = 100
TOTAL_CAP      = 300

class ReZeroSession:
    def __init__(self, goal: str, use_llm: bool = False, ratio: float = 0.20):
        self.goal = goal
        self.use_llm = use_llm
        self.ratio = ratio
        self.history: list[dict] = []
        self.trauma_extractor = TraumaExtractor(use_llm=use_llm)
        self.checkpoint: str = ""      # populated by CheckpointBuilder in Step 4
        self.checkpoint_stack = None   # wired in Step 4
        self.echidna = None            # wired in Step 5
        self.context_builder = None    # wired in Step 6
        self.turn_count: int = 0
        self.turns_since_checkpoint: int = 0

        # seed trauma with goal immediately
        self.trauma_extractor.update(goal)

    def add_turn(self, user: str, assistant: str) -> None:
        """Record a completed turn. Checkpoint logic added in Step 4+."""
        self.trauma_extractor.update(user)
        self.trauma_extractor.update(assistant)
        self.history.append({"role": "user",      "content": user})
        self.history.append({"role": "assistant", "content": assistant})
        self.turn_count += 1
        self.turns_since_checkpoint += 1

    def prompt_for_solver(self) -> str:
        """Assemble [TRAUMA][CHECKPOINT][DELTA] prompt. Always ≤300 tokens."""
        trauma     = self._enforce(self.trauma_extractor.get(), TRAUMA_CAP)
        checkpoint = self._enforce(self._get_checkpoint(),      CHECKPOINT_CAP)
        delta      = self._enforce(self._get_delta(),           DELTA_CAP)

        prompt = self._assemble(trauma, checkpoint, delta)

        # enforce total cap — trim checkpoint first
        while count_tokens(prompt) > TOTAL_CAP:
            words = checkpoint.split()
            if not words:
                break
            checkpoint = " ".join(words[:-5])
            prompt = self._assemble(trauma, checkpoint, delta)

        return prompt

    def token_count(self) -> int:
        return count_tokens(self.prompt_for_solver())

    def list_checkpoints(self) -> list[int]:
        return self.checkpoint_stack.list_ids() if self.checkpoint_stack else []

    def revert_to(self, checkpoint_id: int) -> None:
        """User-initiated revert. Trauma memory is NOT reverted."""
        if self.checkpoint_stack:
            self.checkpoint_stack.revert_to(checkpoint_id)
            self.turns_since_checkpoint = 0

    # ── INTERNAL ────────────────────────────────────────────────────────────

    def _get_checkpoint(self) -> str:
        """Pure read — no side effects. Echidna drives all checkpoint creation."""
        if self.checkpoint_stack is None:
            return self.checkpoint
        return self.checkpoint_stack.current()

    def _get_delta(self) -> str:
        if not self.history:
            return ""
        last_user = next(
            (m["content"] for m in reversed(self.history) if m["role"] == "user"), ""
        )
        return self._enforce(last_user, DELTA_CAP)

    def _assemble(self, trauma: str, checkpoint: str, delta: str) -> str:
        return f"[TRAUMA]\n{trauma}\n\n[CHECKPOINT]\n{checkpoint}\n\n[DELTA]\n{delta}"

    def _enforce(self, text: str, cap: int) -> str:
        words = text.split()
        while count_tokens(" ".join(words)) > cap and words:
            words.pop()
        return " ".join(words)
```

---

## Tests — `tests/test_session.py`

```python
from rezero.session import ReZeroSession

def test_prompt_has_three_sections():
    s = ReZeroSession(goal="Test goal")
    s.add_turn("Hello", "Hi there")
    prompt = s.prompt_for_solver()
    assert "[TRAUMA]"     in prompt
    assert "[CHECKPOINT]" in prompt
    assert "[DELTA]"      in prompt

def test_token_count_under_300():
    s = ReZeroSession(goal="Find the founder of the company that NASA works with most")
    for i in range(10):
        s.add_turn(
            f"User message number {i} about SpaceX and NASA contracts with extra context",
            f"Assistant reply {i} explaining the relationship between SpaceX and government"
        )
    assert s.token_count() <= 300

def test_delta_is_most_recent_user_turn():
    s = ReZeroSession(goal="Test")
    s.add_turn("First question here", "First answer")
    s.add_turn("Second question here", "Second answer")
    prompt = s.prompt_for_solver()
    delta_section = prompt.split("[DELTA]")[1]
    assert "Second question" in delta_section

def test_goal_seeded_into_trauma():
    s = ReZeroSession(goal="Find Alice the Chief Executive of Tech Corp")
    prompt = s.prompt_for_solver()
    trauma_section = prompt.split("[CHECKPOINT]")[0].replace("[TRAUMA]", "").strip()
    assert "Alice" in trauma_section or "Tech" in trauma_section

def test_token_flat_over_many_turns():
    N_TURNS = 25
    s = ReZeroSession(goal="Research Alice and Tech Corp funding history")
    counts = []
    for i in range(N_TURNS):
        s.add_turn(
            f"Question {i} about Tech Corp funding rounds in Silicon Valley this year",
            f"Answer {i}: Tech Corp raised several million dollars from investors recently"
        )
        counts.append(s.token_count())
    assert all(c <= 300 for c in counts), f"Exceeded 300 at some turn: {list(enumerate(counts))}"
```

---

## Run tests

```bash
pytest tests/test_session.py -v
```

---

## Done when

- All 5 tests pass
- `token_count()` never exceeds 300 across any number of turns
- All three `[TRAUMA]` `[CHECKPOINT]` `[DELTA]` sections always present
