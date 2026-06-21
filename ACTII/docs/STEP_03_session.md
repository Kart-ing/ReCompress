# Step 3 — ReZeroSession Skeleton

> **Goal:** Build the main `ReZeroSession` class. At this step it holds state and produces a valid `[TRAUMA][CHECKPOINT][DELTA]` prompt. No real compression yet — checkpoint is empty string. Full interface is testable end-to-end immediately.

---

## What to build

`rezero/session.py`

---

## Code

```python
from act1.tokens import count_tokens
from rezero.trauma import TraumaExtractor

TRAUMA_CAP     = 50
CHECKPOINT_CAP = 150
DELTA_CAP      = 100
TOTAL_CAP      = 300

class ReZeroSession:
    def __init__(self, goal: str, use_llm: bool = False):
        self.goal = goal
        self.use_llm = use_llm
        self.history: list[dict] = []        # {"role": "user"|"assistant", "content": str}
        self.trauma_extractor = TraumaExtractor(use_llm=use_llm)
        self.checkpoint: str = ""            # populated by CheckpointBuilder in Step 4
        self.checkpoint_stack = None         # wired in Step 4
        self.echidna = None                  # wired in Step 5
        self.turn_count: int = 0
        self.turns_since_checkpoint: int = 0

        # seed trauma with the user's goal immediately
        self.trauma_extractor.update(goal)

    def add_turn(self, user: str, assistant: str) -> None:
        """Record a completed turn. Updates trauma. Checkpoint logic added in Step 4+."""
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

    # ── INTERNAL ────────────────────────────────────────────────────────────

    def _get_checkpoint(self) -> str:
        """Returns current checkpoint. Overridden in Step 4 with real builder."""
        return self.checkpoint

    def _get_delta(self) -> str:
        """Returns the most recent user turn, truncated to DELTA_CAP."""
        if not self.history:
            return ""
        last_user = next(
            (m["content"] for m in reversed(self.history) if m["role"] == "user"),
            ""
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
from pathlib import Path
import json
from rezero.session import ReZeroSession

def test_prompt_has_three_sections():
    s = ReZeroSession(goal="Test goal")
    s.add_turn("Hello", "Hi there")
    prompt = s.prompt_for_solver()
    assert "[TRAUMA]"     in prompt
    assert "[CHECKPOINT]" in prompt
    assert "[DELTA]"      in prompt

def test_token_count_under_300():
    # FIX: use longer messages so the cap is actually exercised
    s = ReZeroSession(goal="Find the founder of the company that NASA works with most")
    for i in range(10):
        s.add_turn(
            f"User message number {i} about SpaceX and NASA contracts with extra words to fill tokens",
            f"Assistant reply {i} explaining the relationship between SpaceX and government agencies"
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
    # FIX: use a goal with clear capitalized named entities the mock extractor will catch
    s = ReZeroSession(goal="Find Alice the Chief Executive of Tech Corp")
    prompt = s.prompt_for_solver()
    trauma_section = prompt.split("[CHECKPOINT]")[0].replace("[TRAUMA]", "").strip()
    # "Alice" and "TechCorp" are capitalized — mock extractor will pin them
    assert "Alice" in trauma_section or "Tech" in trauma_section

def test_token_flat_over_15_turns():
    # FIX: use substantive messages so naive would grow but RbD stays flat
    s = ReZeroSession(goal="Research Alice and Tech Corp funding history")
    counts = []
    for i in range(15):
        s.add_turn(
            f"Question {i} about TechCorp funding rounds in Silicon Valley this year",
            f"Answer {i}: TechCorp raised several million dollars from investors recently"
        )
        counts.append(s.token_count())
    assert all(c <= 300 for c in counts), f"Exceeded 300: {counts}"
```

---

## Run tests

```bash
pytest tests/test_session.py -v
```

---

## Done when

- All 5 tests pass
- `token_count()` never exceeds 300 across 15 turns
- Prompt always has all three `[TRAUMA]` `[CHECKPOINT]` `[DELTA]` sections
- `[TRAUMA]` section contains named entities from the goal
