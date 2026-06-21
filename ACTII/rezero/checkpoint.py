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
