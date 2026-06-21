from engine.tokens import count_tokens
from rezero.trauma import TraumaExtractor
from rezero.checkpoint import CheckpointBuilder, CheckpointStack
from rezero.echidna import Echidna
from rezero.context_builder import ContextBuilder

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
        self.checkpoint_builder = CheckpointBuilder(goal=goal, ratio=ratio)
        self.checkpoint_stack = CheckpointStack()
        self.echidna = Echidna(use_llm=use_llm)
        self.context_builder = ContextBuilder()
        self.turn_count: int = 0
        self.turns_since_checkpoint: int = 0

        self.trauma_extractor.update(goal)

    def add_turn(self, user: str, assistant: str) -> None:
        self.trauma_extractor.update(user)
        self.trauma_extractor.update(assistant)

        self.history.append({"role": "user",      "content": user})
        self.history.append({"role": "assistant", "content": assistant})
        self.turn_count += 1

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

    def prompt_for_solver(self) -> str:
        return self.context_builder.build(
            trauma     = self.trauma_extractor.get(),
            checkpoint = self._get_checkpoint(),
            delta      = self._get_delta(),
        )

    def token_count(self) -> int:
        return count_tokens(self.prompt_for_solver())

    def list_checkpoints(self) -> list[int]:
        return self.checkpoint_stack.list_ids()

    def revert_to(self, checkpoint_id: int) -> None:
        self.checkpoint_stack.revert_to(checkpoint_id)
        self.turns_since_checkpoint = 0

    def _get_checkpoint(self) -> str:
        return self.checkpoint_stack.current()

    def _get_delta(self) -> str:
        if not self.history:
            return ""
        last_user = next(
            (m["content"] for m in reversed(self.history) if m["role"] == "user"), ""
        )
        return self._enforce(last_user, DELTA_CAP)

    def _enforce(self, text: str, cap: int) -> str:
        words = text.split()
        while count_tokens(" ".join(words)) > cap and words:
            words.pop()
        return " ".join(words)
