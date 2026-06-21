from engine.tokens import count_tokens
from rezero.trauma import TraumaExtractor
from rezero.checkpoint import CheckpointBuilder, CheckpointStack
from rezero.echidna import Echidna, EchidnaDecision
from rezero.context_builder import ContextBuilder
import time

TRAUMA_CAP     = 50
CHECKPOINT_CAP = 150
DELTA_CAP      = 100
TOTAL_CAP      = 300
CHECKPOINT_COOLDOWN = 2
MIN_HISTORY_FOR_CHECKPOINT = 400

class ReZeroSession:
    def __init__(self, goal: str, use_llm: bool = False, ratio: float = 0.20, verbose: bool = False,
                 backend: str = "deepseek", echidna_mode: str | None = None):
        self.goal = goal
        self.use_llm = use_llm
        self.ratio = ratio
        self.verbose = verbose
        self.backend = backend   # compressor backend: deepseek | naive | distilled | bear
        # echidna_mode: "llm" | "clf" | "mock". None -> follow use_llm (back-compat).
        self.echidna_mode = echidna_mode or ("llm" if use_llm else "mock")
        self.history: list[dict] = []
        self.trauma_extractor = TraumaExtractor(goal=goal, use_llm=use_llm, verbose=verbose)
        self.checkpoint_builder = CheckpointBuilder(goal=goal, ratio=ratio, use_llm=use_llm, backend=backend)
        self.checkpoint_stack = CheckpointStack()
        self.echidna = Echidna(use_llm=use_llm, verbose=verbose, mode=self.echidna_mode)
        self.context_builder = ContextBuilder()
        self.turn_count: int = 0
        self.turns_since_checkpoint: int = 0

        if verbose:
            from engine.deepseek import set_verbose
            set_verbose(True)

    def add_turn(self, user: str, assistant: str) -> None:
        turn_t0 = time.time()

        if self.verbose:
            print(f"\n  ── Turn {self.turn_count + 1} ──")
            print(f"  [USER] {user[:80]}{'...' if len(user) > 80 else ''}")

        t0 = time.time()
        self.trauma_extractor.update(user)
        if len(assistant.split()) > 3:
            self.trauma_extractor.update(assistant)
        elif self.verbose:
            print(f"  [TRAUMA] skipped assistant (trivial: '{assistant}')")
        trauma_time = time.time() - t0

        self.history.append({"role": "user",      "content": user})
        self.history.append({"role": "assistant", "content": assistant})
        self.turn_count += 1

        t0 = time.time()
        decision = self.echidna.decide(
            history                = self.history,
            trauma                 = self.trauma_extractor.get(),
            checkpoint_summary     = self.checkpoint_stack.summary(),
            turns_since_checkpoint = self.turns_since_checkpoint,
            available_checkpoints  = self.checkpoint_stack.list_ids(),
        )
        echidna_time = time.time() - t0

        if decision.action == "checkpoint" and self.echidna_mode in ("llm", "clf"):
            history_tokens = count_tokens(" ".join(m["content"] for m in self.history))
            if self.turns_since_checkpoint < CHECKPOINT_COOLDOWN:
                if self.verbose:
                    print(f"  [GUARDRAIL] override checkpoint → pass (cooldown: {self.turns_since_checkpoint} < {CHECKPOINT_COOLDOWN} turns)")
                decision = EchidnaDecision("pass", None, "override: cooldown", "low")
            elif history_tokens < MIN_HISTORY_FOR_CHECKPOINT:
                if self.verbose:
                    print(f"  [GUARDRAIL] override checkpoint → pass (history {history_tokens} tok < {MIN_HISTORY_FOR_CHECKPOINT})")
                decision = EchidnaDecision("pass", None, "override: history too small", "low")

        cp_time = 0.0
        if decision.action == "checkpoint":
            t0 = time.time()
            new_cp = self.checkpoint_builder.build(
                self.history, self.trauma_extractor.get()
            )
            cp_time = time.time() - t0
            self.checkpoint_stack.push(new_cp, self.turn_count, reason=decision.reason)
            self.turns_since_checkpoint = 0
            if self.verbose:
                print(f"  [CHECKPOINT] built ({count_tokens(new_cp)} tok, {cp_time:.1f}s): {new_cp[:100]}{'...' if len(new_cp) > 100 else ''}")
        elif decision.action == "revert" and decision.revert_to is not None:
            self.checkpoint_stack.revert_to(decision.revert_to)
            self.turns_since_checkpoint = 0
            if self.verbose:
                print(f"  [REVERT] to checkpoint {decision.revert_to}")
        else:
            self.turns_since_checkpoint += 1

        turn_time = time.time() - turn_t0
        if self.verbose:
            print(f"  [STATE] trauma='{self.trauma_extractor.get()[:60]}' | cp_stack={self.checkpoint_stack.list_ids()} | tok={self.token_count()}")
            print(f"  [TIME] turn={turn_time:.1f}s (trauma={trauma_time:.1f}s echidna={echidna_time:.1f}s checkpoint={cp_time:.1f}s)")

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
