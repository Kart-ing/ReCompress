import json
from dataclasses import dataclass
from engine.tokens import count_tokens

TOKEN_THRESHOLD = 800
TURN_CADENCE    = 5

@dataclass
class EchidnaDecision:
    action: str
    revert_to: int | None
    reason: str
    urgency: str


class Echidna:
    def __init__(self, use_llm: bool = False, verbose: bool = False):
        self.use_llm = use_llm
        self.verbose = verbose

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

    def _mock_decide(self, history: list[dict], turns_since_checkpoint: int) -> EchidnaDecision:
        total_tokens = count_tokens(" ".join(m["content"] for m in history))
        if total_tokens > TOKEN_THRESHOLD:
            d = EchidnaDecision("checkpoint", None, "token threshold exceeded", "high")
        elif turns_since_checkpoint >= TURN_CADENCE:
            d = EchidnaDecision("checkpoint", None, f"cadence: every {TURN_CADENCE} turns", "medium")
        else:
            d = EchidnaDecision("pass", None, "within budget", "low")
        if self.verbose:
            print(f"  [ECHIDNA] {d.action} ({d.urgency}) — {d.reason} (history={total_tokens} tok)")
        return d

    def _llm_decide(
        self,
        history: list[dict],
        trauma: str,
        checkpoint_summary: str,
        turns_since_checkpoint: int,
        available_checkpoints: list[int],
    ) -> EchidnaDecision:
        from engine.deepseek import call, repair_json

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

        raw = call(system, prompt, max_tokens=200)
        parsed = repair_json(raw)
        if parsed is None:
            if self.verbose:
                preview = raw[:100].replace("\n", "\\n")
                print(f"  [ECHIDNA] WARNING: JSON parse failed, raw='{preview}'")
            return EchidnaDecision("pass", None, "parse error — defaulting to pass", "low")

        d = EchidnaDecision(
            action    = parsed.get("action",    "pass"),
            revert_to = parsed.get("revert_to"),
            reason    = parsed.get("reason",    ""),
            urgency   = parsed.get("urgency",   "low"),
        )
        if self.verbose:
            print(f"  [ECHIDNA] {d.action} ({d.urgency}) — {d.reason}")
        return d
