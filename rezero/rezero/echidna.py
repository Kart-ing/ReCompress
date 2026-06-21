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


_CLF_BUNDLE = None
_CLF_PATH = None  # set lazily to data/echidna/echidna_clf.joblib


def _load_clf():
    """Lazy-load the trained classifier bundle (scaler+model+features)."""
    global _CLF_BUNDLE, _CLF_PATH
    if _CLF_BUNDLE is None:
        import os, joblib
        if _CLF_PATH is None:
            _CLF_PATH = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "echidna", "echidna_clf.joblib")
        _CLF_BUNDLE = joblib.load(_CLF_PATH)
    return _CLF_BUNDLE


class Echidna:
    def __init__(self, use_llm: bool = False, verbose: bool = False, mode: str | None = None):
        # mode: "llm" | "clf" | "mock". Back-compat: if mode is None, fall back to use_llm.
        self.use_llm = use_llm
        self.verbose = verbose
        self.mode = mode or ("llm" if use_llm else "mock")

    def decide(
        self,
        history: list[dict],
        trauma: str,
        checkpoint_summary: str,
        turns_since_checkpoint: int,
        available_checkpoints: list[int],
    ) -> EchidnaDecision:
        if self.mode == "llm":
            return self._llm_decide(
                history, trauma, checkpoint_summary,
                turns_since_checkpoint, available_checkpoints
            )
        if self.mode == "clf":
            return self._clf_decide(
                history, trauma, checkpoint_summary,
                turns_since_checkpoint, available_checkpoints
            )
        return self._mock_decide(history, turns_since_checkpoint)

    def _clf_decide(
        self,
        history: list[dict],
        trauma: str,
        checkpoint_summary: str,
        turns_since_checkpoint: int,
        available_checkpoints: list[int],
    ) -> EchidnaDecision:
        """Lightweight learned trigger — no LLM call, ~0 tokens. Mimics the LLM Echidna's
        checkpoint/pass behavior from cheap features (rezero/echidna_features.py)."""
        from rezero.echidna_features import extract_features
        bundle = _load_clf()
        feats = extract_features(history, trauma, checkpoint_summary,
                                 turns_since_checkpoint, available_checkpoints)
        import numpy as np
        x = bundle["scaler"].transform(np.array([feats], dtype=float))
        p = float(bundle["model"].predict_proba(x)[0][1])
        action = "checkpoint" if p >= bundle.get("threshold", 0.5) else "pass"
        d = EchidnaDecision(action, None, f"clf p={p:.2f}",
                            "high" if p > 0.8 else "medium" if p > 0.5 else "low")
        if self.verbose:
            print(f"  [ECHIDNA-CLF] {d.action} (p={p:.2f})")
        return d

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
