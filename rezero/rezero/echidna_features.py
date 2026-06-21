"""Feature extraction for the lightweight Echidna classifier.

Echidna decides checkpoint / revert / pass each turn. The LLM version sends ~900 tokens/turn
to DeepSeek; the classifier replaces that with a near-zero-cost decision from cheap features
computable entirely from what decide() already receives. This module is the SINGLE source of
truth for those features, shared by the data generator and the inference path so they can't
drift.

Features (all cheap, no LLM):
  - history_tokens          : total tokens in the conversation so far (the dominant LLM signal)
  - turns_since_checkpoint  : cadence pressure
  - n_checkpoints           : depth of the checkpoint stack (revert availability)
  - trauma_tokens           : how full the protected buffer is
  - checkpoint_summary_tokens
  - last_user_tokens        : size of the most recent user message
  - delta_tokens            : tokens accumulated since the last checkpoint (history - summary proxy)
  - new_entity_ratio        : fraction of capitalized tokens in the last user msg not already in trauma
                              (cheap proxy for 'a new bridging fact appeared' -> topic shift)
"""
from __future__ import annotations
import re
from engine.tokens import count_tokens

# stable feature order — the classifier is trained on exactly this vector
FEATURE_NAMES = [
    "history_tokens",
    "turns_since_checkpoint",
    "n_checkpoints",
    "trauma_tokens",
    "checkpoint_summary_tokens",
    "last_user_tokens",
    "delta_tokens",
    "new_entity_ratio",
]

_CAP = re.compile(r"\b[A-Z][a-zA-Z]+\b")


def extract_features(history, trauma, checkpoint_summary,
                     turns_since_checkpoint, available_checkpoints) -> list[float]:
    """Return a feature vector in FEATURE_NAMES order. Pure function of decide()'s inputs."""
    hist_text = " ".join(m["content"] for m in history)
    history_tokens = count_tokens(hist_text)
    trauma_tokens = count_tokens(trauma) if trauma else 0
    cps_tokens = count_tokens(checkpoint_summary) if checkpoint_summary else 0
    last_user = ""
    for m in reversed(history):
        if m["role"] == "user":
            last_user = m["content"]
            break
    last_user_tokens = count_tokens(last_user) if last_user else 0
    # delta ≈ history not yet folded into a checkpoint (cheap proxy)
    delta_tokens = max(0, history_tokens - cps_tokens)
    # new-entity ratio: capitalized tokens in last user msg not already pinned in trauma
    ents = set(_CAP.findall(last_user))
    new = [e for e in ents if e.lower() not in (trauma or "").lower()]
    new_entity_ratio = (len(new) / len(ents)) if ents else 0.0

    return [
        float(history_tokens),
        float(turns_since_checkpoint),
        float(len(available_checkpoints)),
        float(trauma_tokens),
        float(cps_tokens),
        float(last_user_tokens),
        float(delta_tokens),
        float(new_entity_ratio),
    ]
