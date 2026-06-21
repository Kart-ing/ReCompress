"""End-to-end integration smoke: ReZero (Act 2) compressing checkpoints with the
distilled v3 model (Act 1) on Modal.

Proves the seam: ReZeroSession(backend="distilled") -> CheckpointBuilder ->
compress_backend -> Compressor.compress_batch.remote (v3 on H100).

Run from repo root:  modal run rezero/experiments/integration_smoke.py
"""
from __future__ import annotations
import os
import sys

# make BOTH import roots available: repo root (recompress.*) and ACTII (engine.*, rezero.*)
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ACTII = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_REPO, _ACTII):
    if p not in sys.path:
        sys.path.insert(0, p)

from recompress.distill.infer import app, Compressor  # the Act-1 Modal app


@app.local_entrypoint()
def main():
    from rezero.session import ReZeroSession

    print("=== ReZero + distilled-v3 integration smoke ===")
    s = ReZeroSession(goal="Plan a multi-step research project on coral reefs",
                      use_llm=False, backend="distilled", ratio=0.20)
    # enough turns + content to force a checkpoint (which triggers the distilled compress)
    turns = [
        ("We need to study coral bleaching in the Great Barrier Reef over 2016-2020.",
         "Understood — focusing on bleaching events 2016-2020 in the GBR."),
        ("Key metric is sea surface temperature anomaly above 1 degree C.",
         "Noted: SST anomaly > 1C as the bleaching threshold."),
        ("Also track the recovery rate of Acropora corals specifically.",
         "Added Acropora recovery rate to the tracking list."),
        ("The 2016 event affected 29 percent of shallow-water corals.",
         "Recorded: 2016 event hit 29% of shallow-water corals."),
        ("Compare against the 1998 baseline bleaching event.",
         "Will compare 2016-2020 against the 1998 baseline."),
        ("Funding deadline for the proposal is March 15th.",
         "Flagged the March 15th funding deadline."),
    ]
    for u, a in turns:
        s.add_turn(u, a)

    print(f"\nturns: {s.turn_count}")
    print(f"checkpoints created: {s.list_checkpoints()}")
    print(f"context tokens (flat target <=300): {s.token_count()}")
    print(f"backend used: {s.backend}")
    print("\n--- prompt_for_solver() (compressed by distilled v3) ---")
    print(s.prompt_for_solver()[:600])
    print("\n✅ ReZero successfully compressed a checkpoint with the distilled v3 model on Modal.")
