"""Generate training data for the lightweight Echidna classifier.

Runs real HotpotQA multi-turn conversations with the LLM Echidna making the decisions, and
logs (features, decision) at every turn. The classifier then learns to mimic the LLM Echidna's
checkpoint/pass behavior from cheap features — removing ~900 DeepSeek tokens/turn at inference.

We instrument by monkey-patching Echidna.decide to record the feature vector + the action the
LLM chose, without changing session logic. Revert is extremely rare in practice; we collapse
the label space to {checkpoint, pass} (revert -> pass for the classifier; the LLM path still
handles revert when enabled).

Run from repo root:  python -m rezero.experiments.gen_echidna_data --n 120 --max-turns 8
"""
from __future__ import annotations
import os, sys, json, argparse

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ACTII = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in (_REPO, _ACTII):
    if p not in sys.path:
        sys.path.insert(0, p)


def _build_convo(sample, max_turns):
    docs = [f"{t}: {' '.join(s)}" for t, s in
            zip(sample["context"]["title"], sample["context"]["sentences"])]
    return docs[:max_turns]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--max-turns", type=int, default=8)
    ap.add_argument("--out", default="data/echidna/echidna_train.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    from rezero.session import ReZeroSession
    from rezero.echidna import Echidna
    from rezero.echidna_features import extract_features, FEATURE_NAMES
    import random

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation", streaming=True)
    rows = list(ds.take(max(args.n * 20, 1500)))
    random.Random(123).shuffle(rows)   # different seed than the eval window
    rows = rows[:args.n]
    print(f"generating Echidna decisions over {len(rows)} conversations x {args.max_turns} turns")

    samples = []  # list.append is thread-safe under CPython's GIL
    from concurrent.futures import ThreadPoolExecutor
    import threading
    _done = {"n": 0}
    _lock = threading.Lock()

    # monkey-patch decide to log features + the LLM's chosen action (global, but the
    # logged append is GIL-safe; the conversations themselves are independent)
    orig_decide = Echidna.decide
    def logging_decide(self, history, trauma, checkpoint_summary,
                       turns_since_checkpoint, available_checkpoints):
        feats = extract_features(history, trauma, checkpoint_summary,
                                 turns_since_checkpoint, available_checkpoints)
        d = orig_decide(self, history, trauma, checkpoint_summary,
                        turns_since_checkpoint, available_checkpoints)
        label = "checkpoint" if d.action == "checkpoint" else "pass"  # collapse revert->pass
        samples.append({"features": feats, "label": label})
        return d
    Echidna.decide = logging_decide

    def run_one(s):
        q = s["question"]
        docs = _build_convo(s, args.max_turns)
        sess = ReZeroSession(goal=q, use_llm=True, backend="deepseek")  # LLM Echidna on
        for i, d in enumerate(docs):
            sess.add_turn(f"Document {i+1}: {d}", "Noted.")
        with _lock:
            _done["n"] += 1
            if _done["n"] % 10 == 0:
                ck = sum(1 for x in samples if x["label"] == "checkpoint")
                print(f"  [{_done['n']}/{args.n}] {len(samples)} decisions, {ck} checkpoints")

    try:
        # conversations are independent -> run concurrently (big speedup over sequential API)
        with ThreadPoolExecutor(max_workers=12) as ex:
            list(ex.map(run_one, rows))
    finally:
        Echidna.decide = orig_decide  # restore

    os.makedirs(os.path.join(_REPO, os.path.dirname(args.out)), exist_ok=True)
    path = os.path.join(_REPO, args.out)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    n_ck = sum(1 for s in samples if s["label"] == "checkpoint")
    print(f"\nwrote {len(samples)} decisions to {args.out}")
    print(f"  checkpoint: {n_ck} ({n_ck/max(1,len(samples)):.1%}) | pass: {len(samples)-n_ck}")
    print(f"  features: {FEATURE_NAMES}")


if __name__ == "__main__":
    main()
