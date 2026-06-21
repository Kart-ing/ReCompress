"""Bear-improver training data generation.

NEW OBJECTIVE (vs the standalone distill in gen_data.py):
  Train the SLM so that bear(SLM(text)) — i.e. our rewrite THEN bear's blind deletion —
  produces a context the solver can still answer from. The SLM learns to write
  DELETION-ROBUST text: front-load the answer's evidence and keep it redundant enough
  that bear's token-dropping doesn't destroy it.

Strategy — teacher-for-bear distillation with a survival filter:
  1. DeepSeek produces a deletion-robust rewrite of the context (special prompt).
  2. We ACTUALLY RUN bear on that rewrite.
  3. We score survival: does solve(bear(rewrite), q) get F1 >= keep_threshold vs gold?
  4. Keep only (text -> rewrite) pairs whose rewrite SURVIVES bear. Distill the SLM on those.

So every training target is proven to yield a good answer AFTER bear — which is exactly
the objective the standalone v3 model was never trained for (and why ours->bear washed out).

Run: python -m src.distill.gen_data_bearimprover --n 1500 --out data/distill/bearimprover.jsonl
"""
from __future__ import annotations
import json
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from recompress.act1.client import get_client
from recompress.config import CFG
from recompress.act1.data import load_hotpotqa, context_to_text
from recompress.act1.tokens import count_tokens, truncate_to_tokens
from recompress.act1.bear import compress_bear
from recompress.act1.solve import solve
from recompress.act1.metrics import qa_f1

MAX_WORKERS = 96  # bumped from 8: DeepSeek allows ~2500 concurrent
KEEP_THRESHOLD = 0.5   # keep a pair only if the answer survives bear with F1 >= this
_MIN_OUT_TOKENS = 8

# Deletion-robust rewrite prompt: the teacher writes text ENGINEERED to survive bear's
# blind token deletion. Different from the standalone-compressor prompt.
_SYSTEM_BEAR = (
    "You rewrite a long CONTEXT into a shorter version that will later be passed through "
    "an automatic token-deletion compressor (which blindly drops low-information tokens). "
    "Your rewrite must remain answerable for the QUESTION EVEN AFTER aggressive token deletion.\n\n"
    "Rules:\n"
    "1. DROP passages irrelevant to the question (distractors).\n"
    "2. For every fact needed to answer, state it PLAINLY and EARLY, using the key entities/"
    "numbers/names verbatim — these survive deletion best. Prefer 'X is Y. X did Z.' over long clauses.\n"
    "3. Build in light REDUNDANCY for the critical answer fact (mention the key entity/relation "
    "in two short simple sentences) so that if deletion drops one mention, another survives.\n"
    "4. Keep multi-hop links explicit: name BOTH ends of each hop in the same sentence.\n"
    "5. Do NOT answer the question. Output CONTEXT (plain declarative sentences), not an answer.\n"
    "6. Be concise (aim ~{ratio:.0%} of input) but NEVER sacrifice an answer-supporting fact to be shorter.\n"
    "7. Output ONLY the rewritten context."
)

_TRANSIENT_MARKERS = ("timeout", "timed out", "connection", "rate limit", "429")


def _rewrite_robust(text: str, question: str, ratio: float, max_attempts: int = 4) -> str:
    last = None
    for attempt in range(max_attempts):
        try:
            resp = get_client().chat.completions.create(
                model=CFG.compressor_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_BEAR.format(ratio=ratio)},
                    {"role": "user", "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{text}"},
                ],
                temperature=0.0,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            last = e
            import time
            time.sleep(min(2.0 * (attempt + 1), 12.0))
    raise last


def _gen_one(instance: dict, ratio: float) -> dict:
    """Rewrite -> bear -> solve -> score survival. Returns pair dict with survival info."""
    text = context_to_text(instance)
    q = instance["question"]
    gold = instance["answer"]
    try:
        rewrite = _rewrite_robust(text, q, ratio)
        rewrite = truncate_to_tokens(rewrite, max(1, int(count_tokens(text) * ratio)))
        # the crux: actually run bear on our rewrite, then see if the answer survives
        after_bear = compress_bear(rewrite, ratio)
        pred = solve(after_bear, q)
        f1 = qa_f1(pred, gold)
        return {
            "question": q, "text": text, "compressed": rewrite,
            "n_in": count_tokens(text), "n_out": count_tokens(rewrite),
            "post_bear_f1": f1, "pred": pred, "error": None,
        }
    except Exception as e:
        return {"question": q, "text": text, "compressed": "", "n_in": count_tokens(text),
                "n_out": 0, "post_bear_f1": 0.0, "pred": "", "error": f"{type(e).__name__}: {e}"}


def gen_to_file(n: int, out: str, ratio: float = 0.3, max_workers: int = MAX_WORKERS,
                keep_threshold: float = KEEP_THRESHOLD, skip: int = 0) -> dict:
    total = skip + n
    print(f"loading {total} hotpotqa instances, taking [{skip}:{total}]...")
    instances = load_hotpotqa(n=total)[skip:]
    n = len(instances)
    print(f"loaded {n}; generating deletion-robust rewrites + bear-survival filter "
          f"(keep if post-bear F1 >= {keep_threshold}) with {max_workers} workers...")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    results, n_done, n_kept, n_drop_f1, n_err = [], 0, 0, 0, 0
    surv_f1_sum = 0.0

    with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_gen_one, inst, ratio): i for i, inst in enumerate(instances)}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            n_done += 1
            if r["error"]:
                n_err += 1
            elif r["n_out"] < _MIN_OUT_TOKENS or r["post_bear_f1"] < keep_threshold:
                n_drop_f1 += 1  # rewrite did NOT survive bear well enough — don't train on it
            else:
                n_kept += 1
                surv_f1_sum += r["post_bear_f1"]
                with lock:
                    f.write(json.dumps({k: r[k] for k in
                            ("question", "text", "compressed", "n_in", "n_out", "post_bear_f1")}) + "\n")
                    f.flush()
            if n_done % 10 == 0 or n_done == n:
                print(f"  [{n_done}/{n}] kept={n_kept} dropped(low post-bear F1)={n_drop_f1} err={n_err}")

    keep_rate = n_kept / max(1, n_done - n_err)
    avg_surv = surv_f1_sum / max(1, n_kept)
    print(f"\nKEPT {n_kept} pairs that survive bear (avg post-bear F1 of kept = {avg_surv:.3f})")
    print(f"keep-rate among non-errored = {keep_rate:.1%}  (this is itself a finding: how often "
          f"can DeepSeek write bear-survivable text?)")
    return {"n_kept": n_kept, "n_dropped": n_drop_f1, "n_err": n_err,
            "keep_rate": keep_rate, "avg_post_bear_f1": avg_surv}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--out", default="data/distill/bearimprover.jsonl")
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    p.add_argument("--keep-threshold", type=float, default=KEEP_THRESHOLD)
    p.add_argument("--skip", type=int, default=0)
    args = p.parse_args()
    stats = gen_to_file(n=args.n, out=args.out, ratio=args.ratio, max_workers=args.workers,
                        keep_threshold=args.keep_threshold, skip=args.skip)
    print(f"\nwrote survivors to {args.out}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
