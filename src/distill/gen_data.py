"""Generate teacher training data: run compress_ours() via DeepSeek API on (text, question) pairs.
Outputs JSONL of {text, question, compressed, n_in, n_out} for Modal LoRA training.

The teacher makes ONE DeepSeek call per instance, so generation is I/O-bound. We fan the
calls out across a ThreadPoolExecutor (default 8 workers) and append each finished pair to the
output JSONL immediately (flush), so the run is resumable and a crash never loses progress.

Run: python -m src.distill.gen_data --n 1000 --out data/distill/train.jsonl
"""
from __future__ import annotations
import json
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.act1.data import load_hotpotqa, context_to_text
from src.act1.compress import compress_ours
from src.act1.tokens import count_tokens

MAX_WORKERS = 8


def _gen_one(instance: dict, ratio: float, distill: bool = True) -> dict:
    """Compress one instance via the teacher. Returns a pair dict (error field None on success)."""
    text = context_to_text(instance)
    q = instance["question"]
    try:
        compressed = compress_ours(text, q, ratio=ratio, distill=distill)
        return {
            "question": q,
            "text": text,
            "compressed": compressed,
            "n_in": count_tokens(text),
            "n_out": count_tokens(compressed),
            "error": None,
        }
    except Exception as e:
        return {
            "question": q,
            "text": text,
            "compressed": "",
            "n_in": count_tokens(text),
            "n_out": 0,
            "error": f"{type(e).__name__}: {e}",
        }


# Drop bare-answer-leak pairs at write time: anything this short is an answer span,
# not compressed context (matches the post-hoc filter we applied to the v1 dataset).
_MIN_OUT_TOKENS = 8


def gen_pairs_to_file(
    n: int,
    out: str,
    ratio: float = 0.3,
    max_workers: int = MAX_WORKERS,
    distill: bool = True,
    skip: int = 0,
) -> list[dict]:
    """Generate n (text, question, compressed) pairs from HotpotQA via the teacher.

    Pairs are computed in parallel and appended to ``out`` (one JSON line each) as soon as they
    finish, so partial progress is durable. Only successful pairs are written. Returns the list of
    all result dicts (including any that errored, for the caller's summary).

    ``skip`` drops the first ``skip`` instances and generates the next ``n``. Because
    load_hotpotqa(n) is prefix-stable, skip=1000,n=1500 yields exactly the 1500 instances
    AFTER the first 1000 — used to extend an existing dataset without regenerating it.
    """
    total = skip + n
    print(f"loading {total} hotpotqa instances (streaming), then taking [{skip}:{total}]...")
    instances = load_hotpotqa(n=total)[skip:]
    n = len(instances)
    print(f"loaded {n} instances; compressing with {max_workers} workers...")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    write_lock = threading.Lock()
    n_done = n_ok = n_err = n_skip = 0

    # Append mode + flush after every write => resumable / inspectable mid-run.
    with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_gen_one, inst, ratio, distill): i for i, inst in enumerate(instances)}
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            n_done += 1
            if r["error"]:
                n_err += 1
                print(f"  [{n_done}/{n}] ERROR: {r['error']}")
            elif r["n_out"] < _MIN_OUT_TOKENS:
                n_skip += 1  # bare-answer leak — don't poison the training set
                r["error"] = f"too_short ({r['n_out']} tok) — skipped"
            else:
                n_ok += 1
                with write_lock:
                    f.write(json.dumps({k: r[k] for k in ("question", "text", "compressed", "n_in", "n_out")}) + "\n")
                    f.flush()
            if n_done % 10 == 0 or n_done == n:
                print(f"  [{n_done}/{n}] done (ok={n_ok} skip={n_skip} err={n_err})")

    return results


def gen_pairs(n: int, ratio: float = 0.3) -> list[dict]:
    """Single-threaded, in-memory generation (legacy path; kept for parity).

    Returns successful pairs only. Prefer ``gen_pairs_to_file`` for real runs.
    """
    print(f"loading {n} hotpotqa instances (streaming)...")
    instances = load_hotpotqa(n=n)
    pairs = []
    for i, ex in enumerate(instances):
        text = context_to_text(ex)
        q = ex["question"]
        print(f"  [{i+1}/{len(instances)}] compressing ({count_tokens(text)} tok)...")
        compressed = compress_ours(text, q, ratio=ratio)
        pairs.append({
            "question": q,
            "text": text,
            "compressed": compressed,
            "n_in": count_tokens(text),
            "n_out": count_tokens(compressed),
        })
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--out", default="data/distill/train.jsonl")
    p.add_argument("--workers", type=int, default=MAX_WORKERS,
                   help="parallel teacher calls (set 1 for the single-threaded path)")
    p.add_argument("--skip", type=int, default=0,
                   help="skip the first N instances (to extend an existing dataset without overlap)")
    args = p.parse_args()

    if args.workers <= 1:
        # legacy single-threaded path, then write the file once
        pairs = gen_pairs(n=args.n, ratio=args.ratio)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            for p_ in pairs:
                f.write(json.dumps(p_) + "\n")
        results = pairs
    else:
        results = gen_pairs_to_file(n=args.n, out=args.out, ratio=args.ratio,
                                    max_workers=args.workers, skip=args.skip)

    ok = [r for r in results if not r.get("error")]
    n_err = len(results) - len(ok)
    if not ok:
        print(f"\nwrote 0 pairs to {args.out} ({n_err} errored)")
        return

    avg_in = sum(r["n_in"] for r in ok) / len(ok)
    avg_out = sum(r["n_out"] for r in ok) / len(ok)
    print(f"\nwrote {len(ok)} pairs to {args.out}" + (f" ({n_err} errored, skipped)" if n_err else ""))
    print(f"avg input: {avg_in:.0f} tok | avg output: {avg_out:.0f} tok | avg ratio: {avg_out/avg_in:.2%}")


if __name__ == "__main__":
    main()
