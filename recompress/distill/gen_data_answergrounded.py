"""Answer-grounded distillation data generation (NOVEL objective).

Standard distillation (LLMLingua-2, our v3): train the student to IMITATE the teacher's
compressed text — a text-fidelity loss. The student learns to copy what the teacher wrote.

Answer-grounded distillation (this file): train the student on compressions that are PROVEN
to let the frozen solver produce the correct ANSWER. For each example we:
  1. Generate N candidate compressions from the teacher at varied temperature (diversity).
  2. Run the FROZEN SOLVER on each candidate.
  3. Score each by QA-F1 vs the gold answer.
  4. Keep the BEST candidate, and only if its answer-F1 >= keep_threshold.
Distill the student on these answer-survivors. The training signal is "what the SOLVER can
answer from," not "what the TEACHER happened to write." Different objective, different data.

Why it should help: text-fidelity distillation propagates the teacher's mistakes (a fluent
compression that drops the answer fact still gets imitated). Answer-grounding filters those
out and selects, per example, the compression that actually works downstream — best-of-N
rejection sampling against the real task metric.

Run: python -m src.distill.gen_data_answergrounded --n 1500 --candidates 4 --out data/distill/answergrounded.jsonl
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
from recompress.act1.solve import solve
from recompress.act1.metrics import qa_f1

MAX_WORKERS = 96   # DeepSeek V4 Flash allows ~2500 concurrent; 8 was a needless bottleneck.
                   # 96 threads keeps ThreadPoolExecutor efficient (async needed beyond ~128).
KEEP_THRESHOLD = 0.5   # keep an example only if its best candidate's answer-F1 >= this
_MIN_OUT_TOKENS = 8

# Same query-aware compression instruction as the standalone teacher — the novelty is the
# SELECTION (answer-survival), not the prompt. We vary temperature to get diverse candidates.
_SYSTEM = (
    "You are a context compressor. Given a long context and a QUESTION, produce a compressed "
    "context that lets a downstream QA agent answer the QUESTION correctly.\n\n"
    "Rules:\n"
    "1. DROP passages irrelevant to the question (distractors).\n"
    "2. DENSIFY verbose-but-relevant prose into terse, information-dense sentences. Paraphrase freely.\n"
    "3. Preserve ALL facts needed to answer: entities, numbers, relations, every multi-hop link.\n"
    "4. Do NOT answer the question. Output CONTEXT (full sentences), not the answer.\n"
    "5. Aim for ~{ratio:.0%} of the input tokens; never drop an answer-supporting fact to be shorter.\n"
    "6. Output ONLY the compressed context."
)


def _one_candidate(text: str, question: str, ratio: float, temperature: float) -> str:
    resp = get_client().chat.completions.create(
        model=CFG.compressor_model,
        messages=[
            {"role": "system", "content": _SYSTEM.format(ratio=ratio)},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{text}"},
        ],
        temperature=temperature,
    )
    out = resp.choices[0].message.content.strip()
    return truncate_to_tokens(out, max(1, int(count_tokens(text) * ratio)))


def _gen_one(instance: dict, ratio: float, n_candidates: int) -> dict:
    """Best-of-N: generate candidates, solve each, keep the one with highest answer-F1."""
    text = context_to_text(instance)
    q = instance["question"]
    gold = instance["answer"]
    # temperatures spread for diversity (first one greedy for a strong default)
    temps = [0.0] + [0.7] * (n_candidates - 1)
    try:
        scored = []
        for t in temps[:n_candidates]:
            cand = _one_candidate(text, q, ratio, t)
            pred = solve(cand, q)
            scored.append((qa_f1(pred, gold), cand, pred))
        scored.sort(key=lambda s: s[0], reverse=True)
        best_f1, best_cand, best_pred = scored[0]
        return {
            "question": q, "text": text, "compressed": best_cand,
            "n_in": count_tokens(text), "n_out": count_tokens(best_cand),
            "answer_f1": best_f1, "pred": best_pred,
            "all_f1s": [s[0] for s in scored], "error": None,
        }
    except Exception as e:
        return {"question": q, "text": text, "compressed": "", "n_in": count_tokens(text),
                "n_out": 0, "answer_f1": 0.0, "pred": "", "all_f1s": [], "error": f"{type(e).__name__}: {e}"}


def gen_to_file(n: int, out: str, ratio: float = 0.3, n_candidates: int = 4,
                max_workers: int = MAX_WORKERS, keep_threshold: float = KEEP_THRESHOLD,
                skip: int = 0) -> dict:
    total = skip + n
    print(f"loading {total} hotpotqa instances, taking [{skip}:{total}]...")
    instances = load_hotpotqa(n=total)[skip:]
    n = len(instances)
    print(f"loaded {n}; answer-grounded best-of-{n_candidates} (keep if best answer-F1 >= "
          f"{keep_threshold}), {max_workers} workers...")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    results, n_done, n_kept, n_drop, n_err = [], 0, 0, 0, 0
    kept_f1_sum = 0.0
    # track the lift from best-of-N: how often is the best candidate better than the greedy one?
    greedy_f1_sum = best_f1_sum = 0.0

    with open(out_path, "a") as f, ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_gen_one, inst, ratio, n_candidates): i for i, inst in enumerate(instances)}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            n_done += 1
            if r["error"]:
                n_err += 1
            elif r["n_out"] < _MIN_OUT_TOKENS or r["answer_f1"] < keep_threshold:
                n_drop += 1
            else:
                n_kept += 1
                kept_f1_sum += r["answer_f1"]
                if r["all_f1s"]:
                    greedy_f1_sum += r["all_f1s"][0] if len(r["all_f1s"]) == 1 else min(r["all_f1s"])
                    best_f1_sum += r["answer_f1"]
                with lock:
                    f.write(json.dumps({k: r[k] for k in
                            ("question", "text", "compressed", "n_in", "n_out", "answer_f1")}) + "\n")
                    f.flush()
            if n_done % 10 == 0 or n_done == n:
                print(f"  [{n_done}/{n}] kept={n_kept} dropped(low ans-F1)={n_drop} err={n_err}")

    keep_rate = n_kept / max(1, n_done - n_err)
    avg_kept = kept_f1_sum / max(1, n_kept)
    print(f"\nKEPT {n_kept} answer-grounded pairs (avg answer-F1 of kept = {avg_kept:.3f})")
    print(f"keep-rate among non-errored = {keep_rate:.1%}")
    return {"n_kept": n_kept, "n_dropped": n_drop, "n_err": n_err,
            "keep_rate": keep_rate, "avg_answer_f1": avg_kept}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--candidates", type=int, default=4, help="best-of-N candidates per example")
    p.add_argument("--out", default="data/distill/answergrounded.jsonl")
    p.add_argument("--workers", type=int, default=MAX_WORKERS)
    p.add_argument("--keep-threshold", type=float, default=KEEP_THRESHOLD)
    p.add_argument("--skip", type=int, default=0)
    args = p.parse_args()
    stats = gen_to_file(n=args.n, out=args.out, ratio=args.ratio, n_candidates=args.candidates,
                        max_workers=args.workers, keep_threshold=args.keep_threshold, skip=args.skip)
    print(f"\nwrote answer-grounded survivors to {args.out}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
