import sys
import time
import random
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, ".")
Path("results").mkdir(exist_ok=True)

from datasets import load_dataset
from rouge_score import rouge_scorer

from baselines.naive import NaiveSession
from baselines.coin_toss import coin_toss_solve
from baselines.token_company import tc_solve, init_tc_clients
from engine.deepseek import solve, set_model
from engine.tokens import count_tokens
from engine import ratelimit

SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

_progress_lock = threading.Lock()
_progress_count = 0


def f1(prediction: str, ground_truth: str) -> float:
    return SCORER.score(ground_truth, prediction)["rougeL"].fmeasure


def run_single_sample(sample, use_llm, model, idx, total, rates, tc_aggrs, base_seed):
    global _progress_count
    try:
        question = sample["question"]
        answer = sample["answer"]

        docs = [
            f"{title}: {' '.join(sentences)}"
            for title, sentences in zip(
                sample["context"]["title"],
                sample["context"]["sentences"],
            )
        ]

        session = NaiveSession(goal=question)
        for i, doc in enumerate(docs[:6]):
            session.add_turn(f"Document {i+1}: {doc}", "Noted.")

        prompt = session.prompt_for_solver()
        naive_tok = session.token_count()

        token_company = {}
        coin_toss = {}
        if use_llm:
            naive_res = solve(prompt, question)
            naive_answer = naive_res["answer"]
            naive_prompt_tok = naive_res["prompt_tokens"]

            for a in tc_aggrs:
                tc = tc_solve(prompt, question, aggressiveness=a, model=model)
                token_company[a] = {
                    "answer": tc["answer"],
                    "f1": f1(tc["answer"], answer),
                    "prompt_tok": tc["prompt_tokens"],
                }

            for r in rates:
                ct = coin_toss_solve(prompt, question, removal_rate=r, seed=base_seed + int(r * 100))
                coin_toss[r] = {
                    "answer": ct["answer"],
                    "f1": f1(ct["answer"], answer),
                    "prompt_tok": ct["prompt_tokens"],
                    "compressed_tok": ct["post_compression_tokens"],
                }
        else:
            naive_answer = "PLACEHOLDER"
            naive_prompt_tok = 0
            for a in tc_aggrs:
                token_company[a] = {"answer": "PLACEHOLDER", "f1": 0.0, "prompt_tok": 0}
            words = prompt.split()
            for r in rates:
                rng = random.Random(base_seed + int(r * 100))
                kept = [w for w in words if rng.random() > r] or words[:5]
                coin_toss[r] = {
                    "answer": "PLACEHOLDER",
                    "f1": 0.0,
                    "prompt_tok": 0,
                    "compressed_tok": count_tokens(" ".join(kept)),
                }

        result = {
            "idx": idx,
            "question": question,
            "gt": answer,
            "naive_answer": naive_answer,
            "naive_f1": f1(naive_answer, answer),
            "naive_tok": naive_tok,
            "naive_prompt_tok": naive_prompt_tok,
            "token_company": token_company,
            "coin_toss": coin_toss,
        }

        with _progress_lock:
            _progress_count += 1
            print(f"  Sample {_progress_count}/{total} complete (#{idx+1}: {question[:50]}...)")

        return result

    except Exception as e:
        with _progress_lock:
            _progress_count += 1
            print(f"  Sample {_progress_count}/{total} FAILED (#{idx+1}): {e}")
        return None


def run(n_samples, use_llm, model, workers, tc_aggrs, rates, seed):
    global _progress_count
    _progress_count = 0

    print("Loading HotpotQA dataset...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    samples = list(dataset)[:n_samples]
    print(f"Loaded {len(samples)} samples. Running with {workers} workers...\n")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_single_sample, sample, use_llm, model, idx, len(samples), rates, tc_aggrs, seed): idx
            for idx, sample in enumerate(samples)
        }
        results = [f.result() for f in as_completed(futures)]
    results = [r for r in results if r is not None]
    elapsed = time.time() - t0
    results.sort(key=lambda r: r["idx"])

    if not results:
        print("All samples failed.")
        return

    n = len(results)
    base_tok = sum(r["naive_tok"] for r in results) / n
    naive_f1 = sum(r["naive_f1"] for r in results) / n

    tc_f1 = {}
    tc_post = {}
    for a in tc_aggrs:
        tc_f1[a] = sum(r["token_company"][a]["f1"] for r in results) / n
        tc_post[a] = sum(r["token_company"][a]["prompt_tok"] for r in results) / n

    ct_f1 = {}
    ct_post = {}
    for r in rates:
        ct_f1[r] = sum(s["coin_toss"][r]["f1"] for s in results) / n
        ct_post[r] = sum(s["coin_toss"][r]["compressed_tok"] for s in results) / n

    print(f"\n{'='*64}")
    print(f"Results over {n}/{n_samples} samples ({elapsed:.1f}s total, {workers} workers):")
    print(f"  Naive                    avg F1: {naive_f1:.3f}  avg_tok: {base_tok:.0f}")
    print()
    for a in tc_aggrs:
        print(f"  Token Company aggr={a:.2f}  avg F1: {tc_f1[a]:.3f}  avg_tok: {base_tok:.0f}→{tc_post[a]:.0f}")
    print()
    for r in rates:
        print(f"  Coin Toss     r={r:.2f}     avg F1: {ct_f1[r]:.3f}  avg_tok: {base_tok:.0f}→{ct_post[r]:.0f}")

    print()
    print("  Token Company vs Coin Toss (closest random-removal equivalent):")
    for a in tc_aggrs:
        best = min(rates, key=lambda r: abs(ct_f1[r] - tc_f1[a]))
        print(f"  >> TC aggr={a:.2f} (F1 {tc_f1[a]:.3f}) ≈ Coin Toss r={best:.2f} (F1 {ct_f1[best]:.3f})  [random removal of {int(best*100)}% of words]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--model", type=str, default="deepseek-chat")
    p.add_argument("--tc-aggressiveness", type=str, default="0.1,0.3,0.5,0.7,0.9")
    p.add_argument("--rates", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rpm", type=float, default=60, help="Max DeepSeek requests per minute (0 = unlimited)")
    args = p.parse_args()

    tc_aggrs = [float(x) for x in args.tc_aggressiveness.split(",")]
    rates = [float(x) for x in args.rates.split(",")]
    set_model(args.model)
    ratelimit.set_rpm(args.rpm)
    init_tc_clients(tc_aggrs)
    run(
        n_samples=args.n,
        use_llm=not args.no_llm,
        model=args.model,
        workers=args.workers,
        tc_aggrs=tc_aggrs,
        rates=rates,
        seed=args.seed,
    )
