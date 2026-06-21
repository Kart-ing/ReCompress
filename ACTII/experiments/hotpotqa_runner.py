"""
Evaluates RbD-Compress vs Token Company vs Naive on HotpotQA distractor setting.
Simulates multi-turn: each supporting document is a separate turn.

Run: python experiments/hotpotqa_runner.py --n 5        (quick test)
     python experiments/hotpotqa_runner.py --n 30 --workers 30  (parallel)
     python experiments/hotpotqa_runner.py --n 3 --no-llm  (dry run)
"""
import sys
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, ".")
Path("results").mkdir(exist_ok=True)

from datasets import load_dataset
from rouge_score import rouge_scorer

from rezero.session import ReZeroSession
from baselines.naive import NaiveSession
from baselines.token_company import TokenCompanySession, tc_solve, init_tc_client
from engine.deepseek import solve, set_model
from engine.tokens import count_tokens

SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

_progress_lock = threading.Lock()
_progress_count = 0


def f1(prediction: str, ground_truth: str) -> float:
    return SCORER.score(ground_truth, prediction)["rougeL"].fmeasure


def run_single_sample(sample: dict, use_llm: bool, model: str, idx: int, total: int) -> dict | None:
    global _progress_count
    try:
        question = sample["question"]
        answer   = sample["answer"]

        docs = [
            f"{title}: {' '.join(sentences)}"
            for title, sentences in zip(
                sample["context"]["title"],
                sample["context"]["sentences"]
            )
        ]

        rbd_t0 = time.time()
        rbd   = ReZeroSession(goal=question, use_llm=use_llm)
        naive = NaiveSession(goal=question)
        tc    = TokenCompanySession(goal=question)

        for i, doc in enumerate(docs[:6]):
            rbd.add_turn(f"Document {i+1}: {doc}", "Noted.")
            naive.add_turn(f"Document {i+1}: {doc}", "Noted.")
            tc.add_turn(f"Document {i+1}: {doc}", "Noted.")
        rbd_build_time = time.time() - rbd_t0

        rbd_prompt   = rbd.prompt_for_solver()
        naive_prompt = naive.prompt_for_solver()
        tc_prompt    = tc.prompt_for_solver()

        if use_llm:
            t0 = time.time()
            rbd_answer   = solve(rbd_prompt,   question)
            rbd_solve_time = time.time() - t0
            t0 = time.time()
            naive_answer = solve(naive_prompt, question)
            naive_time = time.time() - t0
            t0 = time.time()
            tc_result    = tc_solve(tc_prompt, question, model=model)
            tc_time = time.time() - t0
            tc_answer    = tc_result["answer"]
            tc_actual_tok = tc_result["prompt_tokens"]
        else:
            rbd_answer   = "PLACEHOLDER"
            naive_answer = "PLACEHOLDER"
            tc_answer    = "PLACEHOLDER"
            tc_actual_tok = 0
            rbd_build_time = rbd_solve_time = naive_time = tc_time = 0.0

        rbd_time = rbd_build_time + rbd_solve_time

        result = {
            "idx":          idx,
            "question":     question,
            "gt":           answer,
            "rbd_answer":   rbd_answer,
            "naive_answer": naive_answer,
            "tc_answer":    tc_answer,
            "rbd_f1":       f1(rbd_answer, answer),
            "naive_f1":     f1(naive_answer, answer),
            "tc_f1":        f1(tc_answer, answer),
            "rbd_tok":      rbd.token_count(),
            "naive_tok":    naive.token_count(),
            "tc_tok":       tc.token_count(),
            "tc_actual_tok": tc_actual_tok,
            "rbd_time":     rbd_time,
            "naive_time":   naive_time,
            "tc_time":      tc_time,
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


def run(n_samples: int = 20, use_llm: bool = True, model: str = "deepseek-chat",
        workers: int = 10, tc_aggressiveness: float = 0.2):
    global _progress_count
    _progress_count = 0

    print(f"Loading HotpotQA dataset...")
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    samples = list(dataset)[:n_samples]
    print(f"Loaded {len(samples)} samples. Running with {workers} workers...\n")

    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_single_sample, sample, use_llm, model, idx, len(samples)): idx
            for idx, sample in enumerate(samples)
        }
        results = []
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    elapsed = time.time() - t0
    results.sort(key=lambda r: r["idx"])

    print(f"\n{'='*60}")
    for r in results:
        print(f"[{r['idx']+1}/{n_samples}] Q: {r['question'][:60]}...")
        print(f"  GT:    {r['gt']}")
        print(f"  RbD:   {r['rbd_answer']}  (F1={r['rbd_f1']:.3f}, tok={r['rbd_tok']}, {r['rbd_time']:.1f}s)")
        print(f"  TC:    {r['tc_answer']}  (F1={r['tc_f1']:.3f}, tok={r['tc_tok']}→{r['tc_actual_tok']}, {r['tc_time']:.1f}s)")
        print(f"  Naive: {r['naive_answer']}  (F1={r['naive_f1']:.3f}, tok={r['naive_tok']}, {r['naive_time']:.1f}s)")
        print()

    if results:
        n = len(results)
        print("=" * 60)
        print(f"Results over {n}/{n_samples} samples ({elapsed:.1f}s total, {workers} workers):")
        print(f"  RbD-Compress    avg F1: {sum(r['rbd_f1'] for r in results)/n:.3f}  avg_tok: {sum(r['rbd_tok'] for r in results)/n:.0f}  avg_time: {sum(r['rbd_time'] for r in results)/n:.1f}s  (model={model})")
        print(f"  Token Company   avg F1: {sum(r['tc_f1'] for r in results)/n:.3f}  avg_tok: {sum(r['tc_tok'] for r in results)/n:.0f}→{sum(r['tc_actual_tok'] for r in results)/n:.0f}  avg_time: {sum(r['tc_time'] for r in results)/n:.1f}s  (aggressiveness={tc_aggressiveness})")
        print(f"  Naive           avg F1: {sum(r['naive_f1'] for r in results)/n:.3f}  avg_tok: {sum(r['naive_tok'] for r in results)/n:.0f}  avg_time: {sum(r['naive_time'] for r in results)/n:.1f}s  (model={model})")
        print(f"\n  Token Company tok = uncompressed→compressed (prompt_tokens from DeepSeek response)")
    else:
        print("All samples failed.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n",       type=int, default=20)
    p.add_argument("--no-llm",  action="store_true")
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--model",   type=str, default="deepseek-chat",
                   help="DeepSeek model name (e.g. deepseek-chat, deepseek-v4-flash)")
    p.add_argument("--tc-aggressiveness", type=float, default=0.2,
                   help="Token Company compression aggressiveness (0.0-1.0, default 0.2)")
    args = p.parse_args()
    set_model(args.model)
    init_tc_client(aggressiveness=args.tc_aggressiveness)
    run(n_samples=args.n, use_llm=not args.no_llm, model=args.model,
        workers=args.workers, tc_aggressiveness=args.tc_aggressiveness)
