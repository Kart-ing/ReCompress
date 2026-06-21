"""
Evaluates RbD-Compress vs Naive on HotpotQA distractor setting.
Simulates multi-turn: each supporting document is a separate turn.

Run: python experiments/hotpotqa_runner.py --n 5        (quick test)
     python experiments/hotpotqa_runner.py --n 20       (full eval)
     python experiments/hotpotqa_runner.py --n 3 --no-llm  (dry run)
"""
import sys
from pathlib import Path
import time
sys.path.insert(0, ".")
Path("results").mkdir(exist_ok=True)

from datasets import load_dataset
from rouge_score import rouge_scorer

from rezero.session import ReZeroSession
from baselines.naive import NaiveSession
from engine.deepseek import solve, set_model
from engine.tokens import count_tokens

SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def f1(prediction: str, ground_truth: str) -> float:
    return SCORER.score(ground_truth, prediction)["rougeL"].fmeasure


def print_prompt_breakdown(label: str, prompt: str):
    import re
    parts = {}
    for key in ["TRAUMA", "CHECKPOINT", "DELTA", "GOAL", "HISTORY", "SUMMARY"]:
        match = re.search(rf'\[{key}\]\n(.*?)(?=\n\[|$)', prompt, re.DOTALL)
        if match:
            parts[key] = match.group(1).strip()
    print(f"  --- {label} prompt breakdown ---")
    for key, text in parts.items():
        tok = count_tokens(text) if text else 0
        preview = text[:100].replace("\n", "\\n") if text else "(empty)"
        print(f"    [{key}] ({tok} tok): {preview}{'...' if len(text) > 100 else ''}")
    print(f"    TOTAL: {count_tokens(prompt)} tok")


def run(n_samples: int = 20, use_llm: bool = True, verbose: bool = False):
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    samples = list(dataset)[:n_samples]

    rbd_f1_scores, naive_f1_scores = [], []
    rbd_token_counts, naive_token_counts = [], []

    for idx, sample in enumerate(samples):
        question = sample["question"]
        answer   = sample["answer"]

        docs = [
            f"{title}: {' '.join(sentences)}"
            for title, sentences in zip(
                sample["context"]["title"],
                sample["context"]["sentences"]
            )
        ]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Sample {idx+1}/{n_samples}: {question}")
            print(f"GT: {answer}")
            print(f"Docs: {len(docs)} total, using first 6")

        rbd_t0 = time.time()
        rbd   = ReZeroSession(goal=question, use_llm=use_llm, verbose=verbose)
        naive = NaiveSession(goal=question)

        for i, doc in enumerate(docs[:6]):
            rbd.add_turn(f"Document {i+1}: {doc}", "Noted.")
            naive.add_turn(f"Document {i+1}: {doc}", "Noted.")
        rbd_build_time = time.time() - rbd_t0

        rbd_prompt   = rbd.prompt_for_solver()
        naive_prompt = naive.prompt_for_solver()

        if verbose:
            print_prompt_breakdown("RbD", rbd_prompt)
            print_prompt_breakdown("Naive", naive_prompt)

        if use_llm:
            t0 = time.time()
            rbd_answer   = solve(rbd_prompt,   question)
            rbd_solve_time = time.time() - t0
            t0 = time.time()
            naive_answer = solve(naive_prompt, question)
            naive_solve_time = time.time() - t0
        else:
            rbd_answer   = "PLACEHOLDER"
            naive_answer = "PLACEHOLDER"
            rbd_solve_time = naive_solve_time = 0.0

        rbd_total_time = rbd_build_time + rbd_solve_time

        rbd_score   = f1(rbd_answer,   answer)
        naive_score = f1(naive_answer, answer)

        rbd_f1_scores.append(rbd_score)
        naive_f1_scores.append(naive_score)
        rbd_token_counts.append(rbd.token_count())
        naive_token_counts.append(naive.token_count())

        print(f"\n[{idx+1}/{n_samples}] Q: {question[:60]}...")
        print(f"  GT:    {answer}")
        print(f"  RbD:   {rbd_answer}  (F1={rbd_score:.3f}, tok={rbd.token_count()}, build={rbd_build_time:.1f}s solve={rbd_solve_time:.1f}s total={rbd_total_time:.1f}s)")
        print(f"  Naive: {naive_answer}  (F1={naive_score:.3f}, tok={naive.token_count()}, solve={naive_solve_time:.1f}s)")
        print()

    print("=" * 60)
    print(f"Results over {n_samples} samples:")
    print(f"  RbD-Compress  avg F1: {sum(rbd_f1_scores)/len(rbd_f1_scores):.3f}  avg_tok: {sum(rbd_token_counts)/len(rbd_token_counts):.0f}")
    print(f"  Naive         avg F1: {sum(naive_f1_scores)/len(naive_f1_scores):.3f}  avg_tok: {sum(naive_token_counts)/len(naive_token_counts):.0f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n",      type=int,  default=20)
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--model",  type=str,  default="deepseek-chat",
                   help="DeepSeek model name (e.g. deepseek-chat, deepseek-v4-flash)")
    args = p.parse_args()
    set_model(args.model)
    run(n_samples=args.n, use_llm=not args.no_llm, verbose=args.verbose)
