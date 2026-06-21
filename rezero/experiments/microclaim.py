"""
§3.5 Micro-Claim: Protected Buffer (Variant A) vs Single Summary (Variant B)

Sweep r ∈ {0.10, 0.15, 0.20, 0.25, 0.30}
Metric: QA-F1 averaged over N_SEEDS HotpotQA conversations
Output: ratio, variant_a_f1, variant_b_f1, a_wins

Run: python experiments/microclaim.py > results/microclaim.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
Path("results").mkdir(exist_ok=True)

from datasets import load_dataset
from rouge_score import rouge_scorer

from rezero.session import ReZeroSession
from engine.compressor import compress
from engine.deepseek import solve
from engine.tokens import count_tokens

RATIOS  = [0.10, 0.15, 0.20, 0.25, 0.30]
N_SEEDS = 5
SCORER  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def f1(prediction: str, ground_truth: str) -> float:
    return SCORER.score(ground_truth, prediction)["rougeL"].fmeasure


def variant_b_prompt(history: list[dict], delta: str, ratio: float, goal: str) -> str:
    full_text    = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
    history_toks = count_tokens(full_text)
    cp_budget    = int(history_toks * ratio)
    total_budget = 50 + cp_budget + 100

    summary = compress(full_text, question=goal, ratio=ratio)
    words   = summary.split()
    while count_tokens(" ".join(words)) > total_budget and words:
        words.pop()

    return f"[SUMMARY]\n{' '.join(words)}\n\n[DELTA]\n{delta}"


def run():
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    samples = list(dataset)[:N_SEEDS]

    print("ratio,variant_a_f1,variant_b_f1,a_wins")
    for ratio in RATIOS:
        a_scores, b_scores = [], []

        for sample in samples:
            question = sample["question"]
            answer   = sample["answer"]
            docs = [
                f"{title}: {' '.join(sentences)}"
                for title, sentences in zip(
                    sample["context"]["title"],
                    sample["context"]["sentences"]
                )
            ]

            session = ReZeroSession(goal=question, use_llm=True, ratio=ratio)
            history = []
            for i, doc in enumerate(docs[:8]):
                user = f"Document {i+1}: {doc}"
                session.add_turn(user, "Noted.")
                history.append({"role": "user",      "content": user})
                history.append({"role": "assistant", "content": "Noted."})

            prompt_a = session.prompt_for_solver()
            delta    = docs[-1] if docs else ""

            prompt_b = variant_b_prompt(history, delta, ratio, question)

            ans_a = solve(prompt_a, question)
            ans_b = solve(prompt_b, question)

            a_scores.append(f1(ans_a, answer))
            b_scores.append(f1(ans_b, answer))

        avg_a  = sum(a_scores) / len(a_scores)
        avg_b  = sum(b_scores) / len(b_scores)
        print(f"{ratio},{avg_a:.3f},{avg_b:.3f},{avg_a > avg_b}")


if __name__ == "__main__":
    run()
