# Step 8 — HotpotQA Evaluation Runner

> **Goal:** Build the HotpotQA data loader and evaluation runner. Compare RbD-Compress vs Naive on multi-hop QA using our `engine/deepseek.py` solver.

---

## What to build

`experiments/hotpotqa_runner.py`

---

## Verify engine is working first

Run these before touching HotpotQA:

```bash
# 1. Solver sanity check
python - <<'EOF'
from engine.deepseek import solve
print(solve("The Eiffel Tower is in Paris, France.", "Where is the Eiffel Tower?"))
EOF
# expect: Paris / Paris, France

# 2. Compressor sanity check
python - <<'EOF'
from engine.compressor import compress
result = compress(
    "Alice founded Tech Corp in 2010. She raised 50 million dollars from Sequoia Capital.",
    "Who founded Tech Corp?",
    ratio=0.5
)
print(result)
print(f"Words: {len(result.split())}")
EOF
# expect: shorter text still mentioning Alice and Tech Corp

# 3. Dry run with no LLM calls — verifies HotpotQA data loads correctly
python experiments/hotpotqa_runner.py --n 3 --no-llm
# expect: 3 samples, placeholder answers, F1 scores logged
```

---

## experiments/hotpotqa_runner.py

```python
"""
Evaluates RbD-Compress vs Naive on HotpotQA distractor setting.
Simulates multi-turn: each supporting document is a separate turn.

Run: python experiments/hotpotqa_runner.py --n 5        (quick test)
     python experiments/hotpotqa_runner.py --n 20       (full eval)
     python experiments/hotpotqa_runner.py --n 3 --no-llm  (dry run)
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
Path("results").mkdir(exist_ok=True)

from datasets import load_dataset
from rouge_score import rouge_scorer

from rezero.session import ReZeroSession
from baselines.naive import NaiveSession
from engine.deepseek import solve

SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def f1(prediction: str, ground_truth: str) -> float:
    return SCORER.score(ground_truth, prediction)["rougeL"].fmeasure


def run(n_samples: int = 20, use_llm: bool = True):
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
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

        rbd   = ReZeroSession(goal=question, use_llm=use_llm)
        naive = NaiveSession(goal=question)

        # feed each document as a turn
        for i, doc in enumerate(docs[:6]):
            rbd.add_turn(f"Document {i+1}: {doc}", "Noted.")
            naive.add_turn(f"Document {i+1}: {doc}", "Noted.")

        rbd_prompt   = rbd.prompt_for_solver()
        naive_prompt = naive.prompt_for_solver()

        if use_llm:
            rbd_answer   = solve(rbd_prompt,   question)
            naive_answer = solve(naive_prompt, question)
        else:
            rbd_answer   = "PLACEHOLDER"
            naive_answer = "PLACEHOLDER"

        rbd_score   = f1(rbd_answer,   answer)
        naive_score = f1(naive_answer, answer)

        rbd_f1_scores.append(rbd_score)
        naive_f1_scores.append(naive_score)
        rbd_token_counts.append(rbd.token_count())
        naive_token_counts.append(naive.token_count())

        print(f"[{idx+1}/{n_samples}] Q: {question[:60]}...")
        print(f"  GT:    {answer}")
        print(f"  RbD:   {rbd_answer}  (F1={rbd_score:.3f}, tok={rbd.token_count()})")
        print(f"  Naive: {naive_answer}  (F1={naive_score:.3f}, tok={naive.token_count()})")
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
    args = p.parse_args()
    run(n_samples=args.n, use_llm=not args.no_llm)
```

---

## Run full evaluation

```bash
python experiments/hotpotqa_runner.py --n 20 | tee results/hotpotqa_results.txt
```

---

## Done when

- Dry run (`--no-llm`) completes without errors — data loads correctly
- Full run produces non-empty answers with F1 > 0 on at least some samples
- RbD `avg_tok` stays near 300; Naive `avg_tok` is significantly higher
