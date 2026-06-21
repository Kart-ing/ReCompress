# Step 8 — Swap Real DeepSeek + HotpotQA Runner

> **Goal:** Replace all mock stubs with real DeepSeek API calls. Build the HotpotQA data loader and evaluation runner. Verify the token curve is still flat with real compression.

---

## What to build

- Replace `act1/compress.py`, `act1/solve.py`, `act1/tokens.py` with real implementations
- Build `experiments/hotpotqa_runner.py`

---

## act1/solve.py (real)

```python
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

# NOTE: act1/solve.py must NEVER import from act1/compress.py
# compress.py imports _deepseek_call from here — importing back would be circular
client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

def _deepseek_call(
    system: str,
    prompt: str,
    max_tokens: int = 512,
    fast: bool = False,
) -> str:
    """
    Shared internal call used by Echidna, TraumaExtractor, and Compressor.
    fast=True routes to same model for now — swap to a flash endpoint if available.

    Import chain: compress.py → solve.py → (nowhere else in act1)
    Never import compress.py from solve.py — that creates a circular import.
    """
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def solve(context: str, question: str) -> str:
    """Frozen solver — answers question given context. Returns concise answer string."""
    system = "Answer the question using only the provided context. Be as concise as possible — one sentence or fewer."
    prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return _deepseek_call(system, prompt, max_tokens=64)
```

---

## act1/tokens.py (real)

```python
def count_tokens(text: str) -> int:
    """
    Approximate token count using word-count × 1.3 BPE ratio.
    DeepSeek tokenizer not publicly available; this is a conservative upper bound.
    A word like "TechCorp" counts as 1 word but may tokenize to 2 BPE tokens.
    """
    return int(len(text.split()) * 1.3)
```

---

## act1/compress.py (real)

```python
# FIX: compress.py imports from solve.py (one direction only — never the reverse)
from act1.solve import _deepseek_call
from act1.tokens import count_tokens

def compress_ours(text: str, question: str, ratio: float) -> str:
    """
    Query-aware compressor. Keeps roughly `ratio` fraction of tokens.
    Uses DeepSeek to produce a faithful compressed version.
    """
    target_tokens = max(20, int(count_tokens(text) * ratio))
    system = f"""You are a precise text compressor.
Compress the following text to approximately {target_tokens} tokens.
The compressed text will be used to answer: "{question}"
Preserve all named entities, numbers, dates, and facts critical to answering the question.
Return only the compressed text — no preamble, no explanation."""
    return _deepseek_call(system, text, max_tokens=target_tokens + 20)
```

---

## Verify before running HotpotQA

Run these in order. Do not proceed to HotpotQA until all pass:

```bash
# 1. Sanity check — one real API call
python - <<'EOF'
from act1.solve import solve
print(solve("The Eiffel Tower is in Paris, France.", "Where is the Eiffel Tower?"))
EOF
# expect: Paris / Paris, France / something geographically correct

# 2. Check compress works
python - <<'EOF'
from act1.compress import compress_ours
result = compress_ours("Alice founded TechCorp in 2010. She raised 50 million dollars from Sequoia Capital.", "Who founded TechCorp?", ratio=0.5)
print(result)
print(f"Words: {len(result.split())}")
EOF
# expect: shorter text that still mentions Alice and TechCorp

# 3. Dry-run with mock (no API cost) — verify data loads correctly
python experiments/hotpotqa_runner.py --n 3 --no-llm
# expect: 3 samples, MOCK_ANSWER for both systems, F1=0 (expected with mock)
```

---

## experiments/hotpotqa_runner.py

```python
"""
Evaluates RbD-Compress vs Naive on HotpotQA distractor setting.
Simulates multi-turn: each supporting document is fed as a separate turn.

Run: python experiments/hotpotqa_runner.py --n 5        (quick test)
     python experiments/hotpotqa_runner.py --n 20       (full eval)
     python experiments/hotpotqa_runner.py --n 3 --no-llm  (dry run, no API cost)
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
Path("results").mkdir(exist_ok=True)

from datasets import load_dataset
from rouge_score import rouge_scorer

from rezero.session import ReZeroSession
from baselines.naive import NaiveSession
from act1.solve import solve

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

        for i, doc in enumerate(docs[:6]):
            rbd.add_turn(f"Document {i+1}: {doc}", "Noted.")
            naive.add_turn(f"Document {i+1}: {doc}", "Noted.")

        rbd_prompt   = rbd.prompt_for_solver()
        naive_prompt = naive.prompt_for_solver()

        rbd_answer   = solve(rbd_prompt,   question)
        naive_answer = solve(naive_prompt, question)

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
mkdir -p results

# Full run — save results
python experiments/hotpotqa_runner.py --n 20 | tee results/hotpotqa_results.txt
```

---

## Done when

- Sanity checks above pass before running HotpotQA
- Dry-run (`--no-llm`) completes without errors — confirms data loading works
- Full run produces non-empty answers with F1 > 0 on at least some samples
- RbD `avg_tok` stays near 300; Naive `avg_tok` is significantly higher
