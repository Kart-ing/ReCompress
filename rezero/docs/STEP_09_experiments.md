# Step 9 — §3.5 Micro-Claim & Turn-15 Probe

> **Goal:** Run the two key experiments that prove (or honestly refute) our novelty claim. Do not modify results — honest reporting is the depth-of-research signal.

---

## What to build

`experiments/microclaim.py` and `experiments/turn15_probe.py`

---

## The Claims Being Tested

**§3.5 claim:** Separating trauma memory into a protected buffer allows the checkpoint to be compressed more aggressively at equal QA-F1 than a single undifferentiated summary at the same total token budget.

**Turn-15 probe:** A 14-turn scripted conversation where the final question references a specific numeric detail from turn 4 (1-indexed) that the Trauma Extractor was NOT designed to flag. We expect naive to win. Shipping this failure is intentional.

---

## experiments/microclaim.py

```python
"""
§3.5 Micro-Claim: Protected Buffer (Variant A) vs Single Summary (Variant B)

Sweep r ∈ {0.10, 0.15, 0.20, 0.25, 0.30}
Metric: QA-F1 averaged over N_SEEDS HotpotQA conversations
Output CSV: ratio, variant_a_f1, variant_b_f1, a_wins

Run: python experiments/microclaim.py > results/microclaim.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
Path("results").mkdir(exist_ok=True)

from datasets import load_dataset
from rouge_score import rouge_scorer

from rezero.session import ReZeroSession
from act1.compress import compress_ours
from act1.solve import solve
from act1.tokens import count_tokens

RATIOS  = [0.10, 0.15, 0.20, 0.25, 0.30]
N_SEEDS = 5
SCORER  = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def f1(prediction: str, ground_truth: str) -> float:
    return SCORER.score(ground_truth, prediction)["rougeL"].fmeasure


def variant_b_prompt(history: list[dict], delta: str, ratio: float, goal: str) -> str:
    """
    Control: single undifferentiated summary at same total budget as Variant A.
    Budget = 50 (trauma budget) + checkpoint_budget(r) + 100 (delta budget).
    No protected layer — everything goes through one compressor pass.
    """
    full_text    = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in history)
    history_toks = count_tokens(full_text)
    cp_budget    = int(history_toks * ratio)
    total_budget = 50 + cp_budget + 100

    summary = compress_ours(full_text, question=goal, ratio=ratio)
    words   = summary.split()
    while count_tokens(" ".join(words)) > total_budget and words:
        words.pop()

    return f"[SUMMARY]\n{' '.join(words)}\n\n[DELTA]\n{delta}"


def run():
    dataset = load_dataset("hotpot_qa", "distractor", split="validation")
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

            # ── Variant A: RbD-Compress ──────────────────────────────────
            # FIX: pass ratio at init, not mid-session
            # Setting checkpoint_builder.ratio after Echidna may have already
            # fired a checkpoint at the default ratio — pass it at construction.
            session = ReZeroSession(goal=question, use_llm=True, ratio=ratio)
            history = []
            for i, doc in enumerate(docs[:8]):
                user = f"Document {i+1}: {doc}"
                session.add_turn(user, "Noted.")
                history.append({"role": "user",      "content": user})
                history.append({"role": "assistant", "content": "Noted."})

            prompt_a = session.prompt_for_solver()
            delta    = docs[-1] if docs else ""

            # ── Variant B: single summary ────────────────────────────────
            prompt_b = variant_b_prompt(history, delta, ratio, question)

            ans_a = solve(prompt_a, question)
            ans_b = solve(prompt_b, question)

            a_scores.append(f1(ans_a, answer))
            b_scores.append(f1(ans_b, answer))

        avg_a  = sum(a_scores) / len(a_scores)
        avg_b  = sum(b_scores) / len(b_scores)
        a_wins = avg_a > avg_b
        print(f"{ratio},{avg_a:.3f},{avg_b:.3f},{a_wins}")


if __name__ == "__main__":
    run()
```

> **Required session change:** `ReZeroSession.__init__` must accept a `ratio` kwarg and pass it to `CheckpointBuilder`:
> ```python
> def __init__(self, goal: str, use_llm: bool = False, ratio: float = 0.20):
>     ...
>     self.checkpoint_builder = CheckpointBuilder(goal=goal, ratio=ratio)
> ```
> Add this to `rezero/session.py` before running microclaim.

---

## experiments/turn15_probe.py

```python
"""
Honesty test: The final question references a specific numeric detail from
turn 4 (1-indexed) that the Trauma Extractor was NOT designed to flag.

NOTE on turn numbering: PROBE_CONVO is a 14-element list (indices 0–13).
- Turn 4 (1-indexed) = index 3: plants the Louvre 9M visitors figure
- Turn 14 (1-indexed) = index 13: references that buried detail

We expect naive to win this probe. We ship this failure ourselves.

Run: python experiments/turn15_probe.py | tee results/turn15_probe.txt
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
Path("results").mkdir(exist_ok=True)

from rezero.session import ReZeroSession
from baselines.naive import NaiveSession
from act1.solve import solve
from rouge_score import rouge_scorer

SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

# 14 turns total (indices 0–13, 1-indexed as turns 1–14)
PROBE_CONVO = [
    # Turn 1
    ("What is the capital of France?",
     "Paris is the capital of France."),
    # Turn 2
    ("What is France known for?",
     "France is known for art, cuisine, and the Eiffel Tower."),
    # Turn 3
    ("Who is the current French president?",
     "Emmanuel Macron is the current French president."),
    # Turn 4 ← plants 9 million visitors (index 3)
    ("How many people visit the Louvre per year?",
     "The Louvre attracts approximately 9 million visitors annually."),
    # Turn 5
    ("What is the most famous painting there?",
     "The Mona Lisa by Leonardo da Vinci is the most famous."),
    # Turn 6
    ("When was the Mona Lisa painted?",
     "The Mona Lisa was painted between 1503 and 1519."),
    # Turn 7
    ("Where is da Vinci from?",
     "Leonardo da Vinci was born in Vinci, Tuscany, Italy in 1452."),
    # Turn 8
    ("What other works did he create?",
     "He also created The Last Supper and Vitruvian Man."),
    # Turn 9
    ("Is The Last Supper in the Louvre?",
     "No, The Last Supper is in Milan, Italy, at Santa Maria delle Grazie."),
    # Turn 10
    ("What technique did da Vinci use?",
     "Da Vinci used sfumato, a technique of blending tones without hard edges."),
    # Turn 11
    ("How did he learn painting?",
     "He apprenticed under Andrea del Verrocchio in Florence."),
    # Turn 12
    ("What else was da Vinci known for?",
     "He was also a scientist, engineer, and anatomist."),
    # Turn 13
    ("Did he have any famous students?",
     "Yes, Giovanni Antonio Boltraffio was one of his notable students."),
    # Turn 14 ← references turn 4's buried detail (index 13)
    ("So combining what we know — how does the Louvre visitor count compare to the Vatican Museums?",
     "The Vatican Museums attract about 6 million visitors, so the Louvre's 9 million is about 50% higher."),
]

PROBE_QUESTION = "How many more annual visitors does the Louvre receive compared to the Vatican Museums?"
GROUND_TRUTH   = "approximately 3 million more"
GOAL           = "Learn about French art and museums"


def run():
    rbd   = ReZeroSession(goal=GOAL, use_llm=True)
    naive = NaiveSession(goal=GOAL)

    for user, assistant in PROBE_CONVO:
        rbd.add_turn(user,   assistant)
        naive.add_turn(user, assistant)

    rbd_ans   = solve(rbd.prompt_for_solver(),   PROBE_QUESTION)
    naive_ans = solve(naive.prompt_for_solver(), PROBE_QUESTION)

    rbd_f1   = SCORER.score(GROUND_TRUTH, rbd_ans)["rougeL"].fmeasure
    naive_f1 = SCORER.score(GROUND_TRUTH, naive_ans)["rougeL"].fmeasure

    print("\n" + "=" * 60)
    print("TURN-15 PROBE — Honesty Test")
    print("=" * 60)
    print(f"Question:     {PROBE_QUESTION}")
    print(f"Ground truth: {GROUND_TRUTH}")
    print(f"RbD answer:   {rbd_ans}")
    print(f"Naive answer: {naive_ans}")
    print(f"RbD F1:       {rbd_f1:.3f}")
    print(f"Naive F1:     {naive_f1:.3f}")
    print()
    if naive_f1 > rbd_f1:
        print(">> RESULT: Naive wins (expected — the detail was not flagged as critical)")
        print("   This is honest. We ship this failure ourselves.")
    elif rbd_f1 > naive_f1:
        print(">> RESULT: RbD-Compress wins (Trauma Extractor caught the numeric detail)")
        print("   Unexpected positive — verify trauma content below.")
    else:
        print(">> RESULT: Tie")

    print(f"\nTrauma memory at end: {rbd.trauma_extractor.get()}")
    print(f"Checkpoint IDs:       {rbd.list_checkpoints()}")
    print(f"RbD tokens:           {rbd.token_count()}")
    print(f"Naive tokens:         {naive.token_count()}")


if __name__ == "__main__":
    run()
```

---

## Run experiments

```bash
mkdir -p results

# §3.5 sweep (5 samples × 5 ratios × 2 variants × 1 API call each = ~50 API calls)
python experiments/microclaim.py | tee results/microclaim.csv

# Turn-15 probe
python experiments/turn15_probe.py | tee results/turn15_probe.txt
```

---

## Interpreting §3.5 results

| Outcome | Meaning | Action |
|---|---|---|
| Variant A F1 > Variant B at lower r | **Pass** — protected buffer works | Lead with this in the paper |
| Variant A ≈ Variant B | Inconclusive | Report honestly; lean on token curve + revert |
| Variant B F1 > Variant A | **Fail** | Concede §3.5; do not overclaim on stage |

---

## Done when

- `microclaim.csv` has 5 rows (one per ratio) with both F1 columns populated
- `turn15_probe.txt` shows both answers and F1 scores
- Neither script crashes
- `ReZeroSession` accepts `ratio` kwarg (added to session.py before running)
- Results reported exactly as measured — no rounding to look better
