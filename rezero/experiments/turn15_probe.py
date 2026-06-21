"""
Honesty test: The final question references a specific numeric detail from
turn 4 (1-indexed) that the Trauma Extractor was NOT designed to flag.

14 turns total (indices 0-13, 1-indexed as turns 1-14).
Turn 4 = index 3: plants the Louvre 9M visitors figure.
Turn 14 = index 13: references that buried detail.

We expect naive to win. We ship this failure ourselves.

Run: python experiments/turn15_probe.py | tee results/turn15_probe.txt
"""
import sys
from pathlib import Path
sys.path.insert(0, ".")
Path("results").mkdir(exist_ok=True)

from rezero.session import ReZeroSession
from baselines.naive import NaiveSession
from engine.deepseek import solve
from rouge_score import rouge_scorer

SCORER = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

PROBE_CONVO = [
    ("What is the capital of France?",            "Paris is the capital of France."),
    ("What is France known for?",                 "France is known for art, cuisine, and the Eiffel Tower."),
    ("Who is the current French president?",      "Emmanuel Macron is the current French president."),
    ("How many people visit the Louvre per year?","The Louvre attracts approximately 9 million visitors annually."),
    ("What is the most famous painting there?",   "The Mona Lisa by Leonardo da Vinci is the most famous."),
    ("When was the Mona Lisa painted?",           "The Mona Lisa was painted between 1503 and 1519."),
    ("Where is da Vinci from?",                   "Leonardo da Vinci was born in Vinci, Tuscany, Italy in 1452."),
    ("What other works did he create?",           "He also created The Last Supper and Vitruvian Man."),
    ("Is The Last Supper in the Louvre?",         "No, The Last Supper is in Milan, Italy."),
    ("What technique did da Vinci use?",          "Da Vinci used sfumato, blending tones without hard edges."),
    ("How did he learn painting?",                "He apprenticed under Andrea del Verrocchio in Florence."),
    ("What else was da Vinci known for?",         "He was also a scientist, engineer, and anatomist."),
    ("Did he have any famous students?",          "Yes, Giovanni Antonio Boltraffio was one of his notable students."),
    ("How does the Louvre visitor count compare to the Vatican Museums?",
     "The Vatican Museums attract about 6 million visitors, so the Louvre's 9 million is about 50% higher."),
]

PROBE_QUESTION = "How many more annual visitors does the Louvre receive compared to the Vatican Museums?"
GROUND_TRUTH   = "approximately 3 million more"
GOAL           = "Learn about French art and museums"


def run():
    rbd   = ReZeroSession(goal=GOAL, use_llm=True)
    naive = NaiveSession(goal=GOAL)

    for user, assistant in PROBE_CONVO:
        rbd.add_turn(user, assistant)
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
        print(">> RESULT: Naive wins (expected — detail not flagged as critical)")
        print("   This is honest. We ship this failure ourselves.")
    elif rbd_f1 > naive_f1:
        print(">> RESULT: RbD-Compress wins (Trauma Extractor caught the numeric detail)")
    else:
        print(">> RESULT: Tie")

    print(f"\nTrauma memory at end: {rbd.trauma_extractor.get()}")
    print(f"Checkpoint IDs:       {rbd.list_checkpoints()}")
    print(f"RbD tokens:           {rbd.token_count()}")
    print(f"Naive tokens:         {naive.token_count()}")


if __name__ == "__main__":
    run()
