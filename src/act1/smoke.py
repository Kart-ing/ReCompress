"""1-instance smoke test: verify bear + deepseek (compressor + solver) all wire up.
Run: python -m src.act1.smoke
"""
from __future__ import annotations
from src.act1.data import load_hotpotqa, context_to_text
from src.act1.tokens import count_tokens
from src.act1.compress import compress_ours
from src.act1.bear import compress_bear
from src.act1.solve import solve
from src.act1.metrics import qa_f1


def main():
    print("loading 1 hotpotqa instance...")
    inst = load_hotpotqa(n=1)[0]
    text = context_to_text(inst)
    q = inst["question"]
    gold = inst["answer"]
    n_in = count_tokens(text)
    print(f"Q: {q}")
    print(f"gold: {gold}")
    print(f"input tokens: {n_in}")

    ratio = 0.3
    print(f"\n--- bear (blind deletion) @ ratio={ratio} ---")
    bear_out = compress_bear(text, ratio=ratio)
    print(f"bear out tokens: {count_tokens(bear_out)}")
    bear_ans = solve(bear_out, q)
    print(f"bear answer: {bear_ans}  (F1={qa_f1(bear_ans, gold):.2f})")

    print(f"\n--- ours (query-aware) @ ratio={ratio} ---")
    ours_out = compress_ours(text, q, ratio=ratio)
    print(f"ours out tokens: {count_tokens(ours_out)}")
    ours_ans = solve(ours_out, q)
    print(f"ours answer: {ours_ans}  (F1={qa_f1(ours_ans, gold):.2f})")

    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
