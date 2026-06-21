import random
from engine.deepseek import solve
from engine.tokens import count_tokens

def coin_toss_solve(context: str, question: str, removal_rate: float = 0.5, seed: int = 42) -> dict:
    rng = random.Random(seed)
    words = context.split()
    kept = [w for w in words if rng.random() > removal_rate]
    if not kept:
        kept = words[:5]
    thinned_context = " ".join(kept)
    result = solve(thinned_context, question)
    return {
        "answer": result["answer"],
        "prompt_tokens": result["prompt_tokens"],
        "pre_compression_tokens": count_tokens(context),
        "post_compression_tokens": count_tokens(thinned_context),
    }
