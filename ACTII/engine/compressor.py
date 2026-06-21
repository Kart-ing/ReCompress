from engine.deepseek import call
from engine.tokens import count_tokens

def compress(text: str, question: str, ratio: float) -> str:
    target = max(20, int(count_tokens(text) * ratio))
    system = (
        f"You are a precise text compressor. "
        f"Compress the following text to approximately {target} tokens. "
        f'The result will be used to answer: "{question}". '
        f"Preserve all named entities, numbers, dates, and facts critical to the question. "
        f"Return only the compressed text — no preamble, no explanation."
    )
    return call(system, text, max_tokens=target + 20)
