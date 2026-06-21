from engine.deepseek import call
from engine.tokens import count_tokens

def compress(text: str, question: str, ratio: float, use_llm: bool = True, exclude: str = "") -> str:
    target = max(20, int(count_tokens(text) * ratio))
    if not use_llm:
        words = text.split()
        while count_tokens(" ".join(words)) > target and words:
            words.pop()
        return " ".join(words)
    system = (
        f"You are a precise text compressor. "
        f"Compress the following text to approximately {target} tokens. "
        f'The result will be used to answer: "{question}". '
        f"Preserve all named entities, numbers, dates, and facts critical to the question. "
        f"Return only the compressed text — no preamble, no explanation."
    )
    if exclude:
        system += (
            f"\nDo NOT include these facts (they are already stored separately): {exclude}"
            f"\nFocus on context, relationships, and details NOT covered above."
        )
    return call(system, text, max_tokens=target + 50)
