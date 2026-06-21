def count_tokens(text: str) -> int:
    """
    Approximate token count: word count × 1.3 BPE ratio.
    DeepSeek tokenizer is not publicly available.
    1.3 is a conservative upper bound for English text.
    """
    return int(len(text.split()) * 1.3)
