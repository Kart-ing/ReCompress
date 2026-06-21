# Step 1 — Project Skeleton & Mock Stubs

> **Goal:** Create folder structure, install deps, wire mock stubs for all Act 1 interfaces so every downstream step is unblocked immediately. Zero real API calls in this step.

---

## What to build

```
rbd_compress/
├── act1/
│   ├── __init__.py
│   ├── compress.py     ← MOCK
│   ├── solve.py        ← MOCK
│   └── tokens.py       ← MOCK
├── rezero/
│   └── __init__.py
├── baselines/
│   └── __init__.py
├── experiments/
│   └── __init__.py
├── demo/
│   └── scripted_convo.jsonl
├── tests/
│   └── __init__.py
├── requirements.txt
└── .env
```

---

## requirements.txt

```
openai>=1.0.0
python-dotenv
datasets
rouge-score
pytest
```

---

## .env

```
DEEPSEEK_API_KEY=your_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

---

## act1/compress.py (mock)

```python
def compress_ours(text: str, question: str, ratio: float) -> str:
    """
    MOCK: truncates to ratio fraction of words.
    Swap for real DeepSeek call in Step 8.
    """
    words = text.split()
    keep = max(1, int(len(words) * ratio))
    return " ".join(words[:keep])
```

---

## act1/tokens.py (mock)

```python
def count_tokens(text: str) -> int:
    """
    MOCK: word count proxy.
    Real version multiplies by 1.3 BPE ratio in Step 8.
    """
    return len(text.split())
```

---

## act1/solve.py (mock)

```python
def solve(context: str, question: str) -> str:
    """
    MOCK: returns fixed string.
    Swap for real DeepSeek call in Step 8.
    """
    return "MOCK_ANSWER"

def _deepseek_call(system: str, prompt: str, max_tokens: int = 512, fast: bool = False) -> str:
    """
    MOCK: stub for internal LLM calls used by Echidna and TraumaExtractor.
    Swap for real implementation in Step 8.
    """
    return '{"action": "pass", "revert_to": null, "reason": "mock", "urgency": "low"}'
```

---

## demo/scripted_convo.jsonl

Create this file with one JSON object per line. Each line is one turn:

```jsonl
{"goal": "Research Elon Musk companies and their government relationships", "user": "What companies were founded by Elon Musk?", "assistant": "Elon Musk founded Tesla, SpaceX, and Neuralink."}
{"goal": null, "user": "Which of these focus on space?", "assistant": "SpaceX focuses on space exploration and launch services."}
{"goal": null, "user": "Who co-founded SpaceX with Musk?", "assistant": "Musk founded SpaceX but brought in Gwynne Shotwell as COO."}
{"goal": null, "user": "Does SpaceX have government contracts?", "assistant": "Yes, SpaceX has major NASA and DoD contracts."}
{"goal": null, "user": "What is the latest SpaceX rocket?", "assistant": "Starship is SpaceX latest and most powerful rocket."}
{"goal": null, "user": "Has Starship reached orbit?", "assistant": "Starship completed its first successful orbital flight in 2024."}
{"goal": null, "user": "What fuel does Starship use?", "assistant": "Starship uses liquid methane and liquid oxygen."}
{"goal": null, "user": "How does this compare to the Falcon 9?", "assistant": "Falcon 9 uses RP-1 kerosene and liquid oxygen."}
{"goal": null, "user": "Which is reusable?", "assistant": "Both are reusable. Falcon 9 lands its first stage; Starship aims for full reuse."}
{"goal": null, "user": "Going back to Musk — what is Neuralink?", "assistant": "Neuralink is a brain-computer interface company founded by Musk in 2016."}
{"goal": null, "user": "Has Neuralink done human trials?", "assistant": "Yes, Neuralink implanted its first human patient in January 2024."}
{"goal": null, "user": "What was the patient name?", "assistant": "The first patient was Noland Arbaugh, a 29-year-old quadriplegic."}
{"goal": null, "user": "What was the outcome?", "assistant": "Arbaugh could control a computer cursor with his thoughts after implantation."}
{"goal": null, "user": "Did NASA fund any Neuralink work?", "assistant": "No, NASA has not funded Neuralink; it primarily funds SpaceX for launch services."}
{"goal": null, "user": "So which Musk company has the most government money?", "assistant": "SpaceX by far — billions in NASA, Air Force, and DoD contracts."}
```

---

## Install & verify

```bash
pip install -r requirements.txt --break-system-packages

python -c "from act1.compress import compress_ours; print(compress_ours('hello world foo bar baz', 'q', 0.5))"
# expect: hello world foo

python -c "from act1.tokens import count_tokens; print(count_tokens('hello world'))"
# expect: 2

python -c "from act1.solve import solve; print(solve('ctx', 'q'))"
# expect: MOCK_ANSWER
```

---

## Done when

- All three imports work without errors
- Mock outputs match expected values above
- `demo/scripted_convo.jsonl` exists with 15 lines
