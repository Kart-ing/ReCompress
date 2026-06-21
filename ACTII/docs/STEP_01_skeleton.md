# Step 1 — Project Skeleton

> **Goal:** Create the full folder structure, install deps, write the DeepSeek client and token counter. Everything we build is ours — no external team dependency.

---

## Folder structure

```
rbd_compress/
├── engine/
│   ├── __init__.py
│   ├── deepseek.py      ← DeepSeek API client + solver
│   ├── compressor.py    ← query-aware text compressor
│   └── tokens.py        ← token counter
├── rezero/
│   ├── __init__.py
│   ├── trauma.py
│   ├── checkpoint.py
│   ├── echidna.py
│   ├── context_builder.py
│   └── session.py
├── baselines/
│   ├── __init__.py
│   └── naive.py
├── experiments/
│   ├── __init__.py
│   ├── token_curve.py
│   ├── hotpotqa_runner.py
│   ├── microclaim.py
│   ├── turn15_probe.py
│   ├── plot_curve.py
│   └── generate_panel.py
├── demo/
│   ├── scripted_convo.jsonl
│   ├── panel.html
│   └── panel_live.html
├── tests/
│   ├── __init__.py
│   ├── test_trauma.py
│   ├── test_checkpoint.py
│   ├── test_echidna.py
│   ├── test_session.py
│   ├── test_budget.py
│   └── test_naive.py
├── notebooks/
├── results/
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

## engine/tokens.py

```python
def count_tokens(text: str) -> int:
    """
    Approximate token count: word count × 1.3 BPE ratio.
    DeepSeek tokenizer is not publicly available.
    1.3 is a conservative upper bound for English text.
    """
    return int(len(text.split()) * 1.3)
```

---

## engine/deepseek.py

```python
import os
import json
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

def call(
    system: str,
    prompt: str,
    max_tokens: int = 512,
) -> str:
    """
    Core DeepSeek call. Used by all components:
    Echidna, TraumaExtractor, Compressor, and Solver.
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
    """
    Answer a question given context. Used in evaluation.
    Returns a concise answer string.
    """
    system = (
        "Answer the question using only the provided context. "
        "Be as concise as possible — one sentence or fewer."
    )
    return call(system, f"Context:\n{context}\n\nQuestion: {question}", max_tokens=64)
```

---

## engine/compressor.py

```python
from engine.deepseek import call
from engine.tokens import count_tokens

def compress(text: str, question: str, ratio: float) -> str:
    """
    Query-aware compressor. Keeps roughly `ratio` fraction of tokens.
    Preserves facts critical to answering `question`.
    """
    target = max(20, int(count_tokens(text) * ratio))
    system = (
        f"You are a precise text compressor. "
        f"Compress the following text to approximately {target} tokens. "
        f'The result will be used to answer: "{question}". '
        f"Preserve all named entities, numbers, dates, and facts critical to the question. "
        f"Return only the compressed text — no preamble, no explanation."
    )
    return call(system, text, max_tokens=target + 20)
```

---

## demo/scripted_convo.jsonl

Create with one JSON object per line — one turn per line:

```jsonl
{"goal": "Research Elon Musk companies and their government relationships", "user": "What companies were founded by Elon Musk?", "assistant": "Elon Musk founded Tesla, SpaceX, and Neuralink."}
{"goal": null, "user": "Which of these focus on space?", "assistant": "SpaceX focuses on space exploration and launch services."}
{"goal": null, "user": "Who co-founded SpaceX with Musk?", "assistant": "Musk founded SpaceX but brought in Gwynne Shotwell as COO."}
{"goal": null, "user": "Does SpaceX have government contracts?", "assistant": "Yes, SpaceX has major NASA and DoD contracts."}
{"goal": null, "user": "What is the latest SpaceX rocket?", "assistant": "Starship is the latest and most powerful SpaceX rocket."}
{"goal": null, "user": "Has Starship reached orbit?", "assistant": "Starship completed its first successful orbital flight in 2024."}
{"goal": null, "user": "What fuel does Starship use?", "assistant": "Starship uses liquid methane and liquid oxygen."}
{"goal": null, "user": "How does this compare to the Falcon 9?", "assistant": "Falcon 9 uses RP-1 kerosene and liquid oxygen."}
{"goal": null, "user": "Which is reusable?", "assistant": "Both are reusable. Falcon 9 lands its first stage; Starship aims for full reuse."}
{"goal": null, "user": "What is Neuralink?", "assistant": "Neuralink is a brain-computer interface company founded by Musk in 2016."}
{"goal": null, "user": "Has Neuralink done human trials?", "assistant": "Yes, Neuralink implanted its first human patient in January 2024."}
{"goal": null, "user": "What was the patient name?", "assistant": "The first patient was Noland Arbaugh, a 29-year-old quadriplegic."}
{"goal": null, "user": "What was the outcome?", "assistant": "Arbaugh could control a computer cursor with his thoughts."}
{"goal": null, "user": "Did NASA fund any Neuralink work?", "assistant": "No, NASA has not funded Neuralink. It primarily funds SpaceX."}
{"goal": null, "user": "Which Musk company has the most government money?", "assistant": "SpaceX by far — billions in NASA, Air Force, and DoD contracts."}
```

---

## Install & verify

```bash
pip install -r requirements.txt --break-system-packages

# verify DeepSeek is wired
python - <<'EOF'
from engine.deepseek import solve
print(solve("The Eiffel Tower is in Paris, France.", "Where is the Eiffel Tower?"))
EOF
# expect: Paris / Paris, France

# verify token counter
python - <<'EOF'
from engine.tokens import count_tokens
print(count_tokens("hello world"))
EOF
# expect: 2 or 3 (word count × 1.3, rounded)

# verify compressor
python - <<'EOF'
from engine.compressor import compress
result = compress("Alice founded Tech Corp in 2010. She raised 50 million dollars.", "Who founded Tech Corp?", ratio=0.5)
print(result)
EOF
# expect: shorter text mentioning Alice and Tech Corp
```

---

## Done when

- All three imports work
- DeepSeek returns a sensible answer on the sanity check
- `demo/scripted_convo.jsonl` exists with at least 10 lines
