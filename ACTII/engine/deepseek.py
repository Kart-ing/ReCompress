import os
import json
import time
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

VERBOSE = False
MAX_RETRIES = 2
MIN_MAX_TOKENS = 1024
MODEL = "deepseek-chat"

def set_verbose(v: bool):
    global VERBOSE
    VERBOSE = v

def set_model(m: str):
    global MODEL
    MODEL = m

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

def call(
    system: str,
    prompt: str,
    max_tokens: int = 512,
) -> str:
    max_tokens = max(max_tokens, MIN_MAX_TOKENS)
    for attempt in range(MAX_RETRIES):
        t0 = time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=max_tokens,
            temperature=0.0,
        )
        elapsed = time.time() - t0
        result = resp.choices[0].message.content
        finish_reason = resp.choices[0].finish_reason
        if VERBOSE and resp.usage:
            print(f"  [DEEPSEEK] {elapsed:.1f}s | usage: prompt={resp.usage.prompt_tokens} completion={resp.usage.completion_tokens} total={resp.usage.total_tokens}")
        if result is None or result.strip() == "":
            if VERBOSE:
                print(f"  [DEEPSEEK] empty response (attempt {attempt+1}/{MAX_RETRIES}, finish_reason={finish_reason})")
            if attempt < MAX_RETRIES - 1:
                if finish_reason == "length":
                    max_tokens = max_tokens * 2
                    if VERBOSE:
                        print(f"  [DEEPSEEK] doubling max_tokens to {max_tokens}")
                time.sleep(0.3)
                continue
            return ""
        result = result.strip()
        if VERBOSE:
            preview = result[:120].replace("\n", "\\n")
            print(f"  [DEEPSEEK] ({len(result)} chars) {preview}{'...' if len(result) > 120 else ''}")
        return result
    return ""


def repair_json(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(l for l in lines if not l.startswith("```"))
        raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    for suffix in ["}", '"}'  , '"}', '"}']:
        try:
            return json.loads(raw + suffix)
        except json.JSONDecodeError:
            continue
    try:
        return json.loads(raw + '"}')
    except json.JSONDecodeError:
        pass
    return None


def solve(context: str, question: str) -> str:
    system = (
        "Answer the question using only the provided context. "
        "Be as concise as possible — one sentence or fewer."
    )
    return call(system, f"Context:\n{context}\n\nQuestion: {question}", max_tokens=128)
