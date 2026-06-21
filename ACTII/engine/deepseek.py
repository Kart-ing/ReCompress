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
    system = (
        "Answer the question using only the provided context. "
        "Be as concise as possible — one sentence or fewer."
    )
    return call(system, f"Context:\n{context}\n\nQuestion: {question}", max_tokens=64)
