import os
from openai import OpenAI
from dotenv import load_dotenv
from engine import ratelimit
load_dotenv()

MODEL = "deepseek-chat"

client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
)

def set_model(m: str):
    global MODEL
    MODEL = m

def solve(context: str, question: str) -> dict:
    ratelimit.wait()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Answer the question using only the provided context. Be as concise as possible — one sentence or fewer."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        max_tokens=128,
        temperature=0.0,
    )
    result = resp.choices[0].message.content
    prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
    return {
        "answer": result.strip() if result else "",
        "prompt_tokens": prompt_tokens,
    }
