import os
import threading
from openai import OpenAI
from thetokencompany.openai import with_compression
from dotenv import load_dotenv
from engine.tokens import count_tokens
from engine import ratelimit
load_dotenv()

tc_clients = {}
_tc_lock = threading.Lock()

def _build_tc_client(aggressiveness: float):
    return with_compression(
        OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        ),
        compression_api_key=os.environ["TOKEN_COMPANY_API_KEY"],
        aggressiveness=aggressiveness,
    )

def init_tc_clients(aggressivenesses):
    for a in aggressivenesses:
        tc_clients[a] = _build_tc_client(a)

def init_tc_client(aggressiveness: float = 0.2):
    init_tc_clients([aggressiveness])

def _get_client(aggressiveness: float):
    client = tc_clients.get(aggressiveness)
    if client is None:
        with _tc_lock:
            client = tc_clients.get(aggressiveness)
            if client is None:
                client = _build_tc_client(aggressiveness)
                tc_clients[aggressiveness] = client
    return client

def tc_solve(context: str, question: str, aggressiveness: float = 0.5, model: str = "deepseek-chat") -> dict:
    client = _get_client(aggressiveness)
    ratelimit.wait()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Answer the question using only the provided context. Be as concise as possible — one sentence or fewer."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        max_tokens=128,
        temperature=0.0,
    )
    result = resp.choices[0].message.content
    prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
    return {"answer": result.strip() if result else "", "prompt_tokens": prompt_tokens}
