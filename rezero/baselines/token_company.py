import os
from openai import OpenAI
from thetokencompany.openai import with_compression
from dotenv import load_dotenv
from engine.tokens import count_tokens
load_dotenv()

tc_client = None
TC_AGGRESSIVENESS = 0.2


def init_tc_client(aggressiveness: float = 0.2):
    global tc_client, TC_AGGRESSIVENESS
    TC_AGGRESSIVENESS = aggressiveness
    tc_client = with_compression(
        OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        ),
        compression_api_key=os.environ["TOKEN_COMPANY_API_KEY"],
        aggressiveness=aggressiveness,
    )


def tc_solve(context: str, question: str, model: str = "deepseek-chat") -> dict:
    global tc_client
    if tc_client is None:
        init_tc_client()
    resp = tc_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": (
                "Answer the question using only the provided context. "
                "Be as concise as possible — one sentence or fewer."
            )},
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


class TokenCompanySession:
    def __init__(self, goal: str):
        self.goal = goal
        self.history: list[dict] = []

    def add_turn(self, user: str, assistant: str) -> None:
        self.history.append({"role": "user",      "content": user})
        self.history.append({"role": "assistant", "content": assistant})

    def prompt_for_solver(self) -> str:
        full = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in self.history)
        return f"[GOAL]\n{self.goal}\n\n[HISTORY]\n{full}"

    def token_count(self) -> int:
        return count_tokens(self.prompt_for_solver())
