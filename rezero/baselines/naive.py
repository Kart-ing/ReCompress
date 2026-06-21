from engine.tokens import count_tokens

class NaiveSession:
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
