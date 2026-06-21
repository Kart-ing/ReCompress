import re
import json
from engine.tokens import count_tokens

TRAUMA_CAP = 50

class TraumaExtractor:
    def __init__(self, use_llm: bool = False):
        self.use_llm = use_llm
        self.trauma: str = ""

    def update(self, message: str) -> str:
        if self.use_llm:
            self.trauma = self._llm_extract(message)
        else:
            self.trauma = self._mock_extract(message)
        return self.trauma

    def get(self) -> str:
        return self.trauma

    def _mock_extract(self, message: str) -> str:
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', message)
        new_facts = ", ".join(dict.fromkeys(entities))
        combined = f"{self.trauma} {new_facts}".strip()
        return self._enforce_cap(combined)

    def _llm_extract(self, message: str) -> str:
        from engine.deepseek import call
        system = """You extract and maintain critical facts that must never be lost.
Given a new message and current trauma memory, identify NEW critical facts:
named entities, bridge facts, numbers, dates, the user's core goal.
Return JSON only: {"update": true/false, "trauma_memory": "..."}
Keep trauma memory under 50 tokens. Never duplicate existing facts.
For HotpotQA: always pin entity names and answers to sub-questions."""
        prompt = f"Current trauma memory: {self.trauma}\nNew message: {message}"
        raw = call(system, prompt, max_tokens=100)
        try:
            parsed = json.loads(raw)
            if parsed.get("update"):
                self.trauma = self._enforce_cap(parsed["trauma_memory"])
        except json.JSONDecodeError:
            pass
        return self.trauma

    def _enforce_cap(self, text: str) -> str:
        words = text.split()
        while count_tokens(" ".join(words)) > TRAUMA_CAP and len(words) > 1:
            words.pop()
        return " ".join(words)
