import re
from engine.tokens import count_tokens

TRAUMA_CAP = 50
PINNED_CAP = 15
BUFFER_CAP = 30

class TraumaExtractor:
    def __init__(self, goal: str = "", use_llm: bool = False, verbose: bool = False):
        self.use_llm = use_llm
        self.verbose = verbose
        self.pinned: str = self._enforce(goal, PINNED_CAP) if goal else ""
        self.buffer: str = ""

    def update(self, message: str) -> str:
        if self.use_llm:
            self._llm_extract(message)
        else:
            self._mock_extract(message)
        return self.get()

    def get(self) -> str:
        if self.pinned and self.buffer:
            return f"{self.pinned} | {self.buffer}"
        return self.pinned or self.buffer

    def _mock_extract(self, message: str) -> None:
        entities = re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b', message)
        new_facts = ", ".join(dict.fromkeys(entities))
        combined = f"{self.buffer} {new_facts}".strip()
        self.buffer = self._enforce(combined, BUFFER_CAP)

    def _llm_extract(self, message: str) -> None:
        from engine.deepseek import call, repair_json
        system = f"""You extract and maintain critical facts that must never be lost.
PROTECTED FACTS (do not repeat or modify): {self.pinned}
Given a new message and current buffer, identify NEW critical facts
not already in the protected section: named entities, bridge facts, numbers, dates.
Return JSON only: {{"update": true/false, "buffer": "..."}}
Keep buffer under 30 tokens. Never duplicate protected facts.
For HotpotQA: pin entity names and answers to sub-questions."""
        prompt = f"Current buffer: {self.buffer}\nNew message: {message}"
        raw = call(system, prompt, max_tokens=200)
        parsed = repair_json(raw)
        if parsed is not None:
            if parsed.get("update"):
                old = self.buffer
                new_val = parsed.get("buffer", parsed.get("trauma_memory", ""))
                self.buffer = self._enforce(new_val, BUFFER_CAP)
                if self.verbose:
                    print(f"  [TRAUMA] buffer updated: '{old}' → '{self.buffer}'")
            elif self.verbose:
                print(f"  [TRAUMA] no update (LLM said update=false)")
        else:
            if self.verbose:
                preview = raw[:100].replace("\n", "\\n")
                print(f"  [TRAUMA] WARNING: JSON parse failed, raw='{preview}'")

    def _enforce(self, text: str, cap: int) -> str:
        words = text.split()
        while count_tokens(" ".join(words)) > cap and len(words) > 1:
            words.pop()
        return " ".join(words)
