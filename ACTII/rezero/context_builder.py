from engine.tokens import count_tokens

TRAUMA_CAP     = 50
CHECKPOINT_CAP = 150
DELTA_CAP      = 100
TOTAL_CAP      = 300

class ContextBuilder:
    def build(self, trauma: str, checkpoint: str, delta: str) -> str:
        trauma     = self._enforce(trauma,     TRAUMA_CAP)
        checkpoint = self._enforce(checkpoint, CHECKPOINT_CAP)
        delta      = self._enforce(delta,      DELTA_CAP)

        prompt = self._assemble(trauma, checkpoint, delta)

        while count_tokens(prompt) > TOTAL_CAP:
            words = checkpoint.split()
            if not words:
                break
            checkpoint = " ".join(words[:-5])
            prompt = self._assemble(trauma, checkpoint, delta)

        return prompt

    def _assemble(self, trauma: str, checkpoint: str, delta: str) -> str:
        return f"[TRAUMA]\n{trauma}\n\n[CHECKPOINT]\n{checkpoint}\n\n[DELTA]\n{delta}"

    def _enforce(self, text: str, cap: int) -> str:
        words = text.split()
        while count_tokens(" ".join(words)) > cap and words:
            words.pop()
        return " ".join(words)
