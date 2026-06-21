"""Inference with the distilled 1.5B model. Loads the LoRA adapter ONCE (Modal class
with @enter) and compresses a whole batch of (text, question) pairs per remote call —
so the 5-bar eval doesn't pay a cold model load on every instance.

Drop-in for the API-based compress_ours(), but batched.
For the 5-bar re-eval we run this on Modal as a deployed class.
"""
from __future__ import annotations
import modal

APP_NAME = "recompress-distill"
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    # Match train.py: modern unsloth requires peft>=0.18.0, so the old
    # peft==0.13.2 pin broke the build. Let unsloth resolve a compatible stack.
    .pip_install(
        "unsloth",
        "unsloth_zoo",
        "peft",
        "transformers",
        "accelerate",
        "bitsandbytes",
        "tiktoken",   # recompress.act1.tokens.count_tokens uses tiktoken at runtime
    )
    # Batch methods import recompress.act1.tokens at runtime, so the local `recompress`
    # package must be available inside the container (modal>=1.0 no longer auto-mounts).
    .add_local_python_source("recompress")
)

app = modal.App(APP_NAME + "-infer", image=IMAGE)
VOL = modal.Volume.from_name("recompress-distill", create_if_missing=True)

_SYSTEM_PROMPT = (
    "You are a context compressor. Given a long context and a QUESTION, produce the MINIMAL "
    "compressed context that still lets a downstream QA agent answer the QUESTION correctly.\n\n"
    "Rules:\n"
    "1. DROP any passage irrelevant to the question (distractors).\n"
    "2. DENSIFY verbose-but-relevant prose into terse, information-dense sentences. Paraphrase freely.\n"
    "3. Preserve all facts needed to answer: entities, numbers, relations, multi-hop links.\n"
    "4. Do NOT answer the question. Do NOT add reasoning. Output ONLY compressed context.\n"
)


@app.cls(gpu="H100", volumes={"/vol": VOL}, timeout=3600, scaledown_window=300)
class Compressor:
    @modal.enter()
    def load(self):
        """Load the distilled adapter once per container."""
        import os
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
        from unsloth import FastLanguageModel

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="/vol/adapter",
            max_seq_length=4096,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)

    def _compress_one(self, text: str, question: str, ratio: float) -> str:
        from recompress.act1.tokens import count_tokens, truncate_to_tokens

        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{text}"},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        out = self.model.generate(
            input_ids=inputs,
            max_new_tokens=512,
            do_sample=False,   # greedy; do NOT pass temperature with greedy decoding
        )
        new_tokens = out[0][inputs.shape[1]:]
        result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        target_tokens = max(1, int(count_tokens(text) * ratio))
        return truncate_to_tokens(result, target_tokens)

    @modal.method()
    def compress_batch(self, items: list[dict], ratio: float = 0.3) -> list[str]:
        """items: list of {"text":..., "question":...}. Returns compressed strings,
        in order. Model is already loaded (via @enter), so no per-item cold start."""
        results = []
        for i, it in enumerate(items):
            results.append(self._compress_one(it["text"], it["question"], ratio))
            if (i + 1) % 10 == 0:
                print(f"  [distilled] compressed {i + 1}/{len(items)}")
        return results

    @modal.method()
    def compress_batch_timed(self, items: list[dict], ratio: float = 0.3) -> list[dict]:
        """Like compress_batch but returns per-item latency for the latency benchmark.
        Model load is excluded (it happened in @enter), so latency_s is pure generation."""
        import time as _t
        from recompress.act1.tokens import count_tokens
        out = []
        for i, it in enumerate(items):
            t0 = _t.perf_counter()
            comp = self._compress_one(it["text"], it["question"], ratio)
            dt = _t.perf_counter() - t0
            out.append({"compressed": comp, "latency_s": dt, "n_out": count_tokens(comp)})
            if (i + 1) % 10 == 0:
                print(f"  [distilled-timed] {i + 1}/{len(items)}")
        return out


@app.local_entrypoint()
def test():
    """Quick test of the distilled model on 2 HotpotQA instances (one batched call)."""
    from recompress.act1.data import load_hotpotqa, context_to_text
    from recompress.act1.tokens import count_tokens

    insts = load_hotpotqa(n=2)
    items = [{"text": context_to_text(i), "question": i["question"]} for i in insts]
    outs = Compressor().compress_batch.remote(items, ratio=0.3)
    for inst, out in zip(insts, outs):
        print(f"\nQ: {inst['question']}")
        print(f"gold: {inst['answer']}")
        print(f"distilled ({count_tokens(out)} tok): {out[:300]}")
