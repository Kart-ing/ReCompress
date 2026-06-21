"""Modal app: LoRA fine-tune Qwen2.5-1.5B-Instruct on query-aware compression pairs.
Uses Unsloth on an H100. Teacher data = DeepSeek API outputs (gen_data.py).

Run: modal run src/distill/train.py --data data/distill/train.jsonl
"""
from __future__ import annotations
import modal
import json

APP_NAME = "recompress-distill"
IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    # Let unsloth pull a self-consistent torch/trl/peft/transformers stack.
    # NOTE: modern unsloth requires trl>=0.18.2 and peft>=0.18.0 — the old
    # trl==0.11.4 / peft==0.13.2 pins were unsatisfiable alongside `unsloth` and
    # broke the image build. We install unsloth (which pulls a compatible
    # unsloth_zoo / trl / peft / torch) and let it resolve the rest.
    .pip_install(
        "unsloth",
        "unsloth_zoo",
        "trl",
        "peft",
        "transformers",
        "datasets",
        "accelerate",
        "bitsandbytes",
    )
    .pip_install("huggingface_hub[hf_transfer]")
)

app = modal.App(APP_NAME, image=IMAGE)
VOL = modal.Volume.from_name("recompress-distill", create_if_missing=True)


def format_chat(text: str, question: str, compressed: str) -> dict:
    """Format one training example as a chat sequence for Qwen2.5-Instruct."""
    return {
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"QUESTION:\n{question}\n\nCONTEXT:\n{text}"},
            {"role": "assistant", "content": compressed},
        ]
    }


_SYSTEM_PROMPT = (
    "You are a context compressor. Given a long context and a QUESTION, produce the MINIMAL "
    "compressed context that still lets a downstream QA agent answer the QUESTION correctly.\n\n"
    "Rules:\n"
    "1. DROP any passage irrelevant to the question (distractors).\n"
    "2. DENSIFY verbose-but-relevant prose into terse, information-dense sentences. Paraphrase freely.\n"
    "3. Preserve all facts needed to answer: entities, numbers, relations, multi-hop links.\n"
    "4. Do NOT answer the question. Do NOT add reasoning. Output ONLY compressed context.\n"
)


@app.function(
    gpu="H100",
    volumes={"/vol": VOL},
    timeout=3600,
    memory=32 * 1024,
)
def train(data_rows: list[dict], epochs: int = 3, lr: float = 2e-4, lora_r: int = 16,
          max_seq_len: int = 4096, eval_frac: float = 0.1):
    """`data_rows` is a list of {"text": <fully-rendered ChatML string>} — the chat
    template is applied LOCALLY in the entrypoint (validated against the real Qwen
    template), so the container just trains on a plain text column. This deliberately
    avoids Unsloth's brittle `messages`-column / `formatting_func` probe path.
    """
    import os
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    # reduce CUDA fragmentation (v3 OOM'd during eval at the epoch boundary)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    # --- plain pre-rendered text column + held-out eval split (overfitting signal) ---
    texts = [{"text": ex["text_rendered"]} for ex in data_rows]
    n_eval = max(1, int(len(texts) * eval_frac))
    eval_rows, train_rows = texts[:n_eval], texts[n_eval:]
    ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)
    print(f"loaded {len(ds)} train + {len(eval_ds)} eval examples (pre-rendered text column)")

    # --- load model + LoRA ---
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        max_seq_length=max_seq_len,
        dtype=None,           # auto
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=lora_r * 2,
        lora_dropout=0.1,   # regularization: was 0.05; raised to fight overfitting (v2 r=64 overfit hard)
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # --- train (eval each epoch; load_best_model_at_end keeps the best-eval checkpoint) ---
    # SPEED: batch 32 × accum 2 = eff. 64 (v2's r=64/6ep overfit AND was slow at eff.32).
    #   Bigger batch → fewer steps; on 5000 ex × 3 ep ≈ 5000*0.9/64*3 ≈ 211 steps (~12-15 min).
    # ANTI-OVERFIT (v2 r=64/6ep diverged: eval_loss bottomed @epoch2=1.654, rose to 1.861@epoch5):
    #   - 3 epochs not 6 (eval bottomed ~epoch2)
    #   - load_best_model_at_end → commit the BEST-eval adapter, not the overfit final one
    #   - weight_decay + (r=32, dropout=0.1 set above)
    # NOTE: packing=True FAILED on this image (needs flash_attention_2; xformers broken here).
    cfg = SFTConfig(
        output_dir="/vol/outputs",
        num_train_epochs=epochs,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,
        learning_rate=lr,
        max_length=max_seq_len,
        weight_decay=0.01,           # regularization
        dataset_text_field="text",   # train on the pre-rendered text column directly
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        per_device_eval_batch_size=4,    # small eval batch
        # ROOT-CAUSE FIX for the eval OOM: the 53GB alloc was _convert_to_fp32 casting the
        # full logits [bs, seq, vocab=151936] to float32. prediction_loss_only skips logit
        # accumulation/return entirely — we only need eval_loss for the overfitting check.
        prediction_loss_only=True,
        eval_accumulation_steps=1,
        load_best_model_at_end=True,     # keep best-eval checkpoint (combats overfitting)
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        seed=42,
        report_to="none",
    )
    # No formatting_func / no messages column: the text is already ChatML-rendered,
    # so SFTTrainer just tokenizes `text`. This sidesteps Unsloth's probe entirely.
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=ds,
        eval_dataset=eval_ds,
        args=cfg,
    )
    trainer_stats = trainer.train()
    print(f"training done. train_loss={trainer_stats.training_loss:.4f}")

    # --- per-epoch train vs eval loss: the overfitting verdict ---
    hist = trainer.state.log_history
    train_losses = [(h.get("epoch"), h["loss"]) for h in hist if "loss" in h]
    eval_losses = [(h.get("epoch"), h["eval_loss"]) for h in hist if "eval_loss" in h]
    print("=== loss curve (overfitting check) ===")
    for ep, el in eval_losses:
        # nearest train loss logged at/just before this epoch
        tl = min((t for t in train_losses if t[0] and t[0] <= ep + 1e-6),
                 key=lambda t: abs(t[0] - ep), default=(None, None))[1]
        gap = (el - tl) if tl is not None else float("nan")
        print(f"  epoch {ep:.2f}: train_loss={tl}  eval_loss={el:.4f}  gap(eval-train)={gap:+.4f}")
    overfit = (len(eval_losses) >= 2 and eval_losses[-1][1] > eval_losses[-2][1] + 1e-3)
    print(f"OVERFITTING SIGNAL: {'YES — eval_loss rose on the last epoch' if overfit else 'no — eval_loss did not rise'}")

    # --- save adapter ---
    adapter_dir = "/vol/adapter"
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    VOL.commit()
    print(f"adapter saved to {adapter_dir}")
    return {
        "train_loss": trainer_stats.training_loss,
        "eval_losses": eval_losses,
        "train_losses": train_losses,
        "overfit": overfit,
        "n_train": len(ds), "n_eval": len(eval_ds),
    }


@app.local_entrypoint()
def main(data: str = "data/distill/train_v3.jsonl", epochs: int = 3, lr: float = 2e-4, lora_r: int = 32, max_seq_len: int = 3072):
    # Read the JSONL LOCALLY, render the ChatML text column LOCALLY (validated against
    # the real Qwen template), and ship rows that already contain the final text.
    # The container never touches a local path and never runs the chat template, so
    # neither the FileNotFoundError nor Unsloth's formatting_func probe can bite.
    from transformers import AutoTokenizer
    with open(data) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    rendered = []
    for r in rows:
        msgs = format_chat(r["text"], r["question"], r["compressed"])["messages"]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        rendered.append({"text_rendered": text})
    print(f"rendered {len(rendered)} ChatML examples locally; shipping to Modal from {data}")
    stats = train.remote(rendered, epochs, lr, lora_r, max_seq_len)
    print("\n=== TRAINING SUMMARY ===")
    print(f"  train examples: {stats['n_train']} | eval examples: {stats['n_eval']}")
    print(f"  final train_loss: {stats['train_loss']:.4f}")
    print(f"  eval_loss per epoch: {[round(e[1], 4) for e in stats['eval_losses']]}")
    print(f"  OVERFITTING: {'⚠️ YES (eval_loss rose)' if stats['overfit'] else '✅ no (eval_loss stable/falling)'}")
