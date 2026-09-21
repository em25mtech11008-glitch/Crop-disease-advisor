"""
QLoRA fine-tuning script for LLaMA 3 8B Instruct on crop disease dataset.
Prompt 2.2 — QLoRA Fine-tuning Setup
"""

import json
import os
import time
import argparse
from pathlib import Path

import torch
import yaml
import wandb
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Prompt Formatting ─────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert agricultural advisor with deep knowledge of plant diseases, "
    "organic and chemical treatments, and regional farming practices across India. "
    "Always respond with a valid JSON object following the specified schema."
)

def format_prompt(example: dict) -> dict:
    """Format as LLaMA 3 chat template."""
    text = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{SYSTEM_PROMPT}\n"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{example['instruction']}\n\nContext: {example['input']}\n"
        "<|start_header_id|>assistant<|end_header_id|>\n"
        f"{example['output']}"
        "<|end_of_text|>"
    )
    return {"text": text}


# ── JSON Validity Callback ────────────────────────────────────────────────────

class JSONValidityCallback(TrainerCallback):
    """Evaluate JSON validity of model outputs every N steps."""

    def __init__(self, model, tokenizer, eval_samples, device, check_every=500):
        self.model        = model
        self.tokenizer    = tokenizer
        self.eval_samples = eval_samples
        self.device       = device
        self.check_every  = check_every

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.check_every != 0:
            return

        valid = 0
        for sample in self.eval_samples[:5]:
            prompt = format_prompt(sample)["text"].split("<|start_header_id|>assistant<|end_header_id|>")[0]
            prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
            inputs  = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output = self.model.generate(
                    **inputs, max_new_tokens=256,
                    do_sample=False, temperature=1.0,
                )
            decoded = self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            try:
                json.loads(decoded.strip())
                valid += 1
            except json.JSONDecodeError:
                pass

        ratio = valid / min(5, len(self.eval_samples))
        print(f"\n[Step {state.global_step}] JSON validity: {valid}/5 ({ratio:.0%})")
        wandb.log({"json_validity_ratio": ratio}, step=state.global_step)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/llm_config.yaml")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── W&B ───────────────────────────────────────────────────────────────────
    wandb.init(
        project=cfg.get("wandb_project", "crop-disease-advisor-llm"),
        name=f"qlora_llama3_{time.strftime('%Y%m%d_%H%M%S')}",
        config=cfg,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    raw = load_dataset("json", data_files=cfg["instruction_dataset"], split="train")
    split = raw.train_test_split(test_size=1 - cfg["train_split"], seed=cfg["seed"])
    train_ds = split["train"].map(format_prompt)
    val_ds   = split["test"].map(format_prompt)
    print(f"✓ Dataset: {len(train_ds)} train / {len(val_ds)} val")

    # ── Quantization ──────────────────────────────────────────────────────────
    compute_dtype = torch.bfloat16 if cfg.get("bf16", False) else torch.float16

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["load_in_4bit"],
        bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
    )

    # ── Model + Tokenizer ─────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=compute_dtype,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # ── LoRA ──────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        target_modules=cfg["lora_target_modules"],
        lora_dropout=cfg["lora_dropout"],
        bias=cfg["lora_bias"],
        task_type=TaskType.CAUSAL_LM,
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training Arguments ────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        fp16=cfg.get("fp16", False),
        bf16=cfg.get("bf16", False),
        optim="paged_adamw_8bit",
        logging_steps=cfg["logging_steps"],
        save_steps=cfg["save_steps"],
        eval_strategy="steps",
        eval_steps=cfg["save_steps"],
        report_to="wandb",
        seed=cfg["seed"],
        dataset_text_field="text",
        packing=False,
        max_length=2048,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    eval_samples = val_ds.select(range(min(10, len(val_ds)))).to_list()
    json_callback = JSONValidityCallback(model, tokenizer, eval_samples, device)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        args=training_args,
        callbacks=[json_callback],
    )

    print("\n✓ Starting QLoRA fine-tuning...")
    trainer.train()

    # ── Save adapter ──────────────────────────────────────────────────────────
    adapter_path = cfg.get("adapter_path", "models/llm/llama3_qlora_adapter")
    Path(adapter_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\n✓ Adapter saved → {adapter_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
