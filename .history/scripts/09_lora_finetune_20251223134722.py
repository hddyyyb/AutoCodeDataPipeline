#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[1]


def build_text_from_messages(messages: List[Dict[str, Any]]) -> str:
    """
    将messages格式样本拼成单一训练文本
    说明：这里不用复杂chat template，保持与你Step08输出一致的可读拼接即可。
    """
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<{role}>\n{content}")
    return "\n".join(parts)


def pick_existing(path_candidates: List[Path]) -> Optional[str]:
    for p in path_candidates:
        if p.exists():
            return str(p)
    return None


def main():
    # 基座模型（建议用Instruct）
    model_name = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    out_dir = os.environ.get("OUT_DIR", str(ROOT / "outputs/lora_qwen25_demo"))

    # 数据路径：默认用Step08输出
    train_path = os.environ.get("TRAIN_PATH") or pick_existing([
        ROOT / "data/sft/train.jsonl",
        ROOT / "data/sft/train.messages.jsonl",
    ])
    dev_path = os.environ.get("DEV_PATH") or pick_existing([
        ROOT / "data/sft/dev.jsonl",
        ROOT / "data/sft/dev.messages.jsonl",
    ])

    if not train_path:
        raise FileNotFoundError("未找到SFT训练数据，请先运行scripts/08_prepare_sft_data.py")

    # LoRA超参
    lora_r = int(os.environ.get("LORA_R", "16"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", "32"))
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))

    # 训练超参（面试演示：小步数即可）
    max_len = int(os.environ.get("MAX_LEN", "512"))
    lr = float(os.environ.get("LR", "2e-4"))
    batch_size = int(os.environ.get("BS", "1"))
    grad_acc = int(os.environ.get("GA", "4"))
    max_steps = int(os.environ.get("MAX_STEPS", "120"))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", "10"))
    eval_steps = int(os.environ.get("EVAL_STEPS", "20"))
    save_steps = int(os.environ.get("SAVE_STEPS", str(max(20, eval_steps))))

    print(f"BASE_MODEL={model_name}")
    print(f"TRAIN_PATH={train_path}")
    print(f"DEV_PATH={dev_path or '(none)'}")
    print(f"OUT_DIR={out_dir}")
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu_name =", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    files = {"train": train_path}
    if dev_path:
        files["validation"] = dev_path
    ds = load_dataset("json", data_files=files)

    def tok_fn(ex):
        # 支持两种格式：text 或 messages
        if "messages" in ex and ex["messages"]:
            text = build_text_from_messages(ex["messages"])
        else:
            text = ex.get("text", "")

        x = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        # 简化：全量语言模型学习（面试demo足够）
        x["labels"] = x["input_ids"].copy()
        return x

    train_ds = ds["train"].map(tok_fn, remove_columns=ds["train"].column_names)
    eval_ds = None
    if "validation" in ds:
        eval_ds = ds["validation"].map(tok_fn, remove_columns=ds["validation"].column_names)

    # 加载模型（LoRA，保持简单，不做4bit）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    # LoRA注入模块（Qwen2.5常用）
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        logging_steps=5,
        save_steps=save_steps,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
        optim="adamw_torch",
        evaluation_strategy="steps" if eval_ds is not None else "no",
        eval_steps=eval_steps if eval_ds is not None else None,
        load_best_model_at_end=True if eval_ds is not None else False,
        metric_for_best_model="eval_loss" if eval_ds is not None else None,
        greater_is_better=False if eval_ds is not None else None,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()

    # 保存LoRA适配器权重+tokenizer
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("训练完成，LoRA权重已保存:", out_dir)


if __name__ == "__main__":
    main()
