#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parents[1]

def build_text_from_messages(messages):
    """将messages格式样本拼成单一训练文本(兼容Step08的messages输出)"""
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<{role}>\n{content}")
    return "\n".join(parts)



def main():
    # 基座模型
    model_name = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B")
    out_dir = os.environ.get("OUT_DIR", str(ROOT / "outputs/lora_qwen25_lora"))
    data_path = os.environ.get("DATA_PATH", str(ROOT / "data/sft/train.jsonl"))

    # LoRA超参
    lora_r = int(os.environ.get("LORA_R", "16"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", "32"))
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))

    # 训练超参
    max_len = int(os.environ.get("MAX_LEN", "384"))
    lr = float(os.environ.get("LR", "2e-4"))
    batch_size = int(os.environ.get("BS", "1"))
    grad_acc = int(os.environ.get("GA", "4"))
    max_steps = int(os.environ.get("MAX_STEPS", "100"))

    print(f"BASE_MODEL={model_name}")
    print(f"DATA_PATH={data_path}")
    print(f"OUT_DIR={out_dir}")
    print("torch.cuda.is_available() =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu_name =", torch.cuda.get_device_name(0))

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    ds = load_dataset("json", data_files={"train": data_path})["train"]

    def tok_fn(ex):
        if "messages" in ex and ex["messages"]:
            text = build_text_from_messages(ex["messages"])
        else:
            text = ex["text"]
        x = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        x["labels"] = x["input_ids"].copy()
        return x

    ds = ds.map(tok_fn, remove_columns=ds.column_names)

    # 加载模型(仅LoRA,不含QLoRA/4bit)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    if torch.cuda.is_available():
        model.to("cuda")

    # LoRA注入模块
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

    # 设备自检(确保不是CPU)
    p = next(model.parameters())
    print("model param device =", p.device)

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_acc,
        learning_rate=lr,
        max_steps=max_steps,
        logging_steps=5,
        save_steps=max(10, max_steps),
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        no_cuda=False,
        report_to=[],
        optim="adamw_torch",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
    )

    trainer.train()

    # 保存LoRA适配器权重+tokenizer
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("训练完成，LoRA权重已保存:", out_dir)


if __name__ == "__main__":
    main()
