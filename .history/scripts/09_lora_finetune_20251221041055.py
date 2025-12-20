#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

ROOT = Path(__file__).resolve().parents[1]

def main():
    # 你可以改成更大的Qwen2.5，但建议从0.5B/1.5B起步
    model_name = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B")
    use_4bit = os.environ.get("USE_4BIT", "1") == "0"  # 默认LoRA
    out_dir = os.environ.get("OUT_DIR", str(ROOT / "outputs/lora_qwen25"))
    data_path = os.environ.get("DATA_PATH", str(ROOT / "data/sft/train.jsonl"))

    # LoRA超参(先保守，跑通为主)
    lora_r = int(os.environ.get("LORA_R", "16"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", "32"))
    lora_dropout = float(os.environ.get("LORA_DROPOUT", "0.05"))

    # 训练超参
    max_len = int(os.environ.get("MAX_LEN", "512"))
    lr = float(os.environ.get("LR", "2e-4"))
    batch_size = int(os.environ.get("BS", "1"))
    grad_acc = int(os.environ.get("GA", "4"))
    max_steps = int(os.environ.get("MAX_STEPS", "50"))

    print(f"BASE_MODEL={model_name}")
    print(f"USE_4BIT={use_4bit}")
    print(f"DATA_PATH={data_path}")
    print(f"OUT_DIR={out_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载数据集
    ds = load_dataset("json", data_files={"train": data_path})["train"]

    def tok_fn(ex):
        x = tokenizer(
            ex["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        x["labels"] = x["input_ids"].copy()
        return x

    ds = ds.map(tok_fn, remove_columns=ds.column_names)

    # 加载模型
    if use_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

    # 选择LoRA注入模块：Qwen类模型一般对q_proj/k_proj/v_proj/o_proj以及up/down/gate有效
    target_modules = ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj"]

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
        logging_steps=5,
        save_steps=max(10, max_steps),
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
    )

    from transformers import Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
    )

    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("训练完成，LoRA权重已保存:", out_dir)

if __name__ == "__main__":
    main()
