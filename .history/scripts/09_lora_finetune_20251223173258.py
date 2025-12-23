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


# -----------------------------
# Utils
# -----------------------------
def build_text_from_messages(messages: List[Dict[str, Any]]) -> str:
    parts = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        parts.append(f"<{role}>\n{content}")
    return "\n".join(parts)


def pick_existing(paths: List[Path]) -> Optional[str]:
    for p in paths:
        if p.exists():
            return str(p)
    return None


def pick_sft_path(split: str, train_lang: str, prefer_format: str = "auto") -> Optional[str]:
    base = ROOT / "data/sft"
    lang = (train_lang or "").strip().lower()
    if lang not in ("zh", "en"):
        lang = ""

    cands_text, cands_msg = [], []

    if lang:
        cands_text.append(base / f"{split}_{lang}.jsonl")
        cands_msg.append(base / f"{split}_{lang}.messages.jsonl")

    cands_text.append(base / f"{split}.jsonl")
    cands_msg.append(base / f"{split}.messages.jsonl")

    if prefer_format == "text":
        return pick_existing(cands_text)
    if prefer_format == "messages":
        return pick_existing(cands_msg)
    return pick_existing(cands_text) or pick_existing(cands_msg)


def detect_format(path: str) -> str:
    return "messages" if path.endswith(".messages.jsonl") else "text"


# -----------------------------
# 🔥关键修复：训练模板包装
# -----------------------------
def wrap_to_infer_template(text: str, train_lang: str) -> str:
    if not text:
        return text

    if train_lang == "zh":
        if "【结论】" in text and "【证据引用】" in text:
            return text

        return (
            "### Instruction\n"
            "请基于下方草稿内容，整理为【严格模板输出】。\n"
            "必须包含且仅包含以下标题：\n"
            "【结论】【关键判断与处理分支】【证据引用】【推理过程】\n"
            "禁止输出JSON。\n\n"
            "### Draft\n"
            + text.strip()
            + "\n\n### Response\n"
            "【结论】\n\n"
            "【关键判断与处理分支】\n\n"
            "【证据引用】\n\n"
            "【推理过程】\n"
        )

    # 英文可后续扩展
    return text


# -----------------------------
# Main
# -----------------------------
def main():
    model_name = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    out_dir = os.environ.get("OUT_DIR", str(ROOT / "outputs/lora_qwen25_demo"))

    train_lang = os.environ.get("TRAIN_LANG", "").strip().lower()
    prefer_format = os.environ.get("SFT_FORMAT", "auto").strip().lower()

    train_path = os.environ.get("TRAIN_PATH") or pick_sft_path("train", train_lang, prefer_format)
    dev_path = os.environ.get("DEV_PATH") or pick_sft_path("dev", train_lang, prefer_format)

    if not train_path:
        raise FileNotFoundError("未找到SFT数据，请先运行08")

    print(f"TRAIN_LANG={train_lang}")
    print(f"TRAIN_PATH={train_path}")
    print(f"DEV_PATH={dev_path or '(none)'}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    files = {"train": train_path}
    if dev_path:
        files["validation"] = dev_path
    ds = load_dataset("json", data_files=files)

    max_len = int(os.environ.get("MAX_LEN", "512"))

    def tok_fn(ex):
        if "messages" in ex and ex["messages"]:
            text = build_text_from_messages(ex["messages"])
        else:
            text = ex.get("text", "")

        text = wrap_to_infer_template(text, train_lang)

        x = tokenizer(
            text,
            truncation=True,
            max_length=max_len,
            padding="max_length",
        )
        x["labels"] = x["input_ids"].copy()
        return x

    train_ds = ds["train"].map(tok_fn, remove_columns=ds["train"].column_names)
    eval_ds = ds["validation"].map(tok_fn, remove_columns=ds["validation"].column_names) if "validation" in ds else None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    peft_config = LoraConfig(
        r=int(os.environ.get("LORA_R", "16")),
        lora_alpha=int(os.environ.get("LORA_ALPHA", "32")),
        lora_dropout=float(os.environ.get("LORA_DROPOUT", "0.05")),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=int(os.environ.get("BS", "1")),
        gradient_accumulation_steps=int(os.environ.get("GA", "4")),
        learning_rate=float(os.environ.get("LR", "2e-4")),
        max_steps=int(os.environ.get("MAX_STEPS", "120")),
        logging_steps=5,
        save_steps=int(os.environ.get("SAVE_STEPS", "20")),
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to=[],
    )

    Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    ).train()

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("LoRA训练完成:", out_dir)


if __name__ == "__main__":
    main()
