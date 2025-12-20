#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step08 (Optional)
Lightweight SFT sanity check for Qwen2.5-style instruction tuning.

Purpose:
- Verify dataset compatibility
- Validate instruction/response formatting
- Ensure no runtime errors in the fine-tuning pipeline

This script is NOT intended to train a performant model.
"""

import json
import random
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]


def load_qa_samples(path: Path, max_samples: int = 10):
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if len(samples) >= max_samples:
                break
            obj = json.loads(line)
            samples.append(obj)
    return samples


def format_sft_example(sample):
    """
    Qwen-style instruction formatting (simplified).
    """
    instruction = sample["question"]
    response = sample["answer"]

    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model:{model_name} on {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)

    qa_samples = load_qa_samples(ROOT / "data/samples/qa_samples.jsonl", max_samples=10)
    random.shuffle(qa_samples)

    texts = [format_sft_example(s) for s in qa_samples]

    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

    class SimpleDataset(torch.utils.data.Dataset):
        def __init__(self, encodings):
            self.encodings = encodings

        def __len__(self):
            return self.encodings["input_ids"].size(0)

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = item["input_ids"].clone()
            return item

    dataset = SimpleDataset(encodings)

    args = TrainingArguments(
        output_dir=str(ROOT / "tmp_sft_check"),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=10,
        logging_steps=1,
        save_steps=20,
        learning_rate=5e-5,
        report_to=[],
        fp16=(device == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Starting SFT sanity check...")
    trainer.train()
    print("SFT sanity check finished successfully.")


if __name__ == "__main__":
    main()
