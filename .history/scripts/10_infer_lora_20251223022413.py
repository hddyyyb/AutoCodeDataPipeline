#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from pathlib import Path

def load_repo_index(repo_index_path: str):
    rows = []
    with open(repo_index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def retrieve_topk_chunks(rows, query: str, topk: int = 3):
    q = query.lower()
    keywords = ["stock", "库存", "lock", "锁定", "reserve", "deduct", "reduce", "释放", "unlock"]
    scored = []
    for r in rows:
        text = (r.get("content") or "").lower()
        fp = (r.get("file_path") or "").lower()
        score = 0
        # query命中加分
        if any(w in q for w in keywords):
            for w in keywords:
                if w in text or w in fp:
                    score += 2
        # 基础：content命中query片段加分
        for token in keywords:
            if token in q and token in text:
                score += 3
        if score > 0:
            scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:topk]]


def main():
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B")
    lora_dir = os.environ.get("LORA_DIR", "outputs/lora_qwen25_lora")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    # 关键：加载LoRA适配器
    model = PeftModel.from_pretrained(base, lora_dir)
    model.eval()
    print("peft_loaded=", hasattr(model, "peft_config"), "lora_dir=", lora_dir)

    question = os.environ.get("QUESTION", "库存锁定规则在代码中是如何实现的？请结合代码说明关键判断与处理分支。")
    repo_index_path = os.environ.get("REPO_INDEX", "data/raw_index/repo_index.jsonl")
    rows = load_repo_index(repo_index_path)
    top_chunks = retrieve_topk_chunks(rows, question, topk=3)

    evidence_text = "\n\n".join([
        f"[chunk_id={c['chunk_id']}] {c['file_path']}:L{c['start_line']}-{c['end_line']}\n{c['content']}"
        for c in top_chunks
    ]) if top_chunks else "N/A"


    prompt = f"""### Instruction:
    你是代码仓分析助手，请严格基于【可用代码证据】回答，禁止编造未出现的API/返回值/字符串。

    【回答要求】
    1. 必须包含【结论】【证据引用】【推理过程】
    2. 【证据引用】只能引用【可用代码证据】里出现的chunk_id与文件行号
    3. 若证据不足，请明确说“证据不足”并说明缺少哪段代码

    【问题】
    {question}

    【可用代码证据】
    {evidence_text}

    ### Response:
    """


    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,  # 先用贪心，便于稳定验证
        )

    gen_ids = out[0][prompt_len:]
    print("=== QUESTION ===")
    print(question)
    print("=== PROMPT (sent to model) ===")
    print(prompt)
    print("=== MODEL OUTPUT ===")

    print(tokenizer.decode(gen_ids, skip_special_tokens=True))

if __name__ == "__main__":
    main()
