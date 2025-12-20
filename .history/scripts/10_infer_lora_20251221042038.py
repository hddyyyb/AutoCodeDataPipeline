#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def main():
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B")
    lora_dir = os.environ.get("LORA_DIR", "outputs/lora_qwen25")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()

    prompt = "### Instruction:\n库存锁定规则在代码中是如何实现的？请结合代码说明关键判断与处理分支。\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    print(tokenizer.decode(out[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
