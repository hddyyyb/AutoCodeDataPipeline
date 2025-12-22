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

    prompt = f"""### Instruction:
你是代码仓分析助手，请按以下格式回答，必须包含证据引用与推理过程，禁止空泛描述。

【结论】
...

【证据引用】
- file_path:Lx-Ly(chunk_id=...)

【推理过程】
1. ...
2. ...
3. ...

问题:{question}

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
    print(tokenizer.decode(gen_ids, skip_special_tokens=True))

if __name__ == "__main__":
    main()
