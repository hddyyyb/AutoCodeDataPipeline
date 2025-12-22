#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_repo_index(repo_index_path: str) -> List[Dict[str, Any]]:
    rows = []
    p = Path(repo_index_path)
    if not p.exists():
        raise FileNotFoundError(f"repo_index not found: {repo_index_path}")
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _score_chunk(text: str, fp: str, query: str) -> int:
    """
    关键词启发式打分：优先拿到 lockStock/hasStock/mapper SQL 等“直接证据”
    """
    t = (text or "").lower()
    p = (fp or "").lower()
    q = (query or "").lower()

    # 强指示词：直接锁库/校验库存/DAO与Mapper
    strong = [
        "lockstock(", "hasstock(", "updateskustock", "lockstockbyskuid",
        "releaseskustocklock", "reduceskustock", "releasestockbyskuid",
        "pms_sku_stock", "lock_stock", "lockstock", "sku_stock",
        "mapper", "mybatis", "<update", "<select",
    ]
    # 一般指示词
    mid = ["stock", "库存", "lock", "锁定", "reserve", "deduct", "reduce", "release", "unlock", "timeout", "cancel"]

    score = 0
    # query相关
    for w in strong:
        if w in q and (w in t or w in p):
            score += 15
    for w in mid:
        wl = w.lower()
        if wl in q and (wl in t or wl in p):
            score += 5

    # chunk自身强命中
    for w in strong:
        if w in t or w in p:
            score += 6
    for w in mid:
        wl = w.lower()
        if wl in t or wl in p:
            score += 2

    # 结构加分：出现if/throw/return代表有分支/校验(更适合“关键判断与处理分支”)
    if re.search(r"\bif\b|\belse\b|\bthrow\b|\breturn\b", t):
        score += 3

    # 轻微偏好：service/dao/mapper相关文件
    if any(x in p for x in ["/service/", "/dao/", "/mapper/", "mapper.xml"]):
        score += 2

    return score


def retrieve_topk_chunks(rows: List[Dict[str, Any]], query: str, topk: int = 4) -> List[Dict[str, Any]]:
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for r in rows:
        s = _score_chunk(r.get("content", ""), r.get("file_path", ""), query)
        if s > 0:
            scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    # 去重：同一文件优先保留高分chunk，避免4个chunk都来自同一文件
    seen_fp = set()
    picked = []
    for s, r in scored:
        fp = r.get("file_path", "")
        if fp in seen_fp:
            continue
        seen_fp.add(fp)
        picked.append(r)
        if len(picked) >= topk:
            break
    return picked


def format_evidence_text(chunks: List[Dict[str, Any]]) -> str:
    parts = []
    for c in chunks:
        parts.append(
            f"[chunk_id={c['chunk_id']}] {c['file_path']}:L{c['start_line']}-{c['end_line']}\n"
            f"{c['content']}"
        )
    return "\n\n".join(parts) if parts else "N/A"


def format_whitelist(chunks: List[Dict[str, Any]]) -> str:
    lines = []
    for c in chunks:
        lines.append(f"- chunk_id={c['chunk_id']} {c['file_path']}:L{c['start_line']}-{c['end_line']}")
    return "\n".join(lines) if lines else "- (empty)"


def hard_check_output(output: str, allowed_chunks: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    硬校验：输出里若出现不在白名单中的file_path或chunk_id，则判定“不可审计”
    """
    allowed_cids = {c["chunk_id"] for c in allowed_chunks}
    allowed_fps = {c["file_path"] for c in allowed_chunks}

    # 抽chunk_id=xxxx
    found_cids = set(re.findall(r"chunk_id\s*=\s*([0-9a-fA-F]+)", output))
    bad_cids = [x for x in found_cids if x not in allowed_cids]

    # 粗抽文件路径(只要包含.java或.xml的相对路径)
    found_fps = set(re.findall(r"([\w\-/\\.]+?\.(?:java|xml))", output))
    # 规范化斜杠
    found_fps_norm = {x.replace("\\", "/") for x in found_fps}
    allowed_fps_norm = {x.replace("\\", "/") for x in allowed_fps}
    bad_fps = [x for x in found_fps_norm if x not in allowed_fps_norm]

    if bad_cids or bad_fps:
        msg = "输出中检测到不在白名单中的引用："
        if bad_cids:
            msg += f"\n- 非法chunk_id: {bad_cids}"
        if bad_fps:
            msg += f"\n- 非法file_path: {bad_fps}"
        msg += "\n请重试或提高检索证据覆盖(例如加入lockStock/hasStock方法本体或mapper SQL片段)。"
        return False, msg

    return True, "OK"


def main():
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    lora_dir = os.environ.get("LORA_DIR", "outputs/lora_qwen25_lora")
    repo_index_path = os.environ.get("REPO_INDEX", "data/raw_index/repo_index.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, lora_dir)
    model.eval()
    print("peft_loaded=", hasattr(model, "peft_config"), "lora_dir=", lora_dir)

    question = os.environ.get("QUESTION", "库存锁定规则在代码中是如何实现的？请结合代码说明关键判断与处理分支。")

    rows = load_repo_index(repo_index_path)
    top_chunks = retrieve_topk_chunks(rows, question, topk=4)

    evidence_text = format_evidence_text(top_chunks)
    whitelist_text = format_whitelist(top_chunks)

    prompt = f"""### Instruction:
你是代码仓分析助手，请严格基于【可用代码证据】回答，禁止编造未出现的API/返回值/字符串。

【回答要求】
1. 必须包含【结论】【证据引用】【推理过程】
2. 【证据引用】必须逐条写成“- chunk_id=... file_path:Lx-Ly”，且file_path必须与【允许引用的证据清单】完全一致
3. 【推理过程】每一步末尾必须标注对应chunk_id，例如“…(chunk_id=xxxx)”
4. 任何未出现在【允许引用的证据清单】中的文件名/类名/返回值/字符串都视为编造，禁止输出
5. 若证据不足，请输出“证据不足”，并说明缺少哪段代码(例如lockStock方法实现或mapper SQL)

【问题】
{question}

【可用代码证据】
{evidence_text}

【允许引用的证据清单(只能从这里选)】
{whitelist_text}

### Response:
"""

    print("=== QUESTION ===")
    print(question)
    print("=== PROMPT (sent to model) ===")
    print(prompt)
    print("=== MODEL OUTPUT ===")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=320,
            do_sample=False,  # 先用贪心，便于稳定复现
        )

    gen_ids = out[0][prompt_len:]
    output = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(output)

    ok, msg = hard_check_output(output, top_chunks)
    if not ok:
        print("\n=== HARD CHECK FAILED ===")
        print(msg)


if __name__ == "__main__":
    main()
