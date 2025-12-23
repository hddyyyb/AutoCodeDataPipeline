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
    rows: List[Dict[str, Any]] = []
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
    t = (text or "").lower()
    p = (fp or "").lower()
    q = (query or "").lower()

    strong = [
        "lockstock(", "hasstock(", "updateskustock", "lockstockbyskuid",
        "releaseskustocklock", "reduceskustock", "releasestockbyskuid",
        "pms_sku_stock", "lock_stock", "lockstock", "sku_stock",
        "mapper", "mybatis", "<update", "<select",
        "update pms_sku_stock", "and lock_stock", "and stock",
    ]
    mid = ["stock", "库存", "lock", "锁定", "reserve", "deduct", "reduce", "release", "unlock", "timeout", "cancel"]

    score = 0
    for w in strong:
        if w in q and (w in t or w in p):
            score += 18
    for w in mid:
        wl = w.lower()
        if wl in q and (wl in t or wl in p):
            score += 6

    for w in strong:
        if w in t or w in p:
            score += 8
    for w in mid:
        wl = w.lower()
        if wl in t or wl in p:
            score += 2

    if re.search(r"\bif\b|\belse\b|\bthrow\b|\breturn\b", t):
        score += 4
    if any(x in p for x in ["/service/", "/dao/", "/mapper/", "mapper.xml", "/resources/dao/"]):
        score += 3
    return score


def retrieve_topk_chunks(rows: List[Dict[str, Any]], query: str, topk: int = 6) -> List[Dict[str, Any]]:
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for r in rows:
        s = _score_chunk(r.get("content", ""), r.get("file_path", ""), query)
        if s > 0:
            scored.append((s, r))
    scored.sort(key=lambda x: x[0], reverse=True)

    per_fp_cnt: Dict[str, int] = {}
    picked: List[Dict[str, Any]] = []
    for s, r in scored:
        fp = r.get("file_path", "")
        per_fp_cnt[fp] = per_fp_cnt.get(fp, 0) + 1
        if per_fp_cnt[fp] > 2:
            continue
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
        lines.append(f"- {c['file_path']}:L{c['start_line']}-{c['end_line']} (chunk_id={c['chunk_id']})")
    return "\n".join(lines) if lines else "- (empty)"


def hard_check_output(output: str, allowed_chunks: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """
    白名单硬校验：输出里若出现不在白名单中的file_path或chunk_id，则判定“不可审计”
    """
    allowed_cids = {str(c["chunk_id"]) for c in allowed_chunks}
    allowed_fps = {str(c["file_path"]).replace("\\", "/") for c in allowed_chunks}

    # 兼容：chunk_id可能是10位sha1前缀或更长
    found_cids = set(re.findall(r"chunk_id\s*=\s*([0-9a-fA-F]{6,64})", output))
    bad_cids = [x for x in found_cids if x not in allowed_cids]

    found_fps = set(re.findall(r"([\w\-/\\.]+?\.(?:java|xml))", output))
    found_fps_norm = {x.replace("\\", "/") for x in found_fps}
    bad_fps = [x for x in found_fps_norm if x not in allowed_fps]

    if bad_cids or bad_fps:
        msg = "输出中检测到不在白名单中的引用："
        if bad_cids:
            msg += f"\n- 非法chunk_id:{bad_cids}"
        if bad_fps:
            msg += f"\n- 非法file_path:{bad_fps}"
        msg += "\n请重试或提高检索证据覆盖(例如加入lockStock/hasStock方法本体或mapperSQL片段)。"
        return False, msg

    return True, "OK"


def hard_check_content(output: str) -> Tuple[bool, str]:
    """
    内容硬校验：防止“格式对了但在胡扯”
    必须满足：
    - 推理过程每步有(chunk_id=...)
    - 关键判断分支包含if/AND/CASE/UPDATE等原句痕迹
    - 与库存锁定主题相关关键词出现
    """
    must_have_patterns = [
        r"\(chunk_id=[0-9a-fA-F]{6,64}\)",            # 绑定chunk_id
        r"\bAND\b|\bif\s*\(|\bCASE\b|\bUPDATE\b",     # 条件/分支/SQL更新
        r"lock_stock|库存|hasStock|lockStock|pms_sku_stock",  # 主题关键词
    ]
    missing = []
    for p in must_have_patterns:
        if not re.search(p, output):
            missing.append(p)
    if missing:
        return False, f"内容硬校验失败，缺少关键要素:{missing}"
    return True, "OK"


def main():
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    lora_dir = os.environ.get("LORA_DIR", "outputs/lora_qwen25_demo")
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
    topk = int(os.environ.get("TOPK", "6"))

    rows = load_repo_index(repo_index_path)
    top_chunks = retrieve_topk_chunks(rows, question, topk=topk)

    evidence_text = format_evidence_text(top_chunks)
    whitelist_text = format_whitelist(top_chunks)

    # 对齐你训练集常见输出格式：结论/证据引用/原文代码段/推理过程
    prompt = f"""### Instruction
你是代码仓分析助手，请严格基于【可用代码证据】回答，禁止编造未出现的API/返回值/字符串。

【回答要求】
1.必须包含【结论】【关键判断与处理分支】【证据引用】【推理过程】
2.【证据引用】必须逐条写成“- file_path:Lx-Ly (chunk_id=...)”，且file_path与chunk_id必须来自【允许引用的证据清单】
3.【推理过程】每一步末尾必须标注对应chunk_id，例如“…(chunk_id=xxxx)”
4.任何未出现在【允许引用的证据清单】中的文件名/类名/返回值/字符串都视为编造，禁止输出
5.若证据不足，请输出“证据不足”，并说明缺少哪段代码(例如lockStock方法实现或mapperSQL)

【问题】
{question}

【可用代码证据】
{evidence_text}

【允许引用的证据清单(只能从这里选)】
{whitelist_text}

【请严格按下面模板作答，不得增删标题】

【结论】
(用1-2句总结库存锁定如何实现)

【关键判断与处理分支】
- 原句1:(从证据原文复制一条SQL/if/AND/CASE/UPDATE原句)(chunk_id=xxxx)
- 原句2:(再复制一条原句)(chunk_id=xxxx)
- 原句3:(可选)(chunk_id=xxxx)

【证据引用】
- file_path:Lx-Ly (chunk_id=xxxx)
- file_path:Lx-Ly (chunk_id=xxxx)

【推理过程】
1. ...(chunk_id=xxxx)
2. ...(chunk_id=xxxx)
3. ...(chunk_id=xxxx)

### Response
"""

    print("=== QUESTION ===")
    print(question)
    print("=== MODEL OUTPUT ===")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "260")),
            do_sample=False,
            repetition_penalty=float(os.environ.get("REP_PENALTY", "1.05")),
        )

    gen_ids = out[0][prompt_len:]
    output = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(output)

    ok1, msg1 = hard_check_output(output, top_chunks)
    if not ok1:
        print("\n=== HARD CHECK FAILED ===")
        print(msg1)
        return

    ok2, msg2 = hard_check_content(output)
    if not ok2:
        print("\n=== CONTENT CHECK FAILED ===")
        print(msg2)
        print("建议：提高检索覆盖，优先抓到lockStock/hasStock方法本体或包含AND条件的SQL片段。")
        return

    print("\n=== HARD CHECK PASSED ===")
    print("输出满足可审计引用与内容关键要素。")


if __name__ == "__main__":
    main()
