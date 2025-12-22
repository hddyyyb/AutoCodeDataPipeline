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
    """
    关键词启发式打分：优先拿到lockStock/hasStock/mapperSQL等“直接证据”
    """
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

    # query相关强命中
    for w in strong:
        if w in q and (w in t or w in p):
            score += 18
    for w in mid:
        wl = w.lower()
        if wl in q and (wl in t or wl in p):
            score += 6

    # chunk自身强命中
    for w in strong:
        if w in t or w in p:
            score += 8
    for w in mid:
        wl = w.lower()
        if wl in t or wl in p:
            score += 2

    # 结构加分：出现if/throw/return代表有分支/校验
    if re.search(r"\bif\b|\belse\b|\bthrow\b|\breturn\b", t):
        score += 4

    # 文件位置偏好
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

    # 允许同文件最多2个chunk，避免把lockStock实现切丢
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
            f"[{c['file_path']}:L{c['start_line']}-{c['end_line']}] (chunk_id={c['chunk_id']})\n"
            f"{c['content']}"
        )
    return "\n\n".join(parts) if parts else "N/A"


def build_evidence_alias(chunks: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Dict[str, Any]]]:
    """
    给证据分配稳定别名E1/E2/...
    让模型只输出E编号，避免chunk_id手写出错
    """
    alias_map: Dict[str, Dict[str, Any]] = {}
    lines = []
    for i, c in enumerate(chunks, 1):
        key = f"E{i}"
        alias_map[key] = c
        lines.append(f"- {key}: {c['file_path']}:L{c['start_line']}-{c['end_line']}")
    return "\n".join(lines) if lines else "- (empty)", alias_map


def hard_check_alias(output: str, alias_map: Dict[str, Dict[str, Any]]) -> Tuple[bool, str]:
    """
    硬校验：输出只能引用E编号(E1/E2/...)；不得出现未知E编号；必须至少引用一次
    """
    allowed = set(alias_map.keys())
    used = set(re.findall(r"\bE\d+\b", output))
    bad = [x for x in used if x not in allowed]
    if bad:
        return False, f"输出中检测到不在白名单中的证据编号:{bad}"
    if not used:
        return False, "输出未引用任何证据编号(E1/E2/...)"
    return True, "OK"


def hard_check_content(output: str) -> Tuple[bool, str]:
    """
    内容硬校验：防止“形式有了但内容在胡扯”
    必须满足：
    - 推理过程每步有(E编号)
    - 关键判断分支必须包含if/AND/CASE/UPDATE等原句痕迹
    - 与库存锁定主题相关关键词出现
    """
    must_have_patterns = [
        r"\bE\d+\b",                                  # 至少引用E编号
        r"\bAND\b|\bif\s*\(|\bCASE\b|\bUPDATE\b",     # 条件/分支/SQL更新
        r"lock_stock|库存|hasStock|lockStock|pms_sku_stock",
    ]
    missing = []
    for p in must_have_patterns:
        if not re.search(p, output):
            missing.append(p)
    if missing:
        return False, f"内容硬校验失败，缺少关键要素:{missing}"
    return True, "OK"


def render_sources_from_aliases(output: str, alias_map: Dict[str, Dict[str, Any]]) -> str:
    """
    把模型输出中的E编号，映射成可审计的文件路径与chunk_id，方便你最终展示/写报告
    """
    used = sorted(set(re.findall(r"\bE\d+\b", output)), key=lambda x: int(x[1:]))
    lines = []
    for e in used:
        c = alias_map.get(e)
        if not c:
            continue
        loc = f"{c['file_path']}:L{c['start_line']}-{c['end_line']}"
        lines.append(f"- {e} -> {loc} (chunk_id={c['chunk_id']})")
    return "\n".join(lines) if lines else "- N/A"


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
    top_chunks = retrieve_topk_chunks(rows, question, topk=6)

    evidence_text = format_evidence_text(top_chunks)
    whitelist_text, alias_map = build_evidence_alias(top_chunks)

    prompt = f"""### Instruction:
你是代码仓分析助手，请严格基于【可用代码证据】回答，禁止编造未出现的API/返回值/字符串。

【回答要求】
1.必须包含【结论】【关键判断与处理分支】【证据引用】【推理过程】
2.【证据引用】只能引用【允许引用的证据清单】中的E编号(例如E1/E2)，禁止输出chunk_id
3.【推理过程】每一步末尾必须标注对应E编号，例如“…(E1)”
4.任何未出现在【允许引用的证据清单】中的文件名/类名/返回值/字符串都视为编造，禁止输出
5.若证据不足，请输出“证据不足”，并说明缺少哪段代码(例如lockStock方法实现或mapperSQL)
6.【关键判断与处理分支】必须逐条引用证据中的原始条件/SQL片段/if判断(原文拷贝)，每条后面标注(E编号)
7.若无法从证据中找到if条件/AND条件/CASE更新/UPDATE语句等原句，则必须输出“证据不足”，禁止凭空总结

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
- 原句1: (从证据原文复制一条SQL/if/AND/CASE/UPDATE原句) (E?)
- 原句2: (再复制一条原句) (E?)
- 原句3: (可选) (E?)

【证据引用】
- E?
- E?

【推理过程】
1. ...(E?)
2. ...(E?)
3. ...(E?)

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
            max_new_tokens=280,
            do_sample=False,
            repetition_penalty=1.05,
        )

    gen_ids = out[0][prompt_len:]
    output = tokenizer.decode(gen_ids, skip_special_tokens=True)
    print(output)

    okA, msgA = hard_check_alias(output, alias_map)
    if not okA:
        print("\n=== HARD CHECK FAILED ===")
        print(msgA)
        return

    okC, msgC = hard_check_content(output)
    if not okC:
        print("\n=== CONTENT CHECK FAILED ===")
        print(msgC)
        print("建议：提高检索覆盖，优先抓到lockStock/hasStock方法本体或包含AND条件的SQL片段。")
        return

    print("\n=== AUDITABLE SOURCES (E -> file/lines/chunk_id) ===")
    print(render_sources_from_aliases(output, alias_map))


if __name__ == "__main__":
    main()
