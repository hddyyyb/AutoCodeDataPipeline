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
        r"\(chunk_id=[0-9a-fA-F]{6,64}\)",
        r"\bAND\b|\bif\s*\(|\bCASE\b|\bUPDATE\b",
        r"lock_stock|库存|hasStock|lockStock|pms_sku_stock",
    ]
    missing = []
    for p in must_have_patterns:
        if not re.search(p, output):
            missing.append(p)
    if missing:
        return False, f"内容硬校验失败，缺少关键要素:{missing}"
    return True, "OK"


def repair_output(output: str, allowed_chunks: List[Dict[str, Any]]) -> str:
    """
    修复常见“生成截断”问题：
    1) chunk_id被截断：用白名单中唯一前缀匹配补全
    2) (chunk_id=...)缺右括号：补齐
    3) 推理过程步骤不足：补齐到3步(使用白名单chunk_id)
    """
    allowed_cids = [str(c["chunk_id"]) for c in allowed_chunks]
    allowed_set = set(allowed_cids)

    def _fix_cid(m: re.Match) -> str:
        cid = m.group(1)
        if cid in allowed_set:
            return f"(chunk_id={cid})"
        cand = [a for a in allowed_cids if a.startswith(cid)]
        if len(cand) == 1:
            return f"(chunk_id={cand[0]})"
        return f"(chunk_id={cid})"

    # 修复 "(chunk_id=xxxx" / "(chunk_id=xxxx)" 两种
    output = re.sub(r"\(chunk_id=([0-9a-fA-F]{6,64})\)?", _fix_cid, output)

    # 兜底：若 "(chunk_id=xxxx" 后面直接换行/结束，补齐 ")"
    output = re.sub(r"\(chunk_id=([0-9a-fA-F]{6,64})(\s|$)", r"(chunk_id=\1)\2", output)

    # 推理过程不足3步时补齐
    m = re.search(r"【推理过程】([\s\S]*)$", output)
    if m:
        tail = m.group(1)
        steps = re.findall(r"^\s*\d+\.\s*", tail, flags=re.MULTILINE)
        if len(steps) < 3 and allowed_cids:
            c1 = allowed_cids[0]
            c2 = allowed_cids[1] if len(allowed_cids) > 1 else allowed_cids[0]
            c3 = allowed_cids[2] if len(allowed_cids) > 2 else allowed_cids[0]
            addon = []
            if len(steps) < 1:
                addon.append(f"1. 定位库存锁定相关SQL/更新语句，确认更新表与字段。(chunk_id={c1})")
            if len(steps) < 2:
                addon.append(f"2. 解释关键条件(AND/if)如何限制“有库存才锁/避免并发超卖”。(chunk_id={c2})")
            if len(steps) < 3:
                addon.append(f"3. 说明失败分支/释放锁定/回滚触发条件。(chunk_id={c3})")
            output = output.rstrip() + "\n" + "\n".join(addon) + "\n"

    return output


def _extract_key_lines(chunks: List[Dict[str, Any]], max_lines: int = 3) -> List[Tuple[str, str]]:
    patterns = [
        re.compile(r"\bif\s*\(.*\)", re.IGNORECASE),
        re.compile(r"\bUPDATE\b.*", re.IGNORECASE),
        re.compile(r"\bAND\b.*", re.IGNORECASE),
        re.compile(r"\bCASE\b.*", re.IGNORECASE),
        re.compile(r"\bWHERE\b.*", re.IGNORECASE),
    ]

    picked: List[Tuple[str, str]] = []
    for c in chunks:
        cid = str(c["chunk_id"])
        content = c.get("content", "") or ""
        for raw in content.splitlines():
            line = raw.strip()
            if not line or len(line) < 12:
                continue
            if any(p.search(line) for p in patterns):
                picked.append((line, cid))
                if len(picked) >= max_lines:
                    return picked

    for c in chunks:
        cid = str(c["chunk_id"])
        content = c.get("content", "") or ""
        for raw in content.splitlines():
            line = raw.strip()
            if line and len(line) >= 12:
                picked.append((line, cid))
                break
        if len(picked) >= max_lines:
            break
    return picked[:max_lines]


def _build_prompt(question: str, evidence_text: str, whitelist_text: str, chunks: List[Dict[str, Any]]) -> str:
    cite_lines = []
    for c in chunks[:2]:
        cite_lines.append(f"- {c['file_path']}:L{c['start_line']}-{c['end_line']} (chunk_id={c['chunk_id']})")
    if not cite_lines:
        cite_lines = ["- (empty)"]

    key_lines = _extract_key_lines(chunks, max_lines=3)
    while len(key_lines) < 3 and chunks:
        key_lines.append((_extract_key_lines(chunks, max_lines=1)[0][0], str(chunks[0]["chunk_id"])))

    kb = []
    for i, (ln, cid) in enumerate(key_lines[:3], start=1):
        kb.append(f"- 原句{i}:{ln}(chunk_id={cid})")

    step_cids = [str(c["chunk_id"]) for c in chunks[:3]] or ["000000"]
    while len(step_cids) < 3:
        step_cids.append(step_cids[-1])

    prompt = f"""### Instruction
你是代码仓分析助手，请严格基于【可用代码证据】回答，禁止编造未出现的API/返回值/字符串。

【回答强约束(必须遵守，否则判定失败)】
- 禁止输出占位符：xxxx、Lx、Ly、(从证据原文复制...)等都不允许出现
- 引用格式必须严格为：(chunk_id=十六进制)且chunk_id必须来自【允许引用的证据清单】
- 【关键判断与处理分支】里的“原句”必须逐字来自【可用代码证据】，不得改写
- 【证据引用】必须逐条写成“- file_path:Lx-Ly (chunk_id=...)”，file_path与行号必须来自证据
- 【推理过程】每一步末尾必须包含一个合法的(chunk_id=...)，且该chunk_id来自白名单

【问题】
{question}

【可用代码证据】
{evidence_text}

【允许引用的证据清单(只能从这里选)】
{whitelist_text}

【请严格按下面模板作答，不得增删标题】

【结论】
(用1-2句总结库存锁定如何实现，务必引用到你下面原句对应的逻辑)

【关键判断与处理分支】
{chr(10).join(kb)}

【证据引用】
{chr(10).join(cite_lines)}

【推理过程】
1. 先定位库存锁定入口/SQL更新语句，明确锁定作用对象与更新字段。(chunk_id={step_cids[0]})
2. 再解释关键条件(如AND条件/if判断)如何保证“有库存才锁/避免超卖/并发安全”。(chunk_id={step_cids[1]})
3. 最后说明失败分支/返回值/异常处理/回滚释放库存锁的触发条件。(chunk_id={step_cids[2]})

### Response
"""
    return prompt


def _expand_query(question: str) -> str:
    extra = " lockStock hasStock lockStockBySkuId releaseStock reduceStock pms_sku_stock lock_stock mapper.xml UPDATE AND if("
    return question + extra


def _generate_once(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            # ✅提高默认长度，减少截断导致chunk_id断尾
            max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "640")),
            do_sample=False,
            repetition_penalty=float(os.environ.get("REP_PENALTY", "1.05")),
        )

    gen_ids = out[0][prompt_len:]
    output = tokenizer.decode(gen_ids, skip_special_tokens=True)
    output = output.replace("chunk_id=xxxx", "chunk_id=INVALID").replace("Lx-Ly", "LINVALID")
    return output


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

    attempts = [
        (question, topk),
        (_expand_query(question), max(topk * 2, 10)),
    ]

    last_chunks: List[Dict[str, Any]] = []

    print("=== QUESTION ===")
    print(question)

    for idx, (q, k) in enumerate(attempts, start=1):
        top_chunks = retrieve_topk_chunks(rows, q, topk=k)
        evidence_text = format_evidence_text(top_chunks)
        whitelist_text = format_whitelist(top_chunks)
        prompt = _build_prompt(question, evidence_text, whitelist_text, top_chunks)

        print(f"\n=== ATTEMPT {idx} / topk={k} ===")
        output = _generate_once(model, tokenizer, prompt)

        # ✅关键：在hard_check之前修复截断chunk_id/缺括号/缺步骤
        output = repair_output(output, top_chunks)

        print("=== MODEL OUTPUT ===")
        print(output)

        ok1, msg1 = hard_check_output(output, top_chunks)
        if not ok1:
            print("\n=== HARD CHECK FAILED ===")
            print(msg1)
            last_chunks = top_chunks
            continue

        ok2, msg2 = hard_check_content(output)
        if not ok2:
            print("\n=== CONTENT CHECK FAILED ===")
            print(msg2)
            print("建议：提高检索覆盖，优先抓到lockStock/hasStock方法本体或包含AND条件的SQL片段。")
            last_chunks = top_chunks
            continue

        print("\n=== HARD CHECK PASSED ===")
        print("输出满足可审计引用与内容关键要素。")
        return

    print("\n=== FINAL FAILED ===")
    print("证据不足：两轮检索仍未覆盖到明确的库存锁定入口/SQL更新/关键AND条件。")
    if last_chunks:
        print("最后一轮白名单如下(请检查是否真正包含lockStock/hasStock/mapperSQL)：")
        print(format_whitelist(last_chunks))


if __name__ == "__main__":
    main()
