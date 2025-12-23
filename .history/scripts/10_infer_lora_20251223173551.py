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


# -----------------------------
# Repo index + retrieval
# -----------------------------
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


# -----------------------------
# Strict format + checks (FIXED)
# -----------------------------
_CID16 = r"[0-9a-fA-F]{16}"  # 你的chunk_id是16位hex(从输出里可见)


def is_json_like(s: str) -> bool:
    t = (s or "").lstrip()
    return t.startswith("{") or t.startswith("[")


def extract_evidence_ref_lines(output: str, lang: str) -> List[str]:
    """
    只从“证据引用/Evidence References”区块抽取引用行，避免误把JSON字段/trace_id里的文件名当成引用。
    """
    text = output or ""

    if lang == "en":
        # [Evidence References] ... [Reasoning Trace] or end
        m = re.search(r"\[Evidence References\]\s*([\s\S]*?)(?:\n\[[A-Za-z ].*?\]|\Z)", text)
    else:
        m = re.search(r"【证据引用】\s*([\s\S]*?)(?:\n【.*?】|\Z)", text)

    block = m.group(1) if m else ""
    lines = []
    for raw in block.splitlines():
        line = raw.strip()
        if line.startswith("- "):
            lines.append(line)
    return lines


def hard_check_format(output: str, lang: str) -> Tuple[bool, str]:
    """
    1)必须是纯文本模板，不允许JSON
    2)必须包含固定标题
    """
    if is_json_like(output):
        return False, "格式失败:输出是JSON/类JSON，必须按模板输出纯文本。"

    if lang == "en":
        must = ["[Conclusion]", "[Key Conditions]", "[Evidence References]", "[Reasoning Trace]"]
        for h in must:
            if h not in output:
                return False, f"格式失败:缺少标题{h}"
    else:
        must = ["【结论】", "【关键判断与处理分支】", "【证据引用】", "【推理过程】"]
        for h in must:
            if h not in output:
                return False, f"格式失败:缺少标题{h}"

    return True, "OK"


def hard_check_output(output: str, allowed_chunks: List[Dict[str, Any]], lang: str) -> Tuple[bool, str]:
    """
    白名单硬校验(修复版):
    - 只在“证据引用区块”里校验file_path是否在白名单
    - chunk_id只允许16位hex，且必须来自白名单(在全文里检查chunk_id=... 但要求16位，避免截断误判/误抓)
    """
    allowed_cids = {str(c["chunk_id"]) for c in allowed_chunks}
    allowed_fps = {str(c["file_path"]).replace("\\", "/") for c in allowed_chunks}

    # 1)校验 Evidence References 区块的 file_path
    ref_lines = extract_evidence_ref_lines(output, lang=lang)
    bad_fps = []
    for line in ref_lines:
        # 期望形态: - path:Lx-Ly (chunk_id=xxxx)
        m = re.search(r"-\s+([^\s:]+?\.(?:java|xml|yml|yaml|properties|py|ts|js|sql)):", line)
        if m:
            fp = m.group(1).replace("\\", "/")
            if fp not in allowed_fps:
                bad_fps.append(fp)

    # 2)校验chunk_id(全文) —— 只抓16位，避免“85565e...b8”这种截断被当成合法格式
    found_cids = set(re.findall(rf"chunk_id\s*=\s*({_CID16})", output))
    bad_cids = [x for x in found_cids if x not in allowed_cids]

    if bad_fps or bad_cids:
        msg = "输出中检测到不在白名单中的引用："
        if bad_fps:
            msg += f"\n- 非法file_path:{bad_fps}"
        if bad_cids:
            msg += f"\n- 非法chunk_id:{bad_cids}"
        msg += "\n请重试或提高证据覆盖(例如加入关键方法本体或mapperSQL片段)。"
        return False, msg

    # 3)额外：要求证据引用区块至少1条
    if len(ref_lines) < 1:
        return False, "输出缺少证据引用区块中的引用行(以'- '开头)。"

    return True, "OK"


def hard_check_content(output: str, lang: str) -> Tuple[bool, str]:
    """
    内容硬校验(更贴合模板):
    - 推理过程每步必须包含(chunk_id=16hex)
    - 关键判断区块必须出现至少2条“原句/Snippet”，并带chunk_id
    - 必须出现主题关键词
    """
    if lang == "en":
        trace_block = re.search(r"\[Reasoning Trace\]\s*([\s\S]*)\Z", output)
        key_block = re.search(r"\[Key Conditions\]\s*([\s\S]*?)\n\[Evidence References\]", output)
        kw_pat = r"stock|lock|reserve|deduct|release|pms_sku_stock|lock_stock|hasStock|lockStock"
    else:
        trace_block = re.search(r"【推理过程】\s*([\s\S]*)\Z", output)
        key_block = re.search(r"【关键判断与处理分支】\s*([\s\S]*?)\n【证据引用】", output)
        kw_pat = r"库存|锁定|stock|lock_stock|pms_sku_stock|hasStock|lockStock"

    if not trace_block:
        return False, "内容失败:找不到推理过程区块。"
    tb = trace_block.group(1)
    steps = re.findall(r"^\s*\d+\.\s+.*?\(chunk_id=" + _CID16 + r"\)\s*$", tb, flags=re.MULTILINE)
    if len(steps) < 3:
        return False, "内容失败:推理过程不足3步或某步缺少合法(chunk_id=16hex)。"

    if not key_block:
        return False, "内容失败:找不到关键判断区块。"
    kb = key_block.group(1)
    cites = re.findall(r"\(chunk_id=" + _CID16 + r"\)", kb)
    if len(cites) < 2:
        return False, "内容失败:关键判断区块至少需要2条带(chunk_id=16hex)的原句/片段。"

    if not re.search(kw_pat, output, re.IGNORECASE):
        return False, "内容失败:缺少主题关键词(库存/锁定/stock/lock等)。"

    return True, "OK"


def repair_output(output: str, allowed_chunks: List[Dict[str, Any]], lang: str) -> str:
    """
    只做轻量修复：补齐少量'(chunk_id=...)'右括号、去掉明显的占位符。
    不做“截断chunk_id补全”(因为现在只承认16位，截断直接重试更可靠)。
    """
    out = output or ""
    out = out.replace("Lx-Ly", "LINVALID").replace("chunk_id=xxxx", "chunk_id=INVALID")
    # 补括号
    out = re.sub(rf"\(chunk_id=({_CID16})\b(?!\))", r"(chunk_id=\1)", out)
    return out


# -----------------------------
# Prompting
# -----------------------------
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


def _build_prompt(question: str, evidence_text: str, whitelist_text: str, chunks: List[Dict[str, Any]], lang: str, strict: bool) -> str:
    cite_lines = []
    for c in chunks[:2]:
        cite_lines.append(f"- {c['file_path']}:L{c['start_line']}-{c['end_line']} (chunk_id={c['chunk_id']})")
    if not cite_lines:
        cite_lines = ["- (empty)"]

    key_lines = _extract_key_lines(chunks, max_lines=3)
    kb = []
    for i, (ln, cid) in enumerate(key_lines[:3], start=1):
        if lang == "en":
            kb.append(f"- Snippet{i}:{ln}(chunk_id={cid})")
        else:
            kb.append(f"- 原句{i}:{ln}(chunk_id={cid})")
    if len(kb) < 2 and chunks:
        # 保底：至少2条
        cid0 = str(chunks[0]["chunk_id"])
        line0 = (chunks[0].get("content") or "").splitlines()[0].strip() if (chunks[0].get("content") or "").splitlines() else ""
        if lang == "en":
            kb.append(f"- SnippetX:{line0}(chunk_id={cid0})")
        else:
            kb.append(f"- 原句X:{line0}(chunk_id={cid0})")

    step_cids = [str(c["chunk_id"]) for c in chunks[:3]] or ["0000000000000000"]
    while len(step_cids) < 3:
        step_cids.append(step_cids[-1])

    # strict模式：额外禁止JSON，禁止出现除证据引用区块之外的任何文件名
    strict_line = ""
    if strict:
        strict_line = (
            "\n[Extra Hard Rule]\n"
            "- Output MUST be plain text with the exact headings below. DO NOT output JSON.\n"
            "- Do NOT mention any filenames anywhere except inside [Evidence References] lines.\n"
            "- chunk_id MUST be exactly 16 hex characters.\n"
        ) if lang == "en" else (
            "\n【额外硬规则】\n"
            "- 必须输出纯文本，严格使用下方标题，禁止输出JSON。\n"
            "- 除【证据引用】区块外，禁止提及任何文件名(包含.json/.js等)。\n"
            "- chunk_id必须严格为16位十六进制。\n"
        )

    if lang == "en":
        return f"""### Instruction
You are a repository analysis assistant. Answer strictly based on the provided evidence. Do NOT fabricate APIs/return values/strings.{strict_line}

[Hard Constraints]
- No placeholders: xxxx, Lx, Ly, etc.
- Citations must be exactly: (chunk_id=16-hex) and chunk_id must come from the whitelist.
- In [Key Conditions], snippets must be copied verbatim from evidence (no paraphrase).
- [Evidence References] must be listed as "- file_path:Lx-Ly (chunk_id=...)"
- In [Reasoning Trace], each step must end with a valid (chunk_id=...) from the whitelist.

[Question]
{question}

[Evidence]
{evidence_text}

[Whitelist]
{whitelist_text}

[Answer Template - do not change headings]

[Conclusion]
(1-2 sentences summarizing how it works, grounded in the snippets below)

[Key Conditions]
{chr(10).join(kb)}

[Evidence References]
{chr(10).join(cite_lines)}

[Reasoning Trace]
1. Locate the stock lock/reserve entry or SQL update and identify target table/fields.(chunk_id={step_cids[0]})
2. Explain key conditions(AND/if) that ensure concurrency safety and prevent oversell.(chunk_id={step_cids[1]})
3. Describe failure branches, rollback/release conditions, and returned status.(chunk_id={step_cids[2]})

### Response
"""
    return f"""### Instruction
你是代码仓分析助手，请严格基于【可用代码证据】回答，禁止编造未出现的API/返回值/字符串。{strict_line}

【回答强约束(必须遵守，否则判定失败)】
- 禁止输出占位符：xxxx、Lx、Ly等都不允许出现
- 引用格式必须严格为：(chunk_id=16位十六进制)且chunk_id必须来自【允许引用的证据清单】
- 【关键判断与处理分支】里的“原句”必须逐字来自【可用代码证据】，不得改写
- 【证据引用】必须逐条写成“- file_path:Lx-Ly (chunk_id=...)”，file_path与行号必须来自证据
- 【推理过程】每一步末尾必须包含一个合法的(chunk_id=...)，且chunk_id来自白名单

【问题】
{question}

【可用代码证据】
{evidence_text}

【允许引用的证据清单(只能从这里选)】
{whitelist_text}

【请严格按下面模板作答，不得增删标题】

【结论】
(用1-2句总结如何实现，务必引用到你下面原句对应的逻辑)

【关键判断与处理分支】
{chr(10).join(kb)}

【证据引用】
{chr(10).join(cite_lines)}

【推理过程】
1. 先定位库存锁定入口或SQL更新语句，明确锁定作用对象与更新字段。(chunk_id={step_cids[0]})
2. 再解释关键条件(如AND条件/if判断)如何保证并发安全与避免超卖。(chunk_id={step_cids[1]})
3. 最后说明失败分支/返回值/回滚释放库存锁的触发条件。(chunk_id={step_cids[2]})

### Response
"""


def _expand_query(question: str) -> str:
    extra = " lockStock hasStock lockStockBySkuId releaseStock reduceStock pms_sku_stock lock_stock mapper.xml UPDATE AND if("
    return question + extra


def _generate_once(model, tokenizer, prompt: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "640")),
            do_sample=False,
            repetition_penalty=float(os.environ.get("REP_PENALTY", "1.05")),
        )

    gen_ids = out[0][prompt_len:]
    output = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return output


# 只展示【新增 / 修改】的关键部分，其余你文件保持不变
# ↓↓↓ 请把下面函数插入到文件中（在 _generate_once 后）↓↓↓

def rewrite_json_to_template(model, tokenizer, bad_output: str, prompt: str, lang: str) -> str:
    if lang == "zh":
        extra = """
【改写任务】
你刚才的输出是JSON/类JSON。
请严格改写为【纯文本模板】，必须包含以下标题：
【结论】【关键判断与处理分支】【证据引用】【推理过程】
禁止输出JSON，禁止新增标题。
【证据引用】只能使用白名单里的文件与chunk_id。
【推理过程】每一步必须以(chunk_id=16位十六进制)结尾。
只输出改写后的正文。
"""
    else:
        extra = """
[REWRITE TASK]
Rewrite the previous JSON-like output into the required plain-text template.
DO NOT output JSON.
Use only whitelist references.
"""

    rewrite_prompt = (
        prompt
        + "\n\n"
        + extra
        + "\n\n[JSON OUTPUT]\n"
        + bad_output
        + "\n\n### Response\n"
    )
    return _generate_once(model, tokenizer, rewrite_prompt)


def load_question() -> str:
    q = os.environ.get("QUESTION", "").strip()
    qf = os.environ.get("QUESTION_FILE", "").strip()
    if q:
        return q
    if qf:
        p = Path(qf)
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return "库存锁定规则在代码中是如何实现的？请结合代码说明关键判断与处理分支。"


# -----------------------------
# Main
# -----------------------------
def main():
    base_model = os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    lora_dir = os.environ.get("LORA_DIR", "outputs/lora_qwen25_demo")
    repo_index_path = os.environ.get("REPO_INDEX", "data/raw_index/repo_index.jsonl")

    infer_lang = os.environ.get("INFER_LANG", "zh").strip().lower()
    if infer_lang not in ("zh", "en"):
        infer_lang = "zh"

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
    print("peft_loaded=", hasattr(model, "peft_config"), "lora_dir=", lora_dir, "INFER_LANG=", infer_lang)

    question = load_question()
    topk = int(os.environ.get("TOPK", "6"))
    rows = load_repo_index(repo_index_path)

    attempts = [
        ("normal", question, topk),
        ("expanded", _expand_query(question), max(topk * 2, 12)),
    ]

    print("=== QUESTION ===")
    print(question)

    last_chunks: List[Dict[str, Any]] = []

    for name, q, k in attempts:
        top_chunks = retrieve_topk_chunks(rows, q, topk=k)
        last_chunks = top_chunks
        evidence_text = format_evidence_text(top_chunks)
        whitelist_text = format_whitelist(top_chunks)

        # 每轮两次：先非strict，失败再strict强约束重试一次
        for strict in (False, True):
            prompt = _build_prompt(question, evidence_text, whitelist_text, top_chunks, infer_lang, strict=strict)

            print(f"\n=== ATTEMPT {name} / topk={k} / strict={int(strict)} ===")
            output = _generate_once(model, tokenizer, prompt)
            output = repair_output(output, top_chunks, infer_lang)

            if is_json_like(output):
                output = rewrite_json_to_template(model, tokenizer, output, prompt, infer_lang)
                output = repair_output(output, top_chunks, infer_lang)

            print("=== MODEL OUTPUT ===")
            print(output)

            okf, msgf = hard_check_format(output, infer_lang)
            if not okf:
                print("\n=== FORMAT CHECK FAILED ===")
                print(msgf)
                continue

            ok1, msg1 = hard_check_output(output, top_chunks, infer_lang)
            if not ok1:
                print("\n=== HARD CHECK FAILED ===")
                print(msg1)
                continue

            ok2, msg2 = hard_check_content(output, infer_lang)
            if not ok2:
                print("\n=== CONTENT CHECK FAILED ===")
                print(msg2)
                continue

            print("\n=== PASSED ===")
            print("输出满足：模板格式 + 白名单引用 + 关键内容要素。")
            return

    print("\n=== FINAL FAILED ===")
    print("两轮检索+strict重试仍未通过。建议：提高TOPK或把问题改成更具体的函数/SQL关键词。")
    if last_chunks:
        print("最后一轮白名单如下：")
        print(format_whitelist(last_chunks))


if __name__ == "__main__":
    main()
