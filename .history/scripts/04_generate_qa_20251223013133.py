#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step04
根据 rules + flows + repo_index 自动生成 QA 样本

-生成的每条QA样本必须包含可审计的evidence(代码段)与trace(推理步骤)
-answer是基于evidence.content抽取关键判断/分支要点(无需LLM)
-flow类问题输出“步骤+模块/代码位置+证据chunk”，满足“指出每一步对应代码位置”
"""

import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import os
import re





def load_templates():
    import json
    p = ROOT / "configs/nlg_templates.json"
    return json.loads(p.read_text(encoding="utf-8"))


def normalize_lang(lang: str) -> str:
    """
    将系统locale/环境变量映射为pipeline内部语言标识: zh/en/bilingual
    例: en_US.UTF-8 -> en, zh_CN.UTF-8 -> zh
    """
    if not lang:
        return "zh"
    l = lang.strip().lower()

    if l in ("bilingual", "bi", "both", "zh+en", "en+zh"):
        return "bilingual"
    if l.startswith("zh"):
        return "zh"
    if l.startswith("en"):
        return "en"
    return "zh"

def resolve_language(default_lang: str = "zh") -> str:
    """
    语言优先级：
    1. 环境变量 LANG（用于临时覆盖）
    2. configs/runtime.yaml
    3. 默认值
    """
    # 1) 环境变量（最高优先级）
    env_lang = os.environ.get("LANG")
    
    # 只有当LANG是“显式业务值”才覆盖配置
    if env_lang:
        norm = normalize_lang(env_lang)
        if env_lang.strip().lower() in ("zh", "en", "bilingual", "bi", "both", "zh+en", "en+zh"):
            return norm
        # 如果是en_US.UTF-8/zh_CN.UTF-8这类locale，不让它压过runtime.yaml
        # 继续往下读配置文件

    # 2) runtime.yaml
    cfg_path = ROOT / "configs/runtime.yaml"
    if cfg_path.exists():
        import yaml  # 这里不再吞异常，读不到就让它报错，方便你定位环境问题
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        lang = (cfg.get("language") or {}).get("mode")
        if lang:
            return normalize_lang(lang)

    # 3) 兜底
    return normalize_lang(default_lang)


ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def make_id(prefix: str, seed: str) -> str:
    return prefix + "_" + hashlib.sha1(seed.encode()).hexdigest()[:10]


def build_evidence(chunk_id: str, index_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    r = index_map[chunk_id]
    return {
        "chunk_id": chunk_id,
        "file_path": r["file_path"],
        "start_line": r["start_line"],
        "end_line": r["end_line"],
        "content": r["content"]
    }


# -----------------------------
# Answer enhancement helpers
# -----------------------------

_KEY_PATTERNS = [
    r"@Transactional",
    r"\bif\b", r"\belse\b", r"\bswitch\b", r"\bcase\b",
    r"\breturn\b", r"\bthrow\b",
    r"status", r"state",
    r"lock", r"reserve", r"deduct", r"reduce", r"release", r"unlock",
    r"timeout", r"cancel", r"refund", r"rollback",
]
_key_re = re.compile("|".join(_KEY_PATTERNS), re.IGNORECASE)


def extract_key_lines(code: str, max_lines: int = 5, max_len: int = 160) -> List[str]:
    if not code:
        return []
    out: List[str] = []
    for raw in code.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _key_re.search(line):
            if len(line) > max_len:
                line = line[: max_len - 3] + "..."
            if line not in out:
                out.append(line)
        if len(out) >= max_lines:
            break
    return out


def build_rule_answer(rule: Dict[str, Any], ev: Dict[str, Any], templates: Dict[str, Any], lang: str) -> str:
    t = templates["qa"][lang]
    base = t["rule_answer_prefix"].format(description=rule["description"])
    key_lines = extract_key_lines(ev.get("content") or "", max_lines=5)

    if not key_lines:
        return base

    if lang == "zh":
        bullets = "\n".join([f"- {x}" for x in key_lines])
        loc = f"{ev['file_path']}:L{ev['start_line']}-L{ev['end_line']}"
        return base + f"\n\n关键判断/处理分支(摘自{loc}):\n{bullets}"
    else:
        bullets = "\n".join([f"- {x}" for x in key_lines])
        loc = f"{ev['file_path']}:L{ev['start_line']}-{ev['end_line']}"
        return base + f"\n\nKey code points (from {loc}):\n{bullets}"


def build_flow_answer(flow: Dict[str, Any], evs: List[Dict[str, Any]], templates: Dict[str, Any], lang: str) -> str:
    t = templates["qa"][lang]
    header = t["flow_answer"]
    steps = flow.get("steps") or []
    if not steps:
        return header

    lines: List[str] = []
    for i, s in enumerate(steps, 1):
        step_name = s.get("name") or s.get("op") or f"step{i}"
        cid = s.get("evidence_chunk")
        ev = None
        if cid:
            for e in evs:
                if e.get("chunk_id") == cid:
                    ev = e
                    break
        if ev:
            loc = f"{ev['file_path']}:L{ev['start_line']}-L{ev['end_line']}"
            lines.append(f"{i}. {step_name} -> {loc} (chunk={cid})")
        else:
            lines.append(f"{i}. {step_name}")

    if lang == "zh":
        return header + "\n\n流程步骤与代码位置:\n" + "\n".join(lines)
    else:
        return header + "\n\nSteps and code locations:\n" + "\n".join(lines)


def generate_rule_qa(rule: Dict[str, Any], index_map: Dict[str, Dict[str, Any]], templates: Dict[str, Any], lang: str) -> Dict[str, Any]:
    cid = rule["evidence_chunks"][0]
    ev = build_evidence(cid, index_map)

    t = templates["qa"][lang]
    q = t["rule_question"].format(title=rule["title"], file_path=ev["file_path"])
    a = t["rule_answer_prefix"].format(description=rule["description"])

    return {
        "sample_id": make_id("qa", f"{rule['rule_id']}|{lang}|{cid}"),
        "task_type": "qa",
        "language": lang,
        "question": q,
        "answer": a,
        "evidence": [ev],
        "trace": {
            "type": "rule_based",
            "rule_ids": [rule["rule_id"]],
            "reasoning_steps": [
                f"定位到规则: {rule['title']}" if lang == "zh" else f"Locate rule: {rule['title']}",
                f"检查关联代码块: {cid}" if lang == "zh" else f"Inspect evidence chunk: {cid}",
                "根据代码逻辑总结规则实现方式" if lang == "zh" else "Summarize the implementation based on the code."
            ]
        },
        "meta": {
            "domain": rule["domain"],
            "qa_type": "rule",
            "difficulty": "medium",
            "generator": "template_v2_multilingual",
            "source": "AutoCodeDataPipeline"
        }
    }


def generate_flow_qa(flow: Dict[str, Any], index_map: Dict[str, Dict[str, Any]], templates: Dict[str, Any], lang: str) -> Dict[str, Any]:
    evs = []
    for s in flow["steps"]:
        cid = s.get("evidence_chunk")
        if cid and cid in index_map:
            evs.append(build_evidence(cid, index_map))

    t = templates["qa"][lang]
    q = t["flow_question"].format(flow_name=flow["name"])
    a = t["flow_answer"]

    return {
        "sample_id": make_id("qa", f"{flow['flow_id']}|{lang}"),
        "task_type": "qa",
        "language": lang,
        "question": q,
        "answer": a,
        "evidence": evs[:2] if evs else [],
        "trace": {
            "type": "flow_based",
            "flow_id": flow["flow_id"],
            "reasoning_steps": [
                "识别流程起点" if lang == "zh" else "Identify the flow entry point.",
                "分析关键业务步骤" if lang == "zh" else "Analyze key business steps.",
                "结合代码顺序总结整体流程" if lang == "zh" else "Summarize the end-to-end flow with code order."
            ]
        },
        "meta": {
            "domain": flow["domain"],
            "qa_type": "flow",
            "difficulty": "hard",
            "generator": "template_v2_multilingual",
            "source": "AutoCodeDataPipeline"
        }
    }



def main():
    index_rows = read_jsonl(ROOT / "data/raw_index/repo_index.jsonl")
    rules = read_jsonl(ROOT / "data/extracted/rules.jsonl")
    flows = read_jsonl(ROOT / "data/extracted/flows.jsonl")

    index_map = {r["chunk_id"]: r for r in index_rows}

    # 增加多语言
    templates = load_templates()
    raw_lang = os.environ.get("LANG", "zh")  # Windows上可能是 en_US.UTF-8
    lang_mode = resolve_language(default_lang="zh")
    langs = ["zh", "en"] if lang_mode == "bilingual" else [lang_mode]
    # 防御性校验，提前报错更好定位
    available = list(templates.get("qa", {}).keys())
    for l in langs:
        if l not in templates.get("qa", {}):
            raise ValueError(f"Unsupported language '{l}' (from LANG={raw_lang}). Available: {available}")

    print("[DEBUG] env LANG =", os.environ.get("LANG"))
    print("[DEBUG] runtime.yaml exists =", (ROOT / "configs/runtime.yaml").exists())
    print("[DEBUG] resolved lang_mode =", lang_mode, "langs =", langs)

    samples = []
    for lang in langs:
        for r in rules[:60]:
            samples.append(generate_rule_qa(r, index_map, templates, lang))
        for f in flows:
            samples.append(generate_flow_qa(f, index_map, templates, lang))


    '''samples = []

    for r in rules[:60]:
        samples.append(generate_rule_qa(r, index_map))

    for f in flows:
        samples.append(generate_flow_qa(f, index_map))'''

    random.shuffle(samples)

    # 划分数据集
    n = len(samples)
    train = samples[: int(n * 0.8)]
    dev = samples[int(n * 0.8): int(n * 0.9)]
    test = samples[int(n * 0.9):]

    write_jsonl(ROOT / "data/dataset/train.jsonl", train)
    write_jsonl(ROOT / "data/dataset/dev.jsonl", dev)
    write_jsonl(ROOT / "data/dataset/test.jsonl", test)
    write_jsonl(ROOT / "data/samples/qa_samples.jsonl", samples[:20])

    print(f"QA生成完成: total={n}, train={len(train)}, dev={len(dev)}, test={len(test)}")


if __name__ == "__main__":
    main()
