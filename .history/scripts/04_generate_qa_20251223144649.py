#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step04
根据 rules + flows + repo_index 自动生成 QA 样本

目标(按你的要求)：
- 每lang规则取前N条(默认200)
- flow全量生成
- 不做比例裁剪(不再用flow_ratio砍rule)
- 每条都带evidence_snippets原文代码段(可截断) + trace
- answer里也包含【原文代码段】代码块，满足“提供原文代码段”审阅习惯
"""

import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import os
import re

ROOT = Path(__file__).resolve().parents[1]

# -----------------------------
# Language hygiene helpers
# -----------------------------
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

def has_cjk(s: str) -> bool:
    return bool(s and _CJK_RE.search(s))

def en_fallback(text: str, fallback: str) -> str:
    # en模式下如果包含汉字则用fallback替换
    return fallback if has_cjk(text) else text



def load_templates():
    p = ROOT / "configs/nlg_templates.json"
    return json.loads(p.read_text(encoding="utf-8"))


def normalize_lang(lang: str) -> str:
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
    # 1) 环境变量LANG(仅当为显式业务值)
    env_lang = os.environ.get("LANG")
    if env_lang:
        norm = normalize_lang(env_lang)
        if env_lang.strip().lower() in ("zh", "en", "bilingual", "bi", "both", "zh+en", "en+zh"):
            return norm

    # 2) runtime.yaml
    cfg_path = ROOT / "configs/runtime.yaml"
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        lang = (cfg.get("language") or {}).get("mode")
        if lang:
            return normalize_lang(lang)

    return normalize_lang(default_lang)


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
    return prefix + "_" + hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:10]


def detect_code_lang(file_path: str) -> str:
    fp = (file_path or "").lower()
    if fp.endswith(".java"):
        return "java"
    if fp.endswith(".xml"):
        return "xml"
    if fp.endswith(".yml") or fp.endswith(".yaml"):
        return "yaml"
    if fp.endswith(".sql"):
        return "sql"
    if fp.endswith(".js"):
        return "javascript"
    if fp.endswith(".ts"):
        return "typescript"
    if fp.endswith(".py"):
        return "python"
    return ""


def trim_code(code: str, max_chars: int) -> str:
    if not code:
        return ""
    if len(code) <= max_chars:
        return code
    return code[: max_chars - 20] + "\n/* ... truncated ... */\n"


def build_evidence(chunk_id: str, index_map: Dict[str, Dict[str, Any]], code_max_chars: int) -> Dict[str, Any]:
    r = index_map[chunk_id]
    return {
        "chunk_id": chunk_id,
        "file_path": r["file_path"],
        "start_line": r["start_line"],
        "end_line": r["end_line"],
        "content": r.get("content") or "",
        "code_lang": detect_code_lang(r["file_path"]),
        "code": trim_code(r.get("content") or "", code_max_chars),
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


def extract_key_lines(code: str, max_lines: int = 6, max_len: int = 180) -> List[str]:
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


def format_code_block(ev: Dict[str, Any]) -> str:
    code = ev.get("code") or ""
    if not code:
        return ""
    lang = ev.get("code_lang") or ""
    return f"```{lang}\n{code}\n```"


def wrap_answer(answer_core: str, evidences: List[Dict[str, Any]], trace_steps: List[str], lang: str) -> str:
    is_zh = (lang == "zh")

    ev_lines = []
    ev_blocks = []
    for ev in (evidences or [])[:3]:
        loc = f"{ev['file_path']}:L{ev['start_line']}-{ev['end_line']}"
        ev_lines.append(f"- {loc} (chunk_id={ev['chunk_id']})")
        ev_blocks.append(format_code_block(ev))

    ev_ref = "\n".join(ev_lines) if ev_lines else "- N/A"
    ev_code = "\n\n".join([b for b in ev_blocks if b]) if ev_blocks else ""

    steps = (trace_steps or [])[:6]
    if not steps:
        steps = [
            "定位与问题相关的实现代码" if is_zh else "Locate relevant implementation",
            "抽取关键条件判断/处理分支" if is_zh else "Extract key branches/conditions",
            "基于证据组织结论" if is_zh else "Synthesize conclusion grounded in evidence",
        ]
    trace_block = "\n".join([f"{i+1}. {st}" for i, st in enumerate(steps)])

    if is_zh:
        return (
            f"【结论】\n{answer_core.strip()}\n\n"
            f"【证据引用】\n{ev_ref}\n\n"
            f"【原文代码段】\n{ev_code if ev_code else 'N/A'}\n\n"
            f"【推理过程】\n{trace_block}"
        )
    else:
        return (
            f"[Conclusion]\n{answer_core.strip()}\n\n"
            f"[Evidence]\n{ev_ref}\n\n"
            f"[Code Snippet]\n{ev_code if ev_code else 'N/A'}\n\n"
            f"[Trace]\n{trace_block}"
        )


def build_rule_answer(rule: Dict[str, Any], ev: Dict[str, Any], templates: Dict[str, Any], lang: str, description: str = "") -> str:
    t = templates["qa"][lang]
    desc = description if description else (rule.get("description", "") or "")
    base = t["rule_answer_prefix"].format(description=desc)

    key_lines = extract_key_lines(ev.get("content") or "", max_lines=6)
    if not key_lines:
        return base

    bullets = "\n".join([f"- {x}" for x in key_lines])
    loc = f"{ev['file_path']}:L{ev['start_line']}-{ev['end_line']}"
    if lang == "zh":
        return base + f"\n\n关键判断/处理分支(摘自{loc}):\n{bullets}"
    else:
        return base + f"\n\nKey code points (from {loc}):\n{bullets}"


def build_flow_answer(flow: Dict[str, Any], evs: List[Dict[str, Any]], templates: Dict[str, Any], lang: str) -> str:
    t = templates["qa"][lang]
    header = str(t["flow_answer"])

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
            loc = f"{ev['file_path']}:L{ev['start_line']}-{ev['end_line']}"
            lines.append(f"{i}. {step_name} -> {loc} (chunk={cid})")
        else:
            lines.append(f"{i}. {step_name}")

    if lang == "zh":
        return header + "\n\n流程步骤与代码位置:\n" + "\n".join(lines)
    else:
        return header + "\n\nSteps and code locations:\n" + "\n".join(lines)


def generate_rule_qa(rule: Dict[str, Any], index_map: Dict[str, Dict[str, Any]], templates: Dict[str, Any],
                     lang: str, code_max_chars: int) -> Dict[str, Any]:
    # 保持“来多少生成多少”的风格：若证据缺失就跳过(避免报错)
    cid = None
    ev_chunks = rule.get("evidence_chunks") or []
    if ev_chunks:
        cid = ev_chunks[0]
    if (not cid) or (cid not in index_map):
        return {}

    ev = build_evidence(cid, index_map, code_max_chars)

    t = templates["qa"][lang]
    q = t["rule_question"].format(title=rule.get("title", ""), file_path=ev["file_path"])

    core = build_rule_answer(rule, ev, templates, lang)
    a = wrap_answer(core, [ev], [
        (f"定位到规则:{rule.get('title','')}" if lang == "zh" else f"Locate rule:{rule.get('title','')}"),
        ("抽取关键判断/分支并与业务描述对齐" if lang == "zh" else "Align key branches/conditions with business description"),
        ("输出结论并给出证据引用与原文代码段" if lang == "zh" else "Output conclusion with evidence and code snippet"),
    ], lang)

    # 统一训练字段text/meta(你后面Step08好合并)
    text = f"### Instruction\n{q}\n\n### Response\n{a}\n"
    meta_v2 = {
        "task_type": "qa",
        "language": lang,
        "domain": rule.get("domain"),
        "qa_type": "rule",
        "difficulty": "medium",
        "evidence": [{
            "chunk_id": ev["chunk_id"],
            "file_path": ev["file_path"],
            "start_line": ev["start_line"],
            "end_line": ev["end_line"],
        }],
        "evidence_snippets": [{
            "chunk_id": ev["chunk_id"],
            "file_path": ev["file_path"],
            "start_line": ev["start_line"],
            "end_line": ev["end_line"],
            "code_lang": ev.get("code_lang"),
            "code": ev.get("code"),
        }],
        "trace_digest": [
            (f"定位到规则:{rule.get('title','')}" if lang == "zh" else f"Locate rule:{rule.get('title','')}"),
            ("抽取关键判断/分支并与业务描述对齐" if lang == "zh" else "Align key branches/conditions with business description"),
            ("给出证据引用与原文代码段" if lang == "zh" else "Provide evidence references and code snippet"),
        ],
        "generator": "step04_topN_no_ratio_v1",
        "source": "AutoCodeDataPipeline",
    }

    return {
        "sample_id": make_id("qa", f"{rule.get('rule_id','')}|{lang}|{cid}"),
        "task_type": "qa",
        "language": lang,
        "question": q,
        "answer": a,
        "evidence": [{
            "chunk_id": ev["chunk_id"],
            "file_path": ev["file_path"],
            "start_line": ev["start_line"],
            "end_line": ev["end_line"],
            "content": ev.get("content") or "",
        }],
        "trace": {
            "type": "rule_based",
            "rule_ids": [rule.get("rule_id")],
            "reasoning_steps": meta_v2["trace_digest"],
        },
        "meta": {
            "domain": rule.get("domain"),
            "qa_type": "rule",
            "difficulty": "medium",
            "generator": "step04_topN_no_ratio_v1",
            "source": "AutoCodeDataPipeline",
        },
        "text": text,
        "meta_v2": meta_v2,
    }


def generate_flow_qa(flow: Dict[str, Any], index_map: Dict[str, Dict[str, Any]], templates: Dict[str, Any],
                     lang: str, code_max_chars: int, flow_max_evs: int) -> Dict[str, Any]:
    evs: List[Dict[str, Any]] = []
    for s in flow.get("steps", []) or []:
        cid = s.get("evidence_chunk")
        if cid and cid in index_map:
            evs.append(build_evidence(cid, index_map, code_max_chars))
        if len(evs) >= flow_max_evs:
            break

    t = templates["qa"][lang]
    q = t["flow_question"].format(flow_name=flow.get("name", ""))

    core = build_flow_answer(flow, evs, templates, lang)
    a = wrap_answer(core, evs, [
        (f"识别流程:{flow.get('name','')}" if lang == "zh" else f"Identify flow:{flow.get('name','')}"),
        ("按调用/职责边界抽取步骤并关联代码位置" if lang == "zh" else "Extract steps and link to code locations"),
        ("输出端到端流程并给出证据与原文代码段" if lang == "zh" else "Summarize end-to-end flow with evidence and code snippet"),
    ], lang)

    text = f"### Instruction\n{q}\n\n### Response\n{a}\n"
    meta_v2 = {
        "task_type": "qa",
        "language": lang,
        "domain": flow.get("domain"),
        "qa_type": "flow",
        "difficulty": "hard",
        "evidence": [{
            "chunk_id": e["chunk_id"],
            "file_path": e["file_path"],
            "start_line": e["start_line"],
            "end_line": e["end_line"],
        } for e in evs],
        "evidence_snippets": [{
            "chunk_id": e["chunk_id"],
            "file_path": e["file_path"],
            "start_line": e["start_line"],
            "end_line": e["end_line"],
            "code_lang": e.get("code_lang"),
            "code": e.get("code"),
        } for e in evs],
        "trace_digest": [
            (f"识别流程:{flow.get('name','')}" if lang == "zh" else f"Identify flow:{flow.get('name','')}"),
            ("抽取步骤并标注代码位置" if lang == "zh" else "Extract steps and annotate code locations"),
            ("汇总为端到端流程并附证据" if lang == "zh" else "Summarize end-to-end flow with evidence"),
        ],
        "generator": "step04_topN_no_ratio_v1",
        "source": "AutoCodeDataPipeline",
    }

    return {
        "sample_id": make_id("qa", f"{flow.get('flow_id','')}|{lang}"),
        "task_type": "qa",
        "language": lang,
        "question": q,
        "answer": a,
        "evidence": [{
            "chunk_id": e["chunk_id"],
            "file_path": e["file_path"],
            "start_line": e["start_line"],
            "end_line": e["end_line"],
            "content": e.get("content") or "",
        } for e in evs],
        "trace": {
            "type": "flow_based",
            "flow_id": flow.get("flow_id"),
            "reasoning_steps": meta_v2["trace_digest"],
        },
        "meta": {
            "domain": flow.get("domain"),
            "qa_type": "flow",
            "difficulty": "hard",
            "generator": "step04_topN_no_ratio_v1",
            "source": "AutoCodeDataPipeline",
        },
        "text": text,
        "meta_v2": meta_v2,
    }


def main():
    index_rows = read_jsonl(ROOT / "data/raw_index/repo_index.jsonl")
    rules = read_jsonl(ROOT / "data/extracted/rules.jsonl")
    flows = read_jsonl(ROOT / "data/extracted/flows.jsonl")
    index_map = {r["chunk_id"]: r for r in index_rows}

    templates = load_templates()
    lang_mode = resolve_language(default_lang="zh")
    langs = ["zh", "en"] if lang_mode == "bilingual" else [lang_mode]

    available = list(templates.get("qa", {}).keys())
    for l in langs:
        if l not in templates.get("qa", {}):
            raise ValueError(f"Unsupported language '{l}'. Available:{available}")

    # 关键参数：每lang规则取前N条，flow全量
    rule_topn = int(os.environ.get("QA_RULE_TOPN", "200"))
    flow_max_evs = int(os.environ.get("QA_FLOW_MAX_EVS", "3"))
    code_max_chars = int(os.environ.get("QA_CODE_MAX_CHARS", "2200"))

    samples: List[Dict[str, Any]] = []
    for lang in langs:
        # 规则：前N条
        for r in rules[:rule_topn]:
            s = generate_rule_qa(r, index_map, templates, lang, code_max_chars)
            if s:
                samples.append(s)

        # flow：全量
        for f in flows:
            s = generate_flow_qa(f, index_map, templates, lang, code_max_chars, flow_max_evs)
            if s:
                samples.append(s)

    random.shuffle(samples)

    n = len(samples)
    train = samples[: int(n * 0.8)]
    dev = samples[int(n * 0.8): int(n * 0.9)]
    test = samples[int(n * 0.9):]

    write_jsonl(ROOT / "data/dataset/train.jsonl", train)
    write_jsonl(ROOT / "data/dataset/dev.jsonl", dev)
    write_jsonl(ROOT / "data/dataset/test.jsonl", test)
    write_jsonl(ROOT / "data/samples/qa_samples.jsonl", samples[:30])

    print(f"QA生成完成: total={n}, train={len(train)}, dev={len(dev)}, test={len(test)}")
    print(f"[Knobs] QA_RULE_TOPN={rule_topn}, QA_FLOW_MAX_EVS={flow_max_evs}, QA_CODE_MAX_CHARS={code_max_chars}, lang_mode={lang_mode}, langs={langs}")


if __name__ == "__main__":
    main()
