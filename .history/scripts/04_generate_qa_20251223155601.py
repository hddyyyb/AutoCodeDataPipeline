#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step04
根据rules+flows+repo_index自动生成QA样本

对齐Step03增强点:
- 英文优先用Step03产出的title_en/description_en/name_en/step.name_en
- 缺失英文时才fallback到证据推断(infer_en_topic_from_evidence)
- rule_question/flow_question支持字符串或列表(多样性)
- 排除低价值/模板文件(可配置)
- 同一rule_id限流+去重(避免刷屏)

保持目标:
- 每lang规则取前N条(默认200)
- flow全量生成
- 每条带evidence_snippets+trace
- answer包含[Code Snippet]/【原文代码段】
"""

import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import os
import re

ROOT = Path(__file__).resolve().parents[1]

# -----------------------------
# Language helpers
# -----------------------------
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def has_cjk(s: str) -> bool:
    return bool(s and _CJK_RE.search(s))


def get_en_strict_level() -> int:
    """
    QA_STRICT_EN:
      0=off(不清洗)
      1=fallback(默认:遇中文用fallback替换,不丢样本)
      2=drop(严格:遇中文直接跳过英文样本)
    """
    v = os.environ.get("QA_STRICT_EN", "1").strip()
    try:
        lv = int(v)
    except Exception:
        lv = 1
    return 0 if lv < 0 else 2 if lv > 2 else lv


def en_pick(text_raw: str, text_en: str, fallback: str, strict_lv: int) -> str:
    if text_en and (not has_cjk(text_en)):
        return text_en
    if text_raw and (not has_cjk(text_raw)):
        return text_raw
    if strict_lv >= 2:
        return ""
    return fallback


# -----------------------------
# IO
# -----------------------------
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
    env_lang = os.environ.get("LANG")
    if env_lang:
        norm = normalize_lang(env_lang)
        if env_lang.strip().lower() in ("zh", "en", "bilingual", "bi", "both", "zh+en", "en+zh"):
            return norm

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
    if not path.exists():
        return out
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


# -----------------------------
# Evidence
# -----------------------------
def detect_code_lang(file_path: str) -> str:
    fp = (file_path or "").lower()
    if fp.endswith(".java"):
        return "java"
    if fp.endswith(".xml"):
        return "xml"
    if fp.endswith(".yml") or fp.endswith(".yaml"):
        return "yaml"
    if fp.endswith(".properties"):
        return "properties"
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


def format_code_block(ev: Dict[str, Any]) -> str:
    code = ev.get("code") or ""
    if not code:
        return ""
    lang = ev.get("code_lang") or ""
    return f"```{lang}\n{code}\n```"


# -----------------------------
# Key line extraction
# -----------------------------
_KEY_PATTERNS = [
    r"@Transactional",
    r"\bif\b", r"\belse\b", r"\bswitch\b", r"\bcase\b",
    r"\breturn\b", r"\bthrow\b",
    r"status", r"state",
    r"lock", r"reserve", r"deduct", r"reduce", r"release", r"unlock",
    r"timeout", r"cancel", r"refund", r"rollback",
    r"@RequestMapping", r"@GetMapping", r"@PostMapping",
    r"CommonResult\.", r"success\(", r"failed\(",
]
_key_re = re.compile("|".join(_KEY_PATTERNS), re.IGNORECASE)


def extract_key_lines(code: str, max_lines: int = 8, max_len: int = 200) -> List[str]:
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


# -----------------------------
# Exclude & diversity
# -----------------------------
def compile_exclude_regex() -> re.Pattern:
    """
    默认排除:
    - mall-mbg(自动生成)
    - /model/目录(大量POJO/模板)
    - *Example.java(MyBatis Generator模板类)
    可用QA_EXCLUDE_PATHS覆盖
    """
    pat = os.environ.get(
        "QA_EXCLUDE_PATHS",
        r"(?:^|/)(mall-mbg)(?:/|$)|(?:^|/)(model)(?:/|$)|Example\.java$"
    ).strip()
    return re.compile(pat, re.IGNORECASE)


def is_excluded_file(file_path: str, ex_re: re.Pattern) -> bool:
    fp = (file_path or "").replace("\\", "/")
    return bool(ex_re.search(fp))


def choose_rule_question(templates_qa_lang: Dict[str, Any], title: str, file_path: str, lang: str) -> str:
    rq = templates_qa_lang.get("rule_question")
    if isinstance(rq, list) and rq:
        fmt = random.choice(rq)
    else:
        fmt = str(rq)
    return fmt.format(title=title, file_path=file_path)


def choose_flow_question(templates_qa_lang: Dict[str, Any], flow_name: str, lang: str) -> str:
    fq = templates_qa_lang.get("flow_question")
    if isinstance(fq, list) and fq:
        fmt = random.choice(fq)
    else:
        fmt = str(fq)
    return fmt.format(flow_name=flow_name)


# -----------------------------
# EN inference fallback
# -----------------------------
_METHOD_RE = re.compile(r"\b(public|private|protected)\s+[\w<>\[\]]+\s+(\w+)\s*\(", re.IGNORECASE)


def infer_en_topic_from_evidence(ev: Dict[str, Any]) -> Tuple[str, str]:
    fp = (ev.get("file_path") or "").replace("\\", "/")
    code = ev.get("content") or ""
    key_lines = extract_key_lines(code, max_lines=12)

    ext = (ev.get("code_lang") or "").lower()
    if ext in ("yaml", "properties"):
        k = ""
        for ln in key_lines:
            m = re.search(r"^\s*([A-Za-z0-9_.-]+)\s*[:=]\s*(.+)$", ln)
            if m:
                k = m.group(1)
                break
        title = "Application configuration"
        desc = "Explain what this configuration controls and what the key settings mean."
        if k:
            title = f"Configuration for {k}"
            desc = f"Explain the purpose of '{k}' and how its value affects runtime behavior."
        return title, desc

    role = "implementation"
    if "/controller/" in fp or fp.endswith("Controller.java"):
        role = "API endpoint behavior"
    elif "/service/" in fp:
        role = "service logic"
    elif "/mapper/" in fp or "Mapper.java" in fp:
        role = "data access(mapper) behavior"
    elif "/dao/" in fp:
        role = "DAO contract"

    method = ""
    for ln in code.splitlines():
        mm = _METHOD_RE.search(ln)
        if mm:
            method = mm.group(2)
            break
    if not method:
        for ln in key_lines:
            mm = re.search(r"\b(\w+)\s*\(", ln)
            if mm:
                method = mm.group(1)
                break

    tags = []
    blob = "\n".join(key_lines).lower()
    if "failed(" in blob or "return 0" in blob or "throw" in blob:
        tags.append("failure handling")
    if "status" in blob:
        tags.append("status update")
    if "cancel" in blob:
        tags.append("cancellation")
    if "lock" in blob or "reserve" in blob:
        tags.append("stock locking")
    if "reduce" in blob or "deduct" in blob:
        tags.append("stock deduction")
    if "transactional" in blob:
        tags.append("transaction boundary")

    tag_str = ", ".join(tags) if tags else "key branches and return paths"
    base_name = fp.split("/")[-1]
    if method:
        title = f"{base_name}:{method} {role}"
        desc = f"Based on the code, explain the {role} of '{method}', focusing on {tag_str}."
    else:
        title = f"{base_name} {role}"
        desc = f"Based on the code, explain the {role}, focusing on {tag_str}."
    return title, desc


# -----------------------------
# Answer wrappers
# -----------------------------
def wrap_answer(answer_core: str, evidences: List[Dict[str, Any]], trace_steps: List[str], lang: str) -> str:
    is_zh = (lang == "zh")

    ev_lines = []
    ev_blocks = []
    for ev in (evidences or [])[:3]:
        loc = f"{ev['file_path']}:L{ev['start_line']}-{ev['end_line']}"
        ev_lines.append(f"- {loc}(chunk_id={ev['chunk_id']})")
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


def build_rule_answer(rule: Dict[str, Any], ev: Dict[str, Any], templates: Dict[str, Any], lang: str, description: str) -> str:
    t = templates["qa"][lang]
    base = t["rule_answer_prefix"].format(description=description)

    key_lines = extract_key_lines(ev.get("content") or "", max_lines=8)
    loc = f"{ev['file_path']}:L{ev['start_line']}-{ev['end_line']}"

    if lang == "zh":
        if key_lines:
            bullets = "\n".join([f"- {x}" for x in key_lines])
            return base + f"\n\n关键判断/处理分支(摘自{loc}):\n{bullets}"
        return base

    if key_lines:
        bullets = "\n".join([f"- {x}" for x in key_lines])
        return base + f"\n\nKey code points(from {loc}):\n{bullets}"
    return base + f"\n\n(Reference:{loc})"


def build_flow_answer(flow: Dict[str, Any], evs: List[Dict[str, Any]], templates: Dict[str, Any], lang: str) -> str:
    t = templates["qa"][lang]
    header = str(t["flow_answer"])

    steps = flow.get("steps") or []
    if not steps:
        return header

    lines: List[str] = []
    for i, s in enumerate(steps, 1):
        if lang == "en":
            step_name = (s.get("name_en") or "").strip()
            if not step_name:
                step_name = (s.get("op") or "").strip() or f"step{i}"
        else:
            step_name = (s.get("name") or "").strip()
            if not step_name:
                step_name = (s.get("why") or "").strip() or (s.get("op") or "").strip() or f"step{i}"

        cid = s.get("evidence_chunk")
        ev = None
        if cid:
            for e in evs:
                if e.get("chunk_id") == cid:
                    ev = e
                    break
        if ev:
            loc = f"{ev['file_path']}:L{ev['start_line']}-{ev['end_line']}"
            lines.append(f"{i}. {step_name}->{loc}(chunk={cid})")
        else:
            lines.append(f"{i}. {step_name}")

    if lang == "zh":
        return header + "\n\n流程步骤与代码位置:\n" + "\n".join(lines)
    return header + "\n\nSteps and code locations:\n" + "\n".join(lines)


# -----------------------------
# QA generators
# -----------------------------
def generate_rule_qa(
    rule: Dict[str, Any],
    index_map: Dict[str, Dict[str, Any]],
    templates: Dict[str, Any],
    lang: str,
    code_max_chars: int,
    ex_re: re.Pattern,
) -> Dict[str, Any]:
    ev_chunks = rule.get("evidence_chunks") or []
    cid = ev_chunks[0] if ev_chunks else None
    if (not cid) or (cid not in index_map):
        return {}

    ev = build_evidence(cid, index_map, code_max_chars)
    if is_excluded_file(ev["file_path"], ex_re):
        return {}

    rule_id = rule.get("rule_id") or rule.get("id") or ""
    title_raw = rule.get("title", "") or ""
    desc_raw = rule.get("description", "") or ""

    strict_lv = get_en_strict_level()

    if lang == "en":
        title_en = (rule.get("title_en") or "").strip()
        desc_en = (rule.get("description_en") or "").strip()

        infer_title, infer_desc = infer_en_topic_from_evidence(ev)
        title_fb = infer_title if infer_title else (f"Rule {rule_id}" if rule_id else "Rule")
        desc_fb = infer_desc if infer_desc else "Explain the implementation based on the evidence code."

        if strict_lv == 0:
            title = title_raw
            desc = desc_raw
        else:
            title = en_pick(title_raw, title_en, title_fb, strict_lv)
            desc = en_pick(desc_raw, desc_en, desc_fb, strict_lv)
            if strict_lv >= 2 and ((not title) or (not desc)):
                return {}

        if (ev.get("code_lang") or "").lower() in ("yaml", "properties"):
            q = f"What does this configuration in {ev['file_path']} control? Explain the key settings and their effects."
        else:
            q = choose_rule_question(templates["qa"][lang], title=title, file_path=ev["file_path"], lang=lang)
    else:
        title = title_raw
        desc = desc_raw
        q = choose_rule_question(templates["qa"][lang], title=title, file_path=ev["file_path"], lang=lang)

    core = build_rule_answer(rule, ev, templates, lang, description=desc)

    trace_steps = [
        (f"定位到规则:{title}" if lang == "zh" else f"Locate rule:{title}"),
        ("抽取关键判断/分支并与业务描述对齐" if lang == "zh" else "Extract key branches/return paths from the evidence"),
        ("给出证据引用与原文代码段" if lang == "zh" else "Provide evidence references and code snippet"),
    ]
    a = wrap_answer(core, [ev], trace_steps, lang)

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
        "trace_digest": trace_steps,
        "generator": "step04_v4_consume_step03_en",
        "source": "AutoCodeDataPipeline",
    }

    return {
        "sample_id": make_id("qa", f"{rule_id}|{lang}|{cid}"),
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
            "rule_ids": [rule_id],
            "reasoning_steps": trace_steps,
        },
        "meta": {
            "domain": rule.get("domain"),
            "qa_type": "rule",
            "difficulty": "medium",
            "generator": "step04_v4_consume_step03_en",
            "source": "AutoCodeDataPipeline",
        },
        "text": text,
        "meta_v2": meta_v2,
    }


def generate_flow_qa(
    flow: Dict[str, Any],
    index_map: Dict[str, Dict[str, Any]],
    templates: Dict[str, Any],
    lang: str,
    code_max_chars: int,
    flow_max_evs: int,
    ex_re: re.Pattern,
) -> Dict[str, Any]:
    evs: List[Dict[str, Any]] = []
    for s in flow.get("steps", []) or []:
        cid = s.get("evidence_chunk")
        if cid and cid in index_map:
            ev = build_evidence(cid, index_map, code_max_chars)
            if is_excluded_file(ev["file_path"], ex_re):
                continue
            evs.append(ev)
        if len(evs) >= flow_max_evs:
            break

    if not evs:
        return {}

    flow_id = flow.get("flow_id") or flow.get("id") or ""
    flow_name_raw = (flow.get("name", "") or "").strip()
    strict_lv = get_en_strict_level()

    if lang == "en":
        flow_name_en = (flow.get("name_en") or "").strip()
        fb = f"Flow {flow_id}" if flow_id else "Business flow"
        if strict_lv == 0:
            flow_name = flow_name_raw
        else:
            flow_name = en_pick(flow_name_raw, flow_name_en, fb, strict_lv)
            if strict_lv >= 2 and (not flow_name):
                return {}
    else:
        flow_name = flow_name_raw

    q = choose_flow_question(templates["qa"][lang], flow_name=flow_name, lang=lang)
    core = build_flow_answer(flow, evs, templates, lang)

    trace_steps = [
        (f"识别流程:{flow_name}" if lang == "zh" else f"Identify flow:{flow_name}"),
        ("按调用/职责边界抽取步骤并关联代码位置" if lang == "zh" else "Extract steps and link to code locations"),
        ("输出端到端流程并给出证据与原文代码段" if lang == "zh" else "Summarize end-to-end flow with evidence and code snippet"),
    ]
    a = wrap_answer(core, evs, trace_steps, lang)

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
        "trace_digest": trace_steps,
        "generator": "step04_v4_consume_step03_en",
        "source": "AutoCodeDataPipeline",
    }

    return {
        "sample_id": make_id("qa", f"{flow_id}|{lang}"),
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
            "flow_id": flow_id,
            "reasoning_steps": trace_steps,
        },
        "meta": {
            "domain": flow.get("domain"),
            "qa_type": "flow",
            "difficulty": "hard",
            "generator": "step04_v4_consume_step03_en",
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

    rule_topn = int(os.environ.get("QA_RULE_TOPN", "200"))
    flow_max_evs = int(os.environ.get("QA_FLOW_MAX_EVS", "3"))
    code_max_chars = int(os.environ.get("QA_CODE_MAX_CHARS", "2200"))

    rule_per_id_cap = int(os.environ.get("QA_RULE_PER_ID_CAP", "3"))
    ex_re = compile_exclude_regex()

    samples: List[Dict[str, Any]] = []
    per_lang_rule_count: Dict[Tuple[str, str], int] = {}
    seen_rule_sig: set = set()
    seen_flow_sig: set = set()

    for lang in langs:
        rules_pool = rules[:rule_topn]
        random.shuffle(rules_pool)

        for r in rules_pool:
            rule_id = r.get("rule_id") or r.get("id") or ""
            if not rule_id:
                continue
            key = (lang, rule_id)
            if per_lang_rule_count.get(key, 0) >= rule_per_id_cap:
                continue

            s = generate_rule_qa(r, index_map, templates, lang, code_max_chars, ex_re)
            if not s:
                continue

            ev0 = (s.get("evidence") or [{}])[0]
            chunk_id = ev0.get("chunk_id") or ""
            sig = (lang, rule_id, chunk_id)
            if chunk_id and sig in seen_rule_sig:
                continue

            seen_rule_sig.add(sig)
            per_lang_rule_count[key] = per_lang_rule_count.get(key, 0) + 1
            samples.append(s)

        for f in flows:
            flow_id = f.get("flow_id") or f.get("id") or ""
            s = generate_flow_qa(f, index_map, templates, lang, code_max_chars, flow_max_evs, ex_re)
            if not s:
                continue
            chunk_ids = tuple([e.get("chunk_id") for e in (s.get("evidence") or []) if e.get("chunk_id")][:flow_max_evs])
            fsig = (lang, flow_id, chunk_ids)
            if fsig in seen_flow_sig:
                continue
            seen_flow_sig.add(fsig)
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
    print(
        f"[Knobs] QA_RULE_TOPN={rule_topn}, QA_RULE_PER_ID_CAP={rule_per_id_cap}, "
        f"QA_FLOW_MAX_EVS={flow_max_evs}, QA_CODE_MAX_CHARS={code_max_chars}, "
        f"QA_STRICT_EN={os.environ.get('QA_STRICT_EN','1')}, "
        f"QA_EXCLUDE_PATHS={os.environ.get('QA_EXCLUDE_PATHS','(default)')}, "
        f"lang_mode={lang_mode}, langs={langs}"
    )


if __name__ == "__main__":
    main()
