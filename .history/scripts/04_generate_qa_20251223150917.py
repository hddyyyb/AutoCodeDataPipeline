#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step04
根据rules+flows+repo_index自动生成QA样本

修复点(解决“基本都是同一类问题”):
- 默认排除mall-mbg等自动生成/模板化代码(可用QA_EXCLUDE_PATHS覆盖)
- 同一rule_id按配额限流(默认每语言最多3条，可用QA_RULE_PER_ID_CAP调)
- (rule_id,chunk_id,lang)去重，避免同题同证据重复
- rule_question支持多模板随机化(可选增强多样性)

仍保持目标:
- 每lang规则取前N条(默认200)
- flow全量生成
- 不做比例裁剪
- 每条带evidence_snippets+trace
- answer包含【原文代码段】代码块
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
# Language hygiene helpers
# -----------------------------
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def has_cjk(s: str) -> bool:
    return bool(s and _CJK_RE.search(s))


def get_en_strict_level() -> int:
    """
    QA_STRICT_EN:
      0=off(不清洗)
      1=fallback(默认:遇中文用英文fallback替换,不丢样本)
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
    if not has_cjk(text_raw):
        return text_raw
    if strict_lv >= 2:
        return ""
    return fallback


# -----------------------------
# IO helpers
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
# Evidence helpers
# -----------------------------
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


# -----------------------------
# Diversity controls (核心修复)
# -----------------------------
def compile_exclude_regex() -> re.Pattern:
    """
    默认排除:
    - mall-mbg(自动生成)
    - /model/目录
    - *Example.java(MyBatis Generator模板类)
    你可用环境变量QA_EXCLUDE_PATHS覆盖(正则)
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
    """
    支持rule_question为字符串或字符串列表:
      - 字符串:直接format
      - 列表:随机选一个
    """
    t = templates_qa_lang
    rq = t.get("rule_question")
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
# QA generators
# -----------------------------
def build_rule_answer(rule: Dict[str, Any], ev: Dict[str, Any], templates: Dict[str, Any], lang: str,
                     description: str = "") -> str:
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
        return base + f"\n\nKey code points(from {loc}):\n{bullets}"


def build_flow_answer(flow: Dict[str, Any], evs: List[Dict[str, Any]], templates: Dict[str, Any], lang: str) -> str:
    t = templates["qa"][lang]
    header = str(t["flow_answer"])

    steps = flow.get("steps") or []
    if not steps:
        return header

    lines: List[str] = []
    for i, s in enumerate(steps, 1):
        step_name_raw = s.get("name") or ""
        op = s.get("op") or ""
        step_name = step_name_raw or op or f"step{i}"
        if lang == "en" and has_cjk(step_name):
            step_name = op or f"step{i}"

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
    else:
        return header + "\n\nSteps and code locations:\n" + "\n".join(lines)


def generate_rule_qa(
    rule: Dict[str, Any],
    index_map: Dict[str, Dict[str, Any]],
    templates: Dict[str, Any],
    lang: str,
    code_max_chars: int,
    ex_re: re.Pattern,
) -> Dict[str, Any]:
    cid = None
    ev_chunks = rule.get("evidence_chunks") or []
    if ev_chunks:
        cid = ev_chunks[0]
    if (not cid) or (cid not in index_map):
        return {}

    ev = build_evidence(cid, index_map, code_max_chars)

    # 排除模板化/低价值文件
    if is_excluded_file(ev["file_path"], ex_re):
        return {}

    rule_id = rule.get("rule_id") or rule.get("id") or ""
    title_raw = rule.get("title", "") or ""
    desc_raw = rule.get("description", "") or ""

    strict_lv = get_en_strict_level()
    if lang == "en":
        title_fb = f"Rule {rule_id}" if rule_id else "Rule"
        desc_fb = "Please explain the rule and its implementation based on the evidence code."
        title_en = rule.get("title_en") or ""
        desc_en = rule.get("description_en") or ""
        if strict_lv == 0:
            title = title_raw
            desc = desc_raw
        else:
            title = en_pick(title_raw, title_en, title_fb, strict_lv)
            desc = en_pick(desc_raw, desc_en, desc_fb, strict_lv)
            if strict_lv >= 2 and ((not title) or (not desc)):
                return {}
    else:
        title = title_raw
        desc = desc_raw

    t_lang = templates["qa"][lang]
    q = choose_rule_question(t_lang, title=title, file_path=ev["file_path"], lang=lang)

    core = build_rule_answer(rule, ev, templates, lang, description=desc)
    a = wrap_answer(core, [ev], [
        (f"定位到规则:{title}" if lang == "zh" else f"Locate rule:{title}"),
        ("抽取关键判断/分支并与业务描述对齐" if lang == "zh" else "Align key branches/conditions with business description"),
        ("给出证据引用与原文代码段" if lang == "zh" else "Provide evidence references and code snippet"),
    ], lang)

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
            (f"定位到规则:{title}" if lang == "zh" else f"Locate rule:{title}"),
            ("抽取关键判断/分支并与业务描述对齐" if lang == "zh" else "Align key branches/conditions with business description"),
            ("给出证据引用与原文代码段" if lang == "zh" else "Provide evidence references and code snippet"),
        ],
        "generator": "step04_diverse_v2",
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
            "reasoning_steps": meta_v2["trace_digest"],
        },
        "meta": {
            "domain": rule.get("domain"),
            "qa_type": "rule",
            "difficulty": "medium",
            "generator": "step04_diverse_v2",
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
    flow_name_raw = flow.get("name", "") or ""
    strict_lv = get_en_strict_level()

    if lang == "en":
        flow_name_fb = f"Flow {flow_id}" if flow_id else "Business flow"
        flow_name_en = flow.get("name_en") or ""
        if strict_lv == 0:
            flow_name = flow_name_raw
        else:
            flow_name = en_pick(flow_name_raw, flow_name_en, flow_name_fb, strict_lv)
            if strict_lv >= 2 and (not flow_name):
                return {}
            if strict_lv >= 2:
                for s in flow.get("steps", []) or []:
                    nm_raw = s.get("name") or ""
                    nm_en = s.get("name_en") or ""
                    if has_cjk(nm_raw) and (not nm_en or has_cjk(nm_en)):
                        return {}
    else:
        flow_name = flow_name_raw

    t_lang = templates["qa"][lang]
    q = choose_flow_question(t_lang, flow_name=flow_name, lang=lang)

    core = build_flow_answer(flow, evs, templates, lang)
    a = wrap_answer(core, evs, [
        (f"识别流程:{flow_name}" if lang == "zh" else f"Identify flow:{flow_name}"),
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
            (f"识别流程:{flow_name}" if lang == "zh" else f"Identify flow:{flow_name}"),
            ("抽取步骤并标注代码位置" if lang == "zh" else "Extract steps and annotate code locations"),
            ("汇总为端到端流程并附证据" if lang == "zh" else "Summarize end-to-end flow with evidence"),
        ],
        "generator": "step04_diverse_v2",
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
            "reasoning_steps": meta_v2["trace_digest"],
        },
        "meta": {
            "domain": flow.get("domain"),
            "qa_type": "flow",
            "difficulty": "hard",
            "generator": "step04_diverse_v2",
            "source": "AutoCodeDataPipeline",
        },
        "text": text,
        "meta_v2": meta_v2,
    }


# -----------------------------
# Main with diversity sampling
# -----------------------------
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

    # knobs
    rule_topn = int(os.environ.get("QA_RULE_TOPN", "200"))
    flow_max_evs = int(os.environ.get("QA_FLOW_MAX_EVS", "3"))
    code_max_chars = int(os.environ.get("QA_CODE_MAX_CHARS", "2200"))

    # diversity knobs
    rule_per_id_cap = int(os.environ.get("QA_RULE_PER_ID_CAP", "3"))  # 每语言每rule_id最多几条
    ex_re = compile_exclude_regex()

    samples: List[Dict[str, Any]] = []

    # 统计器与去重集合
    per_lang_rule_count: Dict[Tuple[str, str], int] = {}  # (lang, rule_id)->count
    seen_rule_sig: set = set()  # (lang, rule_id, chunk_id)
    seen_flow_sig: set = set()  # (lang, flow_id, tuple(chunk_ids[:k]))

    for lang in langs:
        # ---- rules:先打散再限流，避免rules文件头部某类rule霸屏 ----
        rules_pool = rules[:rule_topn]
        random.shuffle(rules_pool)

        for r in rules_pool:
            rule_id = r.get("rule_id") or r.get("id") or ""
            if not rule_id:
                continue

            key = (lang, rule_id)
            if per_lang_rule_count.get(key, 0) >= rule_per_id_cap:
                continue

            # 先尝试生成
            s = generate_rule_qa(r, index_map, templates, lang, code_max_chars, ex_re)
            if not s:
                continue

            # 去重:同一(rule_id,chunk_id,lang)只留一条
            ev0 = (s.get("evidence") or [{}])[0]
            chunk_id = ev0.get("chunk_id") or ""
            sig = (lang, rule_id, chunk_id)
            if chunk_id and sig in seen_rule_sig:
                continue

            seen_rule_sig.add(sig)
            per_lang_rule_count[key] = per_lang_rule_count.get(key, 0) + 1
            samples.append(s)

        # ---- flows:全量生成，但也做基础去重 ----
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
