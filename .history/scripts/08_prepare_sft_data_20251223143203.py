#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step08
将Step04(qa)与Step05(design)的可审计样本整理为可用于微调的SFT数据。

改造目标(对齐面试题意):
-训练数据允许轻量化(light)以控制token成本，但必须保留可回溯的evidence/trace元数据(meta)
-同时支持输出grounded版本:在样本中直接内嵌Evidence(代码段)+Trace(推理步骤)，用于验收展示/可选训练
-支持QA与Design两类样本
"""

import json
from pathlib import Path
import os
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]

try:
    import yaml
except Exception:
    yaml = None


def read_jsonl(p: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not p.exists():
        return out
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(p: Path, rows: List[Dict[str, Any]]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_design_text(sample: Dict[str, Any]) -> str:
    """兼容Step05输出字段: design / design_output(dict)"""
    v = sample.get("design")
    if isinstance(v, str) and v.strip():
        return v.strip()
    v2 = sample.get("design_output")
    if isinstance(v2, str) and v2.strip():
        return v2.strip()
    if isinstance(v2, dict):
        return json.dumps(v2, ensure_ascii=False, indent=2)
    return ""



def load_runtime_cfg() -> Dict[str, Any]:
    cfg_path = ROOT / "configs/runtime.yaml"
    if yaml is None or (not cfg_path.exists()):
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def normalize_lang(lang: str) -> str:
    if not lang:
        return "zh"
    l = lang.strip().lower()
    if l in ("bilingual", "bi", "both", "zh+en", "en+zh"):
        return "bilingual"
    if l.startswith("zh") or l == "zh":
        return "zh"
    if l.startswith("en") or l == "en":
        return "en"
    return "zh"


def pick_lang_filter() -> str:
    """优先级:ENV LANG_FILTER>runtime.yaml.language.mode>zh"""
    env = os.environ.get("LANG_FILTER")
    if env:
        return normalize_lang(env) if env != "all" else "all"
    cfg = load_runtime_cfg()
    mode = ((cfg.get("language") or {}).get("mode")) or "zh"
    return normalize_lang(mode)


def get_sft_cfg() -> Dict[str, Any]:
    cfg = load_runtime_cfg()
    return cfg.get("sft") or {}


def _clip_text(s: str, max_chars: int) -> str:
    s = s or ""
    if max_chars <= 0:
        return s
    return s if len(s) <= max_chars else (s[: max_chars - 3] + "...")


def _content_to_code_block(content: str, lang_hint: str = "java", max_lines: int = 60, max_chars: int = 4000) -> str:
    if not content:
        return ""
    lines = content.splitlines()
    if max_lines > 0:
        lines = lines[:max_lines]
    clipped = "\n".join(lines)
    clipped = _clip_text(clipped, max_chars)
    return f"```{lang_hint}\n{clipped}\n```"


def build_meta(sample: Dict[str, Any], max_trace_steps: int = 6) -> Dict[str, Any]:
    evs = sample.get("evidence") or []
    ev_meta = []
    for ev in evs[:4]:
        ev_meta.append({
            "chunk_id": ev.get("chunk_id"),
            "file_path": ev.get("file_path"),
            "start_line": ev.get("start_line"),
            "end_line": ev.get("end_line"),
        })

    trace = sample.get("trace") or {}
    steps = trace.get("reasoning_steps") or []
    if max_trace_steps > 0:
        steps = steps[:max_trace_steps]

    meta = {
        "sample_id": sample.get("sample_id"),
        "task_type": sample.get("task_type"),
        "language": sample.get("language"),
        "evidence": ev_meta,
        "trace_digest": steps,
        "source": sample.get("meta", {}).get("source") or "AutoCodeDataPipeline",
        "generator": sample.get("meta", {}).get("generator"),
    }
    meta.update({"orig_meta": sample.get("meta") or {}})
    return meta


def to_text_light(sample: Dict[str, Any]) -> str:
    if sample.get("task_type") == "design":
        req = (sample.get("requirement") or "").strip()
        resp = get_design_text(sample)
        return f"### Instruction:\n{req}\n\n### Response:\n{resp}"
    q = (sample.get("question") or "").strip()
    a = (sample.get("answer") or "").strip()
    return f"### Instruction:\n{q}\n\n### Response:\n{a}"


def to_text_grounded(sample: Dict[str, Any], evidence_max: int = 2, evidence_max_lines: int = 60, evidence_max_chars: int = 4000) -> str:
    lang = sample.get("language") or "zh"
    is_zh = lang == "zh"

    if sample.get("task_type") == "design":
        instr = (sample.get("requirement") or "").strip()
        resp = get_design_text(sample)
        title_e = "证据代码" if is_zh else "Evidence"
        title_t = "推理过程" if is_zh else "Trace"
    else:
        instr = (sample.get("question") or "").strip()
        resp = (sample.get("answer") or "").strip()
        title_e = "证据代码" if is_zh else "Evidence"
        title_t = "推理过程" if is_zh else "Trace"

    parts = [f"### Instruction:\n{instr}"]

    evs = sample.get("evidence") or []
    if evs:
        ev_lines: List[str] = [f"\n### {title_e}:"]
        for i, ev in enumerate(evs[: max(1, evidence_max)], 1):
            fp = ev.get("file_path") or ""
            sl = ev.get("start_line")
            el = ev.get("end_line")
            loc = f"{fp}:L{sl}-L{el}" if (sl is not None and el is not None) else fp
            ev_lines.append(f"[{i}] {loc}")
            ev_lines.append(_content_to_code_block(ev.get("content") or "", "java", evidence_max_lines, evidence_max_chars))
        parts.append("\n".join(ev_lines))

    trace = sample.get("trace") or {}
    steps = trace.get("reasoning_steps") or []
    if steps:
        t_lines = [f"\n### {title_t}:"] + [f"{i}. {s}" for i, s in enumerate(steps, 1)]
        parts.append("\n".join(t_lines))

    parts.append(f"\n### Response:\n{resp}")
    return "\n\n".join(parts)


def to_messages_light(sample: Dict[str, Any]) -> List[Dict[str, str]]:
    if sample.get("task_type") == "design":
        user = (sample.get("requirement") or "").strip()
        assistant = get_design_text(sample)
    else:
        user = (sample.get("question") or "").strip()
        assistant = (sample.get("answer") or "").strip()

    system = "你是代码仓库问答与架构设计助手。回答必须基于给定代码仓的实现与约束。" if sample.get("language") == "zh" else \
             "You are a repo-grounded QA & design assistant. Answers must be grounded in the given repository."
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def to_messages_grounded(sample: Dict[str, Any], evidence_max: int = 2, evidence_max_lines: int = 60, evidence_max_chars: int = 4000) -> List[Dict[str, str]]:
    lang = sample.get("language") or "zh"
    is_zh = lang == "zh"

    if sample.get("task_type") == "design":
        instr = (sample.get("requirement") or "").strip()
        assistant = get_design_text(sample)
    else:
        instr = (sample.get("question") or "").strip()
        assistant = (sample.get("answer") or "").strip()

    title_e = "证据代码" if is_zh else "Evidence"
    title_t = "推理过程" if is_zh else "Trace"

    blocks = [instr]

    evs = sample.get("evidence") or []
    if evs:
        ev_lines: List[str] = [f"\n[{title_e}]"]
        for i, ev in enumerate(evs[: max(1, evidence_max)], 1):
            fp = ev.get("file_path") or ""
            sl = ev.get("start_line")
            el = ev.get("end_line")
            loc = f"{fp}:L{sl}-L{el}" if (sl is not None and el is not None) else fp
            ev_lines.append(f"({i}) {loc}")
            ev_lines.append(_content_to_code_block(ev.get("content") or "", "java", evidence_max_lines, evidence_max_chars))
        blocks.append("\n".join(ev_lines))

    trace = sample.get("trace") or {}
    steps = trace.get("reasoning_steps") or []
    if steps:
        t_lines = [f"\n[{title_t}]"] + [f"{i}. {s}" for i, s in enumerate(steps, 1)]
        blocks.append("\n".join(t_lines))

    user = "\n\n".join(blocks)

    system = "你是代码仓库问答与架构设计助手。回答必须基于给定代码仓的实现与约束。" if is_zh else \
             "You are a repo-grounded QA & design assistant. Answers must be grounded in the given repository."

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def build_sft_row(sample: Dict[str, Any], fmt: str, mode: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    evidence_max = int(cfg.get("evidence_max", 2))
    evidence_max_lines = int(cfg.get("evidence_max_lines", 60))
    evidence_max_chars = int(cfg.get("evidence_max_chars", 4000))
    trace_digest_max = int(cfg.get("trace_digest_max", 6))

    meta = build_meta(sample, max_trace_steps=trace_digest_max)

    if fmt == "messages":
        if mode == "grounded":
            messages = to_messages_grounded(sample, evidence_max, evidence_max_lines, evidence_max_chars)
        else:
            messages = to_messages_light(sample)
        return {"messages": messages, "meta": meta}

    if mode == "grounded":
        text = to_text_grounded(sample, evidence_max, evidence_max_lines, evidence_max_chars)
    else:
        text = to_text_light(sample)
    return {"text": text, "meta": meta}


def main():
    lang_filter = pick_lang_filter()  # zh|en|bilingual|all
    cfg = get_sft_cfg()

    sft_mode = os.environ.get("SFT_MODE", str(cfg.get("mode", "light"))).strip().lower()
    sft_fmt = os.environ.get("SFT_FORMAT", str(cfg.get("format", "text"))).strip().lower()
    sft_mode = sft_mode if sft_mode in ("light", "grounded") else "light"
    sft_fmt = sft_fmt if sft_fmt in ("text", "messages") else "text"

    qa_train = ROOT / "data/dataset/train.jsonl"
    qa_dev = ROOT / "data/dataset/dev.jsonl"
    qa_test = ROOT / "data/dataset/test.jsonl"

    design_train = ROOT / "data/dataset/design_train.jsonl"
    design_dev = ROOT / "data/dataset/design_dev.jsonl"
    design_test = ROOT / "data/dataset/design_test.jsonl"

    splits = {
        "train": read_jsonl(qa_train) + read_jsonl(design_train),
        "dev": read_jsonl(qa_dev) + read_jsonl(design_dev),
        "test": read_jsonl(qa_test) + read_jsonl(design_test),
    }

    if not any(splits.values()):
        qa_samples = read_jsonl(ROOT / "data/samples/qa_samples.jsonl")
        design_samples = read_jsonl(ROOT / "data/samples/design_samples.jsonl")
        splits = {"train": qa_samples + design_samples, "dev": [], "test": []}

    def filter_lang(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if lang_filter == "all":
            return rows
        if lang_filter == "bilingual":
            return [r for r in rows if r.get("language") in ("zh", "en")]
        return [r for r in rows if r.get("language") == lang_filter]

    out_dir = ROOT / "data/sft"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for split, rows in splits.items():
        rows = filter_lang(rows)
        if not rows:
            continue

        sft_rows = [build_sft_row(s, sft_fmt, sft_mode, cfg) for s in rows]

        out_path = out_dir / (f"{split}.jsonl" if sft_fmt == "text" else f"{split}.messages.jsonl")
        write_jsonl(out_path, sft_rows)

        if split == "train":
            compat = out_dir / "train.jsonl"
            write_jsonl(compat, sft_rows)

        print(f"[Step08] split={split} mode={sft_mode} format={sft_fmt} count={len(sft_rows)} -> {out_path}")
        total += len(sft_rows)

    print(f"[Step08] done.total={total},lang_filter={lang_filter}")


if __name__ == "__main__":
    main()
