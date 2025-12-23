#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step08(v2)
将可审计样本整理为可用于微调的SFT数据，并按语言拆分输出，便于后续训练只使用一种语言。

输入(优先级):
- Step06输出: data/dataset/final_{train,dev,test}.jsonl
  每条: {"text":..., "meta":...}，meta内含 evidence_snippets(code) + trace_digest
- 若final_*不存在，回退读取Step04/05输出(qa/design)并自动转换为{text,meta}

输出:
- data/sft/{split}_{lang}.jsonl 或 {split}_{lang}.messages.jsonl
  其中lang∈{zh,en,unknown}
- 兼容输出:
  - 若LANG_FILTER=zh或en，则额外写 data/sft/{split}.jsonl(或.messages.jsonl) 和 data/sft/train.jsonl

关键环境变量:
- LANG_FILTER: zh|en|bilingual|all (默认读取runtime.yaml.language.mode,否则zh)
- LANG_SPLIT: 1/0(默认1)。1表示总是按语言拆分写文件
- SFT_MODE: light|grounded (默认读runtime.yaml.sft.mode,否则light)
- SFT_FORMAT: text|messages (默认读runtime.yaml.sft.format,否则text)
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


# -----------------------------
# IO
# -----------------------------
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


# -----------------------------
# Runtime cfg
# -----------------------------
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
    if l == "all":
        return "all"
    return "zh"


def pick_lang_filter() -> str:
    """优先级: ENV LANG_FILTER > runtime.yaml.language.mode > zh"""
    env = os.environ.get("LANG_FILTER")
    if env:
        return normalize_lang(env)
    cfg = load_runtime_cfg()
    mode = ((cfg.get("language") or {}).get("mode")) or "zh"
    return normalize_lang(mode)


def get_sft_cfg() -> Dict[str, Any]:
    cfg = load_runtime_cfg()
    return cfg.get("sft") or {}


def get_lang_split_enabled() -> bool:
    v = os.environ.get("LANG_SPLIT", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


# -----------------------------
# Text helpers
# -----------------------------
def _clip_text(s: str, max_chars: int) -> str:
    s = s or ""
    if max_chars <= 0:
        return s
    return s if len(s) <= max_chars else (s[: max_chars - 3] + "...")


def _code_block(code: str, lang_hint: str = "", max_lines: int = 60, max_chars: int = 4000) -> str:
    if not code:
        return ""
    lines = code.splitlines()
    if max_lines > 0:
        lines = lines[:max_lines]
    clipped = "\n".join(lines)
    clipped = _clip_text(clipped, max_chars)
    return f"```{lang_hint}\n{clipped}\n```"


# -----------------------------
# Sample normalization
# -----------------------------
def get_meta(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step06 final_*: sample["meta"] 就是统一meta
    Step04/05: 优先meta_v2，否则回退meta
    """
    if isinstance(sample.get("meta"), dict) and sample.get("meta", {}).get("task_type"):
        return sample["meta"]
    if isinstance(sample.get("meta_v2"), dict) and sample.get("meta_v2", {}).get("task_type"):
        return sample["meta_v2"]
    return sample.get("meta") or {}


def get_lang(sample: Dict[str, Any]) -> str:
    m = get_meta(sample)
    lang = (m.get("language") or sample.get("language") or "").strip().lower()
    if lang in ("zh", "en"):
        return lang
    return "unknown"


def build_meta(sample: Dict[str, Any], max_trace_steps: int = 6) -> Dict[str, Any]:
    """
    输出给SFT的meta：保留可回溯信息，但尽量轻量化。
    强制带:
    - task_type/language/domain/qa_type/difficulty
    - evidence(定位)
    - evidence_snippets(原文代码段)
    - trace_digest(推理步骤)
    """
    m = get_meta(sample)

    ev_meta = []
    for ev in (m.get("evidence") or [])[:4]:
        ev_meta.append({
            "chunk_id": ev.get("chunk_id"),
            "file_path": ev.get("file_path"),
            "start_line": ev.get("start_line"),
            "end_line": ev.get("end_line"),
        })

    snips = []
    for ev in (m.get("evidence_snippets") or [])[:4]:
        snips.append({
            "chunk_id": ev.get("chunk_id"),
            "file_path": ev.get("file_path"),
            "start_line": ev.get("start_line"),
            "end_line": ev.get("end_line"),
            "code_lang": ev.get("code_lang"),
            "code": ev.get("code"),
        })

    steps = m.get("trace_digest") or []
    if max_trace_steps > 0 and isinstance(steps, list):
        steps = steps[:max_trace_steps]

    out = {
        "task_type": m.get("task_type") or sample.get("task_type"),
        "language": m.get("language") or sample.get("language"),
        "domain": m.get("domain"),
        "qa_type": m.get("qa_type"),
        "difficulty": m.get("difficulty"),
        "evidence": ev_meta,
        "evidence_snippets": snips,
        "trace_digest": steps,
        "source": m.get("source") or "AutoCodeDataPipeline",
        "generator": m.get("generator"),
    }
    if sample.get("sample_id"):
        out["sample_id"] = sample.get("sample_id")
    return out


def to_text_light(sample: Dict[str, Any]) -> str:
    if isinstance(sample.get("text"), str) and sample["text"].strip():
        return sample["text"].strip()

    task_type = sample.get("task_type") or get_meta(sample).get("task_type")
    if task_type == "design":
        req = (sample.get("requirement") or "").strip()
        resp = sample.get("design_output")
        if isinstance(resp, dict):
            resp = json.dumps(resp, ensure_ascii=False, indent=2)
        resp = (resp or "").strip()
        return f"### Instruction\n{req}\n\n### Response\n{resp}\n"

    q = (sample.get("question") or "").strip()
    a = (sample.get("answer") or "").strip()
    return f"### Instruction\n{q}\n\n### Response\n{a}\n"


def to_text_grounded(sample: Dict[str, Any], cfg: Dict[str, Any]) -> str:
    m = get_meta(sample)
    lang = get_lang(sample)
    is_zh = lang == "zh"

    evidence_max = int(cfg.get("evidence_max", 2))
    evidence_max_lines = int(cfg.get("evidence_max_lines", 60))
    evidence_max_chars = int(cfg.get("evidence_max_chars", 4000))
    trace_digest_max = int(cfg.get("trace_digest_max", 6))

    base = to_text_light(sample)

    marker_i = "### Instruction"
    marker_r = "### Response"
    if marker_i in base and marker_r in base:
        idx = base.find(marker_r)
        instr_part = base[:idx].strip()
        resp_part = base[idx:].strip()
    else:
        instr_part = f"### Instruction\n{base.strip()}"
        resp_part = ""

    title_e = "证据代码" if is_zh else "Evidence"
    title_t = "推理过程" if is_zh else "Trace"

    blocks = [instr_part]

    snips = m.get("evidence_snippets") or []
    if snips:
        ev_lines: List[str] = [f"\n### {title_e}:"]
        for i, ev in enumerate(snips[: max(1, evidence_max)], 1):
            fp = ev.get("file_path") or ""
            sl = ev.get("start_line")
            el = ev.get("end_line")
            loc = f"{fp}:L{sl}-L{el}" if (sl is not None and el is not None) else fp
            ev_lines.append(f"[{i}] {loc} (chunk_id={ev.get('chunk_id')})")
            ev_lines.append(_code_block(ev.get("code") or "", ev.get("code_lang") or "", evidence_max_lines, evidence_max_chars))
        blocks.append("\n".join(ev_lines))

    steps = m.get("trace_digest") or []
    if isinstance(steps, list) and steps:
        if trace_digest_max > 0:
            steps = steps[:trace_digest_max]
        t_lines = [f"\n### {title_t}:"] + [f"{i}. {s}" for i, s in enumerate(steps, 1)]
        blocks.append("\n".join(t_lines))

    if resp_part:
        blocks.append("\n" + resp_part)
    return "\n\n".join(blocks) + ("\n" if not base.endswith("\n") else "")


def to_messages(sample: Dict[str, Any], mode: str, cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    lang = get_lang(sample)
    is_zh = lang == "zh"

    system = "你是代码仓库问答与架构设计助手。回答必须基于给定代码仓的实现与约束。" if is_zh else \
             "You are a repo-grounded QA & design assistant. Answers must be grounded in the given repository."

    if mode == "grounded":
        user_text = to_text_grounded(sample, cfg)
        base = to_text_light(sample)
        marker_r = "### Response"
        assistant = ""
        if marker_r in base:
            assistant = base[base.find(marker_r) + len(marker_r):].strip()
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text.strip()},
            {"role": "assistant", "content": assistant},
        ]

    base = to_text_light(sample)
    marker_r = "### Response"
    if marker_r in base:
        idx = base.find(marker_r)
        user = base[:idx].replace("### Instruction", "").strip()
        assistant = base[idx + len(marker_r):].strip()
    else:
        user = base.strip()
        assistant = ""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def build_sft_row(sample: Dict[str, Any], fmt: str, mode: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    trace_digest_max = int(cfg.get("trace_digest_max", 6))
    meta = build_meta(sample, max_trace_steps=trace_digest_max)

    if fmt == "messages":
        return {"messages": to_messages(sample, mode, cfg), "meta": meta}

    text = to_text_grounded(sample, cfg) if mode == "grounded" else to_text_light(sample)
    return {"text": text, "meta": meta}


# -----------------------------
# Dataset loading
# -----------------------------
def load_splits() -> Dict[str, List[Dict[str, Any]]]:
    final_train = ROOT / "data/dataset/final_train.jsonl"
    final_dev = ROOT / "data/dataset/final_dev.jsonl"
    final_test = ROOT / "data/dataset/final_test.jsonl"

    if final_train.exists() or final_dev.exists() or final_test.exists():
        return {
            "train": read_jsonl(final_train),
            "dev": read_jsonl(final_dev),
            "test": read_jsonl(final_test),
        }

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
    return splits


def filter_rows_by_lang_filter(rows: List[Dict[str, Any]], lang_filter: str) -> List[Dict[str, Any]]:
    """
    lang_filter语义:
    - all:不过滤
    - bilingual:只保留zh/en
    - zh/en:只保留对应语言
    """
    if lang_filter == "all":
        return rows
    if lang_filter == "bilingual":
        return [r for r in rows if get_lang(r) in ("zh", "en")]
    if lang_filter in ("zh", "en"):
        return [r for r in rows if get_lang(r) == lang_filter]
    return rows


def group_by_language(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {"zh": [], "en": [], "unknown": []}
    for r in rows:
        out.setdefault(get_lang(r), []).append(r)
    return out


# -----------------------------
# main
# -----------------------------
def main():
    lang_filter = pick_lang_filter()  # zh|en|bilingual|all
    lang_split = get_lang_split_enabled()
    cfg = get_sft_cfg()

    sft_mode = os.environ.get("SFT_MODE", str(cfg.get("mode", "light"))).strip().lower()
    sft_fmt = os.environ.get("SFT_FORMAT", str(cfg.get("format", "text"))).strip().lower()
    sft_mode = sft_mode if sft_mode in ("light", "grounded") else "light"
    sft_fmt = sft_fmt if sft_fmt in ("text", "messages") else "text"

    splits = load_splits()
    out_dir = ROOT / "data/sft"
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = ".jsonl" if sft_fmt == "text" else ".messages.jsonl"

    total = 0
    for split, rows in splits.items():
        if not rows:
            continue

        rows = filter_rows_by_lang_filter(rows, lang_filter)

        if not rows:
            continue

        # 1)按语言拆分写(默认开启)
        if lang_split:
            grouped = group_by_language(rows)
            for lg, sub in grouped.items():
                if not sub:
                    continue
                sft_rows = [build_sft_row(s, sft_fmt, sft_mode, cfg) for s in sub]
                out_path = out_dir / f"{split}_{lg}{suffix}"
                write_jsonl(out_path, sft_rows)
                print(f"[Step08] split={split} lang={lg} mode={sft_mode} format={sft_fmt} count={len(sft_rows)} -> {out_path}")
                total += len(sft_rows)

        # 2)兼容输出:当你明确指定只训练某一种语言时,额外写一份split.jsonl和train.jsonl
        if lang_filter in ("zh", "en"):
            sft_rows = [build_sft_row(s, sft_fmt, sft_mode, cfg) for s in rows]
            compat_path = out_dir / f"{split}{suffix}"
            write_jsonl(compat_path, sft_rows)
            if split == "train":
                write_jsonl(out_dir / f"train.jsonl" if sft_fmt == "text" else out_dir / "train.messages.jsonl", sft_rows)
            print(f"[Step08] compat split={split} lang_filter={lang_filter} -> {compat_path}")

    print(f"[Step08] done.total={total},lang_filter={lang_filter},lang_split={int(lang_split)},mode={sft_mode},format={sft_fmt}")


if __name__ == "__main__":
    main()
