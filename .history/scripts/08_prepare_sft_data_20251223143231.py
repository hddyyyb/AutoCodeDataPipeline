#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step08
将可审计样本整理为可用于微调的SFT数据。

适配你当前改造后的pipeline：
- 默认优先读取Step06输出: data/dataset/final_{train,dev,test}.jsonl
  每条为: {"text":..., "meta":...}，meta内含 evidence_snippets(code) + trace_digest
- 若final_*不存在，回退读取Step04/05输出(qa/design)并自动转换为{text,meta}
- 支持两种模式:
  - light: 只保留指令/回答(低token)
  - grounded: 在user侧附上Evidence(原文代码段)+Trace(推理步骤)，便于验收展示/可选训练
- 支持两种格式:
  - text: {"text": "...", "meta": {...}}
  - messages: {"messages":[...], "meta": {...}}
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
    return "zh"


def pick_lang_filter() -> str:
    """优先级: ENV LANG_FILTER > runtime.yaml.language.mode > zh"""
    env = os.environ.get("LANG_FILTER")
    if env:
        return normalize_lang(env) if env != "all" else "all"
    cfg = load_runtime_cfg()
    mode = ((cfg.get("language") or {}).get("mode")) or "zh"
    return normalize_lang(mode)


def get_sft_cfg() -> Dict[str, Any]:
    cfg = load_runtime_cfg()
    return cfg.get("sft") or {}


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


def build_meta(sample: Dict[str, Any], max_trace_steps: int = 6) -> Dict[str, Any]:
    """
    输出给SFT的meta：保留可回溯信息，但尽量轻量化。
    强制带:
    - task_type/language/domain/qa_type/difficulty
    - evidence(定位)
    - evidence_snippets(原文代码段，可截断由前序控制)
    - trace_digest(推理步骤)
    """
    m = get_meta(sample)

    # evidence(定位)
    ev_meta = []
    for ev in (m.get("evidence") or [])[:4]:
        ev_meta.append({
            "chunk_id": ev.get("chunk_id"),
            "file_path": ev.get("file_path"),
            "start_line": ev.get("start_line"),
            "end_line": ev.get("end_line"),
        })

    # evidence_snippets(原文代码段)
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
    # 额外保留sample_id方便定位
    if sample.get("sample_id"):
        out["sample_id"] = sample.get("sample_id")
    return out


def to_text_light(sample: Dict[str, Any]) -> str:
    """
    优先使用Step06/Step04/Step05已有的sample.text（建议你统一都产出text）。
    若缺失，才回退拼接旧字段。
    """
    if isinstance(sample.get("text"), str) and sample["text"].strip():
        return sample["text"].strip()

    # fallback legacy
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
    """
    grounded模式：user侧附上证据代码段+trace_digest，assistant输出保持原Response。
    证据来源：meta.evidence_snippets[].code（原文代码段）
    """
    m = get_meta(sample)
    lang = (m.get("language") or sample.get("language") or "zh")
    is_zh = lang == "zh"

    evidence_max = int(cfg.get("evidence_max", 2))
    evidence_max_lines = int(cfg.get("evidence_max_lines", 60))
    evidence_max_chars = int(cfg.get("evidence_max_chars", 4000))
    trace_digest_max = int(cfg.get("trace_digest_max", 6))

    # 从light里拿到基础的Instruction/Response结构，避免你重复维护两套拼接
    base = to_text_light(sample)

    # 尝试拆出Instruction和Response（按你统一格式“### Instruction\n...\n\n### Response\n...”）
    # 若拆不出来，就直接在前面加grounding块
    marker_i = "### Instruction"
    marker_r = "### Response"
    if marker_i in base and marker_r in base:
        # 粗糙但足够稳健：找到Response起点
        idx = base.find(marker_r)
        instr_part = base[:idx].strip()
        resp_part = base[idx:].strip()
    else:
        instr_part = f"### Instruction\n{base.strip()}"
        resp_part = ""

    title_e = "证据代码" if is_zh else "Evidence"
    title_t = "推理过程" if is_zh else "Trace"

    blocks = [instr_part]

    # Evidence snippets
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

    # Trace digest
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
    m = get_meta(sample)
    lang = (m.get("language") or sample.get("language") or "zh")
    is_zh = lang == "zh"

    system = "你是代码仓库问答与架构设计助手。回答必须基于给定代码仓的实现与约束。" if is_zh else \
             "You are a repo-grounded QA & design assistant. Answers must be grounded in the given repository."

    if mode == "grounded":
        user_text = to_text_grounded(sample, cfg)
        # grounded里 user_text已经包含Instruction块，这里直接作为user content即可
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user_text.strip()},
            {"role": "assistant", "content": ""},  # 由SFT格式决定是否需要空assistant；下游一般不需要
        ]

    # light: 使用Instruction/Response拆出user/assistant
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
        messages = to_messages(sample, mode, cfg)
        # 上面grounded messages里assistant留空不理想：我们更推荐messages格式时照样输出assistant
        # 所以这里修正：从light文本中提取assistant内容填回去
        if mode == "grounded":
            base = to_text_light(sample)
            marker_r = "### Response"
            assistant = ""
            if marker_r in base:
                assistant = base[base.find(marker_r) + len(marker_r):].strip()
            messages[-1]["content"] = assistant
        return {"messages": messages, "meta": meta}

    # text格式
    if mode == "grounded":
        text = to_text_grounded(sample, cfg)
    else:
        text = to_text_light(sample)
    return {"text": text, "meta": meta}


# -----------------------------
# Dataset loading
# -----------------------------
def load_splits() -> Dict[str, List[Dict[str, Any]]]:
    """
    优先使用Step06输出(final_*),否则回退Step04/05输出。
    """
    final_train = ROOT / "data/dataset/final_train.jsonl"
    final_dev = ROOT / "data/dataset/final_dev.jsonl"
    final_test = ROOT / "data/dataset/final_test.jsonl"

    if final_train.exists() or final_dev.exists() or final_test.exists():
        return {
            "train": read_jsonl(final_train),
            "dev": read_jsonl(final_dev),
            "test": read_jsonl(final_test),
        }

    # fallback: Step04/05
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


def filter_lang(rows: List[Dict[str, Any]], lang_filter: str) -> List[Dict[str, Any]]:
    if lang_filter == "all":
        return rows
    if lang_filter == "bilingual":
        return [r for r in rows if (get_meta(r).get("language") or r.get("language")) in ("zh", "en")]
    return [r for r in rows if (get_meta(r).get("language") or r.get("language")) == lang_filter]


# -----------------------------
# main
# -----------------------------
def main():
    lang_filter = pick_lang_filter()  # zh|en|bilingual|all
    cfg = get_sft_cfg()

    sft_mode = os.environ.get("SFT_MODE", str(cfg.get("mode", "light"))).strip().lower()
    sft_fmt = os.environ.get("SFT_FORMAT", str(cfg.get("format", "text"))).strip().lower()
    sft_mode = sft_mode if sft_mode in ("light", "grounded") else "light"
    sft_fmt = sft_fmt if sft_fmt in ("text", "messages") else "text"

    splits = load_splits()

    out_dir = ROOT / "data/sft"
    out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for split, rows in splits.items():
        rows = filter_lang(rows, lang_filter)
        if not rows:
            continue

        sft_rows = [build_sft_row(s, sft_fmt, sft_mode, cfg) for s in rows]

        out_path = out_dir / (f"{split}.jsonl" if sft_fmt == "text" else f"{split}.messages.jsonl")
        write_jsonl(out_path, sft_rows)

        # 兼容:把train再写一份train.jsonl
        if split == "train":
            compat = out_dir / "train.jsonl"
            write_jsonl(compat, sft_rows)

        print(f"[Step08] split={split} mode={sft_mode} format={sft_fmt} count={len(sft_rows)} -> {out_path}")
        total += len(sft_rows)

    print(f"[Step08] done.total={total},lang_filter={lang_filter}")


if __name__ == "__main__":
    main()
