#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step06
后处理+验证(适配Step04/05新格式):
- 支持样本同时具备 legacy字段(sample_id/question/answer/...) 和 finetune字段(text/meta_v2)
- 强制校验:meta_v2.evidence_snippets必须存在且包含原文code
- evidence校验:允许evidence项只有(content)或(code)之一
- 去重:优先按(text)去重,其次按(question/requirement)去重,并加入(qa_type/domain)防止误杀
- 统计覆盖率与质量指标,输出data/dataset/stats.json
- 生成最终微调数据集(统一为{text,meta})到data/dataset/final_{train,dev,test}.jsonl
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Callable
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def must_have(obj: Dict[str, Any], keys: List[str], where: str) -> None:
    for k in keys:
        if k not in obj:
            raise ValueError(f"{where}缺少字段:{k}")


def _has_any(ev: Dict[str, Any], keys: List[str]) -> bool:
    for k in keys:
        if ev.get(k):
            return True
    return False


def validate_evidence_list(evs: List[Dict[str, Any]], index_map: Dict[str, Dict[str, Any]], where: str) -> Tuple[int, int]:
    """
    返回:(evidence条数,命中chunk_id条数)
    兼容两种证据结构:
    - legacy:要求content存在
    - new:允许只有code(code_lang)+定位信息
    """
    if not isinstance(evs, list) or len(evs) == 0:
        raise ValueError(f"{where}evidence为空或不是list")
    hit = 0
    for i, ev in enumerate(evs):
        must_have(ev, ["chunk_id", "file_path", "start_line", "end_line"], f"{where}.evidence[{i}]")
        if not _has_any(ev, ["content", "code"]):
            raise ValueError(f"{where}.evidence[{i}]缺少content/code(至少一个)")
        cid = ev["chunk_id"]
        if cid in index_map:
            hit += 1
            idx = index_map[cid]
            if idx.get("file_path") and ev.get("file_path") and idx["file_path"] != ev["file_path"]:
                raise ValueError(f"{where}证据file_path不一致:chunk_id={cid}")
    return len(evs), hit


def validate_evidence_snippets(meta_v2: Dict[str, Any], index_map: Dict[str, Dict[str, Any]], where: str) -> Tuple[int, int, int]:
    """
    强制校验原文代码段:
    meta_v2.evidence_snippets: [{chunk_id,file_path,start_line,end_line,code_lang,code}]
    返回:(snippets条数,命中chunk_id条数,code总字符数)
    """
    snips = meta_v2.get("evidence_snippets")
    if not isinstance(snips, list) or len(snips) == 0:
        raise ValueError(f"{where}.meta_v2缺少evidence_snippets或为空(必须包含原文代码段)")
    hit = 0
    code_chars = 0
    for i, ev in enumerate(snips):
        must_have(ev, ["chunk_id", "file_path", "start_line", "end_line", "code"], f"{where}.meta_v2.evidence_snippets[{i}]")
        if not ev.get("code"):
            raise ValueError(f"{where}.meta_v2.evidence_snippets[{i}]code为空")
        code_chars += len(ev.get("code", ""))
        cid = ev["chunk_id"]
        if cid in index_map:
            hit += 1
            idx = index_map[cid]
            if idx.get("file_path") and ev.get("file_path") and idx["file_path"] != ev["file_path"]:
                raise ValueError(f"{where}.meta_v2.evidence_snippetsfile_path不一致:chunk_id={cid}")
    return len(snips), hit, code_chars


def get_meta_v2(s: Dict[str, Any]) -> Dict[str, Any]:
    mv2 = s.get("meta_v2")
    if isinstance(mv2, dict):
        return mv2
    # 兼容你最终可能已经转换成{text,meta}
    mt = s.get("meta")
    if isinstance(mt, dict) and mt.get("task_type") in ("qa", "design"):
        return mt
    return {}


def validate_qa_sample(s: Dict[str, Any], index_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    where = f"QA:{s.get('sample_id') or s.get('id') or 'unknown'}"

    # 基本字段(兼容legacy与finetune)
    must_have(s, ["task_type", "language"], where)
    if s["task_type"] != "qa":
        raise ValueError(f"{where}task_type不是qa")

    # 至少要有一种可训练内容: text 或 question/answer
    has_text = bool(s.get("text"))
    has_qa = bool(s.get("question")) and bool(s.get("answer"))
    if (not has_text) and (not has_qa):
        raise ValueError(f"{where}缺少text或question/answer")

    # evidence与trace(legacy路径)
    if "evidence" in s:
        ev_cnt, hit = validate_evidence_list(s["evidence"], index_map, where)
        if hit < 1:
            raise ValueError(f"{where}evidence的chunk_id未命中repo_index")
    if "trace" in s:
        tr = s["trace"]
        if not isinstance(tr, dict):
            raise ValueError(f"{where}.trace不是dict")
        if "reasoning_steps" in tr:
            if not isinstance(tr["reasoning_steps"], list) or len(tr["reasoning_steps"]) < 2:
                raise ValueError(f"{where}trace.reasoning_steps过短(<2)")

    # meta_v2强制原文代码段
    mv2 = get_meta_v2(s)
    must_have(mv2, ["task_type", "language"], f"{where}.meta_v2")
    if mv2.get("task_type") != "qa":
        raise ValueError(f"{where}.meta_v2.task_type不是qa")
    snip_cnt, snip_hit, code_chars = validate_evidence_snippets(mv2, index_map, where)

    # trace_digest也校验一下(更贴合你新结构)
    td = mv2.get("trace_digest")
    if not isinstance(td, list) or len(td) < 2:
        raise ValueError(f"{where}.meta_v2.trace_digest过短(<2)")

    return {
        "snip_cnt": snip_cnt,
        "snip_hit": snip_hit,
        "code_chars": code_chars,
    }


def validate_design_sample(s: Dict[str, Any], index_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    where = f"Design:{s.get('sample_id') or s.get('id') or 'unknown'}"

    must_have(s, ["task_type", "language"], where)
    if s["task_type"] != "design":
        raise ValueError(f"{where}task_type不是design")

    has_text = bool(s.get("text"))
    has_req = bool(s.get("requirement")) and bool(s.get("design_output"))
    if (not has_text) and (not has_req):
        raise ValueError(f"{where}缺少text或(requirement/design_output)")

    # legacy evidence/trace可选校验(如果存在)
    if "evidence" in s:
        ev_cnt, hit = validate_evidence_list(s["evidence"], index_map, where)
        if hit < 1:
            raise ValueError(f"{where}evidence的chunk_id未命中repo_index")
    if "trace" in s:
        tr = s["trace"]
        if not isinstance(tr, dict):
            raise ValueError(f"{where}.trace不是dict")
        rs = tr.get("reasoning_steps")
        if rs is not None and (not isinstance(rs, list) or len(rs) < 3):
            raise ValueError(f"{where}trace.reasoning_steps过短(<3)")

    mv2 = get_meta_v2(s)
    must_have(mv2, ["task_type", "language"], f"{where}.meta_v2")
    if mv2.get("task_type") != "design":
        raise ValueError(f"{where}.meta_v2.task_type不是design")
    snip_cnt, snip_hit, code_chars = validate_evidence_snippets(mv2, index_map, where)

    td = mv2.get("trace_digest")
    if not isinstance(td, list) or len(td) < 3:
        raise ValueError(f"{where}.meta_v2.trace_digest过短(<3)")

    return {
        "snip_cnt": snip_cnt,
        "snip_hit": snip_hit,
        "code_chars": code_chars,
    }


def dedup_by_key(samples: List[Dict[str, Any]], key_fn: Callable[[Dict[str, Any]], str]) -> Tuple[List[Dict[str, Any]], int]:
    seen = set()
    out = []
    removed = 0
    for s in samples:
        k = key_fn(s)
        if k in seen:
            removed += 1
            continue
        seen.add(k)
        out.append(s)
    return out, removed


def to_finetune_row(s: Dict[str, Any]) -> Dict[str, Any]:
    """
    统一导出为{text,meta}供微调/提交:
    - text:优先使用s.text
    - meta:优先使用meta_v2,否则回退到legacy meta
    """
    text = s.get("text")
    mv2 = get_meta_v2(s)
    if not text:
        # 兜底拼接
        if s.get("task_type") == "qa":
            text = f"### Instruction\n{s.get('question','')}\n\n### Response\n{s.get('answer','')}\n"
        else:
            text = f"### Instruction\n{s.get('requirement','')}\n\n### Response\n{json.dumps(s.get('design_output',{}),ensure_ascii=False)}\n"
    meta = mv2 if mv2 else (s.get("meta") or {})
    return {"text": text, "meta": meta}


def stats_for_samples(samples: List[Dict[str, Any]], kind: str) -> Dict[str, Any]:
    languages = Counter()
    domains = Counter()
    types = Counter()
    diffs = Counter()
    snip_cnts = []
    code_chars = []
    trace_lens = []

    for s in samples:
        mv2 = get_meta_v2(s)
        languages[(mv2.get("language") or s.get("language") or "unknown")] += 1
        domains[(mv2.get("domain") or (s.get("meta") or {}).get("domain") or "unknown")] += 1
        if kind == "qa":
            types[(mv2.get("qa_type") or (s.get("meta") or {}).get("qa_type") or "unknown")] += 1
        diffs[(mv2.get("difficulty") or (s.get("meta") or {}).get("difficulty") or "unknown")] += 1

        # snippet统计
        snips = mv2.get("evidence_snippets") or []
        snip_cnts.append(len(snips))
        code_chars.append(sum(len(x.get("code", "")) for x in snips))

        # trace统计
        td = mv2.get("trace_digest") or []
        trace_lens.append(len(td) if isinstance(td, list) else 0)

    base = {
        "count": len(samples),
        "language_dist": dict(languages),
        "domain_dist": dict(domains),
        "difficulty_dist": dict(diffs),
        "evidence_snippets_avg": round(sum(snip_cnts) / max(1, len(snip_cnts)), 3),
        "code_chars_avg": round(sum(code_chars) / max(1, len(code_chars)), 3),
        "trace_steps_avg": round(sum(trace_lens) / max(1, len(trace_lens)), 3),
        "snippet_coverage_rate": round(sum(1 for x in snip_cnts if x > 0) / max(1, len(snip_cnts)), 3),
    }
    if kind == "qa":
        base["qa_type_dist"] = dict(types)
    return base


def main() -> None:
    # 1)加载repo_index用于证据校验
    index_rows = read_jsonl(ROOT / "data/raw_index/repo_index.jsonl")
    if not index_rows:
        raise FileNotFoundError("repo_index为空,请先运行Step01")
    index_map = {r["chunk_id"]: r for r in index_rows}

    # 2)加载数据集(来自Step04/05)
    qa_train = read_jsonl(ROOT / "data/dataset/train.jsonl")
    qa_dev = read_jsonl(ROOT / "data/dataset/dev.jsonl")
    qa_test = read_jsonl(ROOT / "data/dataset/test.jsonl")
    qa_all = qa_train + qa_dev + qa_test

    design_train = read_jsonl(ROOT / "data/dataset/design_train.jsonl")
    design_dev = read_jsonl(ROOT / "data/dataset/design_dev.jsonl")
    design_test = read_jsonl(ROOT / "data/dataset/design_test.jsonl")
    design_all = design_train + design_dev + design_test

    if not qa_all:
        raise FileNotFoundError("QA数据集为空,请先运行Step04")
    if not design_all:
        raise FileNotFoundError("Design数据集为空,请先运行Step05")

    # 3)校验
    qa_val = []
    for s in qa_all:
        qa_val.append(validate_qa_sample(s, index_map))
    d_val = []
    for s in design_all:
        d_val.append(validate_design_sample(s, index_map))

    # 4)去重
    # QA:优先按text去重,否则按(question+qa_type+domain)
    def qa_key(x: Dict[str, Any]) -> str:
        mv2 = get_meta_v2(x)
        if x.get("text"):
            return sha1(x["text"].strip())
        return sha1(
            (x.get("question", "").strip())
            + "|" + str(mv2.get("qa_type", "unknown"))
            + "|" + str(mv2.get("domain", "unknown"))
        )

    def design_key(x: Dict[str, Any]) -> str:
        mv2 = get_meta_v2(x)
        if x.get("text"):
            return sha1(x["text"].strip())
        return sha1((x.get("requirement", "").strip()) + "|" + str(mv2.get("domain", "unknown")))

    qa_all_dedup, qa_removed = dedup_by_key(qa_all, qa_key)
    design_all_dedup, design_removed = dedup_by_key(design_all, design_key)

    # 5)生成最终finetune数据集(合并qa+design,统一{text,meta})
    final_all = [to_finetune_row(x) for x in (qa_all_dedup + design_all_dedup)]
    # 最终再去重一次(按text)
    final_all_dedup, final_removed = dedup_by_key(final_all, lambda x: sha1(x["text"].strip()))
    import random
    random.shuffle(final_all_dedup)

    n = len(final_all_dedup)
    final_train = final_all_dedup[: int(n * 0.8)]
    final_dev = final_all_dedup[int(n * 0.8): int(n * 0.9)]
    final_test = final_all_dedup[int(n * 0.9):]

    write_jsonl(ROOT / "data/dataset/final_train.jsonl", final_train)
    write_jsonl(ROOT / "data/dataset/final_dev.jsonl", final_dev)
    write_jsonl(ROOT / "data/dataset/final_test.jsonl", final_test)

    # 6)统计
    stats = {
        "repo_index_chunks": len(index_rows),
        "qa": {
            "before_dedup": len(qa_all),
            "after_dedup": len(qa_all_dedup),
            "removed": qa_removed,
            "metrics": stats_for_samples(qa_all_dedup, "qa"),
        },
        "design": {
            "before_dedup": len(design_all),
            "after_dedup": len(design_all_dedup),
            "removed": design_removed,
            "metrics": stats_for_samples(design_all_dedup, "design"),
        },
        "final_finetune": {
            "before_dedup": len(final_all),
            "after_dedup": len(final_all_dedup),
            "removed": final_removed,
            "splits": {"train": len(final_train), "dev": len(final_dev), "test": len(final_test)},
        }
    }

    out_stats = ROOT / "data/dataset/stats.json"
    write_json(out_stats, stats)

    print("Step06验证通过")
    print(f"repo_index_chunks={len(index_rows)}")
    print(f"QA:before={len(qa_all)},after={len(qa_all_dedup)},removed={qa_removed}")
    print(f"Design:before={len(design_all)},after={len(design_all_dedup)},removed={design_removed}")
    print(f"FinalFinetune:before={len(final_all)},after={len(final_all_dedup)},removed={final_removed}")
    print(f"统计输出:{out_stats}")


if __name__ == "__main__":
    main()
