#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step06
后处理+验证:
- 校验QA与Design数据集的schema
- 校验evidence引用的chunk_id是否存在、行号字段是否齐全
- 基础去重(按question/requirement哈希)
- 统计覆盖率与质量指标,输出data/dataset/stats.json
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import Counter, defaultdict

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


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def must_have(obj: Dict[str, Any], keys: List[str], where: str) -> None:
    for k in keys:
        if k not in obj:
            raise ValueError(f"{where}缺少字段:{k}")


def validate_evidence_list(evs: List[Dict[str, Any]], index_map: Dict[str, Dict[str, Any]], where: str) -> Tuple[int, int]:
    """
    返回:(evidence条数,命中chunk_id条数)
    """
    if not isinstance(evs, list) or len(evs) == 0:
        raise ValueError(f"{where}evidence为空或不是list")
    hit = 0
    for i, ev in enumerate(evs):
        must_have(ev, ["chunk_id", "file_path", "start_line", "end_line", "content"], f"{where}.evidence[{i}]")
        cid = ev["chunk_id"]
        if cid in index_map:
            hit += 1
            # 基础一致性检查(不强制完全一致,但文件路径应相同)
            idx = index_map[cid]
            if idx.get("file_path") and ev.get("file_path") and idx["file_path"] != ev["file_path"]:
                raise ValueError(f"{where}证据file_path不一致:chunk_id={cid}")
    return len(evs), hit


def validate_qa_sample(s: Dict[str, Any], index_map: Dict[str, Dict[str, Any]]) -> None:
    must_have(s, ["sample_id", "task_type", "language", "question", "answer", "evidence", "trace", "meta"], f"QA:{s.get('sample_id')}")
    if s["task_type"] != "qa":
        raise ValueError(f"QA:{s.get('sample_id')}task_type不是qa")
    if not s["question"] or not s["answer"]:
        raise ValueError(f"QA:{s.get('sample_id')}question/answer为空")
    ev_cnt, hit = validate_evidence_list(s["evidence"], index_map, f"QA:{s.get('sample_id')}")
    # 允许少量evidence未命中index(比如你后续会做摘要),但至少要命中1条
    if hit < 1:
        raise ValueError(f"QA:{s.get('sample_id')}evidence的chunk_id未命中repo_index")
    # trace校验(可审计,不要求很长)
    tr = s["trace"]
    must_have(tr, ["type", "reasoning_steps"], f"QA:{s.get('sample_id')}.trace")
    if not isinstance(tr["reasoning_steps"], list) or len(tr["reasoning_steps"]) < 2:
        raise ValueError(f"QA:{s.get('sample_id')}trace.reasoning_steps过短")
    # meta校验
    mt = s["meta"]
    must_have(mt, ["domain", "qa_type", "difficulty", "generator", "source"], f"QA:{s.get('sample_id')}.meta")


def validate_design_sample(s: Dict[str, Any], index_map: Dict[str, Dict[str, Any]]) -> None:
    must_have(s, ["sample_id", "task_type", "language", "requirement", "repo_context", "design_output", "evidence", "trace", "meta"], f"Design:{s.get('sample_id')}")
    if s["task_type"] != "design":
        raise ValueError(f"Design:{s.get('sample_id')}task_type不是design")
    if not s["requirement"]:
        raise ValueError(f"Design:{s.get('sample_id')}requirement为空")
    ev_cnt, hit = validate_evidence_list(s["evidence"], index_map, f"Design:{s.get('sample_id')}")
    if hit < 1:
        raise ValueError(f"Design:{s.get('sample_id')}evidence的chunk_id未命中repo_index")
    tr = s["trace"]
    must_have(tr, ["reasoning_steps"], f"Design:{s.get('sample_id')}.trace")
    if not isinstance(tr["reasoning_steps"], list) or len(tr["reasoning_steps"]) < 3:
        raise ValueError(f"Design:{s.get('sample_id')}trace.reasoning_steps过短")
    mt = s["meta"]
    must_have(mt, ["domain", "difficulty", "generator", "source"], f"Design:{s.get('sample_id')}.meta")


def dedup_by_key(samples: List[Dict[str, Any]], key_fn) -> Tuple[List[Dict[str, Any]], int]:
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


def stats_for_qa(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    languages = Counter()
    domains = Counter()
    qa_types = Counter()
    diffs = Counter()
    evidence_cnt = []
    trace_len = []
    for s in samples:
        languages[s.get("language", "unknown")] += 1
        mt = s["meta"]
        domains[mt.get("domain", "unknown")] += 1
        qa_types[mt.get("qa_type", "unknown")] += 1
        diffs[mt.get("difficulty", "unknown")] += 1
        evidence_cnt.append(len(s.get("evidence", [])))
        trace_len.append(len(s.get("trace", {}).get("reasoning_steps", [])))
    return {
        "count": len(samples),
        "language_dist": dict(languages),
        "domain_dist": dict(domains),
        "qa_type_dist": dict(qa_types),
        "difficulty_dist": dict(diffs),
        "evidence_avg": round(sum(evidence_cnt) / max(1, len(evidence_cnt)), 3),
        "trace_steps_avg": round(sum(trace_len) / max(1, len(trace_len)), 3),
    }


def stats_for_design(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    languages = Counter()
    domains = Counter()
    diffs = Counter()
    evidence_cnt = []
    trace_len = []
    for s in samples:
        languages[s.get("language", "unknown")] += 1
        mt = s["meta"]
        domains[mt.get("domain", "unknown")] += 1
        diffs[mt.get("difficulty", "unknown")] += 1
        evidence_cnt.append(len(s.get("evidence", [])))
        trace_len.append(len(s.get("trace", {}).get("reasoning_steps", [])))
    return {
        "count": len(samples),
        "language_dist": dict(languages),
        "domain_dist": dict(domains),
        "difficulty_dist": dict(diffs),
        "evidence_avg": round(sum(evidence_cnt) / max(1, len(evidence_cnt)), 3),
        "trace_steps_avg": round(sum(trace_len) / max(1, len(trace_len)), 3),
    }



def main() -> None:
    # 1)加载repo_index用于证据校验
    index_rows = read_jsonl(ROOT / "data/raw_index/repo_index.jsonl")
    if not index_rows:
        raise FileNotFoundError("repo_index为空,请先运行Step01")
    index_map = {r["chunk_id"]: r for r in index_rows}

    # 2)加载数据集
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

    # 3)校验与去重
    for s in qa_all:
        validate_qa_sample(s, index_map)
    for s in design_all:
        validate_design_sample(s, index_map)

    #qa_all_dedup, qa_removed = dedup_by_key(qa_all, lambda x: sha1(x["question"].strip()))
    qa_all_dedup, qa_removed = dedup_by_key(
        qa_all,
        lambda x: sha1(
            x["question"].strip()
            + "|" + str(x.get("meta", {}).get("qa_type", "unknown"))
            + "|" + str(x.get("meta", {}).get("domain", "unknown"))
        )
    )
    design_all_dedup, design_removed = dedup_by_key(design_all, lambda x: sha1(x["requirement"].strip()))

    # 4)统计
    stats = {
        "generated_at": str(Path.cwd()),
        "repo_index_chunks": len(index_rows),
        "qa": {
            "before_dedup": len(qa_all),
            "after_dedup": len(qa_all_dedup),
            "removed": qa_removed,
            "metrics": stats_for_qa(qa_all_dedup),
        },
        "design": {
            "before_dedup": len(design_all),
            "after_dedup": len(design_all_dedup),
            "removed": design_removed,
            "metrics": stats_for_design(design_all_dedup),
        }
    }

    out_stats = ROOT / "data/dataset/stats.json"
    write_json(out_stats, stats)

    print("Step06验证通过")
    print(f"repo_index_chunks={len(index_rows)}")
    print(f"QA:before={len(qa_all)},after={len(qa_all_dedup)},removed={qa_removed}")
    print(f"Design:before={len(design_all)},after={len(design_all_dedup)},removed={design_removed}")
    print(f"统计输出:{out_stats}")


if __name__ == "__main__":
    main()
