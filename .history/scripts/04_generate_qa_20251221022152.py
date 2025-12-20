#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step04
根据 rules + flows + repo_index 自动生成 QA 样本
"""

import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any

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


def generate_rule_qa(rule: Dict[str, Any], index_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    cid = rule["evidence_chunks"][0]
    ev = build_evidence(cid, index_map)

    q = f"{rule['title']}在代码中是如何实现的？"
    a = f"{rule['description']}具体实现可在对应代码逻辑中看到。"

    return {
        "sample_id": make_id("qa", rule["rule_id"]),
        "task_type": "qa",
        "language": "zh",
        "question": q,
        "answer": a,
        "evidence": [ev],
        "trace": {
            "type": "rule_based",
            "rule_ids": [rule["rule_id"]],
            "reasoning_steps": [
                f"定位到规则: {rule['title']}",
                f"检查关联代码块: {cid}",
                "根据代码逻辑总结规则实现方式"
            ]
        },
        "meta": {
            "domain": rule["domain"],
            "qa_type": "rule",
            "difficulty": "medium",
            "generator": "template_v1",
            "source": "AutoCodeDataPipeline"
        }
    }


def generate_flow_qa(flow: Dict[str, Any], index_map: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    evs = []
    for s in flow["steps"]:
        cid = s.get("evidence_chunk")
        if cid and cid in index_map:
            evs.append(build_evidence(cid, index_map))

    q = f"请说明{flow['name']}流程在系统中的执行步骤。"
    a = "该流程由多个步骤组成，涉及订单与库存等模块的协同处理。"

    return {
        "sample_id": make_id("qa", flow["flow_id"]),
        "task_type": "qa",
        "language": "zh",
        "question": q,
        "answer": a,
        "evidence": evs[:2],
        "trace": {
            "type": "flow_based",
            "flow_id": flow["flow_id"],
            "reasoning_steps": [
                "识别流程起点",
                "分析关键业务步骤",
                "结合代码顺序总结整体流程"
            ]
        },
        "meta": {
            "domain": flow["domain"],
            "qa_type": "flow",
            "difficulty": "hard",
            "generator": "template_v1",
            "source": "AutoCodeDataPipeline"
        }
    }


def main():
    index_rows = read_jsonl(ROOT / "data/raw_index/repo_index.jsonl")
    rules = read_jsonl(ROOT / "data/extracted/rules.jsonl")
    flows = read_jsonl(ROOT / "data/extracted/flows.jsonl")

    index_map = {r["chunk_id"]: r for r in index_rows}

    samples = []

    for r in rules[:60]:
        samples.append(generate_rule_qa(r, index_map))

    for f in flows:
        samples.append(generate_flow_qa(f, index_map))

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
