#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step02
输入:data/raw_index/repo_index.jsonl
输出:data/extracted/domain_map.json

说明:
- 采用启发式规则,先把“订单+库存”相关代码块聚类成领域地图
- 不追求完美调用图,但要保证:可追溯(evidence_chunk_ids)且可扩展
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Any, Set


ROOT = Path(__file__).resolve().parents[1]


# 简单关键词表,用于order/stock二域判别
ORDER_KWS = [
    "order", "oms", "cart", "pay", "payment", "refund", "cancel",
    "receiver", "shipment", "delivery", "trade", "checkout"
]
STOCK_KWS = [
    "stock", "sku", "inventory", "lock", "deduct", "release",
    "ware", "warehouse", "store", "reserve"
]

# 分层边界启发式规则(路径+注解/类名线索)
BOUNDARY_RULES = {
    "controller": ["/controller/", "Controller.java", "@RestController", "@Controller", "RequestMapping", "GetMapping", "PostMapping"],
    "service": ["/service/", "Service.java", "@Service"],
    "mapper": ["/mapper/", "Mapper.java", "@Mapper", "BaseMapper", "MyBatis"],
    "model": ["/model/", "/domain/", "/dto/", "/vo/", "Model.java", "Dto.java", "VO.java"],
    "config": ["/config/", "Configuration", "@Configuration", "application.yml", "application.yaml", "bootstrap.yml"],
}

JAVA_CLASS_RE = re.compile(r"^\s*(public\s+)?(class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
JAVA_METHOD_RE = re.compile(r"^\s*(public|protected|private)\s+([A-Za-z0-9_<>\[\],\s]+)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)


@dataclass
class IndexRow:
    chunk_id: str
    file_path: str
    lang: str
    start_line: int
    end_line: int
    content: str
    symbol: str


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def norm_text(s: str) -> str:
    return s.lower()


def domain_score(text: str) -> Tuple[int, int]:
    t = norm_text(text)
    o = sum(1 for k in ORDER_KWS if k in t)
    s = sum(1 for k in STOCK_KWS if k in t)
    return o, s


def choose_domain(text: str) -> Tuple[str, float]:
    o, s = domain_score(text)
    if o == 0 and s == 0:
        return "other", 0.0
    if o > s:
        conf = o / max(1, o + s)
        return "order", round(conf, 3)
    if s > o:
        conf = s / max(1, o + s)
        return "stock", round(conf, 3)
    return "mixed", 0.5


def infer_boundary(file_path: str, content: str) -> str:
    '''
    把chunk按“架构层次”分桶
    :param file_path: chunk所在的文件路径
    :param content: chunk的源码文本
    :return: 该chunk属于哪一层(controller / service / mapper / model / config / other)
    '''
    fp = file_path.replace("\\", "/")
    t = content
    for b, signals in BOUNDARY_RULES.items():  # 遍历“分层边界规则表”
        for sig in signals:  # 对当前层，逐个尝试它的“识别信号”
            if sig in fp or sig in t:
                return b
    return "other"


def extract_class_names(content: str) -> List[str]:
    '''
    从一段源码文本中，用正则把“类名”抽取出来，作为后续分析的结构线索
    如果 content 是: public class OrderController {
    }

    class OrderParam {
    }
    返回结果就是: ["OrderController", "OrderParam"]

    :param content: 一整个文件或代码块的源码字符串
    :return: 这个源码里出现过的 Java 类名列表
    '''
    names = []
    for m in JAVA_CLASS_RE.finditer(content):
        names.append(m.group(3))
    return names


def extract_method_names(content: str) -> List[str]:
    '''
    从一段源码文本中，用正则把“方法名”抽取出来，作为后续分析的结构线索
    '''
    names = []
    for m in JAVA_METHOD_RE.finditer(content):
        # group(3)=methodName
        mn = m.group(3)
        if mn and len(mn) >= 3:
            names.append(mn)
    return names


def is_entity_name(name: str) -> bool:
    # 在大量名字中，筛出“更可能是领域对象”的那一小撮
    # 判断一个名字是不是“领域实体名候选”:包含Order/Stock/Sku/Inventory
    n = name.lower()
    entity_markers = ["order", "stock", "sku", "inventory", "ware", "warehouse", "refund", "payment", "cart"]
    return any(k in n for k in entity_markers)


def is_operation_name(name: str) -> bool:
    # 操作候选:动词风格,包含create/cancel/pay/lock/deduct/release等
    n = name.lower()
    op_markers = ["create", "submit", "place", "cancel", "pay", "callback", "refund",
                  "lock", "deduct", "release", "reserve", "confirm", "close", "update"]
    return any(k in n for k in op_markers)


def build_candidate_flows(ops: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    非严格调用图,用启发式将操作聚成常见流程骨架.
    """
    # 将操作按domain分桶
    by_domain = defaultdict(list)
    for op in ops:
        by_domain[op["domain"]].append(op)

    flows = []
    # 订单下单流程
    place_ops = [o for o in by_domain["order"] if any(k in o["name"].lower() for k in ["create", "submit", "place", "confirm"])]
    stock_lock_ops = [o for o in ops if any(k in o["name"].lower() for k in ["lock", "reserve"])]
    stock_deduct_ops = [o for o in ops if "deduct" in o["name"].lower() or "reduce" in o["name"].lower()]
    pay_ops = [o for o in by_domain["order"] if any(k in o["name"].lower() for k in ["pay", "callback", "notify"])]
    cancel_ops = [o for o in by_domain["order"] if "cancel" in o["name"].lower() or "close" in o["name"].lower()]
    stock_release_ops = [o for o in ops if "release" in o["name"].lower() or "unlock" in o["name"].lower()]

    def pick_one(cands: List[Dict[str, Any]]) -> Dict[str, Any] | None:
        if not cands:
            return None
        # 选confidence最高,再按证据最多
        cands = sorted(cands, key=lambda x: (x["confidence"], len(x["evidence_chunks"])), reverse=True)
        return cands[0]

    # place_order
    step_list = []
    p1 = pick_one(place_ops)
    p2 = pick_one(stock_lock_ops)
    p3 = pick_one(pay_ops)
    p4 = pick_one(stock_deduct_ops)
    if p1:
        step_list.append(("place_order", p1, "创建订单/提交订单"))
    if p2:
        step_list.append(("place_order", p2, "锁定库存/预占库存"))
    if p3:
        step_list.append(("place_order", p3, "支付回调/支付通知"))
    if p4:
        step_list.append(("place_order", p4, "扣减库存/确认扣减"))

    if len(step_list) >= 2:
        flows.append(_format_flow("place_order", "place_order", "mixed", step_list))

    # cancel_and_release
    step_list = []
    c1 = pick_one(cancel_ops)
    r1 = pick_one(stock_release_ops)
    if c1:
        step_list.append(("cancel_and_release", c1, "取消/关闭订单"))
    if r1:
        step_list.append(("cancel_and_release", r1, "释放库存/解锁库存"))
    if len(step_list) >= 2:
        flows.append(_format_flow("cancel_and_release", "cancel_and_release", "mixed", step_list))

    # pay_callback
    if p3:
        flows.append(_format_flow("pay_callback", "pay_callback", p3["domain"], [("pay_callback", p3, "处理支付回调并驱动状态流转")]))

    return flows


def _format_flow(flow_id: str, name: str, domain: str, step_list: List[Tuple[str, Dict[str, Any], str]]) -> Dict[str, Any]:
    steps = []
    ev = set()
    for idx, (_, op, why) in enumerate(step_list, start=1):
        # evidence_chunk选该op的第一个证据即可
        e = op["evidence_chunks"][0] if op["evidence_chunks"] else ""
        if e:
            ev.add(e)
        steps.append({
            "idx": idx,
            "operation": op["name"],
            "evidence_chunk": e,
            "why": why
        })
    return {
        "flow_id": flow_id,
        "name": name,
        "domain": domain,
        "steps": steps,
        "evidence_chunks": sorted(list(ev))
    }


def main() -> None:
    index_path = ROOT / "data/raw_index/repo_index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError("找不到data/raw_index/repo_index.jsonl,请先运行scripts/01_index_repo.py")

    rows_raw = read_jsonl(index_path)
    rows: List[IndexRow] = []
    for r in rows_raw:
        rows.append(IndexRow(
            chunk_id=r["chunk_id"],
            file_path=r["file_path"],
            lang=r.get("lang", "text"),
            start_line=int(r["start_line"]),
            end_line=int(r["end_line"]),
            content=r["content"],
            symbol=r.get("symbol", "") or ""
        ))

    # boundaries:chunk_id列表
    boundaries: Dict[str, List[str]] = {k: [] for k in list(BOUNDARY_RULES.keys()) + ["other"]}
    # entities/op聚合: name->{evidence_chunks,domain_counter,mentions}
    entity_evs: Dict[str, Set[str]] = defaultdict(set)
    entity_domains: Dict[str, Counter] = defaultdict(Counter)
    entity_mentions: Dict[str, Counter] = defaultdict(Counter)

    op_evs: Dict[str, Set[str]] = defaultdict(set)
    op_domains: Dict[str, Counter] = defaultdict(Counter)
    op_signals: Dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        text = row.file_path + "\n" + row.content + "\n" + (row.symbol or "")
        domain, conf = choose_domain(text)  # 领域判别（order / stock / mixed）, 方法：关键词投票
        boundary = infer_boundary(row.file_path, row.content)
        boundaries.setdefault(boundary, []).append(row.chunk_id)

        if row.lang == "java":
            class_names = extract_class_names(row.content)
            method_names = extract_method_names(row.content)
        else:
            class_names = []
            method_names = []

        # 实体
        for cn in class_names:
            if not is_entity_name(cn):  # cn 基本等价于Order / Stock / Sku / Inventory / Refund 等实体
                continue
            entity_evs[cn].add(row.chunk_id)
            entity_domains[cn][domain] += 1
            entity_mentions[cn][row.file_path] += 1
        '''生成 QA 时能回答
        “Order 实体相关的代码证据有哪些？”

        保证可追溯性（traceability）'''

        # 操作
        for mn in method_names:
            if not is_operation_name(mn):
                continue
            op_evs[mn].add(row.chunk_id)
            op_domains[mn][domain] += 1
            # signals:命中哪些关键词
            lt = norm_text(row.content)
            for k in ORDER_KWS + STOCK_KWS:
                if k in lt:
                    op_signals[mn][k] += 1

    def finalize_items(
        evs: Dict[str, Set[str]],
        domains: Dict[str, Counter],
        mentions: Dict[str, Counter],
        item_type: str
    ) -> List[Dict[str, Any]]:
        items = []
        for name, evset in evs.items():
            dom_cnt = domains[name]
            total = sum(dom_cnt.values()) if dom_cnt else 0
            if total == 0:
                dom = "other"
                conf = 0.0
            else:
                dom, domv = dom_cnt.most_common(1)[0]
                conf = domv / total
            item = {
                "name": name,
                "domain": dom,
                "confidence": round(conf, 3),
                "evidence_chunks": sorted(list(evset)),
            }
            if item_type == "entity":
                # mentions:最常出现的文件top5
                top_files = [fp for fp, _ in mentions[name].most_common(5)]
                item["mentions"] = top_files
            items.append(item)

        # 按confidence+证据数排序
        items.sort(key=lambda x: (x["confidence"], len(x["evidence_chunks"]), x["name"]), reverse=True)
        return items

    entities = finalize_items(entity_evs, entity_domains, entity_mentions, "entity")

    # 操作items单独组装signals
    operations = []
    for name, evset in op_evs.items():
        dom_cnt = op_domains[name]
        total = sum(dom_cnt.values()) if dom_cnt else 0
        if total == 0:
            dom = "other"
            conf = 0.0
        else:
            dom, domv = dom_cnt.most_common(1)[0]
            conf = domv / total
        sigs = [k for k, _ in op_signals[name].most_common(8)]
        operations.append({
            "name": name,
            "domain": dom,
            "confidence": round(conf, 3),
            "evidence_chunks": sorted(list(evset)),
            "signals": sigs
        })
    operations.sort(key=lambda x: (x["confidence"], len(x["evidence_chunks"]), x["name"]), reverse=True)

    candidate_flows = build_candidate_flows(operations)

    out = {
        "repo": {
            "repo_path": "repo/mall",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "source_index": "data/raw_index/repo_index.jsonl"
        },
        "boundaries": boundaries,
        "entities": entities[:200],
        "operations": operations[:300],
        "candidate_flows": candidate_flows
    }

    out_path = ROOT / "data/extracted/domain_map.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 同时生成一个sample便于提交
    sample_path = ROOT / "data/samples/domain_map_sample.json"
    sample = dict(out)
    sample["entities"] = out["entities"][:30]
    sample["operations"] = out["operations"][:40]
    sample_path.write_text(json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")

    print("完成domain_map构建")
    print(f"输出:{out_path}")
    print(f"样例:{sample_path}")
    print(f"entities={len(out['entities'])},operations={len(out['operations'])},flows={len(out['candidate_flows'])}")


if __name__ == "__main__":
    main()
