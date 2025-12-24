#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step03
输入:
- data/raw_index/repo_index.jsonl
- data/extracted/domain_map.json

输出:
- data/extracted/rules.jsonl
- data/extracted/flows.jsonl
- data/samples/rules_sample.jsonl
- data/samples/flows_sample.jsonl

说明:
- 采用启发式规则抽取,强调grounding(每条规则/流程都带evidence_chunks)
- Step03负责把“可引用的规则与流程片段”结构化
- 本版本在最小改动基础上,为rule/flow补充英文title/description/name字段,用于bilingual训练
"""

from __future__ import annotations

import json
import re
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Tuple, Iterable, Set
from collections import defaultdict


ROOT = Path(__file__).resolve().parents[1]


RULE_PATTERNS = [
    ("order_status_guard", re.compile(r"\b(status|Status|orderStatus|OrderStatus)\b.*(==|!=|equals)\b", re.I), ["order", "status"]),
    ("state_transition", re.compile(r"(change|update).*(status|Status)|set.*Status", re.I), ["order", "status"]),
    ("stock_lock", re.compile(r"\b(lock|reserve)\w*\b.*\b(stock|inventory|sku)\b|\b(stock|inventory)\b.*\b(lock|reserve)\w*\b", re.I), ["stock", "lock"]),
    ("stock_deduct", re.compile(r"\b(deduct|reduce|decrease)\w*\b.*\b(stock|inventory|sku)\b|\b(stock|inventory)\b.*\b(deduct|reduce|decrease)\w*\b", re.I), ["stock", "deduct"]),
    ("stock_release", re.compile(r"\b(release|unlock)\w*\b.*\b(stock|inventory|sku)\b|\b(stock|inventory)\b.*\b(release|unlock)\w*\b", re.I), ["stock", "release"]),
    ("idempotency", re.compile(r"\b(idempotent|幂等|duplicate|重复)\b|requestId|uniqueKey|nonce", re.I), ["idempotency"]),
    ("transaction", re.compile(r"@Transactional|\btransaction\b|\brollback\b", re.I), ["transaction"]),
    ("exception", re.compile(r"throw\s+new|RuntimeException|BusinessException|IllegalArgumentException|return\s+false", re.I), ["exception"]),
    ("concurrency", re.compile(r"\block\b|synchronized|redis.*lock|redisson|select\s+for\s+update", re.I), ["concurrency"]),
    ("timeout_close", re.compile(r"\btimeout\b|closeOrder|cancelOrder|auto.*close", re.I), ["order", "timeout"]),
]


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")

def has_cjk(s: str) -> bool:
    return bool(s and _CJK_RE.search(s))


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_domain_map(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_domain_from_tags(tags: List[str]) -> str:
    t = set(tags)
    if "stock" in t:
        return "stock"
    if "order" in t:
        return "order"
    return "mixed"


TITLE_EN_MAP = {
    "order_status_guard": "Order status guard and validation",
    "state_transition": "Order status transition update",
    "stock_lock": "Stock lock/reservation rule",
    "stock_deduct": "Stock deduction rule",
    "stock_release": "Stock release/unlock rule",
    "idempotency": "Idempotency and duplicate request handling",
    "transaction": "Transactional consistency and rollback",
    "exception": "Exception handling and failure branches",
    "concurrency": "Concurrency control and locking",
    "timeout_close": "Timeout close and cancellation",
}

# 业务步骤中文标签 -> 英文(用于flow step对齐;不覆盖name_en(方法名拆词))
BIZ_STEP_ZH_TO_EN = {
    "创建订单/提交订单": "Create/submit order",
    "锁定库存/预占库存": "Lock/reserve stock",
    "支付回调/支付通知": "Payment callback/notification",
    "扣减库存/确认扣减": "Deduct/confirm stock",
    "取消/关闭订单": "Cancel/close order",
    "释放库存/解锁库存": "Release/unlock stock",
    "处理支付回调并驱动状态流转": "Handle payment callback and drive status transition",
}

def biz_step_en(zh: str) -> str:
    zh = (zh or "").strip()
    if not zh:
        return ""
    parts = [p.strip() for p in re.split(r"[\/]+", zh) if p.strip()]
    mapped = []
    for p in parts:
        mapped.append(BIZ_STEP_ZH_TO_EN.get(p, BIZ_STEP_ZH_TO_EN.get(zh, "")))
    mapped = [m for m in mapped if m]
    if mapped:
        seen = set()
        out = []
        for m in mapped:
            if m not in seen:
                out.append(m)
                seen.add(m)
        return " / ".join(out)
    return ""


def title_from_types_zh(types: List[str]) -> str:
    mapping = {
        "order_status_guard": "订单状态约束校验",
        "state_transition": "订单状态流转更新",
        "stock_lock": "库存锁定规则",
        "stock_deduct": "库存扣减规则",
        "stock_release": "库存释放规则",
        "idempotency": "幂等/重复请求处理",
        "transaction": "事务一致性与回滚",
        "exception": "异常处理与失败分支",
        "concurrency": "并发控制与锁",
        "timeout_close": "超时关单与取消",
    }
    parts = [mapping.get(t, t) for t in types]
    return " / ".join(parts)

def title_from_types_en(types: List[str]) -> str:
    parts = [TITLE_EN_MAP.get(t, t) for t in types]
    return " / ".join(parts)

def desc_from_types_zh(types: List[str], tags: List[str]) -> str:
    segs = []
    if "order_status_guard" in types or "state_transition" in types:
        segs.append("涉及订单状态判断或状态更新,需确保状态机约束被满足。")
    if "stock_lock" in types:
        segs.append("涉及库存锁定/预占,需避免超卖并为后续扣减提供保障。")
    if "stock_deduct" in types:
        segs.append("涉及库存扣减,通常需要在支付成功或确认阶段执行,并处理失败回滚。")
    if "stock_release" in types:
        segs.append("涉及库存释放/解锁,常发生在取消/超时关闭等路径,防止库存长期占用。")
    if "transaction" in types:
        segs.append("存在事务边界,需关注原子性与异常回滚。")
    if "idempotency" in types:
        segs.append("存在幂等或重复请求场景,需避免多次扣减/多次状态更新。")
    if "concurrency" in types:
        segs.append("存在并发控制信号,需关注锁粒度与一致性。")
    if "exception" in types:
        segs.append("包含异常/失败分支,需明确失败后的补偿或返回语义。")
    return "".join(segs) if segs else f"命中规则信号tags={tags}"

def desc_from_types_en(types: List[str], tags: List[str]) -> str:
    segs = []
    if "order_status_guard" in types or "state_transition" in types:
        segs.append("Involves order status checks or updates; ensure state-machine constraints are enforced.")
    if "stock_lock" in types:
        segs.append("Involves stock locking/reservation to prevent overselling and prepare for later deduction.")
    if "stock_deduct" in types:
        segs.append("Involves stock deduction, usually after payment/confirmation, and should handle rollback on failure.")
    if "stock_release" in types:
        segs.append("Involves releasing/unlocking stock, commonly on cancel/timeout paths to avoid long-term reservation.")
    if "transaction" in types:
        segs.append("Contains transaction boundaries; pay attention to atomicity and rollback behavior.")
    if "idempotency" in types:
        segs.append("Has idempotency/duplicate request scenarios; avoid repeated deduction or repeated state updates.")
    if "concurrency" in types:
        segs.append("Shows concurrency-control signals; consider lock granularity and consistency.")
    if "exception" in types:
        segs.append("Contains exception/failure branches; clarify compensation or return semantics on failure.")
    return " ".join(segs) if segs else f"Matched rule signals tags={tags}"

_CAMEL_SPLIT = re.compile(r"(?<!^)(?=[A-Z])")

def split_camel(s: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", " ", s or "").strip()
    if not s:
        return ""
    toks: List[str] = []
    for tok in s.split():
        toks.extend(_CAMEL_SPLIT.split(tok))
    toks = [t.lower() for t in toks if t]
    return " ".join(toks).strip()

def flow_name_en_from_zh_or_ops(flow: Dict[str, Any]) -> str:
    name = flow.get("name") or ""
    if name and not has_cjk(name):
        return name.strip()
    ops = []
    for s in flow.get("steps", []) or []:
        op = s.get("op") or ""
        if op:
            ops.append(split_camel(op))
        if len(ops) >= 3:
            break
    dom = (flow.get("domain") or "business").strip()
    if ops:
        return f"{dom.capitalize()} flow: " + " -> ".join([o for o in ops if o])
    fid = flow.get("flow_id") or ""
    return f"{dom.capitalize()} flow({fid[:8]})" if fid else f"{dom.capitalize()} flow"


def build_flow_records(domain_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    flows = []
    for f in domain_map.get("candidate_flows", []):
        flow_id = f.get("flow_id") or sha1_text(f.get("name", ""))[:12]
        steps = f.get("steps", [])
        ev = f.get("evidence_chunks", [])
        domain = f.get("domain", "mixed")

        flow_obj = {
            "flow_id": flow_id,
            "name": f.get("name", flow_id),
            "name_en": "",
            "domain": domain,
            "steps": steps,
            "evidence_chunks": ev,
            "meta": {
                "source": "domain_map.candidate_flows",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
        }

        flow_obj["name_en"] = flow_name_en_from_zh_or_ops(flow_obj)

        for s in flow_obj.get("steps", []) or []:
            op = s.get("op") or s.get("operation") or ""
            if op and ("op" not in s or not s.get("op")):
                s["op"] = op

            if not s.get("name"):
                s["name"] = s.get("why") or op or f"step{s.get('idx') or s.get('index') or ''}"

            # 业务语义英文对齐:不影响中文训练,仅新增why_en字段
            if s.get("why") and not s.get("why_en"):
                s["why_en"] = biz_step_en(s.get("why"))
            if (not s.get("why_en")) and s.get("name") and has_cjk(s.get("name")):
                s["why_en"] = biz_step_en(s.get("name"))

            # 方法名英文可读化(保持原策略)
            if not s.get("name_en"):
                if op:
                    s["name_en"] = split_camel(op) or op
                else:
                    s["name_en"] = "step"

        flows.append(flow_obj)
    return flows


def collect_rule_candidates(index_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    candidates = []
    for r in index_rows:
        content = r.get("content", "")
        fp = r.get("file_path", "")
        if not content:
            continue

        tags_hit: List[str] = []
        types_hit: List[str] = []
        for rule_type, pat, tags in RULE_PATTERNS:
            if pat.search(content):
                types_hit.append(rule_type)
                tags_hit.extend(tags)

        if not types_hit:
            continue

        signature = sha1_text(fp + "|" + "|".join(sorted(set(types_hit))))[:12]
        candidates.append({
            "signature": signature,
            "file_path": fp,
            "chunk_id": r["chunk_id"],
            "types": sorted(set(types_hit)),
            "tags": sorted(set(tags_hit)),
        })
    return candidates


def aggregate_rules(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    buckets: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for c in cands:
        key = (c["signature"], "|".join(c["types"]))
        if key not in buckets:
            rid = sha1_text("rule|" + c["signature"] + "|" + "|".join(c["types"]))[:12]
            types = c["types"]
            tags = c["tags"]
            buckets[key] = {
                "rule_id": rid,
                "title": title_from_types_zh(types),
                "title_en": title_from_types_en(types),
                "description": desc_from_types_zh(types, tags),
                "description_en": desc_from_types_en(types, tags),
                "domain": infer_domain_from_tags(tags),
                "types": types,
                "tags": tags,
                "evidence_chunks": set(),
                "evidence_files": set(),
                "meta": {
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "generator": "heuristic_rule_scanner_v1"
                }
            }
        buckets[key]["evidence_chunks"].add(c["chunk_id"])
        buckets[key]["evidence_files"].add(c["file_path"])

    rules = []
    for _, obj in buckets.items():
        rules.append({
            "rule_id": obj["rule_id"],
            "title": obj["title"],
            "title_en": obj["title_en"],
            "description": obj["description"],
            "description_en": obj["description_en"],
            "domain": obj["domain"],
            "types": obj["types"],
            "tags": obj["tags"],
            "evidence_chunks": sorted(list(obj["evidence_chunks"])),
            "meta": {
                **obj["meta"],
                "evidence_files_top": sorted(list(obj["evidence_files"]))[:10]
            }
        })

    rules.sort(key=lambda x: (len(x["evidence_chunks"]), x["title"]), reverse=True)
    return rules


def build_flow_to_rule_links(flows: List[Dict[str, Any]], rules: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rule_by_chunk = defaultdict(list)
    for r in rules:
        for cid in r["evidence_chunks"]:
            rule_by_chunk[cid].append(r["rule_id"])

    for f in flows:
        rel: Set[str] = set()
        for s in f.get("steps", []):
            cid = s.get("evidence_chunk", "")
            if cid and cid in rule_by_chunk:
                rel.update(rule_by_chunk[cid])
        if not rel:
            dom = f.get("domain", "mixed")
            candidates = [r for r in rules if r["domain"] in (dom, "mixed")]
            candidates = sorted(candidates, key=lambda x: len(x["evidence_chunks"]), reverse=True)[:5]
            rel.update([r["rule_id"] for r in candidates])
        f["related_rules"] = sorted(list(rel))
    return flows


def main() -> None:
    index_path = ROOT / "data/raw_index/repo_index.jsonl"
    domain_map_path = ROOT / "data/extracted/domain_map.json"

    print("DEBUG: loading repo_index:", index_path)
    print("DEBUG: loading domain_map:", domain_map_path)

    if not index_path.exists():
        raise FileNotFoundError("缺少data/raw_index/repo_index.jsonl,请先跑Step01")
    if not domain_map_path.exists():
        raise FileNotFoundError("缺少data/extracted/domain_map.json,请先跑Step02")

    index_rows = read_jsonl(index_path)
    print("DEBUG: repo_index rows =", len(index_rows))

    domain_map = load_domain_map(domain_map_path)
    print("DEBUG: candidate_flows =", len(domain_map.get("candidate_flows", [])))

    flows = build_flow_records(domain_map)

    cands = collect_rule_candidates(index_rows)
    print("DEBUG: rule candidates =", len(cands))
    rules = aggregate_rules(cands)

    flows = build_flow_to_rule_links(flows, rules)

    rules_path = ROOT / "data/extracted/rules.jsonl"
    flows_path = ROOT / "data/extracted/flows.jsonl"
    write_jsonl(rules_path, rules)
    write_jsonl(flows_path, flows)

    sample_rules_path = ROOT / "data/samples/rules_sample.jsonl"
    sample_flows_path = ROOT / "data/samples/flows_sample.jsonl"
    write_jsonl(sample_rules_path, rules[:80])
    write_jsonl(sample_flows_path, flows[:20])

    print("完成rules与flows抽取")
    print(f"rules={len(rules)} -> {rules_path}")
    print(f"flows={len(flows)} -> {flows_path}")
    print(f"sample_rules -> {sample_rules_path}")
    print(f"sample_flows -> {sample_flows_path}")


if __name__ == "__main__":
    main()
