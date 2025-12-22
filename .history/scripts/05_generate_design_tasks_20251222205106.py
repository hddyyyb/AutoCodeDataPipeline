#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step05
场景2:需求 -> 基于本地代码仓架构生成设计方案(可追溯证据+trace)
"""

import json
import random
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List
import os

try:
    import yaml
except Exception:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]


def load_templates():
    import json
    p = ROOT / "configs/nlg_templates.json"
    return json.loads(p.read_text(encoding="utf-8"))



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


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("缺少pyyaml,请pip install pyyaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_evidence(chunk_ids: List[str], index_map: Dict[str, Dict[str, Any]], k: int = 3) -> List[Dict[str, Any]]:
    evs = []
    for cid in chunk_ids:
        if cid in index_map:
            r = index_map[cid]
            evs.append({
                "chunk_id": cid,
                "file_path": r["file_path"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "content": r["content"]
            })
        if len(evs) >= k:
            break
    return evs


def summarize_repo_context(domain_map: Dict[str, Any], domain: str) -> Dict[str, Any]:
    # 选出与domain相关的实体/操作前N个作为上下文摘要
    ents = [e for e in domain_map.get("entities", []) if e.get("domain") in (domain, "mixed")]
    ops = [o for o in domain_map.get("operations", []) if o.get("domain") in (domain, "mixed")]
    return {
        "boundaries_summary": {
            "controller_chunks": len(domain_map.get("boundaries", {}).get("controller", [])),
            "service_chunks": len(domain_map.get("boundaries", {}).get("service", [])),
            "mapper_chunks": len(domain_map.get("boundaries", {}).get("mapper", [])),
        },
        "top_entities": [e["name"] for e in ents[:8]],
        "top_operations": [o["name"] for o in ops[:10]],
        "constraints": [
            "遵循现有Controller/Service/Mapper分层结构",
            "尽量复用现有订单与库存相关Service/Mapper模式",
            "所有关键状态变更与库存变更必须具备幂等与可追溯性"
        ]
    }


def pick_reference_chunks(domain_map: Dict[str, Any], rules: List[Dict[str, Any]], flows: List[Dict[str, Any]], domain: str) -> List[str]:
    # 优先从domain相关的flow步骤中拿证据,其次从rules里拿
    chunk_ids = []
    for f in flows:
        if f.get("domain") in (domain, "mixed"):
            for s in f.get("steps", []):
                cid = s.get("evidence_chunk")
                if cid:
                    chunk_ids.append(cid)
    for r in rules:
        if r.get("domain") in (domain, "mixed"):
            chunk_ids.extend(r.get("evidence_chunks", [])[:2])
    # 去重保持顺序
    seen = set()
    out = []
    for c in chunk_ids:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def generate_design_output(req: str, domain: str, repo_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    模板化生成设计方案骨架.
    面试作业核心:结构完整、与现有分层一致、考虑幂等/并发/事务/补偿.
    """
    # 根据domain给出不同方案重点
    if domain == "stock":
        components = [
            "StockReserveService(新增/扩展):预占库存与TTL管理",
            "StockReserveMapper(新增):预占记录持久化",
            "StockReleaseJob(新增):定时扫描超时预占并释放",
            "OrderService(改造点):下单时写入预占记录,取消/关闭时触发释放"
        ]
        apis = [
            "POST /stock/reserve: 预占库存(返回reserveId/过期时间)",
            "POST /stock/release: 释放预占库存(幂等)",
            "GET  /stock/reserve/{reserveId}: 查询预占状态"
        ]
        data_model = [
            "stock_reserve(id,reserve_id,order_id,sku_id,qty,status,expire_at,created_at,updated_at)",
            "reserve_id作为幂等键,order_id+sku_id可做唯一约束"
        ]
        sequence = [
            "1)下单 -> 调用StockReserveService.reserve",
            "2)reserve成功 -> 继续订单创建/锁定状态",
            "3)支付成功 -> confirm并扣减(将reserve状态标记为CONSUMED)",
            "4)取消/超时 -> release(标记RELEASED并回滚库存)",
            "5)定时任务扫描expire_at < now且status=RESERVED -> release"
        ]
        risks = [
            "并发超卖: reserve采用数据库行锁/乐观锁或Redis原子扣减",
            "幂等: reserveId/请求id作为幂等键,release重复调用无副作用",
            "一致性: 订单创建与预占记录写入建议同事务或采用补偿任务修复"
        ]
    else:
        components = [
            "OrderAuditService(新增):记录订单状态变更审计",
            "OrderAuditMapper(新增):审计表持久化",
            "OrderStatusChangeHook(改造点):在现有状态流转处统一埋点",
            "ReconcileJob(可选):对账/补偿任务,修复回调丢失或重复"
        ]
        apis = [
            "GET /order/audit/{orderId}: 查询订单审计记录",
            "内部: recordStatusChange(orderId,from,to,source,requestId)"
        ]
        data_model = [
            "order_audit(id,order_id,from_status,to_status,source,request_id,created_at)",
            "request_id作为幂等键,避免重复审计写入"
        ]
        sequence = [
            "1)订单状态变更入口(支付回调/取消/超时关闭等)调用hook",
            "2)hook提取from/to/source/requestId并写审计表",
            "3)对账补偿(可选):周期性拉取支付平台结果,驱动最终一致"
        ]
        risks = [
            "重复回调: 使用requestId/订单号+状态作为幂等键",
            "状态机约束: 在变更前校验合法性,非法则拒绝并记录原因",
            "补偿策略: 对账任务应可重入,并记录处理游标"
        ]

    return {
        "architecture_overview": f"基于现有分层架构(Controller/Service/Mapper),在{domain}域新增最小闭环组件,复用现有订单/库存模式。",
        "components": components,
        "apis": apis,
        "data_model": data_model,
        "sequence_flows": sequence,
        "risk_and_mitigation": risks
    }


def build_trace(domain: str, ref_chunks: List[str], repo_context: Dict[str, Any]) -> List[str]:
    steps = [
        f"步骤1:确定需求属于{domain}域,优先复用现有Controller/Service/Mapper分层约束",
        "步骤2:从现有代码证据中识别可复用的服务/状态流转/库存处理模式",
        "步骤3:在不破坏现有调用链的前提下新增组件并定义接口与数据结构",
        "步骤4:补充幂等、并发控制、事务一致性与补偿策略"
    ]
    if ref_chunks:
        steps.insert(2, f"步骤2.1:参考证据chunk={ref_chunks[:3]}")
    return steps


def normalize_lang(lang: str) -> str:
    """
    将系统locale/环境变量映射为pipeline内部使用的语言标识: zh/en/bilingual
    例: en_US.UTF-8 -> en, zh_CN.UTF-8 -> zh
    """
    if not lang:
        return "zh"
    l = lang.strip()
    l_lower = l.lower()

    # 兼容你已有的开关
    if l_lower in ("bilingual", "bi", "both", "zh+en", "en+zh"):
        return "bilingual"

    # 常见locale映射
    if l_lower.startswith("zh"):
        return "zh"
    if l_lower.startswith("en"):
        return "en"

    # 兜底
    return "zh"


def main():

    templates_nlg = load_templates()
    #lang_mode = os.environ.get("LANG", "zh")
    #langs = ["zh", "en"] if lang_mode == "bilingual" else [lang_mode]
    raw_lang = os.environ.get("LANG", "zh")
    lang_mode = normalize_lang(raw_lang)
    langs = ["zh", "en"] if lang_mode == "bilingual" else [lang_mode]
    available_langs = list(templates_nlg.get("design", {}).keys())
    for l in langs:
        if l not in templates_nlg.get("design", {}):
            raise ValueError(f"Unsupported language '{l}' (from LANG={raw_lang}). Available: {available_langs}")


    index_rows = read_jsonl(ROOT / "data/raw_index/repo_index.jsonl")
    index_map = {r["chunk_id"]: r for r in index_rows}

    domain_map = json.loads((ROOT / "data/extracted/domain_map.json").read_text(encoding="utf-8"))
    rules = read_jsonl(ROOT / "data/extracted/rules.jsonl")
    flows = read_jsonl(ROOT / "data/extracted/flows.jsonl")

    cfg_path = ROOT / "configs/design_requirements.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError("缺少configs/design_requirements.yaml(我给你的模板库),请先创建该文件")

    cfg = load_yaml(cfg_path)
    templates = cfg.get("templates", [])
    if not templates:
        raise RuntimeError("design_requirements.yaml中templates为空")

    samples = []
    for t in templates:
        domain = t.get("domain", "mixed")
        req = t.get("requirement", "").strip()
        rid = t.get("id", make_id("req", req))

        repo_context = summarize_repo_context(domain_map, domain)
        ref_chunks = pick_reference_chunks(domain_map, rules, flows, domain)
        evidence = build_evidence(ref_chunks, index_map, k=3)

        for lang in langs:
            design_output = generate_design_output(req, domain, repo_context)

            # 用nlg模板改写overview与trace
            overview = templates_nlg["design"][lang]["overview_prefix"].format(domain=domain)
            design_output["architecture_overview"] = overview

            trace_steps = build_trace(domain, ref_chunks, repo_context)
            if lang == "en":
                trace_steps = [
                    "Step1:Identify the domain and reuse existing layered constraints (Controller/Service/Mapper).",
                    f"Step2:Reference evidence chunks {ref_chunks[:3]}.",
                    "Step3:Introduce minimal new components and define APIs/data model without breaking existing call chains.",
                    "Step4:Address idempotency, concurrency control, transactional consistency and compensations."
                ]

            sample = {
                "sample_id": make_id("design", f"{rid}|{lang}"),
                "task_type": "design",
                "language": lang,
                "requirement": req if lang == "zh" else req,  # 这里先复用原需求文本；要英文需求可再扩展yaml
                "repo_context": repo_context,
                "design_output": design_output,
                "evidence": evidence,
                "trace": {"reasoning_steps": trace_steps},
                "meta": {
                    "domain": domain,
                    "difficulty": t.get("difficulty", "medium"),
                    "generator": "template_arch_v2_multilingual",
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "source": "AutoCodeDataPipeline"
                }
            }
            samples.append(sample)

    '''
    for t in templates:
        domain = t.get("domain", "mixed")
        req = t.get("requirement", "").strip()
        rid = t.get("id", make_id("req", req))

        repo_context = summarize_repo_context(domain_map, domain)
        ref_chunks = pick_reference_chunks(domain_map, rules, flows, domain)
        evidence = build_evidence(ref_chunks, index_map, k=3)

        design_output = generate_design_output(req, domain, repo_context)
        trace_steps = build_trace(domain, ref_chunks, repo_context)

        sample = {
            "sample_id": make_id("design", rid),
            "task_type": "design",
            "language": "zh",
            "requirement": req,
            "repo_context": repo_context,
            "design_output": design_output,
            "evidence": evidence,
            "trace": {
                "reasoning_steps": trace_steps
            },
            "meta": {
                "domain": domain,
                "difficulty": t.get("difficulty", "medium"),
                "generator": "template_arch_v1",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "source": "AutoCodeDataPipeline"
            }
        }
        samples.append(sample)'''

    # 打乱并切分
    random.shuffle(samples)
    n = len(samples)
    train = samples[: int(n * 0.8)] if n >= 3 else samples
    dev = samples[int(n * 0.8): int(n * 0.9)] if n >= 10 else []
    test = samples[int(n * 0.9):] if n >= 10 else []

    write_jsonl(ROOT / "data/dataset/design_train.jsonl", train)
    if dev:
        write_jsonl(ROOT / "data/dataset/design_dev.jsonl", dev)
    if test:
        write_jsonl(ROOT / "data/dataset/design_test.jsonl", test)

    write_jsonl(ROOT / "data/samples/design_samples.jsonl", samples)

    print(f"Design样本生成完成: total={n}, train={len(train)}, dev={len(dev)}, test={len(test)}")
    print("输出:data/dataset/design_train.jsonl 以及 data/samples/design_samples.jsonl")


if __name__ == "__main__":
    main()
