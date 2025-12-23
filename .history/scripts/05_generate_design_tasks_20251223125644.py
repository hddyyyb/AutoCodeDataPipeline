#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step05
场景2:需求 -> 基于本地代码仓架构生成设计方案(可追溯证据+trace)

对齐面试题补强点：
1) 输出必须包含“原文代码段”(evidence_snippets) + 推理trace
2) 覆盖场景2：样本数量/多样性要足够（支持从flows/rules自动扩展需求）
3) 统一产出 text/meta 训练格式（便于Step08合并）
"""

import json
import random
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import os

try:
    import yaml
except Exception:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]


# -----------------------------
# I/O helpers
# -----------------------------
def load_templates():
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
    return prefix + "_" + hashlib.sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:10]


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("缺少pyyaml,请pip install pyyaml")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


# -----------------------------
# Language
# -----------------------------
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


# -----------------------------
# Evidence / snippets
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


def build_evidence(chunk_ids: List[str], index_map: Dict[str, Dict[str, Any]], k: int, code_max_chars: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    返回：
    - evidence_refs：仅定位信息（chunk_id/file_path/行号）
    - evidence_snippets：包含原文代码段（可截断）
    """
    refs = []
    snippets = []
    for cid in chunk_ids:
        if cid in index_map:
            r = index_map[cid]
            refs.append({
                "chunk_id": cid,
                "file_path": r["file_path"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
            })
            snippets.append({
                "chunk_id": cid,
                "file_path": r["file_path"],
                "start_line": r["start_line"],
                "end_line": r["end_line"],
                "code_lang": detect_code_lang(r["file_path"]),
                "code": trim_code(r.get("content") or "", code_max_chars),
            })
        if len(refs) >= k:
            break
    return refs, snippets


# -----------------------------
# Context & chunk picking
# -----------------------------
def summarize_repo_context(domain_map: Dict[str, Any], domain: str) -> Dict[str, Any]:
    ents = [e for e in domain_map.get("entities", []) if e.get("domain") in (domain, "mixed")]
    ops = [o for o in domain_map.get("operations", []) if o.get("domain") in (domain, "mixed")]
    return {
        "boundaries_summary": {
            "controller_chunks": len(domain_map.get("boundaries", {}).get("controller", [])),
            "service_chunks": len(domain_map.get("boundaries", {}).get("service", [])),
            "mapper_chunks": len(domain_map.get("boundaries", {}).get("mapper", [])),
        },
        "top_entities": [e.get("name") for e in ents[:10] if e.get("name")],
        "top_operations": [o.get("name") for o in ops[:12] if o.get("name")],
        "constraints": [
            "遵循现有Controller/Service/Mapper分层结构",
            "尽量复用现有订单与库存相关Service/Mapper模式",
            "关键状态变更与库存变更必须具备幂等与可追溯性",
        ],
    }


def pick_reference_chunks(rules: List[Dict[str, Any]], flows: List[Dict[str, Any]], domain: str) -> List[str]:
    # 优先flow证据，其次rules
    chunk_ids = []
    for f in flows:
        if f.get("domain") in (domain, "mixed"):
            for s in f.get("steps", []) or []:
                cid = s.get("evidence_chunk")
                if cid:
                    chunk_ids.append(cid)

    for r in rules:
        if r.get("domain") in (domain, "mixed"):
            chunk_ids.extend((r.get("evidence_chunks") or [])[:2])

    # 去重保持顺序
    seen = set()
    out = []
    for c in chunk_ids:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


# -----------------------------
# Requirement expansion (关键：补足design数量/多样性)
# -----------------------------
def expand_requirements_from_flows_and_rules(flows: List[Dict[str, Any]], rules: List[Dict[str, Any]], max_n: int) -> List[Dict[str, Any]]:
    """
    从已抽取的flow/rule自动构造更多“可设计化需求”，提高场景2覆盖。
    """
    reqs: List[Dict[str, Any]] = []

    # 1) flow -> 设计改造需求（幂等/补偿/一致性/解耦）
    for f in flows:
        domain = f.get("domain", "mixed")
        name = f.get("name") or f.get("flow_id") or "业务流程"
        base = f"在现有{name}流程上，增加幂等与补偿机制，确保支付回调/库存变更的最终一致性，并给出落地到Controller/Service/Mapper的组件设计。"
        reqs.append({
            "id": make_id("auto_flow_req", f.get("flow_id", name)),
            "domain": domain,
            "requirement": base,
            "difficulty": "hard",
            "source_type": "flow",
            "source_id": f.get("flow_id"),
        })

    # 2) rule -> 设计治理需求（审计/观测/统一校验/状态机）
    for r in rules:
        domain = r.get("domain", "mixed")
        title = r.get("title") or r.get("rule_id") or "业务规则"
        base = f"围绕规则“{title}”，设计一套可扩展的规则治理/校验机制：支持集中校验、审计记录、可观测性，并说明如何在现有分层架构中落地。"
        reqs.append({
            "id": make_id("auto_rule_req", r.get("rule_id", title)),
            "domain": domain,
            "requirement": base,
            "difficulty": "medium",
            "source_type": "rule",
            "source_id": r.get("rule_id"),
        })

    random.shuffle(reqs)
    return reqs[:max_n]


# -----------------------------
# Design generation
# -----------------------------
def generate_design_output(req: str, domain: str, repo_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    模板化生成设计方案骨架：结构完整、与分层一致、考虑幂等/并发/事务/补偿。
    """
    if domain == "stock":
        components = [
            "StockReserveService(新增/扩展):预占库存与TTL管理",
            "StockReserveMapper(新增):预占记录持久化",
            "StockReleaseJob(新增):定时扫描超时预占并释放",
            "OrderService(改造点):下单时写入预占记录,取消/关闭时触发释放",
        ]
        apis = [
            "POST /stock/reserve: 预占库存(返回reserveId/过期时间)",
            "POST /stock/release: 释放预占库存(幂等)",
            "GET  /stock/reserve/{reserveId}: 查询预占状态",
        ]
        data_model = [
            "stock_reserve(id,reserve_id,order_id,sku_id,qty,status,expire_at,created_at,updated_at)",
            "reserve_id作为幂等键,order_id+sku_id可做唯一约束",
        ]
        sequence = [
            "1)下单 -> 调用StockReserveService.reserve",
            "2)reserve成功 -> 继续订单创建/锁定状态",
            "3)支付成功 -> confirm并扣减(将reserve状态标记为CONSUMED)",
            "4)取消/超时 -> release(标记RELEASED并回滚库存)",
            "5)定时任务扫描expire_at < now且status=RESERVED -> release",
        ]
        risks = [
            "并发超卖: reserve采用数据库行锁/乐观锁或Redis原子扣减",
            "幂等: reserveId/请求id作为幂等键,release重复调用无副作用",
            "一致性: 订单创建与预占记录写入建议同事务或采用补偿任务修复",
        ]
    else:
        components = [
            "OrderAuditService(新增):记录订单状态变更审计",
            "OrderAuditMapper(新增):审计表持久化",
            "OrderStatusChangeHook(改造点):在现有状态流转处统一埋点",
            "ReconcileJob(可选):对账/补偿任务,修复回调丢失或重复",
        ]
        apis = [
            "GET /order/audit/{orderId}: 查询订单审计记录",
            "内部: recordStatusChange(orderId,from,to,source,requestId)",
        ]
        data_model = [
            "order_audit(id,order_id,from_status,to_status,source,request_id,created_at)",
            "request_id作为幂等键,避免重复审计写入",
        ]
        sequence = [
            "1)订单状态变更入口(支付回调/取消/超时关闭等)调用hook",
            "2)hook提取from/to/source/requestId并写审计表",
            "3)对账补偿(可选):周期性拉取支付平台结果,驱动最终一致",
        ]
        risks = [
            "重复回调: 使用requestId/订单号+状态作为幂等键",
            "状态机约束: 在变更前校验合法性,非法则拒绝并记录原因",
            "补偿策略: 对账任务应可重入,并记录处理游标",
        ]

    return {
        "architecture_overview": f"基于现有分层架构(Controller/Service/Mapper),在{domain}域新增最小闭环组件,复用现有订单/库存模式。",
        "components": components,
        "apis": apis,
        "data_model": data_model,
        "sequence_flows": sequence,
        "risk_and_mitigation": risks,
        "repo_constraints": repo_context.get("constraints", []),
    }


def build_trace(domain: str, ref_chunks: List[str], repo_context: Dict[str, Any], lang: str) -> List[str]:
    if lang == "en":
        steps = [
            f"Step1:Identify the request as domain={domain} and follow existing layered constraints (Controller/Service/Mapper).",
            f"Step2:Inspect existing evidence chunks {ref_chunks[:3]} to infer reusable patterns.",
            "Step3:Introduce minimal new components, define APIs and data model aligned with existing layers.",
            "Step4:Address idempotency, concurrency control, transactional consistency and compensations.",
        ]
        return steps

    steps = [
        f"步骤1:确定需求属于{domain}域,遵循现有Controller/Service/Mapper分层约束",
        f"步骤2:参考证据chunk={ref_chunks[:3]}识别可复用的服务/状态流转/库存处理模式",
        "步骤3:在不破坏现有调用链的前提下新增组件并定义接口与数据结构",
        "步骤4:补充幂等、并发控制、事务一致性与补偿策略",
    ]
    return steps


def format_code_block(snippet: Dict[str, Any]) -> str:
    lang = snippet.get("code_lang") or ""
    code = snippet.get("code") or ""
    if not code:
        return ""
    return f"```{lang}\n{code}\n```"


def build_training_text(requirement: str, design_output: Dict[str, Any], evidence_snippets: List[Dict[str, Any]], trace_steps: List[str], lang: str) -> str:
    """
    统一输出：Instruction/Response，Response里包含设计+证据代码段+trace。
    """
    if lang == "en":
        ins = f"Generate a design方案 for the requirement based on the repo architecture, and provide evidence code snippets and a reasoning trace.\nRequirement: {requirement}"
        resp = {
            "design": design_output,
            "evidence_code_snippets": evidence_snippets[:3],
            "trace": trace_steps,
        }
        return f"### Instruction\n{ins}\n\n### Response\n{json.dumps(resp, ensure_ascii=False, indent=2)}\n"

    ins = f"基于本地代码仓架构为给定需求生成设计方案，并提供证据代码段与推理trace。\n需求:{requirement}"
    resp = {
        "design": design_output,
        "evidence_code_snippets": evidence_snippets[:3],
        "trace": trace_steps,
    }
    return f"### Instruction\n{ins}\n\n### Response\n{json.dumps(resp, ensure_ascii=False, indent=2)}\n"


def main():
    templates_nlg = load_templates()
    lang_mode = resolve_language(default_lang="zh")
    langs = ["zh", "en"] if lang_mode == "bilingual" else [lang_mode]
    for l in langs:
        if l not in templates_nlg.get("design", {}):
            raise ValueError(f"Unsupported language '{l}'. Available: {list(templates_nlg.get('design', {}).keys())}")

    index_rows = read_jsonl(ROOT / "data/raw_index/repo_index.jsonl")
    index_map = {r["chunk_id"]: r for r in index_rows}

    domain_map = json.loads((ROOT / "data/extracted/domain_map.json").read_text(encoding="utf-8"))
    rules = read_jsonl(ROOT / "data/extracted/rules.jsonl")
    flows = read_jsonl(ROOT / "data/extracted/flows.jsonl")

    # knobs
    code_max_chars = int(os.environ.get("DESIGN_CODE_MAX_CHARS", "2200"))
    evidence_k = int(os.environ.get("DESIGN_EVIDENCE_K", "3"))
    auto_expand = os.environ.get("DESIGN_AUTO_EXPAND", "1").strip() not in ("0", "false", "False")

    # 读取人工模板需求
    cfg_path = ROOT / "configs/design_requirements.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError("缺少configs/design_requirements.yaml,请先创建该文件")
    cfg = load_yaml(cfg_path)
    templates = cfg.get("templates", []) or []

    # 自动扩展需求（关键：补足样本数量与多样性）
    extra_n = int(os.environ.get("DESIGN_AUTO_EXPAND_N", "60"))
    auto_reqs = expand_requirements_from_flows_and_rules(flows, rules, max_n=extra_n) if auto_expand else []

    all_reqs = []
    # 统一结构
    for t in templates:
        all_reqs.append({
            "id": t.get("id") or make_id("req", (t.get("requirement") or "")),
            "domain": t.get("domain", "mixed"),
            "requirement": (t.get("requirement") or "").strip(),
            "difficulty": t.get("difficulty", "medium"),
            "source_type": "yaml",
            "source_id": t.get("id"),
        })
    all_reqs.extend(auto_reqs)

    # 去重（按requirement文本）
    seen_req = set()
    deduped = []
    for r in all_reqs:
        key = (r.get("domain"), r.get("requirement"))
        if key not in seen_req and r.get("requirement"):
            seen_req.add(key)
            deduped.append(r)
    all_reqs = deduped

    samples: List[Dict[str, Any]] = []
    for r in all_reqs:
        domain = r.get("domain", "mixed")
        req = r.get("requirement", "").strip()
        rid = r.get("id") or make_id("req", req)

        repo_context = summarize_repo_context(domain_map, domain)
        ref_chunks = pick_reference_chunks(rules, flows, domain)
        evidence_refs, evidence_snippets = build_evidence(ref_chunks, index_map, k=evidence_k, code_max_chars=code_max_chars)

        for lang in langs:
            design_output = generate_design_output(req, domain, repo_context)

            # nlg改写overview（语言一致）
            overview_prefix = templates_nlg["design"][lang]["overview_prefix"].format(domain=domain)
            design_output["architecture_overview"] = overview_prefix

            trace_steps = build_trace(domain, ref_chunks, repo_context, lang=lang)

            text = build_training_text(req, design_output, evidence_snippets, trace_steps, lang)

            meta_v2 = {
                "task_type": "design",
                "language": lang,
                "domain": domain,
                "difficulty": r.get("difficulty", "medium"),
                "requirement_id": rid,
                "requirement_source": {"type": r.get("source_type"), "id": r.get("source_id")},
                "repo_context": repo_context,
                "evidence": evidence_refs,
                "evidence_snippets": evidence_snippets,
                "trace_digest": trace_steps,
                "generator": "step05_v3_expand_requirements",
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "source": "AutoCodeDataPipeline",
            }

            sample = {
                # 保留你原有结构（便于兼容旧逻辑）
                "sample_id": make_id("design", f"{rid}|{lang}"),
                "task_type": "design",
                "language": lang,
                "requirement": req,
                "repo_context": repo_context,
                "design_output": design_output,
                "evidence": evidence_snippets[:evidence_k],  # 旧字段：直接给含content/code的证据也行
                "trace": {"reasoning_steps": trace_steps},

                # 新增统一训练格式
                "text": text,
                "meta_v2": meta_v2,
            }
            samples.append(sample)

    random.shuffle(samples)
    n = len(samples)
    train = samples[: int(n * 0.8)] if n >= 3 else samples
    dev = samples[int(n * 0.8): int(n * 0.9)]
    test = samples[int(n * 0.9):]

    write_jsonl(ROOT / "data/dataset/design_train.jsonl", train)
    write_jsonl(ROOT / "data/dataset/design_dev.jsonl", dev)
    write_jsonl(ROOT / "data/dataset/design_test.jsonl", test)
    write_jsonl(ROOT / "data/samples/design_samples.jsonl", samples[:40])

    print(f"Design样本生成完成: total={n}, train={len(train)}, dev={len(dev)}, test={len(test)}")
    print(f"[Knobs] DESIGN_AUTO_EXPAND={auto_expand}, DESIGN_AUTO_EXPAND_N={extra_n}, DESIGN_CODE_MAX_CHARS={code_max_chars}, DESIGN_EVIDENCE_K={evidence_k}")


if __name__ == "__main__":
    main()
