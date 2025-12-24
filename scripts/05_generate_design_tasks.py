#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step05(v8)
场景2:需求->基于本地代码仓架构生成设计方案(多样性+代表性+英文纯净)

保留目标:
- 每条样本带evidence_snippets+trace
- bilingual时分别生成zh/en两套
"""

import json
import random
import time
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Tuple
import os
import re
from collections import Counter

try:
    import yaml
except Exception:
    yaml = None

ROOT = Path(__file__).resolve().parents[1]

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CLASS_RE = re.compile(r"\bclass\s+([A-Za-z_][A-Za-z0-9_]*)\b")
_METHOD_RE = re.compile(r"\b(public|private|protected)\s+[\w<>\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")

_ROLE_HINTS = [
    (re.compile(r"/controller/|Controller\.java$", re.IGNORECASE), "controller"),
    (re.compile(r"/service/|Service\.java$", re.IGNORECASE), "service"),
    (re.compile(r"/mapper/|Mapper\.java$", re.IGNORECASE), "mapper"),
    (re.compile(r"/dao/|Dao\.java$", re.IGNORECASE), "mapper"),
    (re.compile(r"/config/|Configuration\.java$|\.yml$|\.yaml$|\.properties$", re.IGNORECASE), "config"),
]


def has_cjk(s: str) -> bool:
    return bool(s and _CJK_RE.search(s))


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def make_id(prefix: str, seed: str) -> str:
    return prefix + "_" + sha1_text(seed)[:10]


def load_templates():
    p = ROOT / "configs/nlg_templates.json"
    return json.loads(p.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists():
        return out
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


def load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError("缺少pyyaml,请pip install pyyaml")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


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
    if env_lang and env_lang.strip().lower() in ("zh", "en", "bilingual", "bi", "both", "zh+en", "en+zh"):
        return normalize_lang(env_lang)
    cfg_path = ROOT / "configs/runtime.yaml"
    if cfg_path.exists():
        import yaml as _yaml
        cfg = _yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        mode = (cfg.get("language") or {}).get("mode")
        if mode:
            return normalize_lang(mode)
    return normalize_lang(default_lang)


def get_strict_en_level() -> int:
    """
    DESIGN_STRICT_EN:
      0=off(不处理)
      1=wrap(保留中文原文但用英文包装,不推荐训练)
      2=drop(默认推荐:英文样本如果requirement含中文则丢弃)
    """
    v = os.environ.get("DESIGN_STRICT_EN", "2").strip()
    try:
        lv = int(v)
    except Exception:
        lv = 2
    return 0 if lv < 0 else 2 if lv > 2 else lv


def detect_code_lang(file_path: str) -> str:
    fp = (file_path or "").lower()
    if fp.endswith(".java"):
        return "java"
    if fp.endswith(".xml"):
        return "xml"
    if fp.endswith(".yml") or fp.endswith(".yaml"):
        return "yaml"
    if fp.endswith(".properties"):
        return "properties"
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


def build_evidence(
    chunk_ids: List[str],
    index_map: Dict[str, Dict[str, Any]],
    k: int,
    code_max_chars: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
                "content": r.get("content") or "",
            })
        if len(refs) >= k:
            break
    return refs, snippets


def infer_role_from_path(fp: str) -> str:
    p = (fp or "").replace("\\", "/")
    for rx, role in _ROLE_HINTS:
        if rx.search(p):
            return role
    return "other"


def extract_names_from_snippets(evidence_snippets: List[Dict[str, Any]]) -> Dict[str, Any]:
    classes: List[str] = []
    methods: List[str] = []
    roles: List[str] = []
    files: List[str] = []
    for ev in (evidence_snippets or [])[:3]:
        fp = ev.get("file_path") or ""
        files.append(fp)
        roles.append(infer_role_from_path(fp))
        content = ev.get("content") or ev.get("code") or ""
        for m in _CLASS_RE.finditer(content):
            n = m.group(1)
            if n and n not in classes:
                classes.append(n)
            if len(classes) >= 6:
                break
        for m in _METHOD_RE.finditer(content):
            n = m.group(2)
            if n and n not in methods:
                methods.append(n)
            if len(methods) >= 8:
                break
    return {
        "classes": classes[:6],
        "methods": methods[:8],
        "roles": roles[:3],
        "files": files[:3],
    }


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


def repo_constraints_en() -> List[str]:
    return [
        "Follow the existing Controller/Service/Mapper layering.",
        "Prefer reusing existing order/stock service and mapper patterns.",
        "State changes and stock changes must be idempotent and auditable(traceable).",
    ]


def pick_reference_chunks(rules: List[Dict[str, Any]], flows: List[Dict[str, Any]], domain: str) -> List[str]:
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
    seen = set()
    out = []
    for c in chunk_ids:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _safe_en_label(domain: str, kind: str, fallback_id: str, zh_text: str) -> str:
    """
    关键修复:严格英文模式下,如果name/title是中文,直接给一个纯英文label,避免requirement_en被strict_en丢弃
    """
    if not zh_text or not has_cjk(zh_text):
        return zh_text
    fid = (fallback_id or "").strip()
    if fid:
        return f"{domain} {kind} {fid}"
    return f"{domain} {kind}"


# -----------------------------
# Strategy library(zh+en)
# -----------------------------
STRATEGIES = [
    {
        "id": "outbox",
        "zh": "Outbox事件表+异步投递(最终一致性)",
        "en": "Outbox table + async dispatcher(eventual consistency)",
        "points_zh": ["outbox_event表", "投递任务", "重试与幂等消费"],
        "points_en": ["outbox_event table", "dispatcher job", "retries and idempotent consumers"],
    },
    {
        "id": "saga",
        "zh": "Saga补偿事务(失败可逆)",
        "en": "Saga with compensations(failure reversible)",
        "points_zh": ["补偿接口", "状态记录", "可重入执行"],
        "points_en": ["compensation APIs", "state tracking", "re-entrant execution"],
    },
    {
        "id": "idempotency_key",
        "zh": "幂等键(requestId)+去重表",
        "en": "Idempotency key(requestId) + dedup table",
        "points_zh": ["dedup表", "唯一约束", "重复无副作用"],
        "points_en": ["dedup table", "unique constraint", "replay has no side effects"],
    },
    {
        "id": "state_machine",
        "zh": "显式状态机校验(禁止非法流转)",
        "en": "Explicit state machine validation(block illegal transitions)",
        "points_zh": ["allowedTransitions", "统一校验入口", "错误语义统一"],
        "points_en": ["allowedTransitions", "centralized guard", "standardized error semantics"],
    },
    {
        "id": "ttl_reserve",
        "zh": "库存预占TTL+超时释放任务",
        "en": "Stock reserve TTL + timeout release job",
        "points_zh": ["reserve记录", "expire_at", "release job扫描"],
        "points_en": ["reserve records", "expire_at", "release job scanner"],
    },
    {
        "id": "optimistic_lock",
        "zh": "乐观锁/条件更新(并发安全)",
        "en": "Optimistic lock/conditional update(concurrency-safe)",
        "points_zh": ["version字段", "where条件更新", "失败重试策略"],
        "points_en": ["version field", "conditional update", "retry-on-conflict policy"],
    },
]
_STR_BY_ID = {s["id"]: s for s in STRATEGIES}


def pick_strategies(domain: str) -> List[Dict[str, Any]]:
    prefer = ["ttl_reserve", "idempotency_key", "optimistic_lock"] if domain == "stock" else ["state_machine", "outbox", "idempotency_key"]
    picked = []
    for pid in prefer:
        if pid in _STR_BY_ID and random.random() < 0.85:
            picked.append(_STR_BY_ID[pid])
    if random.random() < 0.6:
        extra = random.choice(STRATEGIES)
        if extra not in picked:
            picked.append(extra)
    while len(picked) < 2:
        extra = random.choice(STRATEGIES)
        if extra not in picked:
            picked.append(extra)
    return picked[:3]


# -----------------------------
# Requirement expansion(bilingual)
# -----------------------------
def expand_requirements_from_flows_and_rules(flows: List[Dict[str, Any]], rules: List[Dict[str, Any]], max_n: int) -> List[Dict[str, Any]]:
    reqs: List[Dict[str, Any]] = []

    for f in flows:
        domain = f.get("domain", "mixed")
        flow_id = f.get("flow_id") or f.get("id") or ""
        name_zh = (f.get("name") or flow_id or "业务流程").strip()
        raw_name_en = (f.get("name_en") or "").strip()
        name_en = raw_name_en or name_zh
        if has_cjk(name_en):
            name_en = _safe_en_label(domain, "flow", flow_id, name_zh)

        reqs.append({
            "id": make_id("auto_flow_req", flow_id or name_zh),
            "domain": domain,
            "difficulty": "hard",
            "source_type": "flow",
            "source_id": flow_id,
            "requirement_zh": f"围绕{name_zh}流程做增强设计:补齐幂等、失败补偿、可观测性,并落地到Controller/Service/Mapper分层。",
            "requirement_en": f"Enhance the flow '{name_en}' with idempotency, failure compensation, and observability, aligned with Controller/Service/Mapper layers.",
            "context_en": f"Flow:{name_en}",
        })

    for r in rules:
        domain = r.get("domain", "mixed")
        rule_id = r.get("rule_id") or r.get("id") or ""
        title_zh = (r.get("title") or rule_id or "业务规则").strip()
        raw_title_en = (r.get("title_en") or "").strip()
        title_en = raw_title_en or title_zh
        if has_cjk(title_en):
            title_en = _safe_en_label(domain, "rule", rule_id, title_zh)

        reqs.append({
            "id": make_id("auto_rule_req", rule_id or title_zh),
            "domain": domain,
            "difficulty": "medium",
            "source_type": "rule",
            "source_id": rule_id,
            "requirement_zh": f"针对规则“{title_zh}”设计统一校验与审计方案:集中校验、审计记录、异常语义统一、可扩展落地组件。",
            "requirement_en": f"Design unified validation and auditing for the rule '{title_en}': centralized checks, audit logs, consistent error semantics, and concrete components.",
            "context_en": f"Rule:{title_en}",
        })

    random.shuffle(reqs)
    return reqs[:max_n]


# -----------------------------
# Design generation(dynamic, zh/en separated)
# -----------------------------
def generate_design_output(
    requirement: str,
    domain: str,
    repo_context: Dict[str, Any],
    name_hints: Dict[str, Any],
    strategies: List[Dict[str, Any]],
    lang: str
) -> Dict[str, Any]:
    classes = name_hints.get("classes") or []
    methods = name_hints.get("methods") or []
    files = name_hints.get("files") or []

    evidence_hints = []
    if lang == "zh":
        if classes:
            evidence_hints.append(f"复用/改造类:{','.join(classes[:3])}")
        if methods:
            evidence_hints.append(f"关注入口方法:{','.join(methods[:4])}")
        if files:
            evidence_hints.append(f"证据文件:{','.join(files[:2])}")
    else:
        if classes:
            evidence_hints.append(f"Reuse/extend classes:{', '.join(classes[:3])}")
        if methods:
            evidence_hints.append(f"Focus on entry methods:{', '.join(methods[:4])}")
        if files:
            evidence_hints.append(f"Evidence files:{', '.join(files[:2])}")

    strat_text = [(s["zh"] if lang == "zh" else s["en"]) for s in strategies]
    strat_ids = [s["id"] for s in strategies]

    components: List[str] = []
    apis: List[str] = []
    data_model: List[str] = []
    risks: List[str] = []

    if domain == "stock":
        if lang == "zh":
            components += [
                "StockReserveService(新增/扩展):预占/释放/确认入口(幂等)",
                "StockReserveMapper(新增):预占记录持久化",
                "StockReleaseJob(新增):超时释放扫描任务",
            ]
            apis += [
                "POST/stock/reserve:预占库存(返回reserveId/expireAt)",
                "POST/stock/release:释放预占(幂等)",
                "POST/stock/confirm:确认扣减(幂等)",
            ]
        else:
            components += [
                "StockReserveService(new/extend): reserve/release/confirm entry points(idempotent)",
                "StockReserveMapper(new): persistence for reserve records",
                "StockReleaseJob(new): timeout release scanner job",
            ]
            apis += [
                "POST /stock/reserve: reserve stock(returns reserveId/expireAt)",
                "POST /stock/release: release reservation(idempotent)",
                "POST /stock/confirm: confirm deduction(idempotent)",
            ]
        data_model += [
            "stock_reserve(id,reserve_id,order_id,sku_id,qty,status,expire_at,created_at,updated_at)",
        ]
    else:
        if lang == "zh":
            components += [
                "OrderStateGuard(新增):统一状态校验/流转入口",
                "OrderAuditService(新增):状态变更审计",
                "ReconcileJob(可选):对账/补偿任务",
            ]
            apis += [
                "POST/order/state/transition:统一状态流转入口(携带requestId)",
                "GET/order/audit/{orderId}:查询审计记录",
            ]
        else:
            components += [
                "OrderStateGuard(new): centralized state validation/transition entry",
                "OrderAuditService(new): audit trail for state changes",
                "ReconcileJob(optional): reconciliation/compensation job",
            ]
            apis += [
                "POST /order/state/transition: centralized transition entry(carries requestId)",
                "GET /order/audit/{orderId}: query audit records",
            ]
        data_model += [
            "order_audit(id,order_id,from_status,to_status,source,request_id,created_at)",
        ]

    for sid in strat_ids:
        if sid == "outbox":
            if lang == "zh":
                components.append("OutboxEventService(新增):事件记录+投递任务")
                data_model.append("outbox_event(id,event_type,biz_id,payload,status,created_at,updated_at)")
            else:
                components.append("OutboxEventService(new): persist events and dispatch asynchronously")
                data_model.append("outbox_event(id,event_type,biz_id,payload,status,created_at,updated_at)")
        elif sid == "saga":
            components.append("CompensationService(new): a set of re-entrant compensations" if lang == "en" else "CompensationService(新增):补偿动作集合(可重入)")
        elif sid == "idempotency_key":
            if lang == "zh":
                components.append("IdempotencyService(新增):幂等键校验+去重表")
                data_model.append("idempotency_record(id,key,biz_type,biz_id,status,created_at)")
            else:
                components.append("IdempotencyService(new): validate idempotency key and dedup records")
                data_model.append("idempotency_record(id,key,biz_type,biz_id,status,created_at)")
        elif sid == "state_machine":
            components.append(
                "StateMachineConfig(new): allowedTransitions + centralized guard"
                if lang == "en" else
                "StateMachineConfig(新增):allowedTransitions配置+统一校验入口"
            )
        elif sid == "ttl_reserve":
            components.append(
                "ReserveTTLPolicy(new): TTL policy and expiry handling"
                if lang == "en" else
                "ReserveTTLPolicy(新增):TTL策略与过期处理"
            )
        elif sid == "optimistic_lock":
            components.append(
                "VersionedUpdateHelper(new): conditional update and retry-on-conflict"
                if lang == "en" else
                "VersionedUpdateHelper(新增):条件更新/失败重试"
            )

    for s in strategies:
        if lang == "zh":
            pts = s.get("points_zh") or []
            risks.append(f"采用策略:{s['zh']}。落地点:{'、'.join(pts[:3])}")
        else:
            pts = s.get("points_en") or []
            risks.append(f"Strategy:{s['en']}.Key points:{', '.join(pts[:3])}")

    if lang == "zh":
        overview = f"基于现有Controller/Service/Mapper分层,按策略组合({';'.join(strat_text)})做最小侵入式增强,保证幂等、并发安全与最终一致性。"
        sequence = [
            "1)入口携带requestId,先做幂等校验/去重",
            "2)执行核心业务(状态流转/预占扣减),失败进入补偿或回滚",
            "3)记录审计与可观测日志,必要时写入Outbox事件",
            "4)异步任务(对账/释放/投递)保证最终一致性",
        ]
        repo_cons = repo_context.get("constraints", [])
    else:
        overview = f"Follow the existing Controller/Service/Mapper layering and apply the strategy set({'; '.join(strat_text)}) with minimal invasive changes for idempotency, concurrency safety, and eventual consistency."
        sequence = [
            "1)Carry requestId; perform idempotency check/dedup first",
            "2)Run core business(state transition/reserve/deduct); on failure do compensation/rollback",
            "3)Record audit/observability logs; optionally persist Outbox events",
            "4)Async jobs(reconcile/release/dispatch) ensure eventual consistency",
        ]
        repo_cons = repo_constraints_en()

    return {
        "architecture_overview": overview,
        "evidence_hints": evidence_hints,
        "strategy_set": strat_text,
        "strategy_ids": strat_ids,
        "components": components,
        "apis": apis,
        "data_model": data_model,
        "sequence_flows": sequence,
        "risk_and_mitigation": risks,
        "repo_constraints": repo_cons,
    }


def build_trace(domain: str, ref_chunks: List[str], strategy_ids: List[str], lang: str) -> List[str]:
    if lang == "en":
        variants = [
            [
                f"Step1:Set architectural constraints(Controller/Service/Mapper), domain={domain}",
                f"Step2:Ground the design on evidence chunks {ref_chunks[:3]} and extract reusable class/method hints",
                f"Step3:Apply strategies {strategy_ids} to cover concurrency/transaction/failure paths",
                "Step4:Define APIs/data model and describe idempotency/compensation/consistency behaviors",
            ],
            [
                f"Step1:Classify requirement into domain={domain}",
                f"Step2:Select strategy set {strategy_ids} with coverage considerations",
                f"Step3:Map strategies into concrete components and interfaces",
                "Step4:Output design with evidence snippets and a reasoning trace",
            ],
        ]
    else:
        variants = [
            [
                f"步骤1:确定需求属于{domain}域并遵循Controller/Service/Mapper分层",
                f"步骤2:参考证据chunk={ref_chunks[:3]}提取可复用类/方法线索",
                f"步骤3:选择策略组合{strategy_ids}并映射到具体组件落地",
                "步骤4:定义接口/数据结构并补充失败补偿、幂等与一致性说明",
            ],
            [
                f"步骤1:明确架构约束与域={domain}",
                f"步骤2:基于证据chunk={ref_chunks[:2]}做grounding",
                f"步骤3:用策略{strategy_ids}覆盖并发/事务/失败路径",
                "步骤4:输出设计+证据代码段+推理trace",
            ],
        ]
    return random.choice(variants)


def build_training_text(requirement: str, design_output: Dict[str, Any], evidence_snippets: List[Dict[str, Any]], trace_steps: List[str], lang: str, context_en: str = "") -> str:
    if lang == "en":
        ins_pool = [
            "Propose a repository-grounded design and include evidence code snippets and a reasoning trace.",
            "Based on the repo architecture, design a solution and provide evidence snippets plus a reasoning trace.",
            "Design an implementation plan aligned with layered architecture, with evidence snippets and trace.",
        ]
        ctx = f"\nBusiness context:{context_en}" if context_en else ""
        ins = random.choice(ins_pool) + ctx + f"\nRequirement:{requirement}"
        resp = {"design": design_output, "evidence_code_snippets": evidence_snippets[:3], "trace": trace_steps}
        return f"### Instruction\n{ins}\n\n### Response\n{json.dumps(resp, ensure_ascii=False, indent=2)}\n"

    ins_pool = [
        "基于本地代码仓架构生成设计方案,并提供证据代码段与推理trace。",
        "请给出可落地的设计方案(对齐现有分层),同时给出证据代码段与推理过程。",
        "围绕需求输出设计方案,要求可追溯(含原文代码段)并给出推理trace。",
    ]
    ins = random.choice(ins_pool) + f"\n需求:{requirement}"
    resp = {"design": design_output, "evidence_code_snippets": evidence_snippets[:3], "trace": trace_steps}
    return f"### Instruction\n{ins}\n\n### Response\n{json.dumps(resp, ensure_ascii=False, indent=2)}\n"


# -----------------------------
# Quota sampling for representativeness
# -----------------------------
def _quota_counts(target_n: int, ratio: Dict[str, int]) -> Dict[str, int]:
    total = sum(ratio.values()) if ratio else 1
    out = {k: int(target_n * v / total) for k, v in ratio.items()}
    while sum(out.values()) < target_n:
        k = max(ratio, key=lambda x: ratio[x])
        out[k] += 1
    return out


def quota_sample_design(
    samples: List[Dict[str, Any]],
    target_n: int,
    domain_ratio: Dict[str, int],
    strategy_ratio: Dict[str, int],
    per_file_cap: int,
    per_chunk_cap: int,
    seed: int,
) -> List[Dict[str, Any]]:
    if target_n <= 0:
        return []
    if len(samples) <= target_n:
        return list(samples)

    rnd = random.Random(seed)
    pool = list(samples)
    rnd.shuffle(pool)

    dq = _quota_counts(target_n, domain_ratio)
    sq = _quota_counts(target_n, strategy_ratio)

    used_domain = Counter()
    used_strategy = Counter()
    used_file = Counter()
    used_chunk = Counter()

    picked: List[Dict[str, Any]] = []

    def _get_tags(s: Dict[str, Any]) -> Tuple[str, List[str], str, str]:
        mv2 = s.get("meta_v2", {}) or {}
        domain = mv2.get("domain") or "mixed"
        strategy_ids = mv2.get("strategy_ids") or []
        ev0 = (s.get("evidence") or [{}])[0]
        fp = ev0.get("file_path", "") or ""
        cid = ev0.get("chunk_id", "") or ""
        return domain, strategy_ids, fp, cid

    for s in pool:
        if len(picked) >= target_n:
            break
        domain, strategy_ids, fp, cid = _get_tags(s)

        if used_file[fp] >= per_file_cap:
            continue
        if cid and used_chunk[cid] >= per_chunk_cap:
            continue
        if used_domain[domain] >= dq.get(domain, 0):
            continue

        ok_strategy = False
        for sid in strategy_ids:
            if used_strategy[sid] < sq.get(sid, 0):
                ok_strategy = True
                break
        if not ok_strategy and strategy_ids:
            continue

        picked.append(s)
        used_domain[domain] += 1
        used_file[fp] += 1
        if cid:
            used_chunk[cid] += 1
        for sid in strategy_ids:
            if used_strategy[sid] < sq.get(sid, 0):
                used_strategy[sid] += 1

    if len(picked) < target_n:
        for s in pool:
            if len(picked) >= target_n:
                break
            if s in picked:
                continue
            domain, strategy_ids, fp, cid = _get_tags(s)

            if used_file[fp] >= per_file_cap:
                continue
            if cid and used_chunk[cid] >= per_chunk_cap:
                continue
            if used_domain[domain] >= dq.get(domain, 0):
                continue

            picked.append(s)
            used_domain[domain] += 1
            used_file[fp] += 1
            if cid:
                used_chunk[cid] += 1
            for sid in strategy_ids:
                used_strategy[sid] += 1

    if len(picked) < target_n:
        for s in pool:
            if len(picked) >= target_n:
                break
            if s in picked:
                continue
            domain, strategy_ids, fp, cid = _get_tags(s)

            if used_file[fp] >= per_file_cap:
                continue
            if cid and used_chunk[cid] >= per_chunk_cap:
                continue

            picked.append(s)
            used_domain[domain] += 1
            used_file[fp] += 1
            if cid:
                used_chunk[cid] += 1
            for sid in strategy_ids:
                used_strategy[sid] += 1

    return picked


def print_coverage(samples: List[Dict[str, Any]], title: str) -> None:
    d = Counter()
    s = Counter()
    files = set()
    chunks = set()
    for x in samples:
        mv2 = x.get("meta_v2", {}) or {}
        d[mv2.get("domain", "mixed")] += 1
        for sid in (mv2.get("strategy_ids") or []):
            s[sid] += 1
        ev0 = (x.get("evidence") or [{}])[0]
        fp = ev0.get("file_path")
        cid = ev0.get("chunk_id")
        if fp:
            files.add(fp)
        if cid:
            chunks.add(cid)
    print(f"\n[{title}]n={len(samples)},unique_files={len(files)},unique_chunks={len(chunks)}")
    print(f"[{title}]domain={dict(d)}")
    print(f"[{title}]strategy_hit={dict(s)}")


def main():
    templates_nlg = load_templates()
    lang_mode = resolve_language(default_lang="zh")
    langs = ["zh", "en"] if lang_mode == "bilingual" else [lang_mode]
    strict_en = get_strict_en_level()

    index_rows = read_jsonl(ROOT / "data/raw_index/repo_index.jsonl")
    index_map = {r["chunk_id"]: r for r in index_rows}

    domain_map = json.loads((ROOT / "data/extracted/domain_map.json").read_text(encoding="utf-8"))
    rules = read_jsonl(ROOT / "data/extracted/rules.jsonl")
    flows = read_jsonl(ROOT / "data/extracted/flows.jsonl")

    code_max_chars = int(os.environ.get("DESIGN_CODE_MAX_CHARS", "2200"))
    evidence_k = int(os.environ.get("DESIGN_EVIDENCE_K", "3"))
    auto_expand = os.environ.get("DESIGN_AUTO_EXPAND", "1").strip() not in ("0", "false", "False")

    # 关键增量:每条需求每语言生成多个变体
    variants_per_req = int(os.environ.get("DESIGN_VARIANTS_PER_REQ", "3"))

    train_target = int(os.environ.get("DESIGN_TRAIN_N", "0"))
    dev_ratio = float(os.environ.get("DESIGN_DEV_RATIO", "0.1"))
    test_ratio = float(os.environ.get("DESIGN_TEST_RATIO", "0.1"))
    per_file_cap = int(os.environ.get("DESIGN_PER_FILE_CAP", "8"))
    per_chunk_cap = int(os.environ.get("DESIGN_PER_CHUNK_CAP", "2"))
    seed = int(os.environ.get("DESIGN_SEED", "52"))

    domain_ratio = {
        "stock": int(os.environ.get("DESIGN_DQ_STOCK", "45")),
        "order": int(os.environ.get("DESIGN_DQ_ORDER", "35")),
        "mixed": int(os.environ.get("DESIGN_DQ_MIXED", "20")),
    }
    strategy_ratio = {
        "outbox": int(os.environ.get("DESIGN_SQ_OUTBOX", "15")),
        "saga": int(os.environ.get("DESIGN_SQ_SAGA", "15")),
        "idempotency_key": int(os.environ.get("DESIGN_SQ_IDEMPOTENCY", "20")),
        "state_machine": int(os.environ.get("DESIGN_SQ_STATEMACHINE", "15")),
        "ttl_reserve": int(os.environ.get("DESIGN_SQ_TTLRESERVE", "15")),
        "optimistic_lock": int(os.environ.get("DESIGN_SQ_OPTLOCK", "20")),
    }

    cfg_path = ROOT / "configs/design_requirements.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError("缺少configs/design_requirements.yaml,请先创建该文件")
    cfg = load_yaml(cfg_path)
    templates = cfg.get("templates", []) or []

    extra_n = int(os.environ.get("DESIGN_AUTO_EXPAND_N", "120"))
    auto_reqs = expand_requirements_from_flows_and_rules(flows, rules, max_n=extra_n) if auto_expand else []

    all_reqs: List[Dict[str, Any]] = []
    for t in templates:
        req_zh = (t.get("requirement") or "").strip()
        req_en = (t.get("requirement_en") or "").strip()
        ctx_en = (t.get("context_en") or "").strip()

        all_reqs.append({
            "id": t.get("id") or make_id("req", req_zh or req_en),
            "domain": t.get("domain", "mixed"),
            "difficulty": t.get("difficulty", "medium"),
            "source_type": "yaml",
            "source_id": t.get("id"),
            "requirement_zh": req_zh,
            "requirement_en": req_en,
            "context_en": ctx_en,
        })

    all_reqs.extend(auto_reqs)

    # 去重(按文本去重,变体在后面生成)
    seen = set()
    deduped = []
    for r in all_reqs:
        key = (r.get("domain"), r.get("requirement_zh") or "", r.get("requirement_en") or "")
        if key not in seen and (r.get("requirement_zh") or r.get("requirement_en")):
            seen.add(key)
            deduped.append(r)
    all_reqs = deduped

    samples: List[Dict[str, Any]] = []

    for r in all_reqs:
        domain = r.get("domain", "mixed")
        rid = r.get("id") or make_id("req", (r.get("requirement_zh") or r.get("requirement_en") or ""))

        repo_context = summarize_repo_context(domain_map, domain)
        base_ref_chunks = pick_reference_chunks(rules, flows, domain)
        if not base_ref_chunks:
            continue

        # 为了多样性:每次变体打乱证据chunk顺序
        for v in range(max(1, variants_per_req)):
            ref_chunks = list(base_ref_chunks)
            random.shuffle(ref_chunks)

            evidence_refs, evidence_snippets = build_evidence(ref_chunks, index_map, k=evidence_k, code_max_chars=code_max_chars)
            if not evidence_snippets:
                continue

            name_hints = extract_names_from_snippets(evidence_snippets)

            for lang in langs:
                if lang == "en":
                    requirement = (r.get("requirement_en") or "").strip()
                    context_en = (r.get("context_en") or "").strip()

                    if not requirement:
                        if strict_en >= 2:
                            continue
                        if strict_en == 1:
                            zh_raw = (r.get("requirement_zh") or "").strip()
                            if not zh_raw:
                                continue
                            requirement = f"Please design a solution for the following requirement(original in Chinese):\n{zh_raw}"
                        else:
                            requirement = (r.get("requirement_zh") or "").strip()
                            if not requirement:
                                continue

                    if has_cjk(requirement) and strict_en >= 2:
                        continue

                else:
                    requirement = (r.get("requirement_zh") or "").strip() or (r.get("requirement_en") or "").strip()
                    context_en = ""

                if not requirement:
                    continue

                # 每个变体重新抽策略,形成不同样本
                strategies = pick_strategies(domain)

                design_output = generate_design_output(requirement, domain, repo_context, name_hints, strategies, lang)

                if "design" in templates_nlg and lang in templates_nlg["design"]:
                    overview_prefix = templates_nlg["design"][lang].get("overview_prefix")
                    if isinstance(overview_prefix, str) and overview_prefix:
                        if lang == "en" and has_cjk(overview_prefix):
                            pass
                        else:
                            design_output["architecture_overview"] = overview_prefix.format(domain=domain)

                trace_steps = build_trace(domain, ref_chunks, design_output["strategy_ids"], lang)
                text = build_training_text(requirement, design_output, evidence_snippets, trace_steps, lang, context_en=context_en)

                meta_v2 = {
                    "task_type": "design",
                    "language": lang,
                    "domain": domain,
                    "difficulty": r.get("difficulty", "medium"),
                    "requirement_id": rid,
                    "requirement_source": {"type": r.get("source_type"), "id": r.get("source_id")},
                    "repo_context": repo_context,
                    "evidence": evidence_refs,
                    "evidence_snippets": [{
                        "chunk_id": e.get("chunk_id"),
                        "file_path": e.get("file_path"),
                        "start_line": e.get("start_line"),
                        "end_line": e.get("end_line"),
                        "code_lang": e.get("code_lang"),
                        "code": e.get("code"),
                    } for e in (evidence_snippets or [])],
                    "trace_digest": trace_steps,
                    "name_hints": name_hints,
                    "strategy_ids": design_output.get("strategy_ids") or [],
                    "generator": "step05_v8_en_clean_multi_variant",
                    "variant_id": v,
                    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "source": "AutoCodeDataPipeline",
                    "context_en": context_en,
                }

                sample = {
                    "sample_id": make_id("design", f"{rid}|{lang}|v{v}|{','.join(meta_v2['strategy_ids'])}|{sha1_text(text)[:8]}"),
                    "task_type": "design",
                    "language": lang,
                    "requirement": requirement,
                    "repo_context": repo_context,
                    "design_output": design_output,
                    "evidence": [{
                        "chunk_id": e.get("chunk_id"),
                        "file_path": e.get("file_path"),
                        "start_line": e.get("start_line"),
                        "end_line": e.get("end_line"),
                        "code_lang": e.get("code_lang"),
                        "code": e.get("code"),
                    } for e in (evidence_snippets or [])[:evidence_k]],
                    "trace": {"reasoning_steps": trace_steps},
                    "text": text,
                    "meta_v2": meta_v2,
                }
                samples.append(sample)

    random.shuffle(samples)
    n_total = len(samples)
    if n_total == 0:
        raise RuntimeError("未生成任何Design样本:可能英文被DESIGN_STRICT_EN过滤,或evidence抽取为空。")

    if train_target <= 0:
        train_target = int(n_total * (1.0 - dev_ratio - test_ratio))
        train_target = max(1, train_target)

    remaining = list(samples)
    train = quota_sample_design(remaining, train_target, domain_ratio, strategy_ratio, per_file_cap, per_chunk_cap, seed=seed)
    train_set = set([s["sample_id"] for s in train])
    remaining2 = [s for s in remaining if s["sample_id"] not in train_set]

    dev_n = int(len(train) * dev_ratio / max(1e-9, (1.0 - dev_ratio - test_ratio)))
    test_n = int(len(train) * test_ratio / max(1e-9, (1.0 - dev_ratio - test_ratio)))
    dev_n = max(0, min(dev_n, len(remaining2)))
    test_n = max(0, min(test_n, len(remaining2) - dev_n))

    dev = quota_sample_design(remaining2, dev_n, domain_ratio, strategy_ratio, per_file_cap, per_chunk_cap, seed=seed + 1) if dev_n > 0 else []
    dev_set = set([s["sample_id"] for s in dev])
    remaining3 = [s for s in remaining2 if s["sample_id"] not in dev_set]
    test = quota_sample_design(remaining3, test_n, domain_ratio, strategy_ratio, per_file_cap, per_chunk_cap, seed=seed + 2) if test_n > 0 else []

    write_jsonl(ROOT / "data/dataset/design_train.jsonl", train)
    write_jsonl(ROOT / "data/dataset/design_dev.jsonl", dev)
    write_jsonl(ROOT / "data/dataset/design_test.jsonl", test)
    write_jsonl(ROOT / "data/samples/design_samples.jsonl", (train + dev + test)[:40])

    print(f"Design样本生成完成:total={n_total},train={len(train)},dev={len(dev)},test={len(test)},langs={langs},DESIGN_STRICT_EN={strict_en}")
    print(f"[Knobs]DESIGN_AUTO_EXPAND={auto_expand},DESIGN_AUTO_EXPAND_N={os.environ.get('DESIGN_AUTO_EXPAND_N','120')},DESIGN_VARIANTS_PER_REQ={variants_per_req}")
    print(f"[Knobs]DESIGN_CODE_MAX_CHARS={code_max_chars},DESIGN_EVIDENCE_K={evidence_k}")
    print(f"[Knobs]DESIGN_PER_FILE_CAP={per_file_cap},DESIGN_PER_CHUNK_CAP={per_chunk_cap},DESIGN_TRAIN_N={train_target},DESIGN_DEV_RATIO={dev_ratio},DESIGN_TEST_RATIO={test_ratio}")
    print_coverage(train, "design_train_coverage")
    if dev:
        print_coverage(dev, "design_dev_coverage")
    if test:
        print_coverage(test, "design_test_coverage")


if __name__ == "__main__":
    main()
