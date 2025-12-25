#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step04
根据rules+flows+repo_index自动生成QA样本
目标:多样性+代表性(配额覆盖)+可追溯(evidence+trace)

特性:
- 英文优先消费Step03的title_en/description_en/name_en/step.name_en,缺失才fallback推断
- 问法多样性:按focus(transaction/concurrency/stock/status/exception/general)生成多种问法
- 答案多样性:按证据关键行+focus生成不同结论句与检查点
- 代表性:生成后按boundary+focus配额抽样,并对file_path/chunk_id限流防止刷屏
- 输出: data/dataset/{train,dev,test}.jsonl + data/samples/qa_samples.jsonl
"""

import json
import random
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
import os
import re
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[1]

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


# 检查字符串是否包含中文字符。用于英文模式下过滤/兜底。
def has_cjk(s: str) -> bool:
    return bool(s and _CJK_RE.search(s))


# sha1_text(s)
def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


# 返回prefix_+sha1(seed)前10位。用于sample_id稳定且短。
def make_id(prefix: str, seed: str) -> str:
    return prefix + "_" + sha1_text(seed)[:10]


# -----------------------------
# Language config
# -----------------------------
def get_en_strict_level() -> int:
    """
    读取环境变量QA_STRICT_EN并规范到0/1/2。控制英文样本的清洗策略。
    QA_STRICT_EN:
      0=off(不清洗)
      1=fallback(默认:遇中文用fallback替换)
      2=drop(严格:遇中文直接跳过英文样本)
    """
    v = os.environ.get("QA_STRICT_EN", "1").strip()
    try:
        lv = int(v)
    except Exception:
        lv = 1
    return 0 if lv < 0 else 2 if lv > 2 else lv


def en_pick(text_raw: str, text_en: str, fallback: str, strict_lv: int) -> str:
    if text_en and (not has_cjk(text_en)):
        return text_en
    if text_raw and (not has_cjk(text_raw)):
        return text_raw
    if strict_lv >= 2:
        return ""
    return fallback


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
    '''决定最终语言模式的优先级：
    1)优先看环境变量LANG是否明确设置为zh/en/bilingual等
    2)否则读取configs/runtime.yaml里的language.mode
    3)否则用默认值'''
    env_lang = os.environ.get("LANG")
    if env_lang and env_lang.strip().lower() in ("zh", "en", "bilingual", "bi", "both", "zh+en", "en+zh"):
        return normalize_lang(env_lang)
    cfg_path = ROOT / "configs/runtime.yaml"
    if cfg_path.exists():
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        mode = (cfg.get("language") or {}).get("mode")
        if mode:
            return normalize_lang(mode)
    return normalize_lang(default_lang)


# -----------------------------
# IO
# -----------------------------
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


# -----------------------------
# 证据构造相关
# -----------------------------
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


def trim_code(code: str, max_chars: int) -> str:  # 截断代码避免单条样本太长
    if not code:
        return ""
    if len(code) <= max_chars:
        return code
    return code[: max_chars - 20] + "\n/* ... truncated ... */\n"


# build_evidence(chunk_id,index_map,code_max_chars)
def build_evidence(chunk_id: str, index_map: Dict[str, Dict[str, Any]], code_max_chars: int) -> Dict[str, Any]:
    r = index_map[chunk_id]
    return {
        "chunk_id": chunk_id,
        "file_path": r["file_path"],
        "start_line": r["start_line"],
        "end_line": r["end_line"],
        "content": r.get("content") or "",
        "code_lang": detect_code_lang(r["file_path"]),
        "code": trim_code(r.get("content") or "", code_max_chars),
    }


# ev["code"]包装成markdown代码块：lang\ncode\n 用于最终答案里的【原文代码段】/ [Code Snippet]部分。
def format_code_block(ev: Dict[str, Any]) -> str:
    code = ev.get("code") or ""
    if not code:
        return ""
    lang = ev.get("code_lang") or ""
    return f"```{lang}\n{code}\n```"


# -----------------------------
# boundary/focus相关
# -----------------------------
def infer_boundary(file_path: str) -> str:
    # 简单启发式：路径里含/controller/service/mapper/dao/config或后缀Controller.java等，映射到boundary标签。用于代表性配额和统计覆盖。
    fp = (file_path or "").replace("\\", "/").lower()
    if "/controller/" in fp or fp.endswith("controller.java"):
        return "controller"
    if "/service/" in fp or fp.endswith("service.java"):
        return "service"
    if "/mapper/" in fp or fp.endswith("mapper.java") or "/dao/" in fp:
        return "mapper"
    if fp.endswith((".yml", ".yaml", ".properties")) or "/config/" in fp:
        return "config"
    return "other"


# 关键行匹配的词表与正则：事务、锁、库存、状态、异常、分支关键字、timeout、幂等等。用于从证据中抽“信息密度最高的行”。
_KEY_PATTERNS = [
    r"@Transactional", r"\btransaction\b", r"\brollback\b",
    r"\block\b", r"reserve", r"deduct", r"reduce", r"release", r"unlock",
    r"status", r"state",
    r"synchronized", r"redisson", r"redis.*lock", r"select\s+for\s+update",
    r"\bif\b", r"\belse\b", r"\bswitch\b", r"\bcase\b",
    r"\breturn\b", r"\bthrow\b", r"Exception",
    r"timeout", r"cancel", r"closeOrder", r"refund",
    r"idempot", r"requestid", r"unique", r"repeat",
]
_key_re = re.compile("|".join(_KEY_PATTERNS), re.IGNORECASE)


def extract_key_lines(code: str, max_lines: int = 10, max_len: int = 220) -> List[str]:
    '''逐行扫描code，只要命中_key_re就加入out，去重、截长，最多取max_lines行。
        重要：它让答案不是纯模板，而是“模板+证据关键行”。'''
    if not code:
        return []
    out: List[str] = []
    for raw in code.splitlines():
        line = raw.strip()
        if not line:
            continue
        if _key_re.search(line):
            if len(line) > max_len:
                line = line[: max_len - 3] + "..."
            if line not in out:
                out.append(line)
        if len(out) >= max_lines:
            break
    return out


# 把关键行拼成blob后按规则推focus
# focus--从证据代码的“关键行”里推断主题：
def infer_focus_from_key_lines(lines: List[str]) -> str:
    blob = "\n".join(lines).lower()
    if "transactional" in blob or "rollback" in blob or "transaction" in blob:
        return "transaction"
    if "idempot" in blob or "requestid" in blob or "unique" in blob or "repeat" in blob:
        return "idempotency"
    if "synchronized" in blob or "for update" in blob or "redisson" in blob or ("lock" in blob and "unlock" in blob):
        return "concurrency"
    if "reserve" in blob or "deduct" in blob or "reduce" in blob or "release" in blob or "unlock" in blob or "lock" in blob:
        return "stock"
    if "status" in blob or "state" in blob:
        return "status"
    if "throw" in blob or "exception" in blob or "failed(" in blob:
        return "exception"
    return "general"


# 把focus映射成人类可读的中英文短语，供结论句和trace用。
def focus_display(lang: str, focus: str) -> str:
    zh = {
        "transaction": "事务一致性",
        "idempotency": "幂等/重复请求",
        "concurrency": "并发控制",
        "stock": "库存变更",
        "status": "状态机/状态校验",
        "exception": "异常与失败分支",
        "general": "关键逻辑",
    }
    en = {
        "transaction": "transactional consistency",
        "idempotency": "idempotency/duplicate requests",
        "concurrency": "concurrency control",
        "stock": "stock change",
        "status": "state/status guard",
        "exception": "failure/exception branches",
        "general": "key logic",
    }
    return (zh if lang == "zh" else en).get(focus, focus)


# -----------------------------
# 排除低价值文件
# -----------------------------
def compile_exclude_regex() -> re.Pattern:
    """
    默认排除:
    - mall-mbg(自动生成)
    - /model/目录(大量POJO)
    - *Example.java(生成模板)
    可用QA_EXCLUDE_PATHS覆盖
    """
    pat = os.environ.get(
        "QA_EXCLUDE_PATHS",
        r"(?:^|/)(mall-mbg)(?:/|$)|(?:^|/)(model)(?:/|$)|Example\.java$"
    ).strip()
    return re.compile(pat, re.IGNORECASE)


def is_excluded_file(file_path: str, ex_re: re.Pattern) -> bool:
    fp = (file_path or "").replace("\\", "/")
    return bool(ex_re.search(fp))


# -----------------------------
# 英文兜底推断(只在英文模式且缺字段时用)
# -----------------------------
_METHOD_RE = re.compile(r"\b(public|private|protected)\s+[\w<>\[\]]+\s+(\w+)\s*\(", re.IGNORECASE)


def infer_en_topic_from_evidence(ev: Dict[str, Any], key_lines: List[str]) -> Tuple[str, str]:
    # 当rule/flow缺少英文title/desc时，生成一个“可用但一般”的英文topic
    fp = (ev.get("file_path") or "").replace("\\", "/")
    code = ev.get("content") or ""
    ext = (ev.get("code_lang") or "").lower()

    if ext in ("yaml", "properties"):
        k = ""
        for ln in key_lines:
            m = re.search(r"^\s*([A-Za-z0-9_.-]+)\s*[:=]\s*(.+)$", ln)
            if m:
                k = m.group(1)
                break
        title = "Application configuration"
        desc = "Explain what this configuration controls and what the key settings mean."
        if k:
            title = f"Configuration for {k}"
            desc = f"Explain the purpose of '{k}' and how its value affects runtime behavior."
        return title, desc

    role = "implementation"
    if "/controller/" in fp or fp.endswith("Controller.java"):
        role = "API endpoint behavior"
    elif "/service/" in fp:
        role = "service logic"
    elif "/mapper/" in fp or "Mapper.java" in fp:
        role = "data access(mapper) behavior"
    elif "/dao/" in fp:
        role = "DAO contract"

    method = ""
    for ln in code.splitlines():
        mm = _METHOD_RE.search(ln)
        if mm:
            method = mm.group(2)
            break

    base_name = fp.split("/")[-1]
    if method:
        title = f"{base_name}:{method} {role}"
        desc = f"Explain the {role} of '{method}' based on evidence code."
    else:
        title = f"{base_name} {role}"
        desc = f"Explain the {role} based on evidence code."
    return title, desc


# -----------------------------
# 问法多样性池
# -----------------------------

# 按focus分类的规则问句模板池。choose_rule_question会随机挑一个并format注入title和file_path。
_RULE_Q_ZH = {
    "transaction": [
        "这段实现的事务边界在哪里?哪些操作必须保持原子性?请结合{file_path}说明。",
        "代码中如何保证事务一致性与回滚语义?请结合{file_path}的关键分支解释。",
    ],
    "idempotency": [
        "如何避免重复请求导致重复执行?幂等键/去重点在哪?请结合{file_path}说明。",
        "如果同一请求被重放两次会发生什么?代码如何做到无副作用?请引用{file_path}。",
    ],
    "concurrency": [
        "这段代码如何处理并发冲突或超卖风险?请结合{file_path}中的锁/更新语句说明。",
        "并发场景下可能出现什么数据竞争?代码采取了什么控制手段?请引用{file_path}。",
    ],
    "stock": [
        "库存锁定/扣减/释放的关键条件是什么?请结合{file_path}解释触发路径与失败处理。",
        "这段库存相关逻辑如何避免错误扣减或重复释放?请结合{file_path}的关键判断说明。",
    ],
    "status": [
        "订单状态校验/流转的约束是什么?哪些状态会被拒绝?请结合{file_path}解释。",
        "状态机约束在代码里是如何体现的?请从{file_path}提取关键判断并说明后果。",
    ],
    "exception": [
        "失败分支/异常抛出代表什么业务语义?调用方应如何处理?请结合{file_path}说明。",
        "这段代码在失败时返回值/异常如何定义?会触发哪些补偿或中断?请引用{file_path}。",
    ],
    "general": [
        "解释{title}在{file_path}中的实现要点,请给出关键分支与业务含义。",
        "结合{file_path}说明{title}的核心逻辑,并指出关键判断/返回路径。",
    ],
}

_RULE_Q_EN = {
    "transaction": [
        "Where is the transaction boundary in {file_path}?Which operations must be atomic?Explain with evidence.",
        "How does the code enforce transactional consistency and rollback semantics in {file_path}?Explain key branches.",
    ],
    "idempotency": [
        "How does the implementation avoid duplicated execution for repeated requests in {file_path}?Point out idempotency/dedup logic.",
        "If the same request is replayed twice, what happens and why is it safe?Explain using evidence from {file_path}.",
    ],
    "concurrency": [
        "How does this code handle concurrency conflicts or overselling risk in {file_path}?Explain locking/update logic with evidence.",
        "Under concurrency, what race conditions might happen and how does the code mitigate them in {file_path}?",
    ],
    "stock": [
        "What are the key conditions for stock lock/deduct/release in {file_path}?Explain the trigger paths and failure handling.",
        "How does the stock logic avoid double deduction or repeated release in {file_path}?Explain the key checks.",
    ],
    "status": [
        "What state/status constraints are enforced in {file_path}?Which states are rejected and why?",
        "How is the state-machine constraint implemented in {file_path}?Extract key guards and explain consequences.",
    ],
    "exception": [
        "What do the failure branches/exceptions mean in business terms in {file_path}?How should callers handle them?",
        "How are errors represented(return values/exceptions) in {file_path}?What compensation or abort behavior follows?",
    ],
    "general": [
        "Explain how '{title}' is implemented in {file_path}, highlighting key branches and business meaning.",
        "Based on {file_path}, summarize the core logic of '{title}' and point out key conditions/return paths.",
    ],
}

# 流程问句模板池。choose_flow_question随机挑一个注入flow_name。
_FLOW_Q_ZH = [
    "请解释流程“{flow_name}”的端到端步骤,并标注每一步对应的代码位置。",
    "从调用链角度梳理“{flow_name}”包含哪些关键步骤?每一步的证据代码在哪里?",
    "“{flow_name}”流程的关键分支(成功/失败/取消/超时)如何走?请结合证据代码说明。",
]

_FLOW_Q_EN = [
    "Explain the end-to-end steps of the flow '{flow_name}' and annotate the code location for each step.",
    "From the call-chain perspective, list the key steps of '{flow_name}' and point to evidence code for each step.",
    "How do the key branches(success/failure/cancel/timeout) work in '{flow_name}'?Explain with evidence code.",
]


_FLOW_RULE_Q_ZH = [
    "流程“{flow_name}”涉及哪些业务规则/约束?请列出规则并说明它们在代码中的证据位置(强关联/弱关联)。",
    "结合流程“{flow_name}”的执行步骤,梳理其关联的业务规则清单,并标注每条规则的证据代码chunk。",
    "在“{flow_name}”流程中,哪些规则用于约束状态/库存/幂等/失败分支?请用证据代码说明。",
]

_FLOW_RULE_Q_EN = [
    "What business rules/constraints are involved in the flow '{flow_name}'?List the rules and point to evidence chunks(strong/weak links).",
    "Based on the steps of '{flow_name}', summarize the related rules and annotate evidence code chunks for each rule.",
    "In the flow '{flow_name}', which rules constrain state/stock/idempotency/failure branches?Explain with evidence code.",
]



def choose_rule_question(lang: str, focus: str, title: str, file_path: str) -> str:
    pool = _RULE_Q_ZH if lang == "zh" else _RULE_Q_EN
    arr = pool.get(focus) or pool["general"]
    return random.choice(arr).format(title=title, file_path=file_path)


def choose_flow_question(lang: str, flow_name: str) -> str:
    arr = _FLOW_Q_ZH if lang == "zh" else _FLOW_Q_EN
    return random.choice(arr).format(flow_name=flow_name)


def choose_flow_rule_question(lang: str, flow_name: str) -> str:
    arr = _FLOW_RULE_Q_ZH if lang == "zh" else _FLOW_RULE_Q_EN
    return random.choice(arr).format(flow_name=flow_name)


def _rule_title_for_lang(rule: Dict[str, Any], ev: Dict[str, Any], key_lines: List[str], lang: str, strict_lv: int) -> str:
    """给flow*rule QA用：为rule挑一个合适的title(zh/en)，必要时兜底为纯英文label。"""
    rid = rule.get("rule_id") or rule.get("id") or ""
    title_zh = (rule.get("title") or rid or "业务规则").strip()
    title_en = (rule.get("title_en") or "").strip()

    if lang != "en":
        return title_zh

    # EN:尽量用title_en，否则用raw(如果本身不含中文)，否则infer兜底
    if title_en and (not has_cjk(title_en)):
        return title_en
    if title_zh and (not has_cjk(title_zh)):
        return title_zh

    infer_title, _infer_desc = infer_en_topic_from_evidence(ev, key_lines)
    fb = infer_title or (f"Rule {rid}" if rid else "Rule")
    # strict=2时如果还含中文，直接给纯英文label
    if strict_lv >= 2 and has_cjk(fb):
        return f"Rule {rid}" if rid else "Rule"
    return fb


def generate_flow_rule_qa(
    flow: Dict[str, Any],
    rules_by_id: Dict[str, Dict[str, Any]],
    index_map: Dict[str, Dict[str, Any]],
    lang: str,
    code_max_chars: int,
    flow_max_evs: int,
    rule_max_evs: int,
    ex_re: re.Pattern,
) -> Dict[str, Any]:
    """
    新增: 显式flow*rule QA
    - 消费flow.related_rules
    - 输出qa_type=flow_rule
    - 答案里给: 流程步骤+代码位置, 以及关联规则清单(强/弱关联)+证据定位
    """
    related_rule_ids = flow.get("related_rules") or []
    if not related_rule_ids:
        return {}

    # 1) flow侧证据
    flow_evs: List[Dict[str, Any]] = []
    flow_step_chunk_ids: List[str] = []
    for s in flow.get("steps", []) or []:
        cid = s.get("evidence_chunk")
        if cid:
            flow_step_chunk_ids.append(cid)
        if cid and cid in index_map:
            ev = build_evidence(cid, index_map, code_max_chars)
            if is_excluded_file(ev["file_path"], ex_re):
                continue
            flow_evs.append(ev)
        if len(flow_evs) >= flow_max_evs:
            break
    if not flow_evs:
        return {}

    # flow chunk集合用于强/弱关联
    flow_chunk_set = set([c for c in (flow.get("evidence_chunks") or flow_step_chunk_ids) if c])

    flow_id = flow.get("flow_id") or flow.get("id") or ""
    domain = flow.get("domain") or "mixed"

    strict_lv = get_en_strict_level()
    if lang == "en":
        flow_name_raw = (flow.get("name") or "").strip()
        flow_name_en = (flow.get("name_en") or "").strip()
        fb = f"Flow {flow_id}" if flow_id else "Business flow"
        flow_name = flow_name_raw if strict_lv == 0 else en_pick(flow_name_raw, flow_name_en, fb, strict_lv)
        if strict_lv >= 2 and (not flow_name):
            return {}
    else:
        flow_name = (flow.get("name") or "").strip() or (flow_id or "业务流程")

    # 2) rule侧证据
    rule_items = []
    rule_evs: List[Dict[str, Any]] = []

    for rid in related_rule_ids:
        r = rules_by_id.get(rid)
        if not r:
            continue
        r_chunks = r.get("evidence_chunks") or []
        if not r_chunks:
            continue

        # 只取首个chunk做rule证据(聚焦)
        rcid = r_chunks[0]
        if rcid not in index_map:
            continue

        ev = build_evidence(rcid, index_map, code_max_chars)
        if is_excluded_file(ev["file_path"], ex_re):
            continue

        key_lines_r = extract_key_lines(ev.get("content") or "", max_lines=10)
        title = _rule_title_for_lang(r, ev, key_lines_r, lang, strict_lv)

        # 强/弱关联：规则证据chunk是否落在flow证据集合中
        link = "strong" if (rcid in flow_chunk_set or any(c in flow_chunk_set for c in r_chunks[:3])) else "weak"

        rule_items.append({
            "rule_id": rid,
            "title": title,
            "link": link,
            "evidence_chunk": rcid,
            "file_path": ev["file_path"],
            "start_line": ev["start_line"],
            "end_line": ev["end_line"],
        })
        rule_evs.append(ev)

        if len(rule_evs) >= rule_max_evs:
            break

    if not rule_items:
        return {}

    # 3) 问题
    q = choose_flow_rule_question(lang, flow_name=flow_name)

    # 4) focus/boundary沿用flow首证据推断
    key_lines = extract_key_lines(flow_evs[0].get("content") or "", max_lines=10)
    focus = infer_focus_from_key_lines(key_lines)
    boundary = infer_boundary(flow_evs[0]["file_path"])

    # 5) core：流程+规则清单
    flow_part = build_flow_answer(flow, flow_evs, lang)

    if lang == "zh":
        rules_lines = []
        for it in rule_items:
            tag = "强关联" if it["link"] == "strong" else "弱关联"
            rules_lines.append(
                f"- {it['title']}(rule_id={it['rule_id']},{tag})->{it['file_path']}:L{it['start_line']}-{it['end_line']}(chunk_id={it['evidence_chunk']})"
            )
        rules_part = "关联规则清单(含强/弱关联标注):\n" + "\n".join(rules_lines)
        core = f"关注点:{focus_display(lang, focus)}\n\n{flow_part}\n\n{rules_part}"
    else:
        rules_lines = []
        for it in rule_items:
            tag = "strong" if it["link"] == "strong" else "weak"
            rules_lines.append(
                f"- {it['title']}(rule_id={it['rule_id']},{tag})->{it['file_path']}:L{it['start_line']}-{it['end_line']}(chunk_id={it['evidence_chunk']})"
            )
        rules_part = "Related rules(with strong/weak links):\n" + "\n".join(rules_lines)
        core = f"Focus:{focus_display(lang, focus)}\n\n{flow_part}\n\n{rules_part}"

    trace_steps = build_trace(lang, focus, flow_name)

    # wrap_answer只展示前3个代码块，因此把flow+rule证据拼到一起即可
    evidences_all = flow_evs + rule_evs
    a = wrap_answer(core, evidences_all, trace_steps, lang)
    text = f"### Instruction\n{q}\n\n### Response\n{a}\n"

    meta_v2 = {
        "task_type": "qa",
        "language": lang,
        "domain": domain,
        "qa_type": "flow_rule",
        "focus": focus,
        "boundary": boundary,
        "flow_id": flow_id,
        "related_rule_ids": [x["rule_id"] for x in rule_items],
        "rule_links": rule_items,  # 强/弱关联+定位信息
        "evidence": [{
            "chunk_id": e["chunk_id"],
            "file_path": e["file_path"],
            "start_line": e["start_line"],
            "end_line": e["end_line"],
        } for e in evidences_all],
        "evidence_snippets": [{
            "chunk_id": e["chunk_id"],
            "file_path": e["file_path"],
            "start_line": e["start_line"],
            "end_line": e["end_line"],
            "code_lang": e.get("code_lang"),
            "code": e.get("code"),
        } for e in evidences_all],
        "trace_digest": trace_steps,
        "generator": "step04_v6_diversity_quota",
        "source": "AutoCodeDataPipeline",
    }

    return {
        "sample_id": make_id("qa", f"flow_rule|{flow_id}|{lang}|{focus}|{sha1_text(q)[:6]}"),
        "task_type": "qa",
        "language": lang,
        "question": q,
        "answer": a,
        "evidence": [{
            "chunk_id": e["chunk_id"],
            "file_path": e["file_path"],
            "start_line": e["start_line"],
            "end_line": e["end_line"],
            "content": e.get("content") or "",
        } for e in evidences_all],
        "trace": {
            "type": "flow_rule_based",
            "flow_id": flow_id,
            "rule_ids": [x["rule_id"] for x in rule_items],
            "reasoning_steps": trace_steps,
        },
        "meta": {
            "domain": domain,
            "qa_type": "flow_rule",
            "focus": focus,
            "boundary": boundary,
            "generator": "step04_v6_diversity_quota",
            "source": "AutoCodeDataPipeline",
        },
        "text": text,
        "meta_v2": meta_v2,
    }
# -----------------------------
# 答案构造
# -----------------------------



def build_rule_conclusion(lang: str, title: str, focus: str, key_lines: List[str]) -> str:
    '''核心是：按focus输出不同的“结论句+检查点”。
    不逐行解释代码，而是给训练一个更像“架构/规则总结”的答案骨架'''
    f = focus_display(lang, focus)
    if lang == "zh":
        hints = []
        if key_lines:
            hints.append(f"证据显示该段主要关注{f}。")
        if focus == "transaction":
            hints.append("结论:状态变更与持久化应保持原子性,或提供可重试补偿。")
        elif focus == "idempotency":
            hints.append("结论:需要幂等键/去重记录,重复请求不产生副作用。")
        elif focus == "concurrency":
            hints.append("结论:需要锁/条件更新/重试,避免并发下重复执行与超卖。")
        elif focus == "stock":
            hints.append("结论:库存锁定/扣减/释放应具备幂等与失败回滚路径,避免不一致。")
        elif focus == "status":
            hints.append("结论:应先校验合法状态再流转,非法分支要明确返回语义。")
        elif focus == "exception":
            hints.append("结论:失败分支要统一异常/返回语义,并保证上层可感知与可补偿。")
        else:
            hints.append("结论:可从关键条件与返回路径推断业务约束与边界行为。")
        return f"{title}的实现要点如下:\n- " + "\n- ".join(hints)
    else:
        hints = []
        if key_lines:
            hints.append(f"The evidence indicates the main focus is {f}.")
        if focus == "transaction":
            hints.append("Conclusion: keep state changes and persistence atomic, or provide retryable compensations.")
        elif focus == "idempotency":
            hints.append("Conclusion: use idempotency keys/dedup records to make retries safe.")
        elif focus == "concurrency":
            hints.append("Conclusion: use locks/conditional updates/retries to mitigate races and overselling.")
        elif focus == "stock":
            hints.append("Conclusion: make stock lock/deduct/release idempotent with rollback paths.")
        elif focus == "status":
            hints.append("Conclusion: validate allowed transitions first; define clear semantics for rejected paths.")
        elif focus == "exception":
            hints.append("Conclusion: standardize error semantics and ensure callers can observe and compensate.")
        else:
            hints.append("Conclusion: infer constraints from key conditions and return paths.")
        return f"Key points for '{title}':\n- " + "\n- ".join(hints)


# 随机选一组“推理步骤模板”，用于trace_digest和答案的【推理过程】部分
def build_trace(lang: str, focus: str, title_or_flow: str) -> List[str]:
    f = focus_display(lang, focus)
    if lang == "zh":
        variants = [
            [f"识别主题:{title_or_flow}", f"抽取与{f}相关的关键条件/分支", "用证据定位代码段并组织结论"],
            [f"定位实现点:{title_or_flow}", "摘录关键行(条件/返回/异常)", f"围绕{f}解释业务含义与边界行为"],
            [f"确定关注点:{f}", "关联关键行到业务语义", "输出结论+证据引用+原文代码段"],
        ]
    else:
        variants = [
            [f"Identify target:{title_or_flow}", f"Extract key lines related to {f}", "Synthesize a grounded conclusion with citations"],
            [f"Locate implementation:{title_or_flow}", "Quote key branches(conditions/returns/exceptions)", f"Explain behavior with focus on {f}"],
            [f"Set focus:{f}", "Map key lines to business semantics", "Output conclusion+evidence+code snippet"],
        ]
    return random.choice(variants)


# 把核心答案包成统一格式
def wrap_answer(answer_core: str, evidences: List[Dict[str, Any]], trace_steps: List[str], lang: str) -> str:
    is_zh = (lang == "zh")
    ev_lines = []
    ev_blocks = []
    for ev in (evidences or [])[:3]:
        loc = f"{ev['file_path']}:L{ev['start_line']}-{ev['end_line']}"
        ev_lines.append(f"- {loc}(chunk_id={ev['chunk_id']})")
        ev_blocks.append(format_code_block(ev))
    ev_ref = "\n".join(ev_lines) if ev_lines else "- N/A"
    ev_code = "\n\n".join([b for b in ev_blocks if b]) if ev_blocks else ""
    steps = (trace_steps or [])[:6]

    if is_zh:
        return (
            f"【结论】\n{answer_core.strip()}\n\n"
            f"【证据引用】\n{ev_ref}\n\n"
            f"【原文代码段】\n{ev_code if ev_code else 'N/A'}\n\n"
            f"【推理过程】\n" + "\n".join([f"{i+1}. {st}" for i, st in enumerate(steps)])
        )
    return (
        f"[Conclusion]\n{answer_core.strip()}\n\n"
        f"[Evidence]\n{ev_ref}\n\n"
        f"[Code Snippet]\n{ev_code if ev_code else 'N/A'}\n\n"
        f"[Trace]\n" + "\n".join([f"{i+1}. {st}" for i, st in enumerate(steps)])
    )


# 把flow.steps整理成“步骤列表+每步代码位置”：
def build_flow_answer(flow: Dict[str, Any], evs: List[Dict[str, Any]], lang: str) -> str:
    steps = flow.get("steps") or []
    if not steps:
        return "N/A"

    lines: List[str] = []
    for i, s in enumerate(steps, 1):
        if lang == "en":
            step_name = (s.get("name_en") or "").strip() or (s.get("op") or "").strip() or f"step{i}"
        else:
            step_name = (s.get("name") or "").strip() or (s.get("why") or "").strip() or (s.get("op") or "").strip() or f"步骤{i}"

        cid = s.get("evidence_chunk")
        loc = ""
        if cid:
            for e in evs:
                if e.get("chunk_id") == cid:
                    loc = f"{e['file_path']}:L{e['start_line']}-{e['end_line']}"
                    break
        if loc:
            lines.append(f"{i}. {step_name}->{loc}(chunk={cid})")
        else:
            lines.append(f"{i}. {step_name}")

    if lang == "zh":
        return "流程步骤与代码位置:\n" + "\n".join(lines)
    return "Steps and code locations:\n" + "\n".join(lines)


# -----------------------------
# 配额抽样(代表性保障核心) / Quota sampling for representativeness
# -----------------------------
def _quota_counts(target_n: int, ratio: Dict[str, int]) -> Dict[str, int]:
    total = sum(ratio.values()) if ratio else 1
    out = {k: int(target_n * v / total) for k, v in ratio.items()}
    # rounding fix
    while sum(out.values()) < target_n:
        k = max(ratio, key=lambda x: ratio[x])
        out[k] += 1
    # ensure at least 1 for existing keys if target_n allows
    return out


def quota_sample(
    samples: List[Dict[str, Any]],
    target_n: int,
    boundary_ratio: Dict[str, int],
    focus_ratio: Dict[str, int],
    per_file_cap: int,
    per_chunk_cap: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rnd = random.Random(seed)
    pool = list(samples)
    rnd.shuffle(pool)

    bq = _quota_counts(target_n, boundary_ratio)
    fq = _quota_counts(target_n, focus_ratio)

    used_b = Counter()
    used_f = Counter()
    used_file = Counter()
    used_chunk = Counter()

    picked: List[Dict[str, Any]] = []

    def _get_tags(s: Dict[str, Any]) -> Tuple[str, str, str, str]:
        mv2 = s.get("meta_v2", {}) or {}
        focus = mv2.get("focus") or "general"
        boundary = mv2.get("boundary") or "other"
        ev0 = (s.get("evidence") or [{}])[0]
        fp = ev0.get("file_path", "") or ""
        cid = ev0.get("chunk_id", "") or ""
        return boundary, focus, fp, cid

    # pass1:同时满足boundary配额+focus配额+file/chunk限流
    for s in pool:
        if len(picked) >= target_n:
            break
        boundary, focus, fp, cid = _get_tags(s)

        if used_file[fp] >= per_file_cap:
            continue
        if cid and used_chunk[cid] >= per_chunk_cap:
            continue
        if used_b[boundary] >= bq.get(boundary, 0):
            continue
        if used_f[focus] >= fq.get(focus, 0):
            continue

        picked.append(s)
        used_b[boundary] += 1
        used_f[focus] += 1
        used_file[fp] += 1
        if cid:
            used_chunk[cid] += 1

    # pass2:放松focus配额，只保boundary+限流
    if len(picked) < target_n:
        for s in pool:
            if len(picked) >= target_n:
                break
            if s in picked:
                continue
            boundary, focus, fp, cid = _get_tags(s)

            if used_file[fp] >= per_file_cap:
                continue
            if cid and used_chunk[cid] >= per_chunk_cap:
                continue
            if used_b[boundary] >= bq.get(boundary, 0):
                continue

            picked.append(s)
            used_b[boundary] += 1
            used_f[focus] += 1
            used_file[fp] += 1
            if cid:
                used_chunk[cid] += 1

    # pass3:再放松boundary(最后兜底),仍限file/chunk
    if len(picked) < target_n:
        for s in pool:
            if len(picked) >= target_n:
                break
            if s in picked:
                continue
            boundary, focus, fp, cid = _get_tags(s)

            if used_file[fp] >= per_file_cap:
                continue
            if cid and used_chunk[cid] >= per_chunk_cap:
                continue

            picked.append(s)
            used_b[boundary] += 1
            used_f[focus] += 1
            used_file[fp] += 1
            if cid:
                used_chunk[cid] += 1

    return picked


def print_coverage(samples: List[Dict[str, Any]], title: str) -> None:
    # 用于肉眼检查“有没有某层或某类被严重偏置”。
    b = Counter()
    f = Counter()
    d = Counter()
    files = set()
    chunks = set()
    for s in samples:
        mv2 = s.get("meta_v2", {}) or {}
        b[mv2.get("boundary", "other")] += 1
        f[mv2.get("focus", "general")] += 1
        d[mv2.get("domain", "unknown")] += 1
        ev0 = (s.get("evidence") or [{}])[0]
        fp = ev0.get("file_path")
        cid = ev0.get("chunk_id")
        if fp:
            files.add(fp)
        if cid:
            chunks.add(cid)
    print(f"\n[{title}]n={len(samples)},unique_files={len(files)},unique_chunks={len(chunks)}")
    print(f"[{title}]boundary={dict(b)}")
    print(f"[{title}]focus={dict(f)}")
    print(f"[{title}]domain={dict(d)}")


# -----------------------------
# 两类样本生成器 Generators
# -----------------------------
def generate_rule_qa(
    rule: Dict[str, Any],
    index_map: Dict[str, Dict[str, Any]],
    lang: str,
    code_max_chars: int,
    ex_re: re.Pattern,
) -> Dict[str, Any]:
    '''  
    :param: 一条rule+index_map+lang等
    :return: 一个完整QA样本dict(含text和meta_v2)，或{}表示跳过
    '''

    # 1)取rule.evidence_chunks的第一个cid作为证据(只用首个，保证单条规则QA更聚焦)
    ev_chunks = rule.get("evidence_chunks") or []
    cid = ev_chunks[0] if ev_chunks else None
    if (not cid) or (cid not in index_map):
        return {}

    # 2)build_evidence得到ev
    ev = build_evidence(cid, index_map, code_max_chars)
    if is_excluded_file(ev["file_path"], ex_re):
        return {}

    # 3)从ev.content抽关键行key_lines，推focus，再推boundary
    rule_id = rule.get("rule_id") or rule.get("id") or ""
    domain = rule.get("domain") or "mixed"

    title_raw = (rule.get("title") or "").strip()
    desc_raw = (rule.get("description") or "").strip()

    key_lines = extract_key_lines(ev.get("content") or "", max_lines=10)
    focus = infer_focus_from_key_lines(key_lines)
    boundary = infer_boundary(ev["file_path"])

    strict_lv = get_en_strict_level()

    # 4)语言处理：
    if lang == "en":
        title_en = (rule.get("title_en") or "").strip()
        desc_en = (rule.get("description_en") or "").strip()
        infer_title, infer_desc = infer_en_topic_from_evidence(ev, key_lines)
        title_fb = infer_title or (f"Rule {rule_id}" if rule_id else "Rule")
        desc_fb = infer_desc or "Explain the behavior based on evidence code."
        if strict_lv == 0:
            title = title_raw
            desc = desc_raw
        else:
            title = en_pick(title_raw, title_en, title_fb, strict_lv)
            desc = en_pick(desc_raw, desc_en, desc_fb, strict_lv)
            if strict_lv >= 2 and ((not title) or (not desc)):
                return {}
    else:
        title = title_raw or (rule_id or "业务规则")
        desc = desc_raw or ""
    
    # 5)选问题模板
    q = choose_rule_question(lang, focus, title=title, file_path=ev["file_path"])
    # 6)构建结论
    conclusion = build_rule_conclusion(lang, title=title, focus=focus, key_lines=key_lines)
    trace_steps = build_trace(lang, focus, title)
    # 7)组织core：加背景desc、加关键行摘录
    if lang == "zh":
        bg = f"背景:{desc}" if desc else "背景:未提供结构化描述,以下从代码推断。"
        kl = "\n".join([f"- {x}" for x in key_lines]) if key_lines else "- N/A"
        core = f"{bg}\n\n{conclusion}\n\n关键行摘录:\n{kl}"
    else:
        bg = f"Context:{desc}" if desc else "Context:No structured description; infer from code."
        kl = "\n".join([f"- {x}" for x in key_lines]) if key_lines else "- N/A"
        core = f"{bg}\n\n{conclusion}\n\nKey lines:\n{kl}"

    # 8)wrap_answer把core+证据+trace拼成最终answer
    a = wrap_answer(core, [ev], trace_steps, lang)
    text = f"### Instruction\n{q}\n\n### Response\n{a}\n"

    meta_v2 = {
        "task_type": "qa",
        "language": lang,
        "domain": domain,
        "qa_type": "rule",
        "focus": focus,
        "boundary": boundary,
        "evidence": [{
            "chunk_id": ev["chunk_id"],
            "file_path": ev["file_path"],
            "start_line": ev["start_line"],
            "end_line": ev["end_line"],
        }],
        "evidence_snippets": [{
            "chunk_id": ev["chunk_id"],
            "file_path": ev["file_path"],
            "start_line": ev["start_line"],
            "end_line": ev["end_line"],
            "code_lang": ev.get("code_lang"),
            "code": ev.get("code"),
        }],
        "trace_digest": trace_steps,
        "generator": "step04_v6_diversity_quota",
        "source": "AutoCodeDataPipeline",
    }

    # 每条规则QA只挂一个evidence chunk--> 
    # 答案更像“这段实现体现了某类规则”，而不是“规则在全仓库所有证据的全景总结”
    return {
        "sample_id": make_id("qa", f"rule|{rule_id}|{lang}|{cid}|{focus}|{sha1_text(q)[:6]}"),
        "task_type": "qa",
        "language": lang,
        "question": q,
        "answer": a,
        "evidence": [{
            "chunk_id": ev["chunk_id"],
            "file_path": ev["file_path"],
            "start_line": ev["start_line"],
            "end_line": ev["end_line"],
            "content": ev.get("content") or "",
        }],
        "trace": {
            "type": "rule_based",
            "rule_ids": [rule_id],
            "reasoning_steps": trace_steps,
        },
        "meta": {
            "domain": domain,
            "qa_type": "rule",
            "focus": focus,
            "boundary": boundary,
            "generator": "step04_v6_diversity_quota",
            "source": "AutoCodeDataPipeline",
        },
        "text": text,
        "meta_v2": meta_v2,
    } 



def generate_flow_qa(
    flow: Dict[str, Any],
    index_map: Dict[str, Dict[str, Any]],
    lang: str,
    code_max_chars: int,
    flow_max_evs: int,
    ex_re: re.Pattern,
) -> Dict[str, Any]:
    '''
    :param: 一条flow+index_map+lang等
    :return: 一个流程QA样本或{}
    '''
    evs: List[Dict[str, Any]] = []

    # 1)遍历flow.steps，收集每步evidence_chunk对应的ev，最多flow_max_evs条(默认3)
    '''
    ev={
        "chunk_id": chunk_id,
        "file_path": r["file_path"],
        "start_line": r["start_line"],
        "end_line": r["end_line"],
        "content": r.get("content") or "",
        "code_lang": detect_code_lang(r["file_path"]),
        "code": trim_code(r.get("content") or "", code_max_chars),
    }'''
    for s in flow.get("steps", []) or []:
        cid = s.get("evidence_chunk")
        if cid and cid in index_map:
            ev = build_evidence(cid, index_map, code_max_chars)
            if is_excluded_file(ev["file_path"], ex_re):
                continue
            evs.append(ev)
        if len(evs) >= flow_max_evs:
            break
    if not evs:
        return {}

    flow_id = flow.get("flow_id") or flow.get("id") or ""
    domain = flow.get("domain") or "mixed"

    strict_lv = get_en_strict_level()
    if lang == "en":
        flow_name_raw = (flow.get("name") or "").strip()
        flow_name_en = (flow.get("name_en") or "").strip()
        fb = f"Flow {flow_id}" if flow_id else "Business flow"
        flow_name = flow_name_raw if strict_lv == 0 else en_pick(flow_name_raw, flow_name_en, fb, strict_lv)
        if strict_lv >= 2 and (not flow_name):
            return {}
    else:
        flow_name = (flow.get("name") or "").strip() or (flow_id or "业务流程")

    # 2)选择流程问句choose_flow_question
    '''_FLOW_Q_ZH = [
    "请解释流程“{flow_name}”的端到端步骤,并标注每一步对应的代码位置。",
    "从调用链角度梳理“{flow_name}”包含哪些关键步骤?每一步的证据代码在哪里?",
    "“{flow_name}”流程的关键分支(成功/失败/取消/超时)如何走?请结合证据代码说明。",
    ]'''
    q = choose_flow_question(lang, flow_name=flow_name)

    # 3)用evs[0]抽关键行推focus和boundary
    key_lines = extract_key_lines(evs[0].get("content") or "", max_lines=10)  #  让答案不是纯模板，而是“模板+证据关键行”
    focus = infer_focus_from_key_lines(key_lines)
    boundary = infer_boundary(evs[0]["file_path"])

    # 4)build_flow_answer输出步骤与代码位置, 步骤列表+每步代码位置
    core_lines = build_flow_answer(flow, evs, lang)

    if lang == "zh":
        core = f"关注点:{focus_display(lang, focus)}\n\n{core_lines}"
    else:
        core = f"Focus:{focus_display(lang, focus)}\n\n{core_lines}"

    trace_steps = build_trace(lang, focus, flow_name)
    a = wrap_answer(core, evs, trace_steps, lang)

    text = f"### Instruction\n{q}\n\n### Response\n{a}\n"

    meta_v2 = {
        "task_type": "qa",
        "language": lang,
        "domain": domain,
        "qa_type": "flow",
        "focus": focus,
        "boundary": boundary,
        "evidence": [{
            "chunk_id": e["chunk_id"],
            "file_path": e["file_path"],
            "start_line": e["start_line"],
            "end_line": e["end_line"],
        } for e in evs],
        "evidence_snippets": [{
            "chunk_id": e["chunk_id"],
            "file_path": e["file_path"],
            "start_line": e["start_line"],
            "end_line": e["end_line"],
            "code_lang": e.get("code_lang"),
            "code": e.get("code"),
        } for e in evs],
        "trace_digest": trace_steps,
        "generator": "step04_v6_diversity_quota",
        "source": "AutoCodeDataPipeline",
    }

    return {
        "sample_id": make_id("qa", f"flow|{flow_id}|{lang}|{focus}|{sha1_text(q)[:6]}"),
        "task_type": "qa",
        "language": lang,
        "question": q,
        "answer": a,
        "evidence": [{
            "chunk_id": e["chunk_id"],
            "file_path": e["file_path"],
            "start_line": e["start_line"],
            "end_line": e["end_line"],
            "content": e.get("content") or "",
        } for e in evs],
        "trace": {
            "type": "flow_based",
            "flow_id": flow_id,
            "reasoning_steps": trace_steps,
        },
        "meta": {
            "domain": domain,
            "qa_type": "flow",
            "focus": focus,
            "boundary": boundary,
            "generator": "step04_v6_diversity_quota",
            "source": "AutoCodeDataPipeline",
        },
        "text": text,
        "meta_v2": meta_v2,
    }


def main():
    index_rows = read_jsonl(ROOT / "data/raw_index/repo_index.jsonl")
    rules = read_jsonl(ROOT / "data/extracted/rules.jsonl")
    flows = read_jsonl(ROOT / "data/extracted/flows.jsonl")
    index_map = {r["chunk_id"]: r for r in index_rows}
    # =============================
    # ===== [FLOW*RULE ADD] =====
    # rules_by_id: 通过rule_id/id快速取rule
    # =============================
    rules_by_id = {}
    for r in rules:
        rid = r.get("rule_id") or r.get("id")
        if rid:
            rules_by_id[rid] = r


    lang_mode = resolve_language(default_lang="zh")
    langs = ["zh", "en"] if lang_mode == "bilingual" else [lang_mode]

    rule_topn = int(os.environ.get("QA_RULE_TOPN", "200"))
    flow_max_evs = int(os.environ.get("QA_FLOW_MAX_EVS", "3"))
    code_max_chars = int(os.environ.get("QA_CODE_MAX_CHARS", "2200"))
    rule_per_id_cap = int(os.environ.get("QA_RULE_PER_ID_CAP", "3"))
    
    # =============================
    # ===== [FLOW*RULE ADD] =====
    # 控制是否生成flow*rule QA，以及每条flow最多挂多少条rule证据
    # =============================
    flow_rule_enable = os.environ.get("QA_FLOW_RULE_ENABLE", "1").strip() not in ("0", "false", "False")
    flow_rule_max_rules = int(os.environ.get("QA_FLOW_RULE_MAX_RULES", "4"))

    # 代表性控制参数
    train_target = int(os.environ.get("QA_TRAIN_N", "0"))  # 0表示用全部样本按比例切
    dev_ratio = float(os.environ.get("QA_DEV_RATIO", "0.1"))
    test_ratio = float(os.environ.get("QA_TEST_RATIO", "0.1"))
    per_file_cap = int(os.environ.get("QA_PER_FILE_CAP", "10"))
    per_chunk_cap = int(os.environ.get("QA_PER_CHUNK_CAP", "3"))
    seed = int(os.environ.get("QA_SEED", "42"))

    # 配额:boundary与focus
    boundary_ratio = {
        "service": int(os.environ.get("QA_BQ_SERVICE", "45")),
        "controller": int(os.environ.get("QA_BQ_CONTROLLER", "20")),
        "mapper": int(os.environ.get("QA_BQ_MAPPER", "20")),
        "config": int(os.environ.get("QA_BQ_CONFIG", "5")),
        "other": int(os.environ.get("QA_BQ_OTHER", "10")),
    }
    focus_ratio = {
        "stock": int(os.environ.get("QA_FQ_STOCK", "25")),
        "status": int(os.environ.get("QA_FQ_STATUS", "20")),
        "transaction": int(os.environ.get("QA_FQ_TRANSACTION", "15")),
        "concurrency": int(os.environ.get("QA_FQ_CONCURRENCY", "15")),
        "idempotency": int(os.environ.get("QA_FQ_IDEMPOTENCY", "10")),
        "exception": int(os.environ.get("QA_FQ_EXCEPTION", "10")),
        "general": int(os.environ.get("QA_FQ_GENERAL", "5")),
    }

    ex_re = compile_exclude_regex()

    all_samples: List[Dict[str, Any]] = []
    # lang内也做去重,避免同一chunk重复问法过多
    seen_q_sig = set()

    for lang in langs:
        # rules:按rule_id限流+打散
        rules_pool = rules[:rule_topn]
        random.shuffle(rules_pool)
        per_rule = Counter()

        for r in rules_pool:
            rule_id = r.get("rule_id") or r.get("id") or ""
            if not rule_id:
                continue
            if per_rule[(lang, rule_id)] >= rule_per_id_cap:
                continue

            s = generate_rule_qa(r, index_map, lang, code_max_chars, ex_re)
            if not s:
                continue

            sig = (lang, s["question"], (s.get("evidence") or [{}])[0].get("chunk_id"))
            if sig in seen_q_sig:
                continue
            seen_q_sig.add(sig)

            per_rule[(lang, rule_id)] += 1
            all_samples.append(s)

        # flows:全量生成,但去重
        '''for f in flows:
            s = generate_flow_qa(f, index_map, lang, code_max_chars, flow_max_evs, ex_re)
            if not s:
                continue
            sig = (lang, s["question"], tuple([e.get("chunk_id") for e in (s.get("evidence") or [])]))
            if sig in seen_q_sig:
                continue
            seen_q_sig.add(sig)
            all_samples.append(s)'''
                # flows:全量生成,但去重
        for f in flows:
            # 1) 原有: flow QA
            s = generate_flow_qa(f, index_map, lang, code_max_chars, flow_max_evs, ex_re)
            if s:
                sig = (lang, s["question"], tuple([e.get("chunk_id") for e in (s.get("evidence") or [])]))
                if sig not in seen_q_sig:
                    seen_q_sig.add(sig)
                    all_samples.append(s)

            # 2) 新增: flow*rule QA
            if flow_rule_enable:
                sr = generate_flow_rule_qa(
                    f,
                    rules_by_id=rules_by_id,
                    index_map=index_map,
                    lang=lang,
                    code_max_chars=code_max_chars,
                    flow_max_evs=flow_max_evs,
                    rule_max_evs=flow_rule_max_rules,
                    ex_re=ex_re,
                )
                if sr:
                    sig = (lang, sr["question"], tuple([e.get("chunk_id") for e in (sr.get("evidence") or [])]))
                    if sig not in seen_q_sig:
                        seen_q_sig.add(sig)
                        all_samples.append(sr)


    random.shuffle(all_samples)

    # 代表性抽样:先抽train,再从剩余切dev/test
    n_total = len(all_samples)
    if n_total == 0:
        raise RuntimeError("未生成任何QA样本,请检查rules/flows/repo_index是否为空或被排除。")

    if train_target <= 0:
        # 默认按比例切,但仍做配额抽样以提升代表性
        train_target = int(n_total * (1.0 - dev_ratio - test_ratio))
        train_target = max(1, train_target)

    remaining = list(all_samples)
    train = quota_sample(remaining, train_target, boundary_ratio, focus_ratio, per_file_cap, per_chunk_cap, seed=seed)

    train_set = set([s["sample_id"] for s in train])
    remaining2 = [s for s in remaining if s["sample_id"] not in train_set]

    dev_n = int(len(train) * dev_ratio / max(1e-9, (1.0 - dev_ratio - test_ratio)))
    test_n = int(len(train) * test_ratio / max(1e-9, (1.0 - dev_ratio - test_ratio)))
    dev_n = max(0, min(dev_n, len(remaining2)))
    test_n = max(0, min(test_n, len(remaining2) - dev_n))

    dev = quota_sample(remaining2, dev_n, boundary_ratio, focus_ratio, per_file_cap, per_chunk_cap, seed=seed + 1) if dev_n > 0 else []
    dev_set = set([s["sample_id"] for s in dev])
    remaining3 = [s for s in remaining2 if s["sample_id"] not in dev_set]
    test = quota_sample(remaining3, test_n, boundary_ratio, focus_ratio, per_file_cap, per_chunk_cap, seed=seed + 2) if test_n > 0 else []

    # 输出
    write_jsonl(ROOT / "data/dataset/train.jsonl", train)
    write_jsonl(ROOT / "data/dataset/dev.jsonl", dev)
    write_jsonl(ROOT / "data/dataset/test.jsonl", test)
    write_jsonl(ROOT / "data/samples/qa_samples.jsonl", (train + dev + test)[:30])

    print(f"QA生成完成:total={n_total},train={len(train)},dev={len(dev)},test={len(test)},langs={langs}")
    print(f"[Knobs]QA_RULE_TOPN={rule_topn},QA_RULE_PER_ID_CAP={rule_per_id_cap},QA_FLOW_MAX_EVS={flow_max_evs},QA_CODE_MAX_CHARS={code_max_chars}")
    print(f"[Knobs]QA_PER_FILE_CAP={per_file_cap},QA_PER_CHUNK_CAP={per_chunk_cap},QA_TRAIN_N={train_target},QA_DEV_RATIO={dev_ratio},QA_TEST_RATIO={test_ratio}")
    print_coverage(train, "train_coverage")
    if dev:
        print_coverage(dev, "dev_coverage")
    if test:
        print_coverage(test, "test_coverage")


if __name__ == "__main__":
    main()
