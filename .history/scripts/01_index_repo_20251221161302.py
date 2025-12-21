#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step01
目标:扫描repo/mall范围内文件,生成可追溯的repo索引repo_index.jsonl
输出字段包含:file_path,start_line,end_line,content,chunk_id等,为后续证据引用打底
"""

from __future__ import annotations

import os
import re
import json
import hashlib
import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Iterable, Tuple, Optional

try:
    import yaml
except Exception:
    yaml = None


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Chunk:
    chunk_id: str
    file_path: str
    lang: str
    start_line: int
    end_line: int
    content: str
    content_hash: str
    approx_tokens: int
    symbol: str


def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def approx_token_count(s: str) -> int:
    # 粗估token:按字符/4近似,最低为1
    n = max(1, len(s) // 4)
    return n


def load_scope_config(cfg_path: Path) -> Dict:
    if not cfg_path.exists():
        raise FileNotFoundError(f"找不到配置文件:{cfg_path}")
    if yaml is None:
        raise RuntimeError("缺少pyyaml,请先pip install pyyaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_excluded(rel_path: str, exclude_patterns: List[str]) -> bool:
    for pat in exclude_patterns:
        if fnmatch.fnmatch(rel_path, pat):
            return True
    return False


def iter_files(repo_root: Path, include_paths: List[str], exclude_patterns: List[str]) -> Iterable[Path]:
    for inc in include_paths:
        base = repo_root / inc
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if not p.is_file():
                continue
            rel = str(p.relative_to(repo_root)).replace("\\", "/")
            if is_excluded(rel, exclude_patterns):
                continue
            yield p


def detect_lang(p: Path) -> str:
    suf = p.suffix.lower().lstrip(".")
    if suf in ("java", "xml", "yaml", "yml", "md", "properties"):
        return suf
    return "text"


JAVA_CLASS_RE = re.compile(r"^\s*(public\s+)?(class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)", re.M)
JAVA_METHOD_RE = re.compile(
    r"^\s*(public|protected|private)?\s*(static\s+)?[A-Za-z0-9_<>\[\],\s]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.M,
)


def extract_primary_symbol(lang: str, content: str) -> str:
    if lang != "java":
        return ""
    m = JAVA_CLASS_RE.search(content)
    if m:
        return m.group(3)
    return ""


def split_java_chunks(lines: List[str], max_lines: int, min_lines: int) -> List[Tuple[int, int]]:
    """
    简化版切分:
    1)优先按类/方法起始行作为边界
    2)再按max_lines强制分段
    返回(起始行号,结束行号)均为1-based闭区间
    """
    n = len(lines)
    boundaries = set([1, n + 1])

    for i, line in enumerate(lines, start=1):
        if re.match(r"^\s*(public\s+)?(class|interface|enum)\s+", line):
            boundaries.add(i)
        # 方法定义可能跨行,这里用行级正则做近似
        if re.match(r"^\s*(public|protected|private)\s+", line) and "(" in line and ")" in line and "{" in line:
            boundaries.add(i)

    b = sorted(boundaries)
    spans: List[Tuple[int, int]] = []
    # 先按boundary形成片段
    for idx in range(len(b) - 1):
        s = b[idx]
        e = b[idx + 1] - 1
        if s <= e:
            spans.append((s, e))

    # 合并小片段并控制max_lines
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = spans[0] if spans else (1, n)
    for s, e in spans[1:]:
        if (cur_e - cur_s + 1) < min_lines:
            cur_e = e
            continue
        merged.append((cur_s, cur_e))
        cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # 再对过长段落按max_lines切
    final: List[Tuple[int, int]] = []
    for s, e in merged:
        length = e - s + 1
        if length <= max_lines:
            final.append((s, e))
        else:
            cur = s
            while cur <= e:
                end = min(e, cur + max_lines - 1)
                final.append((cur, end))
                cur = end + 1

    # 过滤极小段(仍然可能出现),允许最后一段小于min_lines
    cleaned: List[Tuple[int, int]] = []
    for s, e in final:
        if (e - s + 1) < max(5, min_lines // 3) and cleaned:
            ps, pe = cleaned[-1]
            cleaned[-1] = (ps, e)
        else:
            cleaned.append((s, e))
    return cleaned


def split_text_chunks(lines: List[str], max_lines: int, min_lines: int) -> List[Tuple[int, int]]:
    n = len(lines)
    spans: List[Tuple[int, int]] = []
    cur = 1
    while cur <= n:
        end = min(n, cur + max_lines - 1)
        spans.append((cur, end))
        cur = end + 1

    # 合并过小尾段
    if len(spans) >= 2:
        s_last, e_last = spans[-1]
        if (e_last - s_last + 1) < min_lines:
            s_prev, e_prev = spans[-2]
            spans[-2] = (s_prev, e_last)
            spans.pop()
    return spans


def should_keep_by_keywords(rel_path: str, content: str, keyword_filters: Dict[str, List[str]]) -> bool:
    """
    只要命中order或stock关键词之一就保留.
    命中规则:路径或内容(小写)包含关键词
    """
    rp = rel_path.lower()
    ct = content.lower()
    all_keywords = set()
    for _, ks in keyword_filters.items():
        for k in ks:
            all_keywords.add(k.lower())
    for k in all_keywords:
        if k in rp or k in ct:
            return True
    return False


def build_chunks_for_file(
    repo_root: Path,
    p: Path,
    chunking_cfg: Dict,
    keyword_filters: Dict[str, List[str]],
) -> List[Chunk]:
    rel = str(p.relative_to(repo_root)).replace("\\", "/")
    lang = detect_lang(p)
    try:
        text = p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    # 关键词过滤:只保留与order/stock相关文件
    if not should_keep_by_keywords(rel, text, keyword_filters):
        return []

    lines = text.splitlines()
    if not lines:
        return []

    if lang == "java":
        max_lines = int(chunking_cfg["java"]["max_lines_per_chunk"])
        min_lines = int(chunking_cfg["java"]["min_lines_per_chunk"])
        spans = split_java_chunks(lines, max_lines=max_lines, min_lines=min_lines)
    else:
        max_lines = int(chunking_cfg["text"]["max_lines_per_chunk"])
        min_lines = int(chunking_cfg["text"]["min_lines_per_chunk"])
        spans = split_text_chunks(lines, max_lines=max_lines, min_lines=min_lines)

    symbol = extract_primary_symbol(lang, text)
    chunks: List[Chunk] = []
    for (s, e) in spans:
        content = "\n".join(lines[s - 1 : e])
        chash = sha1_text(content)
        cid = sha1_text(f"{rel}:{s}:{e}:{chash}")[:16]
        chunks.append(
            Chunk(
                chunk_id=cid,
                file_path=rel,
                lang=lang,
                start_line=s,
                end_line=e,
                content=content,
                content_hash=chash,
                approx_tokens=approx_token_count(content),
                symbol=symbol,
            )
        )
    return chunks  # 每个片段生成一个Chunk对象(含chunk_id、行号、content、hash等)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    cfg = load_scope_config(ROOT / "configs" / "scope.yaml")  # 读配置
    repo_path = ROOT / cfg["repo"]["repo_path"]  # 定位repo
    if not repo_path.exists():
        raise FileNotFoundError(f"repo路径不存在:{repo_path},请先把mall代码放到AutoCodeDataPipeline/{cfg['repo']['repo_path']}")

    include_paths = cfg["scope"]["include_paths"]
    exclude_paths = cfg["scope"]["exclude_paths"]
    keyword_filters = cfg["scope"]["keyword_filters"]
    chunking_cfg = cfg["chunking"]

    out_index = ROOT / cfg["output"]["index_file"]
    out_sample = ROOT / cfg["output"]["sample_index_file"]
    max_sample_lines = int(cfg["output"]["max_sample_lines"])

    all_rows: List[Dict] = []
    for fp in iter_files(repo_path, include_paths, exclude_paths):
        # 遍历文件，只遍历include_paths下的文件，排除exclude_patterns命中的
        chunks = build_chunks_for_file(repo_path, fp, chunking_cfg, keyword_filters) # 按文件构建chunks
        for c in chunks:
            all_rows.append(
                {
                    "chunk_id": c.chunk_id,
                    "file_path": c.file_path,
                    "lang": c.lang,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    "content": c.content,
                    "content_hash": c.content_hash,
                    "approx_tokens": c.approx_tokens,
                    "symbol": c.symbol,
                }
            )

    # 稳定排序,便于diff
    all_rows.sort(key=lambda x: (x["file_path"], x["start_line"], x["end_line"]))

    write_jsonl(out_index, all_rows)

    # 提交仓库时建议只提交sample文件
    sample_rows = all_rows[:max_sample_lines]
    write_jsonl(out_sample, sample_rows)

    print(f"完成索引:chunks={len(all_rows)}")
    print(f"索引文件:{out_index}")
    print(f"样例文件:{out_sample}")


if __name__ == "__main__":
    main()
