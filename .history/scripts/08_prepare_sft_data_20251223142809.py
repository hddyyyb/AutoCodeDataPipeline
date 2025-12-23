#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step08
将 validated QA / design task 数据整理为 SFT 所需格式
支持 bilingual 时按语言拆分 zh / en
"""

import os
import json
from pathlib import Path
from typing import List, Dict

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FINAL_DIR = DATA_DIR / "final"
SFT_DIR = DATA_DIR / "sft"
SFT_DIR.mkdir(parents=True, exist_ok=True)

LANG_FILTER = os.getenv("LANG_FILTER", "zh").lower()
SPLIT_BY_LANG = os.getenv("SPLIT_BY_LANG", "1") != "0"


def load_jsonl(p: Path) -> List[Dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def to_sft_sample(rec: Dict) -> Dict:
    return {
        "messages": [
            {"role": "system", "content": rec.get("system", "你是一个代码仓专家")},
            {"role": "user", "content": rec["question"]},
            {"role": "assistant", "content": rec["answer"]},
        ],
        "meta": rec.get("meta", {})
    }


def write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def process_split(split: str):
    src = FINAL_DIR / f"{split}.jsonl"
    records = load_jsonl(src)
    if not records:
        return

    sft_rows = [to_sft_sample(r) for r in records]

    # ===== bilingual 拆分 =====
    if LANG_FILTER == "bilingual" and SPLIT_BY_LANG:
        zh_rows, en_rows = [], []
        for r in sft_rows:
            lang = r["meta"].get("language", "zh")
            if lang == "en":
                en_rows.append(r)
            else:
                zh_rows.append(r)

        if zh_rows:
            write_jsonl(SFT_DIR / f"{split}.zh.jsonl", zh_rows)
        if en_rows:
            write_jsonl(SFT_DIR / f"{split}.en.jsonl", en_rows)

        # 兼容：合并版
        write_jsonl(SFT_DIR / f"{split}.jsonl", zh_rows + en_rows)

    else:
        # 单语言逻辑
        filtered = []
        for r in sft_rows:
            lang = r["meta"].get("language", "zh")
            if LANG_FILTER in ("all", lang):
                filtered.append(r)

        write_jsonl(SFT_DIR / f"{split}.jsonl", filtered)


def main():
    for split in ["train", "dev", "test"]:
        process_split(split)


if __name__ == "__main__":
    main()
