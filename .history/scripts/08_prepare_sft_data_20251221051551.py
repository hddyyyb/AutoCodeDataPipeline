#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def read_jsonl(p: Path):
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def to_text(sample: dict) -> str:
    q = sample["question"].strip()
    a = sample["answer"].strip()
    # 你现在的样本里有evidence/trace，但训练时先把它们“隐式作为约束”即可
    # 如果你想把evidence也喂进去，可把evidence.content拼到Instruction里(后面我也给你可选版)
    return f"### Instruction:\n{q}\n\n### Response:\n{a}"

def main():
    lang_filter = os.environ.get("LANG_FILTER", "zh")  # zh|en|all
    src = ROOT / "data/dataset/train.jsonl"
    if not src.exists():
        src = ROOT / "data/samples/qa_samples.jsonl"

    samples = read_jsonl(src)
    rows = [{"text": to_text(s)} for s in samples]

    out = ROOT / "data/sft/train.jsonl"
    write_jsonl(out, rows)
    print(f"已生成SFT数据:{out},count={len(rows)}")

if __name__ == "__main__":
    main()
