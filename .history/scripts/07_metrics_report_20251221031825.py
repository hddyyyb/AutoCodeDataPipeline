#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step07
将data/dataset/stats.json生成可读Markdown报告到docs/05_demo_and_results.md
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md_kv(d: dict) -> str:
    items = sorted(d.items(), key=lambda x: x[0])
    return "\n".join([f"- {k}:{v}" for k, v in items])


def main() -> None:
    stats_path = ROOT / "data/dataset/stats.json"
    if not stats_path.exists():
        raise FileNotFoundError("缺少data/dataset/stats.json,请先运行scripts/06_postprocess_and_validate.py")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    qa = stats["qa"]
    design = stats["design"]
    qa_m = qa["metrics"]
    d_m = design["metrics"]

    md = []
    md.append("# Demo与结果")
    md.append("")
    md.append("## 数据集概览")
    md.append(f"- repo_index_chunks:{stats['repo_index_chunks']}")
    md.append(f"- QA样本数(去重前/后):{qa['before_dedup']}/{qa['after_dedup']}removed={qa['removed']}")
    md.append(f"- Design样本数(去重前/后):{design['before_dedup']}/{design['after_dedup']}removed={design['removed']}")
    md.append("")

    md.append("## QA样本统计")
    md.append(f"- count:{qa_m['count']}")
    md.append(f"- evidence_avg:{qa_m['evidence_avg']}")
    md.append(f"- trace_steps_avg:{qa_m['trace_steps_avg']}")
    md.append("### domain分布")
    md.append(md_kv(qa_m["domain_dist"]))
    md.append("### qa_type分布")
    md.append(md_kv(qa_m["qa_type_dist"]))
    md.append("### difficulty分布")
    md.append(md_kv(qa_m["difficulty_dist"]))
    md.append("")

    md.append("## Design样本统计")
    md.append(f"- count:{d_m['count']}")
    md.append(f"- evidence_avg:{d_m['evidence_avg']}")
    md.append(f"- trace_steps_avg:{d_m['trace_steps_avg']}")
    md.append("### domain分布")
    md.append(md_kv(d_m["domain_dist"]))
    md.append("### difficulty分布")
    md.append(md_kv(d_m["difficulty_dist"]))
    md.append("")

    md.append("## 质量保证机制")
    md.append("- schema校验:每条样本必须包含核心字段(question/answer或requirement/design_output)、evidence、trace、meta")
    md.append("- grounding校验:evidence的chunk_id必须命中repo_index,且文件路径一致")
    md.append("- trace可审计:QA至少2步,Design至少3步")
    md.append("- 去重策略:基于question文本,并结合问题类型(domain/qa_type)与代码定位信息进行实例级去重,避免语义重复同时保留不同实现位置的有效样本")
    md.append("")

    out = ROOT / "docs/05_demo_and_results.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"报告已生成:{out}")


if __name__ == "__main__":
    main()
