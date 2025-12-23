#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step07
将data/dataset/stats.json生成可读Markdown报告到docs/04_demo_and_results.md
(适配Step06新指标字段,并向后兼容旧字段名)
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def md_kv(d: dict) -> str:
    items = sorted(d.items(), key=lambda x: str(x[0]))
    return "\n".join([f"- {k}:{v}" for k, v in items])


def pick_metric(m: dict, *keys, default="N/A"):
    for k in keys:
        if k in m:
            return m[k]
    return default


def main() -> None:
    stats_path = ROOT / "data/dataset/stats.json"
    if not stats_path.exists():
        raise FileNotFoundError("缺少data/dataset/stats.json,请先运行scripts/06_postprocess_and_validate.py")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))

    qa = stats["qa"]
    design = stats["design"]
    qa_m = qa["metrics"]
    d_m = design["metrics"]

    qa_evi_avg = pick_metric(qa_m, "evidence_snippets_avg", "evidence_avg", default=0)
    d_evi_avg = pick_metric(d_m, "evidence_snippets_avg", "evidence_avg", default=0)

    md = []
    md.append("# Demo与结果")
    md.append("")
    md.append("## 数据集概览")
    md.append(f"- repo_index_chunks:{stats.get('repo_index_chunks','N/A')}")
    md.append(f"- QA样本数(去重前/后):{qa['before_dedup']}/{qa['after_dedup']}removed={qa['removed']}")
    md.append(f"- Design样本数(去重前/后):{design['before_dedup']}/{design['after_dedup']}removed={design['removed']}")
    ff = stats.get("final_finetune", {})
    if ff:
        sp = ff.get("splits", {})
        md.append(f"- FinalFinetune(去重后)split:train={sp.get('train','N/A')},dev={sp.get('dev','N/A')},test={sp.get('test','N/A')}")
    md.append("")

    md.append("## QA样本统计")
    md.append(f"- count:{qa_m.get('count','N/A')}")
    md.append(f"- evidence_snippets_avg:{qa_evi_avg}")
    md.append(f"- code_chars_avg:{qa_m.get('code_chars_avg','N/A')}")
    md.append(f"- snippet_coverage_rate:{qa_m.get('snippet_coverage_rate','N/A')}")
    md.append(f"- trace_steps_avg:{qa_m.get('trace_steps_avg','N/A')}")
    md.append("")
    md.append("### language分布")
    md.append(md_kv(qa_m.get("language_dist", {})))
    md.append("### domain分布")
    md.append(md_kv(qa_m.get("domain_dist", {})))
    md.append("### qa_type分布")
    md.append(md_kv(qa_m.get("qa_type_dist", {})))
    md.append("### difficulty分布")
    md.append(md_kv(qa_m.get("difficulty_dist", {})))
    md.append("")

    md.append("## Design样本统计")
    md.append(f"- count:{d_m.get('count','N/A')}")
    md.append(f"- evidence_snippets_avg:{d_evi_avg}")
    md.append(f"- code_chars_avg:{d_m.get('code_chars_avg','N/A')}")
    md.append(f"- snippet_coverage_rate:{d_m.get('snippet_coverage_rate','N/A')}")
    md.append(f"- trace_steps_avg:{d_m.get('trace_steps_avg','N/A')}")
    md.append("")
    md.append("### language分布")
    md.append(md_kv(d_m.get("language_dist", {})))
    md.append("### domain分布")
    md.append(md_kv(d_m.get("domain_dist", {})))
    md.append("### difficulty分布")
    md.append(md_kv(d_m.get("difficulty_dist", {})))
    md.append("")

    md.append("## 质量保证机制")
    md.append("- schema校验:每条样本必须包含可训练text或核心字段(QA:question/answer,Design:requirement/design_output)")
    md.append("- grounding校验:meta_v2.evidence_snippets强制包含原文code,并校验chunk_id命中repo_index且file_path一致")
    md.append("- trace可审计:meta_v2.trace_digest强制存在(QA>=2步,Design>=3步)")
    md.append("- 去重策略:优先按text去重,否则按(question/requirement+domain/qa_type)去重,避免刷屏同时保留不同定位的有效样本")
    md.append("")

    out = ROOT / "docs/04_demo_and_results.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"报告已生成:{out}")


if __name__ == "__main__":
    main()
