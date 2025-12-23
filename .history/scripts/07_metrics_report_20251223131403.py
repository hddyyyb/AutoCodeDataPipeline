#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AutoCodeDataPipeline Step07
将data/dataset/stats.json生成可读Markdown报告到docs/04_demo_and_results.md
(适配Step06新统计:增加evidence_snippets/最终finetune输出)
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
    final_ft = stats.get("final_finetune", {})

    md = []
    md.append("# Demo与结果")
    md.append("")
    md.append("## 数据集概览")
    md.append(f"- repo_index_chunks:{stats['repo_index_chunks']}")
    md.append(f"- QA样本数(去重前/后):{qa['before_dedup']}/{qa['after_dedup']}removed={qa['removed']}")
    md.append(f"- Design样本数(去重前/后):{design['before_dedup']}/{design['after_dedup']}removed={design['removed']}")
    if final_ft:
        md.append(f"- FinalFinetune样本数(去重前/后):{final_ft.get('before_dedup')}/{final_ft.get('after_dedup')}removed={final_ft.get('removed')}")
        sp = final_ft.get("splits", {})
        if sp:
            md.append(f"- FinalFinetune切分:train={sp.get('train')}dev={sp.get('dev')}test={sp.get('test')}")
            md.append("- FinalFinetune文件:data/dataset/final_train.jsonl,final_dev.jsonl,final_test.jsonl")
    md.append("")

    md.append("## QA样本统计")
    md.append(f"- count:{qa_m['count']}")
    md.append(f"- evidence_snippets_avg:{qa_m.get('evidence_snippets_avg')}")
    md.append(f"- code_chars_avg:{qa_m.get('code_chars_avg')}")
    md.append(f"- snippet_coverage_rate:{qa_m.get('snippet_coverage_rate')}")
    md.append(f"- trace_steps_avg:{qa_m['trace_steps_avg']}")
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
    md.append(f"- count:{d_m['count']}")
    md.append(f"- evidence_snippets_avg:{d_m.get('evidence_snippets_avg')}")
    md.append(f"- code_chars_avg:{d_m.get('code_chars_avg')}")
    md.append(f"- snippet_coverage_rate:{d_m.get('snippet_coverage_rate')}")
    md.append(f"- trace_steps_avg:{d_m['trace_steps_avg']}")
    md.append("")
    md.append("### language分布")
    md.append(md_kv(d_m.get("language_dist", {})))
    md.append("### domain分布")
    md.append(md_kv(d_m.get("domain_dist", {})))
    md.append("### difficulty分布")
    md.append(md_kv(d_m.get("difficulty_dist", {})))
    md.append("")

    md.append("## 质量保证机制")
    md.append("- schema校验:样本必须包含可训练text与可审计meta_v2(meta),并声明task_type/language/domain等")
    md.append("- grounding校验:evidence/evidence_snippets的chunk_id必须命中repo_index且file_path一致")
    md.append("- 原文代码段强制:evidence_snippets必须包含code字段,用于满足“提供原文代码段”要求")
    md.append("- trace可审计:QA至少2步,Design至少3步(通过trace_digest验证)")
    md.append("- 去重策略:优先按text哈希去重,其次按(question/requirement)+类型信息去重,避免语义重复且保留不同实现位置样本")
    md.append("")

    out = ROOT / "docs/04_demo_and_results.md"
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"报告已生成:{out}")


if __name__ == "__main__":
    main()
