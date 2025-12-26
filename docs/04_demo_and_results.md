# Demo与结果

## 数据集概览
- repo_index_chunks:3408
- QA样本数(去重前/后):121/121removed=0
- Design样本数(去重前/后):115/115removed=0
- FinalFinetune(去重后)split:train=188,dev=24,test=24

## QA样本统计
- count:121
- evidence_snippets_avg:1.132
- code_chars_avg:1406.587
- snippet_coverage_rate:1.0
- trace_steps_avg:3.0

### language分布
- en:60
- zh:61
### domain分布
- mixed:16
- order:92
- stock:13
### qa_type分布
- flow:4
- flow_rule:4
- rule:113
### difficulty分布
- unknown:121

## Design样本统计
- count:115
- evidence_snippets_avg:3.0
- code_chars_avg:2788.148
- snippet_coverage_rate:1.0
- trace_steps_avg:4.0

### language分布
- en:54
- zh:61
### domain分布
- mixed:24
- order:52
- stock:39
### difficulty分布
- easy:2
- hard:15
- medium:98

## 质量保证机制
- schema校验:每条样本必须包含可训练text或核心字段(QA:question/answer,Design:requirement/design_output)
- grounding校验:meta_v2.evidence_snippets强制包含原文code,并校验chunk_id命中repo_index且file_path一致
- trace可审计:meta_v2.trace_digest强制存在(QA>=2步,Design>=3步)
- 去重策略:优先按text去重,否则按(question/requirement+domain/qa_type)去重,避免刷屏同时保留不同定位的有效样本
