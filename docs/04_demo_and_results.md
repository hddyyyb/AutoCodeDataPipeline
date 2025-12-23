# Demo与结果

## 数据集概览
- repo_index_chunks:3408
- QA样本数(去重前/后):110/110removed=0
- Design样本数(去重前/后):92/92removed=0
- FinalFinetune(去重后)split:train=161,dev=20,test=21

## QA样本统计
- count:110
- evidence_snippets_avg:1.036
- code_chars_avg:1227.291
- snippet_coverage_rate:1.0
- trace_steps_avg:3.0

### language分布
- en:55
- zh:55
### domain分布
- mixed:16
- order:92
- stock:2
### qa_type分布
- flow:4
- rule:106
### difficulty分布
- unknown:110

## Design样本统计
- count:92
- evidence_snippets_avg:3.0
- code_chars_avg:2504.217
- snippet_coverage_rate:1.0
- trace_steps_avg:4.0

### language分布
- en:42
- zh:50
### domain分布
- mixed:24
- order:53
- stock:15
### difficulty分布
- easy:3
- hard:15
- medium:74

## 质量保证机制
- schema校验:每条样本必须包含可训练text或核心字段(QA:question/answer,Design:requirement/design_output)
- grounding校验:meta_v2.evidence_snippets强制包含原文code,并校验chunk_id命中repo_index且file_path一致
- trace可审计:meta_v2.trace_digest强制存在(QA>=2步,Design>=3步)
- 去重策略:优先按text去重,否则按(question/requirement+domain/qa_type)去重,避免刷屏同时保留不同定位的有效样本
