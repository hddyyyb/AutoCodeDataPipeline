# Demo与结果

## 数据集概览
- repo_index_chunks:3408
- QA样本数(去重前/后):116/116removed=0
- Design样本数(去重前/后):74/74removed=0
- FinalFinetune(去重后)split:train=152,dev=19,test=19

## QA样本统计
- count:116
- evidence_snippets_avg:1.034
- code_chars_avg:1294.966
- snippet_coverage_rate:1.0
- trace_steps_avg:3.0

### language分布
- en:58
- zh:58
### domain分布
- mixed:14
- order:88
- stock:14
### qa_type分布
- flow:4
- rule:112
### difficulty分布
- unknown:116

## Design样本统计
- count:74
- evidence_snippets_avg:3.0
- code_chars_avg:2527.257
- snippet_coverage_rate:1.0
- trace_steps_avg:4.0

### language分布
- en:33
- zh:41
### domain分布
- mixed:23
- order:42
- stock:9
### difficulty分布
- easy:3
- hard:9
- medium:62

## 质量保证机制
- schema校验:每条样本必须包含可训练text或核心字段(QA:question/answer,Design:requirement/design_output)
- grounding校验:meta_v2.evidence_snippets强制包含原文code,并校验chunk_id命中repo_index且file_path一致
- trace可审计:meta_v2.trace_digest强制存在(QA>=2步,Design>=3步)
- 去重策略:优先按text去重,否则按(question/requirement+domain/qa_type)去重,避免刷屏同时保留不同定位的有效样本
