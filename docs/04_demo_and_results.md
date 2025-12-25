# Demo与结果

## 数据集概览
- repo_index_chunks:3408
- QA样本数(去重前/后):57/57removed=0
- Design样本数(去重前/后):46/46removed=0
- FinalFinetune(去重后)split:train=82,dev=10,test=11

## QA样本统计
- count:57
- evidence_snippets_avg:1.035
- code_chars_avg:1295.088
- snippet_coverage_rate:1.0
- trace_steps_avg:3.0

### language分布
- zh:57
### domain分布
- mixed:7
- order:43
- stock:7
### qa_type分布
- flow:2
- rule:55
### difficulty分布
- unknown:57

## Design样本统计
- count:46
- evidence_snippets_avg:3.0
- code_chars_avg:2360.717
- snippet_coverage_rate:1.0
- trace_steps_avg:4.0

### language分布
- zh:46
### domain分布
- mixed:9
- order:19
- stock:18
### difficulty分布
- easy:3
- hard:6
- medium:37

## 质量保证机制
- schema校验:每条样本必须包含可训练text或核心字段(QA:question/answer,Design:requirement/design_output)
- grounding校验:meta_v2.evidence_snippets强制包含原文code,并校验chunk_id命中repo_index且file_path一致
- trace可审计:meta_v2.trace_digest强制存在(QA>=2步,Design>=3步)
- 去重策略:优先按text去重,否则按(question/requirement+domain/qa_type)去重,避免刷屏同时保留不同定位的有效样本
