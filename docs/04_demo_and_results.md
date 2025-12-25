# Demo与结果

## 数据集概览
- repo_index_chunks:3408
- QA样本数(去重前/后):60/60removed=0
- Design样本数(去重前/后):55/55removed=0
- FinalFinetune(去重后)split:train=92,dev=11,test=12

## QA样本统计
- count:60
- evidence_snippets_avg:1.133
- code_chars_avg:1421.55
- snippet_coverage_rate:1.0
- trace_steps_avg:3.0

### language分布
- zh:60
### domain分布
- mixed:7
- order:46
- stock:7
### qa_type分布
- flow:2
- flow_rule:2
- rule:56
### difficulty分布
- unknown:60

## Design样本统计
- count:55
- evidence_snippets_avg:3.0
- code_chars_avg:2446.818
- snippet_coverage_rate:1.0
- trace_steps_avg:4.0

### language分布
- zh:55
### domain分布
- mixed:9
- order:28
- stock:18
### difficulty分布
- easy:3
- hard:9
- medium:43

## 质量保证机制
- schema校验:每条样本必须包含可训练text或核心字段(QA:question/answer,Design:requirement/design_output)
- grounding校验:meta_v2.evidence_snippets强制包含原文code,并校验chunk_id命中repo_index且file_path一致
- trace可审计:meta_v2.trace_digest强制存在(QA>=2步,Design>=3步)
- 去重策略:优先按text去重,否则按(question/requirement+domain/qa_type)去重,避免刷屏同时保留不同定位的有效样本
