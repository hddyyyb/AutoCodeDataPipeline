# Demo与结果

## 数据集概览
- repo_index_chunks:3408
- QA样本数(去重前/后):124/124removed=0
- Design样本数(去重前/后):6/3removed=3

## QA样本统计
- count:124
- evidence_avg:1.016
- trace_steps_avg:3.0
### language分布
- en:62
- zh:62
### domain分布
- mixed:82
- order:42
### qa_type分布
- flow:4
- rule:120
### difficulty分布
- hard:4
- medium:120

## Design样本统计
- count:3
- evidence_avg:3.0
- trace_steps_avg:4.333
### language分布
- en:2
- zh:1
### domain分布
- order:2
- stock:1
### difficulty分布
- easy:1
- hard:1
- medium:1

## 质量保证机制
- schema校验:每条样本必须包含核心字段(question/answer或requirement/design_output)、evidence、trace、meta
- grounding校验:evidence的chunk_id必须命中repo_index,且文件路径一致
- trace可审计:QA至少2步,Design至少3步
- 去重策略:基于question文本,并结合问题类型(domain/qa_type)与代码定位信息进行实例级去重,避免语义重复同时保留不同实现位置的有效样本
