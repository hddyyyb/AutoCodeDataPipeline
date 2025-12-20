{
  "sample_id": "qa_xxxxx",
  "task_type": "qa",
  "language": "zh",
  "question": "...?",
  "answer": "...",
  "evidence": [
    {
      "chunk_id": "abcd1234",
      "file_path": "mall-portal/src/...",
      "start_line": 45,
      "end_line": 98,
      "content": "public void lockStock(...) { ... }"
    }
  ],
  "trace": {
    "type": "rule_based | flow_based",
    "rule_ids": ["rule_xxx"],
    "flow_id": "flow_xxx",
    "reasoning_steps": [
      "步骤1：定位到库存锁定相关规则(rule_xxx)",
      "步骤2：从对应代码段验证锁定条件",
      "步骤3：得出结论"
    ]
  },
  "meta": {
    "domain": "order | stock | mixed",
    "qa_type": "fact | rule | flow",
    "difficulty": "easy | medium | hard",
    "generator": "template_v1",
    "source": "AutoCodeDataPipeline"
  }
}
