在本作业中，代码仓来源并不限定为第三方项目。为保证实验的可控性与可复现性，我构建了一个遵循真实工程实践的最小示例代码仓，并将其作为 GitHub 公开仓库的一部分。该仓库用于模拟企业内部常见的业务流程与规则，使得训练数据生成与 trace 机制能够被清晰验证。系统本身对代码仓来源是解耦的，同样可以无缝适配任意第三方 GitHub 项目。
## Dataset Schema (Draft v1 - frozen)

We use JSONL for training samples. Each record contains:
- scenario: 1 (QA) or 2 (Design)
- evidence: code snippets with file path and line range
- trace: auditable reasoning chain (claims supported by evidence ids)

### Scenario 1 (QA)
```json
{
  "id": "uuid",
  "scenario": 1,
  "repo": { "name": "AutoCodeDataPipeline", "path": "codebase/" },
  "language": "zh",
  "question": "",
  "answer": "",
  "evidence": [
    { "id": "E1", "file": "codebase/app/pipeline.py", "start_line": 1, "end_line": 1, "snippet": "" }
  ],
  "trace": {
    "steps": [{ "claim": "", "support": ["E1"] }],
    "conclusion": ""
  }
}
