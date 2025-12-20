import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def _read_jsonl(p: Path):
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out

def test_rules_and_flows_exist():
    rules_p = ROOT / "data/extracted/rules.jsonl"
    flows_p = ROOT / "data/extracted/flows.jsonl"
    assert rules_p.exists(), "缺少rules.jsonl,请先运行scripts/03_extract_rules_and_flows.py"
    assert flows_p.exists(), "缺少flows.jsonl,请先运行scripts/03_extract_rules_and_flows.py"

    rules = _read_jsonl(rules_p)
    flows = _read_jsonl(flows_p)

    assert isinstance(rules, list) and len(rules) > 0
    assert isinstance(flows, list) and len(flows) > 0

    r0 = rules[0]
    for k in ["rule_id", "title", "description", "domain", "evidence_chunks", "tags", "types"]:
        assert k in r0
    assert len(r0["evidence_chunks"]) > 0

    f0 = flows[0]
    for k in ["flow_id", "name", "domain", "steps", "evidence_chunks", "related_rules"]:
        assert k in f0
