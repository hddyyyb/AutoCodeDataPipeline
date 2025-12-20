import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_domain_map_schema_minimal():
    p = ROOT / "data/extracted/domain_map.json"
    assert p.exists(), "缺少data/extracted/domain_map.json,请先运行scripts/02_build_domain_map.py"

    obj = json.loads(p.read_text(encoding="utf-8"))
    assert "repo" in obj
    assert "boundaries" in obj
    assert "entities" in obj
    assert "operations" in obj
    assert "candidate_flows" in obj

    assert isinstance(obj["entities"], list)
    assert isinstance(obj["operations"], list)

    if obj["entities"]:
        e0 = obj["entities"][0]
        for k in ["name", "domain", "confidence", "evidence_chunks"]:
            assert k in e0

    if obj["operations"]:
        o0 = obj["operations"][0]
        for k in ["name", "domain", "confidence", "evidence_chunks", "signals"]:
            assert k in o0
