from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

def test_stats_json_exists():
    p = ROOT / "data/dataset/stats.json"
    assert p.exists(), "缺少stats.json,请先运行scripts/06_postprocess_and_validate.py"
    obj = json.loads(p.read_text(encoding="utf-8"))
    assert "qa" in obj and "design" in obj
