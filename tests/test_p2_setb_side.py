"""Regression tests for the set-B side analyses cited in the paper (Sections 4.1.4, 4.2.1, 4.3):
summaries/sc_fair_setb.py and summaries/mechanics_setb.py reproduce their stored outputs in summaries/out2/."""
import json, shutil, subprocess, sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script,outputs", [
    ("summaries/sc_fair_setb.py", ["sc_fair_setb.json", "sc_fair_setb.md"]),
    ("summaries/mechanics_setb.py", ["mechanics_setb.json", "mechanics_setb.md"]),
])
def test_side_script_reproduces_stored_output(tmp_path, script, outputs):
    work = tmp_path / "repo"
    for part in ("datasets", "lib", "operators", "summaries"):
        shutil.copytree(ROOT / part, work / part, ignore=shutil.ignore_patterns("__pycache__"))
    for name in outputs:
        (work / "summaries/out2" / name).unlink()
    subprocess.run([sys.executable, "-B", str(work / script)], check=True, capture_output=True, cwd=work)
    for name in outputs:
        assert (work / "summaries/out2" / name).read_bytes() == (ROOT / "summaries/out2" / name).read_bytes(), name


def test_key_numbers():
    s = json.loads((ROOT / "summaries/out2/sc_fair_setb.json").read_text())
    assert (s["glm"]["W"], s["glm"]["L"], s["glm"]["empty_first"]) == (12, 3, 72)
    assert round(s["glm"]["acc_first_answered"], 3) == 0.495 and round(s["glm"]["gain_fair_pp"], 1) == 4.5
    assert round(s["holm_if_replaced"]["glm"], 2) == 0.28 and s["holm_if_replaced"]["n_significant"] == 0
    m = json.loads((ROOT / "summaries/out2/mechanics_setb.json").read_text())
    assert round(m["robust"]["paper_order"]["vote"], 3) == 0.770
    assert round(m["robust"]["tie_half"]["vote"], 3) == 0.755
    assert (m["q2"]["vote7_vs_opus"]["tie_changes"], m["q2"]["vote7_vs_opus"]["changes"]) == (13, 34)
