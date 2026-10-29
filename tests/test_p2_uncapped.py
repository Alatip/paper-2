"""The uncapped twins of the 2026-10-18 rerun (datasets/**/<stem>.uncapped.<ext>).

The rest of the test session runs on the capped files (tests/conftest.py sets P2_DATA=capped, pinned v8 regressions); here the
loaders are switched to the twins and the operators that read them must run and see the rerun rows."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "operators"))
from p2parts import datasrc  # noqa: E402

TWINS = sorted((ROOT / "datasets").rglob("*.uncapped.*"))
pytestmark = pytest.mark.skipif(not TWINS, reason="no uncapped twins installed")


@pytest.fixture
def uncapped():
    from p2parts import tn5 as T
    prev = datasrc.DATA; datasrc.set_data("uncapped"); T._cache.clear()
    yield
    datasrc.set_data(prev); T._cache.clear()


def test_pick_follows_flag():
    p = TWINS[0]; stem, _, ext = p.name.partition(".uncapped.")
    orig = p.with_name(f"{stem}.{ext}")
    prev = datasrc.DATA
    try:
        datasrc.set_data("capped"); assert datasrc.pick(orig) == orig
        datasrc.set_data("uncapped"); assert datasrc.pick(orig) == p
    finally:
        datasrc.set_data(prev)


def test_loaders_see_rerun_rows(uncapped):
    from p2parts import tn5 as T, debx as D, budget as B
    mixed = [r for L in T.HARD for r in T.debate(L)]
    assert any(r.get("src", "").startswith("rerun2") for r in mixed)
    assert sum(r.get("src", "").startswith("p1rerun") for L in T.HARD for m in T.MAIN for r in T.matrix(m, L)) > 0
    assert any(r.get("src", "").startswith("rerun2") for r in D.weak_debate())
    assert any(r.get("src", "").startswith("rerun2") for r in D.weak_judge())
    assert any(r.get("src", "").startswith("rerun2") for r in D.ct_mixed())
    assert any(r.get("src", "").startswith("rerun2") for r in D.six_debate())
    deb = B._csv(B.DEBATES)
    assert any(r["src"].startswith("rerun2") for r in deb)


def test_rehit_rows_are_wrong(uncapped):
    """Paper-1 rule: an answer that hit 65,536 again is graded wrong (not a missing vote)."""
    from p2parts import tn5 as T, debx as D
    rows = [r for L in T.HARD for r in T.debate(L)] + D.weak_debate() + D.ct_mixed()
    rehit = [r for r in rows if str(r.get("src", "")).endswith("-length")]
    assert all(r["correct"] is False for r in rehit)
    assert all(r["pick_correct"] is False for r in D.weak_judge() if str(r.get("src", "")).endswith("-length"))


def test_operators_run_on_twins(uncapped):
    import debate, debate_weak, debate_bounds, budget
    assert debate.run()
    assert debate_weak.run()
    assert debate_bounds.run()
    assert budget.run()
