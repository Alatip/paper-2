"""Dollar-budget mode of operators/budget.py (decided 2026-10-19). The v8 equal-token regressions stay in test_p2_budget.py
(P2_BUDGET_UNIT=tokens pinned in conftest); here the dollar section is run on the uncapped data (its default)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "operators"))
_C = {}


def dollar():
    if "d" not in _C:
        import budget
        from p2parts import datasrc
        prev = datasrc.DATA; datasrc.set_data("uncapped")
        try:
            _C["d"] = budget.compute_dollar()
        finally:
            datasrc.set_data(prev)
    return _C["d"]


def test_call_cost_formula():
    import budget
    pr = {"m": (1e-6, 4e-6)}
    assert budget.call_cost(pr, "m", 1000, 500) == pytest.approx(1000e-6 + 2000e-6)
    assert budget.call_cost(pr, "m", None, 500) == pytest.approx(2000e-6)


def test_best_at_picks_most_accurate_affordable():
    import budget
    pool = {"cheap": ({1: 1.0, 2: 1.0}, {1: 1.0, 2: 0.0}), "dear": ({1: 5.0, 2: 5.0}, {1: 1.0, 2: 1.0})}
    assert budget.best_at(pool, [1, 2], 0.5) is None
    assert budget.best_at(pool, [1, 2], 1.0)[0] == "cheap"
    assert budget.best_at(pool, [1, 2], 5.0)[:2] == ("dear", 1.0)


def test_prices_table():
    from p2parts import budget as IO
    pr = IO.prices()
    assert pr["claude_opus"] == pytest.approx((4e-6, 20e-6))          # list price, flagged in datasets/model-prices
    assert pr["qwen_qwen3.7-plus"][1] == pytest.approx(1.28e-6, rel=1e-3)


def test_dollar_tables_shape_and_invariants():
    d = dollar()
    for L, v in d["ops"].items():
        for k, x in v.items():
            assert x["cost"] > 0 and 0 <= x["acc"][0] <= 1
            if x["best"]: assert x["best"]["cost"] <= x["cost"] + 1e-15
        assert v["draws_16"]["cost"] > v["draws_1"]["cost"]
    for L, sm in d["same_model"].items():   # one model, small input: dollars ~ output tokens within 15 %
        assert 0.85 < sm["k"]["16"]["dollar_over_token"] < 1.15
    for L, v in d["t10_1"].items():
        for B, row in v["B"].items():
            for n, c in row["vote"].items():
                if c: assert c["cost"] <= row["cost_single"] + 1e-15 and c["b"] <= int(B)


def test_dollar_mode_is_default_and_tokens_flag_drops_it():
    import budget
    assert budget.UNIT == "tokens"            # pinned by conftest for the v8 regressions
    import os
    assert os.environ.get("P2_BUDGET_UNIT") == "tokens"
