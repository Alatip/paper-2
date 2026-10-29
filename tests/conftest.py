import os
import sys
from pathlib import Path

# Paper-2 pinned regressions replay frozen outputs of the capped (8,192-token) runs: the test session reads the capped dataset
# files unless P2_DATA is set explicitly. tests/test_p2_uncapped.py switches to the uncapped twins itself (2026-10-18 rerun).
os.environ.setdefault("P2_DATA", "capped")
os.environ.setdefault("P2_DECOMP_BUDGET", "8192")   # §9 regressions = the preregistered 8,192 run
os.environ.setdefault("P2_BUDGET_UNIT", "tokens")   # §10 regressions = v8's equal-token tables; dollar mode: test_p2_budget_dollar.py

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "lib"))
import pytest  # noqa: E402
from p1stats import data  # noqa: E402


@pytest.fixture
def canon_v1(monkeypatch):
    """Swap in the OLD 7-significant-digit identity (data.canon_v1) for tests that replay frozen outputs of paper 1 (tests/expected in the paper-1 repository),
    which were produced before the canon change of 2026-10-17. Every other test uses the current data.canon."""
    monkeypatch.setattr(data, "canon", data.canon_v1)
    monkeypatch.setattr(data, "same_answer", data.same_answer_v1)   # no decimal-rounding rule in the frozen outputs
