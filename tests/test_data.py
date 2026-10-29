"""Tests of lib/p1stats/data.py (corpus loader, label audit, canon()).

Where possible the expected values are recomputed independently from datasets/corpus-*/items.csv and answers.csv with the
csv module, not through the loader. The heldout_best_minus keep()-empties-B guard and median_member even-n conventions
are already covered in tests/test_independent.py (test_heldout_keep_restricts_B_after_choice_and_counts_dropped,
test_median_member_hand*), so they are not repeated here."""
import csv
import json
import shutil

import pytest

from p1stats import data

EXPECTED_N = {"mmlu-pro": 553, "gpqa-diamond": 164, "gsm-hard": 141, "olympiad": 83}
AUDIT_DIR = data.DATASETS / "set_a" / "label-audit"


def _items(layer):
    return list(csv.DictReader(open(data.DATASETS / "set_a" / f"corpus-{layer}" / "items.csv", encoding="utf-8")))


def _answers(layer):
    return list(csv.DictReader(open(data.DATASETS / "set_a" / f"corpus-{layer}" / "answers.csv", encoding="utf-8")))


def _corpus_ids_from_items(layer):
    rows = _items(layer)
    assert "status" in rows[0], "items.csv has no status column"
    return {r["id"] for r in rows if r["status"] == "corpus"}


@pytest.fixture(scope="module")
def loaded():
    return {l: data.load_layer(l) for l in data.LAYERS}


# ---------------------------------------------------------------- per-layer counts and membership

@pytest.mark.parametrize("layer", data.LAYERS)
def test_layer_count(loaded, layer):
    assert len(loaded[layer]) == EXPECTED_N[layer]


def test_total_941(loaded):
    assert sum(len(X) for X in loaded.values()) == 941


@pytest.mark.parametrize("layer", data.LAYERS)
def test_loaded_ids_equal_items_csv_status_corpus(loaded, layer):
    assert set(loaded[layer]) == _corpus_ids_from_items(layer)


@pytest.mark.parametrize("layer", data.LAYERS)
def test_loaded_ids_recomputed_from_answers_csv(loaded, layer):
    """Independent recount: ids with a row for all seven P7 models, not audit-excluded, all correct cells filled."""
    ex = set(json.load(open(AUDIT_DIR / "label-audit-wave1.json"))["excluded"]) \
        | set(json.load(open(AUDIT_DIR / "audit2-wave2.json"))["excluded"])
    per_id = {}
    for r in _answers(layer):
        if r["model"] in data.P7:
            per_id.setdefault(r["id"], {})[r["model"]] = r["correct"]
    want = {i for i, d in per_id.items()
            if len(d) == 7 and i not in ex and all(c != "" for c in d.values())}
    assert set(loaded[layer]) == want


def test_rows_have_all_seven_models_and_fields(loaded):
    for X in loaded.values():
        for i, d in X.items():
            assert set(d) == set(data.P7), i
            for m, row in d.items():
                assert set(row) == {"answer", "correct", "source", "out_tok"}
                assert isinstance(row["correct"], bool), (i, m)


# ---------------------------------------------------------------- all-wrong items

def test_exactly_three_all_wrong_items(loaded):
    all_wrong = {l: [i for i, d in X.items() if not any(r["correct"] for r in d.values())] for l, X in loaded.items()}
    assert sum(len(v) for v in all_wrong.values()) == 3
    assert len(all_wrong["mmlu-pro"]) == 1
    assert len(all_wrong["gpqa-diamond"]) == 2
    assert all_wrong["gsm-hard"] == [] and all_wrong["olympiad"] == []


def test_all_wrong_items_recomputed_from_answers_csv():
    """Same count straight from answers.csv restricted to status == corpus items."""
    n = 0
    for layer in data.LAYERS:
        corpus = _corpus_ids_from_items(layer)
        wrong = {}
        for r in _answers(layer):
            if r["id"] in corpus and r["model"] in data.P7:
                wrong.setdefault(r["id"], []).append(r["correct"] == "1")
        n += sum(1 for i, v in wrong.items() if len(v) == 7 and not any(v))
    assert n == 3


# ---------------------------------------------------------------- pools

@pytest.mark.parametrize("pool", ["P4", "P3"])
@pytest.mark.parametrize("layer", data.LAYERS)
def test_smaller_pool_same_ids_as_p7(loaded, pool, layer):
    models = data.POOLS[pool]
    X = data.load_layer(layer, models=models)
    assert list(X) == list(loaded[layer])          # same ids, same order
    for i, d in X.items():
        assert list(d) == models, i                # only the requested models, in pool order


def test_gpqa_74_absent_in_every_pool():
    """gpqa:74 has rows for 5 models only (Claude Opus 5.5, Claude Sonnet 5.5 missing) -> all five P4 members are present, but membership is
    decided on all seven, so it must not appear even when P4 / P3 is requested."""
    rows = {r["model"] for r in _answers("gpqa-diamond") if r["id"] == "gpqa:74"}
    assert set(data.P4) <= rows and not set(data.P7) <= rows   # precondition of the test
    for pool in ("P7", "P4", "P3"):
        assert "gpqa:74" not in data.load_layer("gpqa-diamond", models=data.POOLS[pool])


@pytest.mark.parametrize("models", [["claude_haiku", "gpt-5"], ["nobody"], ["claude_haiku", "claude_haiku2"]])
def test_models_outside_p7_raise(models):
    with pytest.raises(ValueError):
        data.load_layer("gsm-hard", models=models)


# ---------------------------------------------------------------- label audit

def test_excluded_contains_both_waves():
    w1 = json.load(open(AUDIT_DIR / "label-audit-wave1.json"))["excluded"]
    w2 = json.load(open(AUDIT_DIR / "audit2-wave2.json"))["excluded"]
    assert len(w1) == 22 and len(w2) == 14        # table A.1: excluded 22 / 14
    ex = data.excluded()
    assert ex == set(w1) | set(w2)
    assert len(ex) == len(set(w1) | set(w2))
    assert set(w1) <= ex and set(w2) <= ex


def test_no_excluded_id_loaded(loaded):
    w1 = set(json.load(open(AUDIT_DIR / "label-audit-wave1.json"))["excluded"])
    w2 = set(json.load(open(AUDIT_DIR / "audit2-wave2.json"))["excluded"])
    all_ids = set().union(*loaded.values())
    assert not (all_ids & w1)
    assert not (all_ids & w2)


def test_no_items_csv_excluded_status_loaded(loaded):
    for layer, X in loaded.items():
        bad = {r["id"] for r in _items(layer) if r["status"].startswith("excluded")}
        assert not (set(X) & bad), layer
    # the excluded ids in items.csv and the audit JSONs agree
    from_items = {r["id"] for l in data.LAYERS for r in _items(l) if r["status"].startswith("excluded")}
    assert from_items == data.excluded()


# ---------------------------------------------------------------- run-order

@pytest.mark.parametrize("layer", data.LAYERS)
def test_dict_order_equals_run_order(loaded, layer):
    order = [r["id"] for r in csv.DictReader(open(data.DATASETS / "set_a" / f"corpus-{layer}" / "run-order.csv"))]
    assert len(order) == len(set(order))
    assert set(loaded[layer]) <= set(order)
    assert list(loaded[layer]) == [i for i in order if i in loaded[layer]]


def test_missing_run_order_entry_raises(tmp_path):
    src = data.DATASETS / "set_a" / "corpus-olympiad"
    dst = tmp_path / "set_a" / "corpus-olympiad"
    dst.mkdir(parents=True)
    for f in ("answers.csv", "items.csv"):
        shutil.copy(src / f, dst / f)
    ids = list(data.load_layer("olympiad"))
    victim = ids[len(ids) // 2]
    order = [r["id"] for r in csv.DictReader(open(src / "run-order.csv"))]
    with open(dst / "run-order.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id"])
        w.writerows([i] for i in order if i != victim)
    with pytest.raises(ValueError, match="run-order"):
        data.load_layer("olympiad", root=tmp_path)
    # the untouched copy still loads identically
    (dst / "run-order.csv").write_text((src / "run-order.csv").read_text())
    assert list(data.load_layer("olympiad", root=tmp_path)) == ids


def test_root_argument_is_used(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_layer("olympiad", root=tmp_path)


# ---------------------------------------------------------------- correctness_matrix

@pytest.mark.parametrize("layer", data.LAYERS)
def test_correctness_matrix_shape_and_sorted_ids(loaded, layer):
    M, ids = data.correctness_matrix(layer)
    assert M.shape == (7, EXPECTED_N[layer])
    assert ids == sorted(ids) and set(ids) == set(loaded[layer])
    assert set(M.flatten().tolist()) <= {0, 1}
    for k, m in enumerate(data.P7):
        assert M[k].tolist() == [int(loaded[layer][i][m]["correct"]) for i in ids]


def test_correctness_matrix_unsorted_keeps_run_order(loaded):
    M, ids = data.correctness_matrix("olympiad", sort_ids=False)
    assert ids == list(loaded["olympiad"])
    assert M.shape == (7, 83)


def test_correctness_matrix_subpool():
    M, ids = data.correctness_matrix("gsm-hard", models=data.P3)
    assert M.shape == (3, 141)


def test_correctness_matrix_raises_on_missing_correct(tmp_path):
    """correct = None must raise, not become 0 via int(bool(None)) (decision 2026-10-17)."""
    src = data.DATASETS / "set_a" / "corpus-olympiad"
    dst = tmp_path / "set_a" / "corpus-olympiad"
    dst.mkdir(parents=True)
    for f in ("items.csv", "run-order.csv"):
        shutil.copy(src / f, dst / f)
    victim = sorted(data.load_layer("olympiad"))[0]
    rows = list(csv.DictReader(open(src / "answers.csv")))
    hit = 0
    for r in rows:
        if r["id"] == victim and r["model"] == data.P7[0]:
            r["correct"] = ""; hit += 1
    assert hit == 1
    with open(dst / "answers.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with pytest.raises(ValueError, match="correct = None"):
        data.correctness_matrix("olympiad", root=tmp_path)
    # the untouched data still builds
    assert data.correctness_matrix("olympiad")[0].shape == (7, 83)


# ---------------------------------------------------------------- canon

EQUAL = [
    ("24", "24.00"), ("24", "24.0"), ("2,400", "2400"), ("$2,400", "2400"), ("1,234,567", "1234567"),
    ("24.", "24"), ("0.5", ".5"), ("-3", "-3.0"), ("1e3", "1000"),
    ("\\frac{1}{2}", "1/2"), ("\\dfrac{1}{2}", "1/2"), ("\\sqrt{2}", "√2"), ("\\sqrt2", "√2"),
    ("\\(\\frac{3}{4}\\)", "3/4"), ("$\\frac{1}{2}$", "1/2"), ("A", "a"), (" B ", "b"), ("x + y", "x+y"),
    ("2\\sqrt{3}", "2√3"), ("{a}", "a"),
    # new numeric identity (decision 2026-10-17)
    ("1/2", "0.5"), ("0.5", "5e-1"), ("\\frac{1}{2}", "0.5"), ("24112830.0000041", "24112830"), ("0", "-0.0"),
]


@pytest.mark.parametrize("a, b", EQUAL)
def test_canon_equal(a, b):
    assert data.canon(a) == data.canon(b)


@pytest.mark.parametrize("a, b", [("24", "25"), ("1/2", "1/3"), ("a", "b"), ("√2", "√3"), ("1234567", "1234568"),
                                  ("50627678", "50627679"), ("72768591", "72768592"), ("26073328", "26073332"),
                                  ("42159089/3", "14053030"), ("12345678", "12345679"), ("3.14159265358979", "3.14159265")])
def test_canon_distinct(a, b):
    assert data.canon(a) != data.canon(b)


def test_canon_half_equals_all_forms():
    assert data.canon("1/2") == data.canon("0.5") == data.canon("5e-1") == data.canon("\\frac{1}{2}") == data.canon("\\dfrac{1}{2}")


def test_canon_twelve_significant_digits_intended():
    """INTENDED behaviour (decision 2026-10-17): the key is the exact value rounded to 12 significant digits.
    9099577.92 and 9099577.91666 differ at the 12th significant digit, so they are DIFFERENT answers, although they agree
    to ~1e-8 relative. A difference beyond the 12th digit is merged."""
    assert data.canon("9099577.92") != data.canon("9099577.91666")
    assert data.canon("24112830.0000041") == data.canon("24112830") == "2.41128300000E+7"
    assert data.canon("1.00000000000001") == data.canon("1")


def test_canon_huge_integers():
    a, b = "12345678901234567890123", "12345678901234567890124"
    assert data.canon(a) != data.canon(b) and data.canon(a) == a
    assert data.canon("12,345,678,901,234,567,890,123") == data.canon(a)


def test_canon_key_format():
    assert data.canon("24") == data.canon(24) == "2.40000000000E+1"
    assert data.canon("2,400") == "2.40000000000E+3"
    assert data.canon("0") == "0"


def test_canon_none_and_empty_kept():
    assert data.canon(None) is None
    assert data.canon("") == ""


def test_canon_letters_and_text():
    assert data.canon("C") == "c" and data.canon("UNPARSED") == "unparsed" and data.canon("1/0") == "1/0"


def test_canon_v1_kept_for_regression():
    """The old 7-significant-digit identity, kept only to replay frozen outputs."""
    assert data.canon_v1("12345678") == data.canon_v1("12345679") == "12345680.0"
    assert data.canon_v1("24") == "24.0" and data.canon_v1("1/2") != data.canon_v1("0.5")


def test_canon_thousands_comma_only_before_three_digits():
    assert data.canon("1,23") == "1,23"        # not a thousands separator: not a number, left as text
    assert data.canon("12,345") == data.canon("12345")


def test_canon_task_cases_2026_10_17():
    """Cases named in the decision of 2026-10-17 (maintainer workspace, not published)."""
    c = data.canon
    assert c("1/2") == c("0.5") == c("5e-1") == c("\\frac{1}{2}")
    assert c("50627678") != c("50627679")
    assert c("24112830.0000041") == c("24112830")
    assert c("9099577.92") != c("9099577.91666")          # INTENDED: same value at different rounding is split
    assert c("9,099,577.91666") == c("9099577.91666") and c("5,072,530") == c("5072530")
    assert c("14178334486374317338200") != c("14178334489937770264200")
    assert c("A") == c("a") != c("b")
    assert c(None) is None and c("") == ""


# ---------------------------------------------------------------- decimal-rounding rule + answer_groups (2026-10-17)
from p1stats import voting as _V  # noqa: E402


def _partition(answers):
    g = _V.answer_groups(answers)
    groups = {}
    for a, gi in zip(answers, g):
        if gi is not None:
            groups.setdefault(gi, set()).add(a)
    return {frozenset(s) for s in groups.values()}


def test_groups_decimal_rounding_examples():
    assert len(set(_V.answer_groups(["48179.43", "48179.42857", "48179.42857142857"]))) == 1
    assert len(set(_V.answer_groups(["9099577.92", "9099577.91666"]))) == 1
    assert data.same_answer("48179.42857", "48179.43") and data.same_answer("9099577.91666", "9099577.92")
    assert not data.same_answer("9099577.91666", "9099577.93")


def test_groups_integers_exact():
    assert not data.same_answer("48179", "48179.43")
    assert len(set(_V.answer_groups(["48179", "48179.43", "48179.42857"]))) == 2
    assert not data.same_answer("50627678", "50627679")
    assert not data.same_answer("5072530", "5072529.6")
    assert data.same_answer("48179", "48179.0")          # same exact value: canon key


def test_groups_fractions_and_scientific_exact_only():
    assert not data.decimal_round_match("109194935/12", "9099577.92")
    assert not data.decimal_round_match("4.81794e4", "48179.4")
    assert data.same_answer("1/2", "0.5")                  # exact value via canon


def test_groups_half_up():
    assert data.same_answer("0.125", "0.13") and not data.same_answer("0.125", "0.12")


def test_groups_chaining_closure():
    """0.06 ~ 0.1 and 0.14 ~ 0.1, so the closure joins 0.06 and 0.14 although they are not directly related."""
    assert not data.same_answer("0.06", "0.14")
    assert len(set(_V.answer_groups(["0.06", "0.1", "0.14"]))) == 1
    assert len(set(_V.answer_groups(["0.06", "0.14"]))) == 2


def test_groups_order_independent():
    import itertools
    a = ["0.06", "0.14", "0.1", "48179", "48179.43", "48179.42857", None, "C", "c", "9099577.91666", "9099577.92"]
    ref = _partition(a)
    for p in itertools.islice(itertools.permutations(a), 0, 5000, 7):
        assert _partition(list(p)) == ref


def test_groups_letters_and_none_unaffected():
    assert _V.answer_groups(["A", "a", "B", None, ""]) == [0, 0, 2, None, None]
    assert _V.plurality([("A", True), ("b", False), ("a", True), (None, False)]) == 1.0


def test_plurality_uses_groups():
    # 48179.43 group (3 votes) beats 48179 (2 votes)
    ans = [("48179", False), ("48179", False), ("48179.43", True), ("48179.42857", True), ("48179.42857142857", True)]
    assert _V.plurality(ans) == 1.0
