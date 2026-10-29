"""Loading the uncapped text corpus from datasets/set_a/corpus-<layer>/ (answers.csv, run-order.csv) and the label audit from
datasets/set_a/label-audit/, with the same filters as the original analysis.

Conventions (section 4.1 of paper 1, appendix A):
- an item enters a layer only if all seven P7 models have a row for it, it is not excluded by audit wave 1 or 2,
  and (voting) every row has correct is not None;
- answers are compared via canon() (exact-value numeric identity at 12 significant digits;
  the 7-significant-digit rule is canon_v1, used only to replay frozen reference outputs)."""
import collections
import csv
import json
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, ROUND_HALF_UP, localcontext
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]               # repository root
DATASETS = ROOT / "datasets"
LABEL_AUDIT = DATASETS / "set_a" / "label-audit" / "label-audit-wave1.json"   # wave 1 (appendix A); "excluded" ids
AUDIT2 = DATASETS / "set_a" / "label-audit" / "audit2-wave2.json"             # wave 2
LAYERS = ["mmlu-pro", "gpqa-diamond", "gsm-hard", "olympiad"]
P7 = ["claude_haiku", "claude_opus", "claude_sonnet55", "deepseek_deepseek-v3.2", "mistralai_mistral-large-2512",
      "qwen_qwen3.7-plus", "z-ai_glm-5.3-flash"]
P4 = ["claude_haiku", "deepseek_deepseek-v3.2", "mistralai_mistral-large-2512", "z-ai_glm-5.3-flash"]
P3 = P4[:3]
POOLS = {"P7": P7, "P4": P4, "P3": P3}
NAMES_CSV = DATASETS / "model-names" / "names.csv"   # the one canonical model-name table (datasets/model-names/README.md)


def model_names(scope="corpus"):
    """{raw id: (canonical name, vendor)} for one scope of datasets/model-names/names.csv (corpus | corpus-judge | swe)."""
    return {r["raw"]: (r["canonical"], r["vendor"]) for r in csv.DictReader(open(NAMES_CSV, encoding="utf-8")) if r["scope"] == scope}


_N = model_names("corpus")
SH = {m: _N[m][0] for m in P7}       # display name = canonical name with version (e.g. 'Claude Haiku 4.5')
FAMILY = {m: _N[m][1] for m in P7}   # vendor label (same labels as the SWE vendor grouping)
# short labels printed by the original scripts (frozen outputs in tests/expected); used only to compare with those files
LEGACY = {r["canonical"]: r["legacy_label"] for r in csv.DictReader(open(NAMES_CSV, encoding="utf-8")) if r["scope"] == "corpus"}


_NUM = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
CANON_SIG = 12   # significant digits of the numeric grouping key


def _clean(a):
    """Shared cleanup of canon / canon_v1: strip, drop '$' and spaces, \\( \\) wrappers, thousands commas, trailing '.'."""
    t = str(a).strip().replace("$", "").replace(" ", "")
    t = re.sub(r"^\\\(|\\\)$", "", t)
    return re.sub(r"(?<=\d),(?=\d{3})", "", t).rstrip(".")


def _text(t):
    t = t.replace("\\dfrac", "\\frac")
    t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", t)
    return t.replace("\\sqrt{", "√").replace("\\sqrt", "√").replace("{", "").replace("}", "").lower()


def _exact(t):
    """Exact Fraction if t is a plain number, a decimal/scientific literal, or a/b of two such literals; else None."""
    parts = t.split("/")
    if len(parts) > 2 or not all(_NUM.fullmatch(x) for x in parts):
        return None
    try:
        v = Fraction(Decimal(parts[0]))
        if len(parts) == 2:
            v /= Fraction(Decimal(parts[1]))
        return v
    except (ZeroDivisionError, InvalidOperation):
        return None


def canon(a):
    """Canonical answer identity, used for every plurality count (section 2, plurality vote) and certification `strict`.

    Rule (the 7-significant-digit rule is kept as canon_v1):
    1. cleanup as before: '$', spaces, \\( \\) wrappers, thousands commas and a trailing '.' dropped; \\dfrac/\\frac{a}{b} -> a/b;
    2. if the result is a plain number, a decimal/scientific literal, or a/b of two such literals, it is parsed into an exact
       Fraction (so 1/2 == 0.5 == 5e-1 == \\frac{1}{2});
    3. the key is that exact value rounded (half-even) to 12 significant digits, as a deterministic string
       '<d.ddddddddddd>E<exp>' ('0' for zero). This is a grouping key, i.e. ~1e-12 relative tolerance that stays transitive:
       24112830.0000041 == 24112830, but 50627678 != 50627679 and 9099577.92 != 9099577.91666 (they differ at 12 digits).
       Exception: an exact integer with more than 12 digits is keyed by its full digit string (so two 23-digit integers
       that differ only in the last digit stay distinct); this exception is an implementation choice;
    4. anything else keeps the text normalisation (\\sqrt -> √, braces dropped, lower-cased); letters unchanged
       up to case. None / '' returned as is.
    canon() is the key of EXACT identity. Plurality counts use a coarser relation:
    same_answer = same canon key OR decimal_round_match (two decimals, the longer rounded half-up to the places
    of the shorter equals the shorter), grouped by transitive closure in voting.answer_groups."""
    if a in (None, ""):
        return a
    t = _clean(a)
    t = t.replace("\\dfrac", "\\frac")
    t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", t)
    v = _exact(t)
    if v is not None:
        if v == 0:
            return "0"
        if v.denominator == 1 and len(str(abs(v.numerator))) > CANON_SIG:
            return str(v.numerator)          # long exact integer: kept exact, never merged with a neighbour
        with localcontext() as c:
            c.prec = CANON_SIG
            c.rounding = ROUND_HALF_EVEN
            d = Decimal(v.numerator) / Decimal(v.denominator)
        return f"{d:.{CANON_SIG - 1}E}"
    return _text(t)


_DEC = re.compile(r"[+-]?\d*\.\d+")


def _decimal_literal(a):
    """(Decimal value, number of decimal places) if the cleaned answer is a plain decimal literal with a point and at
    least one digit after it (no exponent, no fraction, no thousands-only integer); else None."""
    if a in (None, ""):
        return None
    t = _clean(a)
    if not _DEC.fullmatch(t):
        return None
    return Decimal(t), len(t.split(".")[1])


def decimal_round_match(a, b):
    """Decimal-rounding rule (refines the 12-digit canon): two DECIMAL answers count as one
    vote if the one with more decimal places, rounded half-up (decimal arithmetic) to the number of decimal places of the
    other, equals the other: 48179.42857 -> 48179.43, 9099577.91666 -> 9099577.92. Integers never take part (48179 vs
    48179.43 stay different; integers are compared exactly by canon); fractions a/b and scientific literals neither
    (exact value only, i.e. canon). This relation is NOT transitive; grouping uses its transitive closure
    (voting.answer_groups)."""
    x, y = _decimal_literal(a), _decimal_literal(b)
    if x is None or y is None:
        return False
    (va, pa), (vb, pb) = sorted([x, y], key=lambda z: z[1])
    with localcontext() as c:
        c.prec = 100
        return vb.quantize(Decimal(1).scaleb(-pa), rounding=ROUND_HALF_UP) == va


def same_answer(a, b):
    """Pairwise answer identity of every plurality count: same canon() key OR decimal_round_match.
    Empty / None answers are never the same as anything."""
    if a in (None, "") or b in (None, ""):
        return False
    return canon(a) == canon(b) or decimal_round_match(a, b)


def same_answer_v1(a, b):
    """OLD pairwise identity (canon_v1 key only), for replaying frozen outputs (tests/conftest.py fixture canon_v1)."""
    if a in (None, "") or b in (None, ""):
        return False
    return canon_v1(a) == canon_v1(b)


def canon_v1(a):
    """7-significant-digit identity: numbers compared at 7 significant
    digits via float, fractions not evaluated. Kept ONLY for regression tests that replay frozen outputs in tests/expected
    and for comparisons with canon(); do not use for new numbers."""
    if a in (None, ""):
        return a
    t = _clean(a)
    try:
        return repr(float(f"{float(t):.7g}"))
    except ValueError:
        pass
    return _text(t)


def excluded(path=LABEL_AUDIT, path2=AUDIT2):
    """Set of item ids dropped by the label audit, BOTH waves (appendix A): keys of wave-1 "excluded" plus the wave-2
    "excluded" list."""
    return set(json.load(open(path))["excluded"]) | set(json.load(open(path2))["excluded"])


def _row(r):
    """answers.csv row -> {answer, correct, source, out_tok}; empty cells -> None (as in the merged jsonl)."""
    return dict(answer=r["answer"] or None, correct=None if r["correct"] == "" else bool(int(r["correct"])),
                source=r["source"], out_tok=int(r["out_tok"]) if r["out_tok"] else None)


def load_layer(layer, root=DATASETS, models=P7, require_correct=True):
    """{item_id: {model: row}} for one layer, filtered as in the original voting analysis: all seven P7 models present (also when
    a smaller `models` pool is requested), not excluded by either audit wave, (require_correct) every correct is not None. Dict order = run-order.csv (order of the first P7 model's
    merged run file), which the split-half RNG conventions depend on. GPQA answers are letters ('UNPARSED' for a GLM-5.3-Flash
    answer that is not a letter; text is not stored)."""
    if not set(models) <= set(P7):
        raise ValueError(f"load_layer: models must be a subset of P7, got {models}")
    ex = excluded()
    folder = Path(root) / "set_a" / f"corpus-{layer}"
    order = [r["id"] for r in csv.DictReader(open(folder / "run-order.csv"))]
    rows = collections.defaultdict(dict)
    for r in csv.DictReader(open(folder / "answers.csv", encoding="utf-8")):
        if r["model"] in P7:
            rows[r["id"]][r["model"]] = _row(r)
    # corpus membership is decided on ALL seven models, whatever subset is requested (section 4.1: P4 and P3 use the
    # same 941 items); only then the rows are projected to `models`
    full = {i for i, d in rows.items() if len(d) == len(P7) and i not in ex
            and (not require_correct or all(x["correct"] is not None for x in d.values()))}
    missing = full - set(order)
    if missing:  # guard: every corpus item must have a place in run-order.csv, the split-half RNG depends on it
        raise ValueError(f"load_layer({layer}): {len(missing)} corpus items missing from run-order.csv, e.g. {sorted(missing)[:3]}")
    return {i: {m: rows[i][m] for m in models} for i in order if i in full}


def correctness_matrix(layer, root=DATASETS, models=P7, sort_ids=True):
    """models x items 0/1 array. ValueError if any cell has correct = None (it used to become 0 via int(bool(None))).
    The matrix is in id-sorted order: fine for deterministic computations and for operators/factor (the original
    convention of the original factor analysis); for split-half or any RNG over items use the load_layer order (sort_ids=False)."""
    X = load_layer(layer, root, models, require_correct=False)
    ids = sorted(X) if sort_ids else list(X)
    miss = [(i, m) for i in ids for m in models if X[i][m]["correct"] is None]
    if miss:
        raise ValueError(f"correctness_matrix({layer}): correct = None in {len(miss)} cell(s), first {miss[:3]}")
    return np.array([[int(bool(X[i][m]["correct"])) for i in ids] for m in models]), ids
