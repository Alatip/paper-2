"""Area tn5: paths, loaders and budget-free helpers for the self-consistency, diversity, debate and synthesis operators
(paper v8 sections 5.7, 5.8, 6.3 and their restatements in 6.7).

Reads ONLY datasets/set_a/tn5-*/ (built once by a maintainer build script (not published) from the maintainer's private raw runs of tech-note-5). Each loader
returns rows shaped like the tech-note-5 raw rows (`tok` is rebuilt as {"in", "out", "think"}), grouped like the raw files
(per model / layer / arm / mode), in the original file order, so that the operator code can be a line-by-line port of
tech-note-5 stats.py / extra.py.

Helpers are verbatim copies of tech-note-5 stats.py (canon with 7 significant digits, phi, tetra (cos-pi), icc1, n_eff, the
plurality with ties to the first row, Wilson interval, exact McNemar, Spearman).
Answer identity (decision 2026-10-18): default = paper 1's rule (p2parts/answer_id.py: 12-digit canon + decimal rounding,
chained groups; a group is correct if any member is). canon() below is the OLD tech-note-5 rule (7 significant digits) and is
used only when answer_id.RULE == "v8" (P2_ANSWER_RULE=v8), which reproduces the v8 numbers.
Truncation policy: TRUNC_AS_WRONG = False (a truncated row, correct is None, is a missing vote), as in v8.
"""
import collections, gzip, itertools, json, math, re, csv
from pathlib import Path

from p2parts import answer_id as AID

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
HARD = ["mmlu-pro", "gpqa-diamond", "gsm-hard", "olympiad"]
MAIN = ["claude_haiku", "claude_opus", "claude_sonnet55", "deepseek_deepseek-v3.2", "mistralai_mistral-large-2512",
        "qwen_qwen3.7-plus", "z-ai_glm-5.3-flash"]
SAMPLE_MODELS = ["claude_haiku", "deepseek_deepseek-v3.2", "z-ai_glm-5.3-flash"]   # sorted dir order of stats.py
DEBATE_AGENTS = ["deepseek/deepseek-v3.2", "z-ai/glm-5.3-flash", "mistralai/mistral-large-2512", "qwen/qwen3.7-plus",
                 "claude:haiku", "claude:sonnet55"]
ARMS = ("plain", "selfonly", "pressure")
TRUNC_AS_WRONG = False
# T=0 matrix source (decided 2026-10-18): "corpus" (default) = paper 1's uncapped matrix
# (datasets/corpus-<layer>/answers.csv via p1stats.data.load_layer: 941 items after label audit wave 1+2, max 65,536 output
# tokens, an answer that hit 65,536 graded wrong); "tn5" = the tech-note-5 matrix capped at 8,192 (datasets/set_a/tn5-matrix), as in
# v8. Env P2_MATRIX (read at import) or set_matrix(). Used by diversity.family_triples and
# self_consistency.untraced_olympiad_deepseek only; debate round 0 always stays on tn5-matrix (it seeded the debates).
import os as _os
MATRIX_SOURCE = _os.environ.get("P2_MATRIX", "corpus")
assert MATRIX_SOURCE in ("corpus", "tn5"), MATRIX_SOURCE


def set_matrix(src):
    global MATRIX_SOURCE
    assert src in ("corpus", "tn5"), src
    MATRIX_SOURCE = src


def corpus_matrix(layer):
    """{id: {model: row}} of paper 1's uncapped T=0 matrix (all 7 models graded on every item; row keys answer, correct,
    source, out_tok)."""
    import sys as _sys
    _sys.path.insert(0, str(ROOT / "lib"))
    from p1stats import data as _D
    return _D.load_layer(layer)


# ---------------- loaders ----------------
def _rows(name):
    out = []
    from p2parts import datasrc
    d = DATASETS / "set_a" / name
    with gzip.open(datasrc.pick(d / "rows.jsonl.gz"), "rt", encoding="utf-8") as f:   # P2_DATA twin
        for line in f:
            r = json.loads(line)
            r["tok"] = {"in": r.pop("tok_in", None), "out": r.pop("tok_out", None), "think": r.pop("tok_think", None)}
            out.append(r)
    return out


_cache = {}


def _grouped(name, *keys):
    k = (name, keys)
    if k not in _cache:
        g = collections.defaultdict(list)
        for r in _rows(name): g[tuple(r[x] for x in keys)].append(r)
        _cache[k] = g
    return _cache[k]


def samples(model, layer):
    """Rows of datasets/set_a/tn5-samples for (model, layer), model in SAMPLE_MODELS; [] if absent."""
    return list(_grouped("tn5-samples", "model", "layer").get((model, layer), []))


def persona_models():
    return sorted({m for (m, _) in _grouped("tn5-personas", "model", "layer")})


def persona_samples(model, layer):
    return list(_grouped("tn5-personas", "model", "layer").get((model, layer), []))


def matrix(model, layer):
    """Rows of datasets/set_a/tn5-matrix for (model, layer): T=0 answers, round 0 of the mixed debate."""
    return list(_grouped("tn5-matrix", "model", "layer").get((model, layer), []))


def debate(layer):
    return list(_grouped("tn5-debate-mixed", "layer").get((layer,), []))


def homo(arm, layer):
    return list(_grouped("tn5-debate-homo", "arm", "layer").get((arm, layer), []))


def aggregate(mode, layer):
    return list(_grouped("tn5-aggregate", "mode", "layer").get((mode, layer), []))


def matrix_v2():
    with open(DATASETS / "set_a" / "tn5-matrix" / "matrix-v2.csv", newline="") as f:
        return list(csv.DictReader(f))


def agent_file(a):
    """Debate agent id -> matrix model id (stats.load_debate: a.replace('/', '_').replace(':', '_'))."""
    return a.replace("/", "_").replace(":", "_")


# ---------------- helpers (verbatim tech-note-5 stats.py) ----------------
def present(r):
    return bool(r) and (TRUNC_AS_WRONG or r.get("correct") is not None)


def cval(r):
    if not present(r): return None
    return bool(r.get("correct"))


def canon(a):
    if a in (None, ""): return a
    t = str(a).strip().replace("$", "").replace(" ", "")
    t = re.sub(r"^\\\(|\\\)$", "", t); t = re.sub(r"(?<=\d),(?=\d{3})", "", t).rstrip(".")
    try: return repr(float(f"{float(t):.7g}"))
    except ValueError: pass
    t = t.replace("\\dfrac", "\\frac"); t = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"\1/\2", t)
    t = t.replace("\\sqrt{", "√").replace("\\sqrt", "√").replace("{", "").replace("}", "").lower()
    return t


def phi(a, b):
    n = len(a); p1 = sum(a) / n; p2 = sum(b) / n
    if p1 in (0, 1) or p2 in (0, 1): return None
    p12 = sum(x and y for x, y in zip(a, b)) / n
    return (p12 - p1 * p2) / math.sqrt(p1 * (1 - p1) * p2 * (1 - p2))


def tetra(a, b):
    n11 = sum(x and y for x, y in zip(a, b)); n00 = sum((not x) and (not y) for x, y in zip(a, b))
    n10 = sum(x and not y for x, y in zip(a, b)); n01 = sum((not x) and y for x, y in zip(a, b))
    if n10 * n01 == 0 or n11 * n00 == 0: return None
    return math.cos(math.pi / (1 + math.sqrt(n11 * n00 / (n10 * n01))))


def n_eff(n, ph):
    return n / (1 + (n - 1) * ph) if ph is not None else None


def icc1(mat):
    n = len(mat); k = len(mat[0])
    gm = sum(sum(r) for r in mat) / (n * k)
    msb = sum(k * (sum(r) / k - gm) ** 2 for r in mat) / (n - 1)
    msw = sum((x - sum(r) / k) ** 2 for r in mat for x in r) / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw) if (msb + (k - 1) * msw) else None


def pairwise_phi(cols):
    ph = []
    for a, b in itertools.combinations(list(cols), 2):
        x = [(u, v) for u, v in zip(cols[a], cols[b]) if u is not None and v is not None]
        if len(x) < 5: continue
        q = phi([u for u, _ in x], [v for _, v in x])
        if q is not None: ph.append(q)
    return (sum(ph) / len(ph) if ph else None), len(ph)


def pairwise_tetra(cols):
    te = []
    for a, b in itertools.combinations(list(cols), 2):
        x = [(u, v) for u, v in zip(cols[a], cols[b]) if u is not None and v is not None]
        if len(x) < 5: continue
        t = tetra([u for u, _ in x], [v for _, v in x])
        if t is not None: te.append(t)
    return (sum(te) / len(te) if te else None), len(te)


def majority(answers):
    """(top key, count) of the plurality, ties to the first answer seen. The key is a canon string (v8) or a paper-1
    group id (p1); compare it only with keys of the SAME vote (vote_keys)."""
    if AID.RULE == "v8":
        c = collections.Counter(canon(a) for a in answers if a not in (None, ""))
        if not c: return None, 0
        top, cnt = c.most_common(1)[0]
        return top, cnt
    top, cnt, _, _ = AID.vote(answers)
    return top, cnt


def vote_keys(answers):
    """Identity key per answer of one vote, consistent with majority(answers) on the same list."""
    return AID.keys(answers, canon)


def maj_correct(rs):
    rs = [r for r in rs if present(r)]
    if AID.RULE == "v8":
        top, _ = majority([r.get("answer") for r in rs])
        if top is None: return False
        return bool(next((r for r in rs if canon(r.get("answer")) == top), {}).get("correct"))
    top, _, ok, _ = AID.vote([r.get("answer") for r in rs], [r.get("correct") for r in rs])
    return bool(ok) if top is not None else False


def split_correct(rows):
    """Plurality with ties split (1/k credit) over rows with an answer. v8: tech-note-5 (a key's correctness = its LAST row);
    p1: paper 1 voting.plurality(rule='split') (a group is correct if any member is)."""
    if AID.RULE == "v8":
        cnt = collections.Counter(canon(r.get("answer")) for r in rows if r.get("answer") not in (None, ""))
        if not cnt: return 0.0
        top = max(cnt.values()); tied = [k for k, v in cnt.items() if v == top]
        corr = {canon(r.get("answer")): bool(r.get("correct")) for r in rows}
        return sum(corr[k] for k in tied) / len(tied)
    rows = [r for r in rows if r.get("answer") not in (None, "")]
    k = AID.keys([r.get("answer") for r in rows])
    cnt = collections.Counter(k)
    if not cnt: return 0.0
    top = max(cnt.values()); tied = [x for x, v in cnt.items() if v == top]
    right = {x for x, r in zip(k, rows) if r.get("correct")}
    return sum(x in right for x in tied) / len(tied)


def wilson(k, n, z=1.96):
    if n == 0: return (0, 0)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def mcnemar(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, x) for x in range(k + 1)) / 2 ** n)


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]: j += 1
            for k in range(i, j + 1): r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    a, b = ranks(x), ranks(y); n = len(a); ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((u - ma) ** 2 for u in a)); sb = math.sqrt(sum((u - mb) ** 2 for u in b))
    return sum((u - ma) * (v - mb) for u, v in zip(a, b)) / (sa * sb) if sa and sb else float("nan")


def load_debate(layers=HARD):
    """item -> {(agent, round): row} with round 0 from the matrix files (stats.load_debate)."""
    items = collections.defaultdict(dict)
    for layer in layers:
        rs = debate(layer)
        if not rs: continue
        for r in rs: items[r["id"]][(r["agent"], r["round"])] = dict(r, layer=layer)
        for a in DEBATE_AGENTS:
            for r in matrix(agent_file(a), layer):
                if r["id"] in items: items[r["id"]][(a, 0)] = dict(r, layer=layer)
    return items


def load_homo(arm, layers=HARD):
    """item -> {(copy, round): row}; round 0 = DeepSeek T=1 draws 0-5 (stats.section_homodebate)."""
    items = collections.defaultdict(dict)
    for layer in layers:
        rs = homo(arm, layer)
        if not rs: continue
        for r in rs: items[r["id"]][(r["agent"], r["round"])] = r
        for r in samples("deepseek_deepseek-v3.2", layer):
            if r["id"] in items and r["sample"] < 6: items[r["id"]][(f"deepseek/deepseek-v3.2#{r['sample']}", 0)] = r
    return items
