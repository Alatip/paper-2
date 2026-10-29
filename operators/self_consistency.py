"""Self-consistency: plurality of k temperature-1 draws of one model vs its mean single draw (paper v8 §5.7, tables A' and A'2
of tech-note-5 results/tables.md).

Method (port of tech-note-5 stats.section_samples and stats.section_samples_deep, line by line):
- Per model (DeepSeek V3.2, Claude Haiku 4.5, GLM 5.3 Flash) and corpus layer, k = 16 if at least 10 items have all of draws
  0..15 answered (not truncated), else 8. "Full" items = items whose draws 0..k-1 are all answered.
- p = mean draw accuracy on full items; MV@k = plurality by answer identity (paper-1 rule by default, p2parts/answer_id.py; v8: canon, 7 significant digits) of draws 0..k-1,
  ties to the lowest draw index; pass@k; all-wrong; within-model φ = mean pairwise φ over draw pairs (pairs with a constant
  draw dropped); n_eff = k / (1 + (k-1) φ) (Kish).
- "all items" column: every item of the layer, the first k draws by index, a truncated draw a missing vote.
- A'2: ICC(1), mean tetrachoric ρ (cos-π) over draw pairs, MV@1/4/8/k, Kish prediction = exact binomial majority (ties split)
  of n_eff independent votes at p, interpolated between floor and floor+1 of n_eff; pass@4 and pass@k.
- untraced.* (new, documented functions; v8 §5.7 numbers that came from a hand analysis): DeepSeek olympiad on the 49 full
  items: accuracy of the temperature-0 matrix answer, accuracy of each draw position 0..15, and MV@16 minus each of them.
Decisions: truncated rows are missing votes (v8 policy); label-audit wave-1 exclusions are already dropped from the dataset.
Output keys: haiku / deepseek / glm (model ids contain dots); each row also carries its model id.
Data: datasets/set_a/tn5-samples, datasets/set_a/tn5-matrix (T=0 answers). Offline, $0.
"""
import collections, itertools, math
from p2parts import tn5 as IO

SHORT = {"claude_haiku": "haiku", "deepseek_deepseek-v3.2": "deepseek", "z-ai_glm-5.3-flash": "glm"}   # output keys (no dots)


def _by(rows):
    by = collections.defaultdict(dict); allby = collections.defaultdict(dict)
    for r in rows:
        allby[r["id"]][r["sample"]] = r
        if r.get("correct") is not None: by[r["id"]][r["sample"]] = r
    return by, allby


def table_a1(model, layer):
    """Table A' row (stats.section_samples)."""
    rows = IO.samples(model, layer)
    if not rows: return None
    by, allby = _by(rows)
    k = 16 if sum(all(j in v for j in range(16)) for v in by.values()) >= 10 else 8
    full = {i: {j: v[j] for j in range(k)} for i, v in by.items() if all(j in v for j in range(k))}
    if len(full) < 10: return None
    canon, majority = IO.canon, IO.majority
    p = sum(bool(r["correct"]) for v in full.values() for r in v.values()) / (k * len(full))
    if IO.AID.RULE == "v8":
        mv = sum(bool(next(r for r in v.values() if canon(r.get("answer")) == majority([r.get("answer") for r in v.values()])[0])["correct"]) for v in full.values()) / len(full)
    else:
        mv = sum(IO.maj_correct(list(v.values())) for v in full.values()) / len(full)
    passk = sum(any(r["correct"] for r in v.values()) for v in full.values()) / len(full)
    allw = sum(not any(r["correct"] for r in v.values()) for v in full.values()) / len(full)
    phis = []
    for s1, s2 in itertools.combinations(range(k), 2):
        a = [bool(v[s1]["correct"]) for v in full.values()]; b = [bool(v[s2]["correct"]) for v in full.values()]
        ph = IO.phi(a, b)
        if ph is not None: phis.append(ph)
    mp = sum(phis) / len(phis) if phis else None
    amv = apk = 0
    for i, v in allby.items():
        rs_ = [v[s] for s in range(k) if s in v]
        amv += IO.maj_correct(rs_); apk += any(bool(r.get("correct")) for r in rs_)
    return {"items": len(full), "k": k, "p": p, "mv": mv, "passk": passk, "all_wrong": allw, "phi": mp, "n_eff": IO.n_eff(k, mp),
            "mv_minus_p": mv - p, "all_n": len(allby), "all_mv": amv / len(allby), "all_passk": apk / len(allby),
            "_full_ids": sorted(full)}


def _pmaj_exact(n, p):
    return sum(math.comb(n, c) * p ** c * (1 - p) ** (n - c) * (1.0 if 2 * c > n else 0.5 if 2 * c == n else 0.0) for c in range(n + 1))


def kish_majority(n, p):
    """P(majority correct) of n (real) independent votes at accuracy p: exact binomial with ties split, linear in n between
    floor(n) and floor(n)+1 (stats.section_samples_deep)."""
    lo = math.floor(n); return _pmaj_exact(lo, p) + (n - lo) * (_pmaj_exact(lo + 1, p) - _pmaj_exact(lo, p))


def table_a2(model, layer):
    """Table A'2 row (stats.section_samples_deep)."""
    rows = IO.samples(model, layer)
    if not rows: return None
    by = collections.defaultdict(dict)
    for r in rows:
        if r.get("correct") is not None: by[r["id"]][r["sample"]] = r
    k = 16 if sum(len(v) >= 16 for v in by.values()) >= 10 else 8
    full = {i: v for i, v in by.items() if all(s in v for s in range(k))}
    if len(full) < 10: return None
    mat = [[bool(v[s]["correct"]) for s in range(k)] for v in full.values()]
    te = [t for s1, s2 in itertools.combinations(range(k), 2) if (t := IO.tetra([m[s1] for m in mat], [m[s2] for m in mat])) is not None]

    def mvk(kk):
        c = 0
        for v in full.values():
            if IO.AID.RULE == "v8":
                top, _ = IO.majority([v[s].get("answer") for s in range(kk)])
                c += bool(next(v[s] for s in range(kk) if IO.canon(v[s].get("answer")) == top).get("correct"))
            else:
                c += IO.maj_correct([v[s] for s in range(kk)])
        return c / len(full)
    p = sum(sum(m) for m in mat) / (k * len(full))
    ph = [q for s1, s2 in itertools.combinations(range(k), 2) if (q := IO.phi([m[s1] for m in mat], [m[s2] for m in mat])) is not None]
    ne = IO.n_eff(k, sum(ph) / len(ph)) if ph else None
    kish = kish_majority(ne, p) if ne else None
    pk = lambda kk: sum(any(m[:kk]) for m in mat) / len(mat)
    mk = mvk(k)
    return {"items": len(full), "k": k, "icc1": IO.icc1(mat), "tetra": (sum(te) / len(te) if te else None),
            "mv1": mvk(1), "mv4": mvk(4), "mv8": mvk(8), "mvk": mk, "kish": kish, "mvk_minus_kish": mk - kish,
            "pass4": pk(4), "passk": pk(k), "p": p}


def untraced_olympiad_deepseek():
    """NEW (v8 §5.7 hand numbers). DeepSeek V3.2, olympiad, the items with all 16 draws answered (table A' 'full' set):
    t0_acc = accuracy of the temperature-0 matrix answer on these items (a truncated or missing T=0 answer scores 0; also
    given on the answered T=0 rows); draw_acc[s] = accuracy of draw s; MV@16 minus t0_acc and minus draw_acc[0]."""
    m, L = "deepseek_deepseek-v3.2", "olympiad"
    a = table_a1(m, L); ids = a["_full_ids"]
    by, _ = _by(IO.samples(m, L))
    t0 = {}
    if IO.MATRIX_SOURCE == "corpus":                     # paper 1's uncapped matrix (default); items not in the corpus (label
        X = IO.corpus_matrix(L)                          # audit exclusions) are dropped from this comparison
        missing = [i for i in ids if i not in X]
        ids = [i for i in ids if i in X]
        for i in ids: t0[i] = X[i][m]
    else:
        missing = []
        for r in IO.matrix(m, L): t0[r["id"]] = r        # last row wins, as everywhere in stats.py
    t0_acc = sum(bool((t0.get(i) or {}).get("correct")) for i in ids) / len(ids)
    ans = [i for i in ids if IO.present(t0.get(i))]
    t0_ans = sum(bool(t0[i]["correct"]) for i in ans) / len(ans)
    draw = [sum(bool(by[i][s]["correct"]) for i in ids) / len(ids) for s in range(16)]
    if missing:   # MV@16 and mean draw on the same reduced item set
        mv = sum(IO.maj_correct([by[i][s] for s in range(16)]) for i in ids) / len(ids)
        a = dict(a, mv=mv, p=sum(draw) / 16)
    return {"items": len(ids), "items_dropped_not_in_corpus": missing, "matrix_source": IO.MATRIX_SOURCE, "mv16": a["mv"], "mean_draw": a["p"], "t0_acc": t0_acc, "t0_answered": len(ans), "t0_acc_answered": t0_ans,
            "draw_acc": draw, "draw_acc_min": min(draw), "draw_acc_max": max(draw), "draw0_acc": draw[0],
            "gain_vs_t0": a["mv"] - t0_acc, "gain_vs_draw0": a["mv"] - draw[0], "gain_vs_draw0_items": round((a["mv"] - draw[0]) * len(ids)),
            "gain_vs_mean": a["mv"] - a["p"]}


def run() -> dict:
    A1 = {}; A2 = {}
    for m in IO.SAMPLE_MODELS:
        for L in IO.HARD:
            r1 = table_a1(m, L); r2 = table_a2(m, L)
            if r1: r1.pop("_full_ids"); A1.setdefault(SHORT[m], {})[L] = dict(r1, model=m)
            if r2: A2.setdefault(SHORT[m], {})[L] = dict(r2, model=m)
    hd = [(m, L) for m in ("haiku", "deepseek") for L in IO.HARD]
    summ = {
        "phi_min_haiku_deepseek": min(A1[m][L]["phi"] for m, L in hd), "phi_max_haiku_deepseek": max(A1[m][L]["phi"] for m, L in hd),
        "neff_min_haiku_deepseek": min(A1[m][L]["n_eff"] for m, L in hd), "neff_max_haiku_deepseek": max(A1[m][L]["n_eff"] for m, L in hd),
        "mv_minus_p_mmlu_gsm_min": min(A1[m][L]["mv_minus_p"] for m in SHORT.values() for L in ("mmlu-pro", "gsm-hard")),
        "mv_minus_p_mmlu_gsm_max": max(A1[m][L]["mv_minus_p"] for m in SHORT.values() for L in ("mmlu-pro", "gsm-hard")),
        "rows_mv_above_kish_haiku_deepseek": sum(A2[m][L]["mvk"] > A2[m][L]["kish"] + 1e-12 for m, L in hd),
        "rows_haiku_deepseek": len(hd),
        "max_mv_minus_kish_haiku_deepseek": max(A2[m][L]["mvk_minus_kish"] for m, L in hd),
        "glm_items_phi1_min": min(A1["glm"][L]["items"] for L in ("mmlu-pro", "gpqa-diamond")),
        "glm_items_phi1_max": max(A1["glm"][L]["items"] for L in ("mmlu-pro", "gpqa-diamond")),
    }
    return {"self_consistency": {"A1": A1, "A2": A2, "summary": summ, "untraced": {"olympiad_deepseek": untraced_olympiad_deepseek()}}}


if __name__ == "__main__":
    import json; print(json.dumps(run(), indent=1)[:4000])
