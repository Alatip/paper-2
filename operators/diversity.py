"""Sources of diversity: model family (triples of different models) and role prompts (personas) vs plain resampling
(paper v8 §5.8; tables A2 and D of tech-note-5 results/tables.md). Temperature (§5.8 third paragraph) is self_consistency.py.

Family (port of stats.section_matrix_deep): on the tech-note-5 matrix (T=0 answers of 7 models, items all seven answered,
matrix-v2.csv correctness), per layer: mean tetrachoric ρ (cos-π) over model pairs and its range, 95% quantile of the
wrong-count, observed and independent all-wrong; for every 3-subset of the 7 models with φ̄ defined (no constant member):
gain = plurality by answer identity of the three (ties to the first in the fixed MAIN order) minus the triple's best member;
best / worst / mean gain and Spearman(gain, φ̄) with tie-averaged ranks.

Matrix source (2026-10-18): default = paper 1's uncapped matrix (IO.MATRIX_SOURCE "corpus", all 941 items, cap hits wrong);
P2_MATRIX=tn5 = the tech-note-5 matrix capped at 8,192 with the 'all seven answered' filter (v8).

Personas (port of stats.section_personas): DeepSeek V3.2 under four role prompts (engineer, formal, skeptic, teacher), draws
0..3, vs the plain model's draws 0..3, on the items where every prompt has 4 answered draws. φ within plain = mean φ over plain
draw pairs; within a persona = mean over the four prompts of their within-draw φ; across = mean φ over pairs of different
personas and all 16 draw pairs; n_eff of 4 personas = Kish at the across φ; MV plain@4 and MV of the 4 personas (one draw per
persona, draw index s, averaged over s). Interval: percentile bootstrap over items of across − within(plain), 2,000
resamples, random.Random(0), resamples with an undefined side dropped and counted (index rule int(0.025 n), int(0.975 n) - 1).
Data: datasets/set_a/tn5-matrix (rows + matrix-v2.csv), datasets/set_a/tn5-samples, datasets/set_a/tn5-personas. Offline, $0.
"""
import collections, itertools, random
from p2parts import tn5 as IO


def _matrix_ids_rows(layer):
    """stats.matrix_ids / raw_matrix: item -> {model: row} for items answered (non-truncated, non-empty) by all 7."""
    rows = collections.defaultdict(dict)
    for m in IO.MAIN:
        rs = IO.matrix(m, layer)
        if not rs: return {}
        for r in rs:
            if r.get("correct") is not None and r.get("answer"): rows[r["id"]][m] = r
    return {i: d for i, d in rows.items() if len(d) == len(IO.MAIN)}


def _corpus_family(layers):
    """Same statistics on paper 1's uncapped matrix (IO.MATRIX_SOURCE == "corpus", default): every corpus item of the layer
    (553 / 164 / 141 / 83), every row graded (a 65,536-cap hit is a wrong vote, as in paper 1), correctness and answers from
    datasets/set_a/corpus-<layer>/answers.csv (label audits as in paper 1). No 'all seven answered' filter is needed."""
    out = {}
    models = list(IO.MAIN)
    for layer in layers:
        X = IO.corpus_matrix(layer); ids = sorted(X)
        cols = {m: [bool(X[i][m]["correct"]) for i in ids] for m in models}
        te = [t for a, b in itertools.combinations(models, 2) if (t := IO.tetra(cols[a], cols[b])) is not None]
        wc = sorted(sum(not cols[m][j] for m in models) for j in range(len(ids)))
        q95 = wc[int(0.95 * (len(wc) - 1))]
        allw = sum(1 for j in range(len(ids)) if all(not cols[m][j] for m in models)) / len(ids)
        ind = 1
        for m in models: ind *= 1 - sum(cols[m]) / len(ids)
        gains = []; phis = []
        for sub in itertools.combinations(models, 3):
            acc = [sum(cols[m]) / len(ids) for m in sub]
            ph = [p for a, b in itertools.combinations(sub, 2) if (p := IO.phi(cols[a], cols[b])) is not None]
            if not ph: continue
            mv = sum(IO.maj_correct([X[i][m] for m in sub]) for i in ids) / len(ids)
            gains.append(mv - max(acc)); phis.append(sum(ph) / len(ph))
        n = len(gains)
        out[layer] = {"n": len(ids), "tetra_mean": sum(te) / len(te), "tetra_min": min(te), "tetra_max": max(te), "wrong_q95": q95,
                      "all_wrong": allw, "all_wrong_indep": ind, "gain_best": max(gains), "gain_worst": min(gains),
                      "gain_mean": sum(gains) / n, "mean_below_best": -sum(gains) / n,
                      "triples": n, "spearman": IO.spearman(gains, phis) if n > 2 else float("nan"), "source": "corpus"}
    return out


def family_triples(layers=IO.HARD):
    if IO.MATRIX_SOURCE == "corpus": return _corpus_family(layers)
    raws = {L: _matrix_ids_rows(L) for L in layers}
    rows = [r for r in IO.matrix_v2() if (r["dataset"] not in raws or r["item_id"] in raws[r["dataset"]])]
    models = [m for m in IO.MAIN if m in rows[0]]
    out = {}
    for layer in layers:
        rs = [r for r in rows if r["dataset"] == layer and all(r.get(m) not in ("", None) for m in models)]
        if len(rs) < 20: continue
        cols = {m: [bool(int(r[m])) for r in rs] for m in models}
        te = [t for a, b in itertools.combinations(models, 2) if (t := IO.tetra(cols[a], cols[b])) is not None]
        wc = sorted(sum(not cols[m][j] for m in models) for j in range(len(rs)))
        q95 = wc[int(0.95 * (len(wc) - 1))]
        allw = sum(1 for j in range(len(rs)) if all(not cols[m][j] for m in models)) / len(rs)
        ind = 1
        for m in models: ind *= 1 - sum(cols[m]) / len(rs)
        raw = raws[layer]; ids = [r["item_id"] for r in rs if r["item_id"] in raw]
        gains = []; phis = []
        for sub in itertools.combinations(models, 3):
            acc = [sum(cols[m]) / len(rs) for m in sub]
            ph = [p for a, b in itertools.combinations(sub, 2) if (p := IO.phi(cols[a], cols[b])) is not None]
            if not ph: continue
            mv = sum(IO.maj_correct([raw[i][m] for m in sub]) for i in ids) / len(ids)
            gains.append(mv - max(acc)); phis.append(sum(ph) / len(ph))
        n = len(gains)
        out[layer] = {"n": len(rs), "tetra_mean": sum(te) / len(te), "tetra_min": min(te), "tetra_max": max(te), "wrong_q95": q95,
                      "all_wrong": allw, "all_wrong_indep": ind, "gain_best": max(gains), "gain_worst": min(gains),
                      "gain_mean": sum(gains) / n, "mean_below_best": -sum(gains) / n,   # v8 prints "X points below"
                      "triples": n, "spearman": IO.spearman(gains, phis) if n > 2 else float("nan")}
    return out


def personas(layers=IO.HARD, n_boot=2000, seed=0):
    base = "deepseek_deepseek-v3.2"; pers = IO.persona_models()
    out = {}
    for layer in layers:
        data = {}; rows_ = {}
        for name, rs in [(base, IO.samples(base, layer))] + [(p, IO.persona_samples(p, layer)) for p in pers]:
            if not rs: continue
            by = collections.defaultdict(dict); byr = collections.defaultdict(dict)
            for r in rs:
                if r.get("correct") is not None and r["sample"] < 4: by[r["id"]][r["sample"]] = bool(r["correct"]); byr[r["id"]][r["sample"]] = r
            data[name] = {i: v for i, v in by.items() if len(v) == 4}; rows_[name] = byr
        names = [n for n in data if "persona" in n]
        if len(names) < 2 or base not in data: continue
        ids = set(data[base])
        for n in names: ids &= set(data[n])
        ids = sorted(ids)
        if len(ids) < 10: continue

        def mphi(cols):
            ph = [p for a, b in itertools.combinations(cols, 2) if (p := IO.phi(a, b)) is not None]
            return sum(ph) / len(ph) if ph else None
        plain = [[data[base][i][s] for i in ids] for s in range(4)]
        within_p = [mphi([[data[n][i][s] for i in ids] for s in range(4)]) for n in names]
        within_p = [x for x in within_p if x is not None]

        def across_phi(sel):
            ph = []
            for n1, n2 in itertools.combinations(names, 2):
                for s1 in range(4):
                    for s2 in range(4):
                        q = IO.phi([data[n1][i][s1] for i in sel], [data[n2][i][s2] for i in sel])
                        if q is not None: ph.append(q)
            return sum(ph) / len(ph) if ph else None

        def within_phi(sel):
            ph = [q for s1, s2 in itertools.combinations(range(4), 2) if (q := IO.phi([data[base][i][s1] for i in sel], [data[base][i][s2] for i in sel])) is not None]
            return sum(ph) / len(ph) if ph else None
        across = across_phi(ids)
        p_plain = sum(sum(c) for c in plain) / (4 * len(ids))
        p_pers = sum(data[n][i][s] for n in names for i in ids for s in range(4)) / (4 * len(names) * len(ids))
        mv_plain = sum(IO.maj_correct([rows_[base][i][s] for s in range(4)]) for i in ids) / len(ids)
        mv_pers = sum(IO.maj_correct([rows_[n][i][s] for n in names]) for i in ids for s in range(4)) / (4 * len(ids))
        wp = within_phi(ids); wpp = sum(within_p) / len(within_p) if within_p else None
        rng = random.Random(seed); diffs = []; undefined = 0
        for _ in range(n_boot):
            sel = [rng.choice(ids) for _ in ids]
            a_, w_ = across_phi(sel), within_phi(sel)
            if a_ is None or w_ is None: undefined += 1; continue
            diffs.append(a_ - w_)
        diffs.sort(); lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs)) - 1]
        out[layer] = {"items": len(ids), "p_plain": p_plain, "p_personas": p_pers, "phi_within_plain": wp, "phi_within_persona": wpp,
                      "phi_across": across, "n_eff_personas": IO.n_eff(len(names), across), "mv_plain4": mv_plain, "mv_personas": mv_pers,
                      "diff": across - wp, "diff_lo": lo, "diff_hi": hi, "undefined": undefined}
    if out:
        vals = list(out.values())
        out["_summary"] = {"n_eff_min": min(v["n_eff_personas"] for v in vals), "n_eff_max": max(v["n_eff_personas"] for v in vals),
                           "items_min": min(v["items"] for v in vals), "items_max": max(v["items"] for v in vals),
                           # v8 "intervals up to ±0.35 in φ": the largest half-width of the bootstrap intervals
                           "max_half_width": max((v["diff_hi"] - v["diff_lo"]) / 2 for v in vals)}
    return out


def run() -> dict:
    return {"diversity": {"family": family_triples(), "personas": personas()}}


if __name__ == "__main__":
    import json; print(json.dumps(run(), indent=1))
