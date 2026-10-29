"""Factor structure: Marchenko-Pastur edge, permutation null for top eigenvalues, 2PL / Rasch IRT (MML-EM, Gauss-Hermite,
), residual correlations, Q3 bias, family excess.
Paper 1, section 2 (eigenvalues, MP edge lambda+), section 5.2 (difficulty factor checks) and appendix C."""
import math

import numpy as np
from scipy.special import expit, log_expit

NODES, W = np.polynomial.hermite_e.hermegauss(41)
W = W / W.sum()
PRIOR_LOGA_SD, PRIOR_B_SD = 0.5, 2.0


def _mstep(X_is_r, n_k, alpha, delta, rasch, iters=8):
    """Internal M-step. Joint Newton on (log alpha_j, delta_j) with priors; r: items x K expected correct, n_k: K."""
    K = len(n_k)
    for _ in range(iters):
        la = np.log(alpha)
        P = expit(alpha[:, None] * NODES[None, :] + delta[:, None])          # J x K
        # log-lik gradient wrt logit per node: (r - n P)
        g_logit = X_is_r - n_k[None, :] * P                                   # J x K
        h_logit = -n_k[None, :] * P * (1 - P)
        # d logit / d la = alpha*theta ; d logit / d delta = 1
        ga = (g_logit * alpha[:, None] * NODES[None, :]).sum(1)
        gd = g_logit.sum(1)
        haa = (h_logit * (alpha[:, None] * NODES[None, :]) ** 2).sum(1) + ga  # second-order of exp
        hdd = h_logit.sum(1)
        had = (h_logit * alpha[:, None] * NODES[None, :]).sum(1)
        # priors: log a ~ N(0, s_a^2); b = -delta/alpha ~ N(0, s_b^2)
        b = -delta / alpha
        lp_b = -b / PRIOR_B_SD ** 2                                           # d/db log prior
        # db/dla = delta/alpha = -b ; db/ddelta = -1/alpha
        ga += -la / PRIOR_LOGA_SD ** 2 + lp_b * (-b)
        gd += lp_b * (-1 / alpha)
        haa += -1 / PRIOR_LOGA_SD ** 2 - (b * b) / PRIOR_B_SD ** 2 + lp_b * b
        hdd += -(1 / alpha ** 2) / PRIOR_B_SD ** 2
        had += -(b / alpha) / PRIOR_B_SD ** 2 + lp_b * (1 / alpha)
        if rasch:
            step_d = -gd / hdd
            delta = delta + np.clip(step_d, -2, 2)
            continue
        det = haa * hdd - had ** 2
        det = np.where(np.abs(det) < 1e-9, -1e-9, det)
        sa = -(hdd * ga - had * gd) / det
        sd = -(-had * ga + haa * gd) / det
        la = la + np.clip(sa, -1, 1)
        delta = delta + np.clip(sd, -2, 2)
        alpha = np.exp(np.clip(la, -4, 3))
    return alpha, delta


def fit(X, rasch=False, max_iter=200, tol=1e-4):
    """2PL logit P = a(theta - b) (Rasch: a = 1), MML-EM over 41 Gauss-Hermite nodes, priors log a ~ N(0, 0.5^2), b ~ N(0, 2^2) (appendix C).
    X: persons x items (0/1). Returns dict with alpha, delta, a, b, theta (EAP), posterior, loglik."""
    X = X.astype(float)
    n, J = X.shape
    alpha = np.ones(J)
    p = X.mean(0).clip(0.02, 0.98)
    delta = np.log(p / (1 - p))
    prev = -np.inf
    for it in range(max_iter):
        logit = alpha[:, None] * NODES[None, :] + delta[:, None]              # J x K
        lp1, lp0 = log_expit(logit), log_expit(-logit)
        LL = X @ lp1 + (1 - X) @ lp0                                          # n x K
        LL += np.log(W)[None, :]
        m = LL.max(1, keepdims=True)
        post = np.exp(LL - m); marg = post.sum(1, keepdims=True); post /= marg
        loglik = float((m[:, 0] + np.log(marg[:, 0])).sum())
        r = X.T @ post                                                        # J x K
        n_k = post.sum(0)
        alpha, delta = _mstep(r, n_k, alpha, delta, rasch)
        if abs(loglik - prev) < tol:
            break
        prev = loglik
    theta = post @ NODES
    a = alpha; b = -delta / alpha
    P = expit(np.outer(theta, a) - (a * b)[None, :])
    return dict(a=a, b=b, alpha=alpha, delta=delta, theta=theta, P=P, loglik=loglik, iters=it + 1)


def eap_theta(X, alpha, delta):
    """EAP ability and marginal log-likelihood per person for fixed item parameters."""
    X = X.astype(float)
    logit = alpha[:, None] * NODES[None, :] + delta[:, None]
    LL = X @ log_expit(logit) + (1 - X) @ log_expit(-logit) + np.log(W)[None, :]
    m = LL.max(1, keepdims=True); post = np.exp(LL - m); marg = post.sum(1, keepdims=True)
    return (post / marg) @ NODES, (m[:, 0] + np.log(marg[:, 0]))


def lz(X, P):
    """Person-fit l_z statistic (standardised log-likelihood) per member."""
    X = X.astype(float)
    l0 = (X * np.log(P) + (1 - X) * np.log(1 - P)).sum(1)
    E = (P * np.log(P) + (1 - P) * np.log(1 - P)).sum(1)
    V = (P * (1 - P) * np.log(P / (1 - P)) ** 2).sum(1)
    return (l0 - E) / np.sqrt(V)


def icc_by_family(E, fam_idx, G):
    """Family ICC (Section 5.2, Appendix C). Pooled one-way ANOVA ICC over items. E: persons x items residuals."""
    n = E.shape[0]
    counts = np.bincount(fam_idx, minlength=G)
    SSB = SSW = 0.0
    for j in range(E.shape[1]):
        e = E[:, j]; gm = e.mean()
        sums = np.bincount(fam_idx, weights=e, minlength=G)
        means = np.where(counts > 0, sums / np.maximum(counts, 1), 0)
        SSB += (counts * (means - gm) ** 2).sum()
        SSW += ((e - means[fam_idx]) ** 2).sum()
    Gn = (counts > 0).sum(); J = E.shape[1]
    MSB = SSB / (J * (Gn - 1)); MSW = SSW / (J * (n - Gn))
    n0 = (n - (counts ** 2).sum() / n) / (Gn - 1)
    return (MSB - MSW) / (MSB + (n0 - 1) * MSW)


def family_stats(E, fam, rng, n_perm=2000):
    """Family excess (section 5.2): R = corrcoef(E) of residuals; T = mean r within same-family pairs - mean r between;
    permutation p = (#{T_perm >= T} + 1)/(n_perm + 1) over permutations of family labels (rng: numpy Generator; ICC
    permutations interleaved on the same stream, as in the original analysis)."""
    n = E.shape[0]
    R = np.corrcoef(E)
    off = ~np.eye(n, dtype=bool)
    fams = sorted(set(fam)); fidx = np.array([fams.index(f) for f in fam]); G = len(fams)
    same = (fidx[:, None] == fidx[None, :]) & off
    def T(fi):
        s = (fi[:, None] == fi[None, :]) & off
        return R[s].mean() - R[~s & off].mean()
    obs = T(fidx); obs_icc = icc_by_family(E, fidx, G)
    perm_T, perm_I = [], []
    for _ in range(n_perm):
        p = rng.permutation(fidx); perm_T.append(T(p)); perm_I.append(icc_by_family(E, p, G))
    perm_T, perm_I = np.array(perm_T), np.array(perm_I)
    return dict(T=obs, T_within=R[same].mean(), T_between=R[~same & off].mean(),
                p_T=float((np.sum(perm_T >= obs) + 1) / (n_perm + 1)), null95_T=float(np.percentile(perm_T, 95)),
                ICC=float(obs_icc), p_ICC=float((np.sum(perm_I >= obs_icc) + 1) / (n_perm + 1)),
                null95_ICC=float(np.percentile(perm_I, 95)), n_families=G)


def mp_edge(n, N):
    """Marchenko-Pastur upper edge lambda+ = (1 + sqrt(n/N))^2 for an n x n correlation matrix from N items (section 2)."""
    return (1 + math.sqrt(n / N)) ** 2


def eigen_check(X, rng, n_perm=500, q=95):
    """Eigen check: drop constant rows; eigenvalues (desc) of corrcoef(X) (X: members x items), MP edge,
    and the q-th percentile of the max eigenvalue under a null that permutes every row independently (rng.permutation per
    row, numpy Generator, n_perm draws). Returns (ev, edge, null_q, n_above = #(ev > max(edge, null_q)))."""
    X = np.asarray(X)[np.asarray(X).std(1) > 0]
    n, N = X.shape
    ev = np.sort(np.linalg.eigvalsh(np.corrcoef(X)))[::-1]
    edge = mp_edge(n, N)
    null = [np.linalg.eigvalsh(np.corrcoef(np.array([rng.permutation(r) for r in X]))).max() for _ in range(n_perm)]
    pq = float(np.percentile(null, q))
    return ev, edge, pq, int((ev > max(edge, pq)).sum())


def q3_bias(n):
    """Expected mean residual correlation under local independence: -1/(n-1) (Yen's Q3 bias; section 5.1)."""
    if n < 2:
        raise ValueError(f"q3_bias: need n >= 2, got {n}")
    return -1.0 / (n - 1)


def residual_correlation(X, rasch=False):
    """Fit 2PL (or Rasch) to X (persons=members x items), residuals E = X - P, R = corrcoef(E); returns
    (mean off-diagonal r, excess = mean r - q3_bias(n), E). Section 5.2 / appendix C."""
    f = fit(X, rasch=rasch)
    E = X - f["P"]
    mr = mean_offdiag_corr(E)
    return mr, mr - q3_bias(E.shape[0]), E


def mean_offdiag_corr(E):
    """Mean of the off-diagonal Pearson correlations between the rows of E (members x items), undefined (nan) pairs
    dropped: the mean residual correlation r-bar of section 5.2 / appendix C when E = X - P-hat (equals phi-bar on 0/1 rows)."""
    E = np.asarray(E, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        R = np.corrcoef(E)
    off = ~np.eye(E.shape[0], dtype=bool)
    return float(np.nanmean(R[off]))


def drop_constant_items(X):
    """Keep items solved by some but not all members (convention of the original analysis)."""
    X = np.asarray(X)
    keep = (X.sum(0) > 0) & (X.sum(0) < X.shape[0])
    return X[:, keep], keep


def cv_persons(X, rng, k=5):
    """Model choice 2PL vs Rasch by k-fold cross-validation over persons (members), as in the IRT scripts of the series:
    persons permuted once by rng.permutation(n) and split by np.array_split; per fold fit on the rest, EAP theta of the
    held-out persons with the fitted items, then per-response held-out log-likelihood at the EAP theta ('eap_per_resp')
    and the marginal log-likelihood ('marg_per_resp'). Returns {'2pl': {...}, 'rasch': {...}}."""
    X = np.asarray(X)
    n = X.shape[0]; idx = rng.permutation(n); folds = np.array_split(idx, k)
    out = {}
    for rasch in (False, True):
        eap_ll, marg_ll, cnt = 0.0, 0.0, 0
        for f in folds:
            tr = np.setdiff1d(idx, f)
            fit_ = fit(X[tr], rasch=rasch)
            th, lm = eap_theta(X[f], fit_["alpha"], fit_["delta"])
            P = expit(np.outer(th, fit_["a"]) - (fit_["a"] * fit_["b"])[None, :])
            Xf = X[f].astype(float)
            eap_ll += (Xf * np.log(P) + (1 - Xf) * np.log(1 - P)).sum(); marg_ll += lm.sum(); cnt += Xf.size
        out["rasch" if rasch else "2pl"] = dict(eap_per_resp=eap_ll / cnt, marg_per_resp=marg_ll / cnt)
    return out
