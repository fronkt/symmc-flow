"""N3 statistics (protocol §2.4): exact paired sign-flip test, Holm, BCa CI, two-stage seed bootstrap, Wilson.

Unit of analysis is the crystal. For arm A with per-crystal, per-seed outcomes h[i, s] in [0, 1] (a
deterministic pick gives 0/1, a RANDOM selector its expectation k/16), h_i = mean_s h[i, s];
d_i = h_i^OURS - h_i^A and Delta = mean_i d_i. Every d_i is a multiple of 1/48 (seed means of k/16 over
3 seeds), so the sign-flip null distribution of sum_i d_i is an exact integer convolution.
"""
import numpy as np

DENOM = 48  # common denominator of 1/3 (seed means) and 1/16 (RANDOM expectations)


def to_units(d, denom=DENOM):
    u = np.asarray(d, float) * denom
    ui = np.rint(u).astype(np.int64)
    if np.max(np.abs(u - ui), initial=0.0) > 1e-6:
        raise ValueError("d_i are not multiples of 1/%d" % denom)
    return ui


def signflip_p(d, denom=DENOM):
    """Exact two-sided paired sign-flip p-value for H0: sum d_i symmetric about 0. Zeros drop out.
    p = P(|T*| >= |T_obs|) under independent random signs. Equals exact McNemar for binary data."""
    u = to_units(d, denom)
    u = np.abs(u[u != 0])
    if u.size == 0:
        return 1.0
    t_obs = abs(int(to_units(d, denom).sum()))
    g = int(np.gcd.reduce(u))                   # exact rescaling: every attainable sum is a multiple of g
    u = u // g
    t_obs = t_obs / g
    total = int(u.sum())
    # distribution of sum of +-u_j with prob 1/2 each, over support [-total, total]; the c elements of
    # equal magnitude v contribute v * (2B - c) with B ~ Binomial(c, 1/2), i.e. one dilated binomial kernel
    from scipy.stats import binom
    dist = np.ones(1)
    for v, cnt in zip(*np.unique(u, return_counts=True)):
        v, cnt = int(v), int(cnt)
        ker = np.zeros(2 * v * cnt + 1)
        ker[::2 * v] = binom.pmf(np.arange(cnt + 1), cnt, 0.5)
        dist = np.convolve(dist, ker)
    support = np.arange(-total, total + 1)
    return float(min(1.0, dist[np.abs(support) >= t_obs - 1e-9].sum()))


def _signflip_p_slow(d, denom=DENOM):
    """Reference implementation (one +-v step per element); used only to test signflip_p."""
    u = to_units(d, denom)
    t_obs = abs(int(u.sum()))
    u = np.abs(u[u != 0])
    if u.size == 0:
        return 1.0
    total = int(u.sum())
    dist = np.zeros(2 * total + 1)
    dist[total] = 1.0
    for v in u:
        dist = 0.5 * (np.roll(dist, int(v)) + np.roll(dist, -int(v)))
    support = np.arange(-total, total + 1)
    return float(min(1.0, dist[np.abs(support) >= t_obs - 1e-9].sum()))


def holm(pvals):
    """Holm step-down adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj


def bca_ci(d, n_boot=10000, seed=0, alpha=0.05):
    """Paired crystal-bootstrap BCa CI for mean(d) (d = per-crystal differences)."""
    from scipy.stats import norm
    d = np.asarray(d, float)
    n = len(d)
    theta = d.mean()
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, n, size=(n_boot, n))].mean(1)
    prop = np.clip((boots < theta).mean() + 0.5 * (boots == theta).mean(), 1e-6, 1 - 1e-6)
    z0 = norm.ppf(prop)
    jack = (d.sum() - d) / (n - 1)
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum()) ** 1.5
    acc = num / den if den > 0 else 0.0
    lo_hi = []
    for q in (alpha / 2, 1 - alpha / 2):
        zq = norm.ppf(q)
        a = norm.cdf(z0 + (z0 + zq) / (1 - acc * (z0 + zq)))
        lo_hi.append(float(np.quantile(boots, np.clip(a, 0, 1))))
    return float(theta), lo_hi[0], lo_hi[1]


def two_stage_ci(h_ours, h_arm, n_boot=10000, seed=0, alpha=0.05):
    """Training-seed sensitivity: resample each arm's seeds (independently per arm), then crystals
    (same indices for both arms). h_* are [n_crystals, m_seeds] outcome arrays. Percentile CI."""
    h1, h2 = np.asarray(h_ours, float), np.asarray(h_arm, float)
    n, m1 = h1.shape
    m2 = h2.shape[1]
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot)
    for b in range(n_boot):
        s1 = rng.integers(0, m1, m1)
        s2 = rng.integers(0, m2, m2)
        idx = rng.integers(0, n, n)
        out[b] = h1[idx][:, s1].mean() - h2[idx][:, s2].mean()
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (float(c - h), float(c + h))


def verdict(delta, p_adj, alpha=0.05):
    if p_adj < alpha:
        return "OURS better" if delta > 0 else "ARM better" if delta < 0 else "not resolved"
    return "not resolved"


def verdict_h1(delta_r, delta_g, p_adj, alpha=0.05):
    """Intersection-union H1 (protocol §2.4)."""
    if p_adj >= alpha:
        return "not resolved"
    if delta_r > 0 and delta_g > 0:
        return "OURS better"
    if delta_r < 0 and delta_g < 0:
        return "MCF better (both variants)"
    return "split"


def _selftest():
    rng = np.random.default_rng(1)
    # binary single-seed data: sign-flip == exact two-sided McNemar
    from scipy.stats import binomtest
    for _ in range(20):
        a = rng.integers(0, 2, 60)
        b = rng.integers(0, 2, 60)
        d = a - b
        n10, n01 = int(((a == 1) & (b == 0)).sum()), int(((a == 0) & (b == 1)).sum())
        p_mc = binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
        assert abs(signflip_p(d) - p_mc) < 1e-9, (signflip_p(d), p_mc)
    # grouped-binomial convolution == element-by-element reference on seed means and RANDOM expectations
    for _ in range(30):
        n = int(rng.integers(20, 300))
        a = rng.integers(0, 49, n) / 48 * (rng.random(n) < 0.3)
        b = rng.integers(0, 4, n) / 3 * (rng.random(n) < 0.3)
        assert abs(signflip_p(a - b) - _signflip_p_slow(a - b)) < 1e-10
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    print("n3_stats selftest ok")


if __name__ == "__main__":
    _selftest()
