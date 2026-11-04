"""Unabhängige Gegenproben (scipy): Quadratur, Kolmogorov–Smirnov, Gauß–Hermite; ORACLE-PLAYBOOK §5."""

import math

import numpy as np
import pytest
from scipy import integrate, optimize, stats

import mlm_algorithm as A
from conftest import STD


def _g(kind, q):
    return {"glatt": lambda x: x, "knick": lambda x: max(x - q, 0.0), "sprung": lambda x: float(x > q)}[kind]


def _quad_truth(kind, a, sigma, q):
    mu = a - sigma**2 / 2
    g = _g(kind, q)
    f = lambda z: g(math.exp(mu + sigma * z)) * stats.norm.pdf(z)
    zq = (math.log(q) - mu) / sigma
    pts = sorted({-12.0, zq, 12.0})
    return sum(integrate.quad(f, lo, hi, epsabs=1e-12, epsrel=1e-12)[0] for lo, hi in zip(pts[:-1], pts[1:]))


@pytest.mark.parametrize("kind", ["glatt", "knick", "sprung"])
@pytest.mark.parametrize("a,sigma,q", [(0.125, 0.5, 1.3), (0.5, 1.0, 1.0), (-0.3, 0.3, 0.8), (0.0, 1.2, 2.0)])
def test_exakter_wert_gegen_quadratur(kind, a, sigma, q):
    assert A.truth(kind, a, sigma, q) == pytest.approx(_quad_truth(kind, a, sigma, q), abs=1e-9)


def _quad_p0(scheme, kind, a, sigma, q):
    def S(z):
        return 1 + a + sigma * z + (0.5 * sigma**2 * (z * z - 1) if scheme == "milstein" else 0.0)
    g = _g(kind, q)
    f = lambda z: g(S(z)) * stats.norm.pdf(z)
    grid = np.linspace(-10, 10, 4001)
    roots = [optimize.brentq(lambda z: S(z) - q, grid[i], grid[i + 1], xtol=1e-14) for i in range(len(grid) - 1) if (S(grid[i]) - q) * (S(grid[i + 1]) - q) < 0]
    pts = sorted({-10.0, 10.0, 0.0, *roots})
    return sum(integrate.quad(f, lo, hi, epsabs=1e-12, epsrel=1e-12, limit=200)[0] for lo, hi in zip(pts[:-1], pts[1:]))


@pytest.mark.parametrize("scheme", ["euler", "milstein"])
@pytest.mark.parametrize("kind", ["glatt", "knick", "sprung"])
@pytest.mark.parametrize("a,sigma,q", [(0.125, 0.5, 1.3), (0.5, 1.0, 1.0), (-0.3, 0.3, 0.8), (0.0, 1.2, 2.0), (0.1, 0.2, 0.5)])
def test_e_p0_gegen_quadratur(scheme, kind, a, sigma, q):
    assert A.e_p0(scheme, kind, a, sigma, q) == pytest.approx(_quad_p0(scheme, kind, a, sigma, q), abs=2e-8)


def test_normalzahlen_gegen_scipy_kolmogorov_smirnov():
    z = A.normals(A.stream_seed(7), 20000)
    assert stats.kstest(z, "norm").pvalue > 0.001


@pytest.mark.parametrize("scheme", ["euler", "milstein"])
def test_erwartung_des_endwerts_ist_eins_plus_ah_hoch_n(scheme):
    # E S_n = (1 + a h)^n für beide Verfahren (E[ΔW] = 0, E[ΔW² − h] = 0, unabhängige Zuwächse)
    N = 400000
    _, _, vp, _, _, mean_p = A.level_stats(scheme, "glatt", 3, N, 11, 0.125, 0.5, 1.3)
    assert abs(mean_p - (1 + 0.125 / 8) ** 8) < 4 * math.sqrt(vp / N)


def _gauss_hermite_mean(scheme, kind, level, a, sigma, q, nodes):
    """E[P_l] als Produkt-Gauß–Hermite-Integral über die 2^l Normalzahlen (nur l ≤ 2)."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    n = 2**level
    h = 1.0 / n
    xs = np.meshgrid(*([x] * n), indexing="ij")
    ws = np.meshgrid(*([w] * n), indexing="ij")
    wts = np.ones_like(xs[0])
    for wi in ws:
        wts = wts * wi
    S = np.ones_like(xs[0])
    for g in xs:
        S = A.step(scheme, S, h, math.sqrt(h) * g, a, sigma)
    return float((A.payoff(kind, S, q) * wts).sum())


@pytest.mark.parametrize("scheme", ["euler", "milstein"])
@pytest.mark.parametrize("kind", ["glatt", "knick"])
@pytest.mark.parametrize("level", [1, 2])
def test_ebenenmittel_gegen_gauss_hermite(scheme, kind, level):
    ref = _gauss_hermite_mean(scheme, kind, level, 0.125, 0.5, 1.3, 120 if level == 1 else 40)
    N = 200000
    st = A.level_stats(scheme, kind, level, N, 5, 0.125, 0.5, 1.3)
    assert abs(st[5] - ref) < 4 * math.sqrt(st[2] / N) + (2e-3 if kind == "knick" else 0.0)


def test_gauss_hermite_selbst_gegen_geschlossene_form():
    # Prüfung der Gegenprobe: Ebene 0 (ein Schritt), Euler, Knick, gegen e_p0
    x, w = np.polynomial.hermite_e.hermegauss(100)
    w = w / w.sum()
    S = A.step("euler", np.ones_like(x), 1.0, x, 0.125, 0.5)
    assert float((A.payoff("knick", S, 1.3) * w).sum()) == pytest.approx(A.e_p0("euler", "knick", 0.125, 0.5, 1.3), abs=1e-3)


@pytest.mark.parametrize("scheme", ["euler", "milstein"])
def test_teleskopsumme_gleicht_dem_mittel_der_feinen_ebene(scheme):
    kind, L, N = "knick", 4, 100000
    stats_l = [A.level_stats(scheme, kind, l, N, 21, **STD) for l in range(L + 1)]
    tele = A.e_p0(scheme, kind, **STD) + sum(s[0] for s in stats_l[1:])
    se = math.sqrt(sum(s[1] / N for s in stats_l[1:])) + math.sqrt(stats_l[L][2] / N)
    assert abs(tele - stats_l[L][5]) < 4 * se


def test_grobe_bahn_gleicht_direkt_gebauter_grober_bahn():
    # Kopplung: die grobe Bahn nutzt die Summen je zweier feiner Zuwächse; dieselben Zahlen auf dem groben Gitter von Hand verbraucht liefern dieselbe Bahn
    z = np.random.default_rng(3).standard_normal((8, 5))
    pf, pc = A.level_pair("euler", "glatt", 3, 5, 0, zfun=lambda i, n: z[i], **STD)
    S = np.ones(5)
    for j in range(4):
        S = A.step("euler", S, 1 / 4, math.sqrt(1 / 8) * (z[2 * j] + z[2 * j + 1]), STD["a"], STD["sigma"])
    assert np.allclose(pc, S)
    fein = np.ones(5)
    for i in range(8):
        fein = A.step("euler", fein, 1 / 8, math.sqrt(1 / 8) * z[i], STD["a"], STD["sigma"])
    assert np.allclose(pf, fein)


def test_pair_general_mit_benachbarten_ebenen_gleicht_level_pair_in_der_verteilung():
    N = 100000
    y1 = np.subtract(*A.level_pair("euler", "knick", 4, N, 31, **STD))
    p, pc = A.pair_general("euler", "knick", 4, 3, N, 32, **STD)
    y2 = p - pc
    assert abs(y1.mean() - y2.mean()) < 5 * math.sqrt((y1.var() + y2.var()) / N)
    assert y1.var() == pytest.approx(y2.var(), rel=0.1)


def test_zufaellige_stoerung_der_zuteilung_ist_nie_billiger():
    rng = np.random.default_rng(1)
    vl, cl, eps = [0.25, 0.01, 0.005, 0.002], [1.0, 3.0, 6.0, 12.0], 0.01
    ns, cost = A.optimal_allocation(vl, cl, eps)
    for _ in range(200):
        trial = np.array(ns) * rng.uniform(0.5, 2.0, len(ns))
        var = sum(v / n for v, n in zip(vl, trial))
        trial = trial * var / (eps**2 / 2)               # auf dieselbe Gesamtvarianz ε²/2 skalieren
        assert sum(n * c for n, c in zip(trial, cl)) >= cost * (1 - 1e-9)
