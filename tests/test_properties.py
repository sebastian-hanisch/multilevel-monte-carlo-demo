"""Eigenschaften, die für jede Wahl gelten müssen (nicht nur an einem Zahlenwert): Raten, Kopplung, Verzerrung, Kosten, Reproduzierbarkeit."""

import math

import numpy as np
import pytest

import mlm_algorithm as A
import mlm_evaluation as ev
from mlm_evaluation import Settings


def _a(scheme="euler", kind="knick", **kw):
    return ev.analyse(Settings(ausgabe=kind, verfahren=scheme, lmax=7, **kw))


def test_gekoppelte_varianz_halbiert_sich_bei_euler_und_viertelt_sich_bei_milstein():
    for kind in ("glatt", "knick"):
        e, m = _a("euler", kind), _a("milstein", kind)
        assert e.levels[7]["var"] / e.levels[6]["var"] == pytest.approx(0.5, abs=0.12)
        assert m.levels[7]["var"] / m.levels[6]["var"] == pytest.approx(0.25, abs=0.08)
        assert 0.7 < e.beta < 1.2 and 1.6 < m.beta < 2.3


def test_ungekoppelte_varianz_faellt_nicht_und_ist_etwa_das_doppelte_der_varianz_der_ausgabe():
    for scheme in ("euler", "milstein"):
        for kind in ("glatt", "knick", "sprung"):
            a = _a(scheme, kind)
            assert abs(a.beta_unc) < 0.3
            assert a.var_unc[7] == pytest.approx(2 * a.levels[7]["var_p"], rel=0.12)


def test_der_sprung_halbiert_die_rate_bei_euler():
    assert _a("euler", "sprung").beta < _a("euler", "knick").beta - 0.25


def test_verzerrung_halbiert_sich_je_ebene_beim_euler_endwert():
    a = _a("euler", "glatt")
    for l in (2, 3):
        assert ev.bias_at(a, l) / ev.bias_at(a, l - 1) == pytest.approx(0.5, abs=0.15)
    # geschlossene Form: E[S_n] = (1 + a/n)^n
    for l in (1, 2):
        exakt = (1 + 0.125 / 2**l) ** 2**l - math.exp(0.125)
        assert ev.bias_at(a, l) == pytest.approx(exakt, abs=6e-4)


def test_ebene_waechst_mit_enger_werdender_toleranz():
    a = _a()
    ls = [ev.level_for(a, e) for e in sorted(ev.C.EPS_GRID, reverse=True)]
    assert ls == sorted(ls) and ls[-1] > ls[0]


def test_kosten_wachsen_mit_enger_werdender_toleranz_und_sind_ueberall_positiv():
    for scheme in ("euler", "milstein"):
        t = ev.cost_table(_a(scheme))
        for lo, hi in zip(t[:-1], t[1:]):
            assert hi["mc"] > lo["mc"] and hi["ml"] > lo["ml"] > 0


def test_milstein_gewinnt_bei_enger_toleranz_mehr_als_euler():
    e, m = ev.cost_table(_a("euler")), ev.cost_table(_a("milstein"))
    assert m[-1]["ratio"] > 3 * e[-1]["ratio"]
    assert m[-1]["ratio"] > m[0]["ratio"] and e[-1]["ratio"] > e[0]["ratio"]


def test_kostenanteile_summieren_sich_zu_eins():
    for r in ev.cost_table(_a("milstein")):
        assert sum(r["shares"]) == pytest.approx(1.0)


def test_ungekoppelt_waere_multilevel_viel_teurer_als_gewoehnlich():
    a = _a("euler", "knick")
    eps = 0.0025
    L = ev.level_for(a, eps)
    vl = [a.var_unc[l] for l in range(L + 1)]
    cl = [A.cost_per_sample(l) for l in range(L + 1)]
    ml = A.optimal_allocation(vl, cl, eps)[1]
    assert ml > 5 * ev.costs_for_level(a, eps, L)["mc"]


def test_gleiche_einstellung_gleiche_zahlen():
    s = Settings(verfahren="milstein", lmax=6)
    a1, a2 = ev.analyse(s), ev.analyse(s)
    assert a1 == a2
    assert ev.cost_table(a1) == ev.cost_table(a2)
    other = ev.analyse(Settings(verfahren="milstein", lmax=6, seed=77))
    assert other.levels[3]["mean"] != a1.levels[3]["mean"]


def test_gelaufener_multilevel_schaetzer_trifft_die_toleranz_und_nutzt_das_verzerrungsbudget_aus():
    s = Settings(verfahren="milstein", eps=0.005, lmax=7)
    a = ev.analyse(s)
    errs = []
    for r in range(40):
        est, cost, L, Nl = A.mlmc("milstein", "knick", s.eps, A.stream_seed(5, r), 0.125, 0.5, 1.3, n0=1000)
        errs.append(est - a.truth)
    errs = np.array(errs)
    assert math.sqrt((errs**2).mean()) < 1.3 * s.eps
    # Das Verfahren darf die Verzerrung bis ε/√2 ausschöpfen (Abbruchtest): der Fehler ist systematisch negativ (Abbruchverzerrung), aber nicht größer als das Budget
    assert -0.9 * s.eps < errs.mean() < 0.0 and errs.std() < 0.9 * s.eps


def test_ein_term_schaetzer_ist_bei_milstein_unverzerrt_und_hat_endliche_varianz():
    mean, var, cost = A.single_term("milstein", "knick", 1.5, 7, 40000, 9, 0.125, 0.5, 1.3)
    assert abs(mean - A.truth("knick", 0.125, 0.5, 1.3)) < 4 * math.sqrt(var / 40000) + 3e-4
    assert var < 0.4 and cost == pytest.approx(2.8, abs=0.3)


def test_ein_term_varianz_waechst_bei_euler_mit_der_abschneideebene_und_bleibt_bei_milstein_beschraenkt():
    e, m = _a("euler"), _a("milstein")
    for s in (1.5, 2.0):
        ve = [ev.single_term_theory(e, s, lm)[0] for lm in (3, 7)]
        assert ve[1] > 3 * ve[0]
    vm = [ev.single_term_theory(m, 1.5, lm)[0] for lm in (3, 7)]
    assert vm[1] < 1.3 * vm[0]


def test_zwei_ebenen_varianz_waechst_mit_dem_abstand_der_ebenen():
    s = Settings(eps=0.0025, lmax=6)
    ex = ev.two_level_experiment(s)
    vs = [r["var"] for r in ex["rows"]]
    assert vs == sorted(vs, reverse=True)           # je näher die grobe Ebene an L, desto kleiner die Varianz der Differenz
