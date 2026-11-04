"""Jede Zahl der README, nachgerechnet über dieselben Auswertungsfunktionen wie in der App (DEMO-PLAYBOOK §4).

Exakte Zahlen (geschlossene Formeln) mit engen Bändern. Gemessene Zahlen stammen aus festen Strömen (50 000 Bahnen je Ebene, Ebenen 0 bis 8, Strom 36) und sind reproduzierbar; sie stehen hier mit Bändern,
die das Rauschen (die Ebenenvarianz ist auf etwa 10 bis 40 % genau, bei Sprung und Milstein wegen seltener Ereignisse schlechter; die Verzerrungskonstante schwankt um etwa 30 %) und kleine
Plattformunterschiede abdecken. Verhältnisse werden mal/durch verglichen."""

import pytest

import mlm_algorithm as A
import mlm_evaluation as ev
from mlm_evaluation import Settings


def _band(wert, soll, mal=1.5):
    return soll / mal <= wert <= soll * mal


def _a(scheme, kind, **kw):
    return ev.analyse(Settings(ausgabe=kind, verfahren=scheme, lmax=8, **kw))


# --- exakte Werte ------------------------------------------------------------------------------------------------------------------------------

def test_exakte_werte():
    assert A.truth("glatt", 0.125, 0.5, 1.3) == pytest.approx(1.13315, abs=5e-6)
    assert A.truth("knick", 0.125, 0.5, 1.3) == pytest.approx(0.16554, abs=5e-6)
    assert A.truth("sprung", 0.125, 0.5, 1.3) == pytest.approx(0.29989, abs=5e-6)


def test_erwartung_der_ersten_ebene():
    assert A.e_p0("euler", "knick", 0.125, 0.5, 1.3) == pytest.approx(0.12407, abs=5e-6)
    assert A.e_p0("milstein", "knick", 0.125, 0.5, 1.3) == pytest.approx(0.14414, abs=5e-6)
    assert A.e_p0("euler", "sprung", 0.125, 0.5, 1.3) == pytest.approx(0.36317, abs=5e-6)
    assert A.e_p0("milstein", "sprung", 0.125, 0.5, 1.3) == pytest.approx(0.29812, abs=5e-6)


# --- Raten ---------------------------------------------------------------------------------------------------------------------------------------

BETA = {("euler", "glatt"): (0.75, 1.15), ("euler", "knick"): (0.75, 1.15), ("euler", "sprung"): (0.2, 0.8),
        ("milstein", "glatt"): (1.6, 2.3), ("milstein", "knick"): (1.6, 2.3), ("milstein", "sprung"): (0.4, 1.4)}


@pytest.mark.parametrize("scheme,kind", list(BETA))
def test_rate_der_ebenenvarianz(scheme, kind):
    a = _a(scheme, kind)
    lo, hi = BETA[(scheme, kind)]
    assert lo < a.beta < hi and abs(a.beta_unc) < 0.3, (scheme, kind, a.beta, a.beta_unc)


def test_varianzverhaeltnis_je_ebene_und_ungekoppelt():
    e, m = _a("euler", "glatt"), _a("milstein", "glatt")
    assert _band(e.levels[8]["var"] / e.levels[7]["var"], 0.50, 1.25) and _band(m.levels[8]["var"] / m.levels[7]["var"], 0.26, 1.35)
    for scheme, kind, soll in (("euler", "glatt", 0.745), ("euler", "knick", 0.331), ("euler", "sprung", 0.422)):
        a = _a(scheme, kind)
        assert _band(a.var_unc[8], soll, 1.15) and a.var_unc[8] == pytest.approx(2 * a.levels[8]["var_p"], rel=0.1)


VERZERRUNG = {("euler", "glatt"): 0.0066, ("euler", "knick"): 0.0231, ("euler", "sprung"): 0.0913, ("milstein", "glatt"): 0.0089, ("milstein", "knick"): 0.0305, ("milstein", "sprung"): 0.0067}


@pytest.mark.parametrize("scheme,kind", list(VERZERRUNG))
def test_verzerrungskonstante(scheme, kind):
    assert _band(_a(scheme, kind).c, VERZERRUNG[(scheme, kind)], 2.2), (scheme, kind)


# --- idealisierte Kosten ------------------------------------------------------------------------------------------------------------------------

RATIO = {("euler", "glatt"): (1.14, 2.52), ("euler", "knick"): (1.29, 5.27), ("euler", "sprung"): (0.56, 1.01),
         ("milstein", "glatt"): (1.62, 16.58), ("milstein", "knick"): (1.64, 57.3), ("milstein", "sprung"): (1.36, 2.61)}


@pytest.mark.parametrize("scheme,kind", list(RATIO))
def test_kostenverhaeltnis_gewoehnlich_durch_multilevel(scheme, kind):
    t = ev.cost_table(_a(scheme, kind))
    mal = 1.9 if kind == "sprung" else 1.6
    assert _band(t[0]["ratio"], RATIO[(scheme, kind)][0], mal) and _band(t[-1]["ratio"], RATIO[(scheme, kind)][1], mal), (scheme, kind, t[0]["ratio"], t[-1]["ratio"])


def test_euler_mit_sprung_multilevel_ist_im_idealfall_nie_klar_billiger():
    t = ev.cost_table(_a("euler", "sprung"))
    assert max(r["ratio"] for r in t) < 1.25 and min(r["ratio"] for r in t) < 0.65
    assert all(r["ratio"] < 1.0 for r in t[:5])               # ε von 0,04 bis 0,0025
    assert all(0.4 < r["ratio"] < 0.8 for r in t[:5])         # README: 0,51 bis 0,65


def test_saettigung_bei_milstein_und_wachstum_bei_euler():
    m, e = ev.cost_table(_a("milstein", "knick")), ev.cost_table(_a("euler", "knick"))
    assert _band(m[0]["eps2_ml"], 0.32, 1.5) and _band(m[-1]["eps2_ml"], 0.72, 1.5) and m[-1]["eps2_ml"] / m[0]["eps2_ml"] < 3.0
    assert _band(m[0]["eps2_mc"], 0.53, 1.5) and _band(m[-1]["eps2_mc"], 41.3, 1.6) and m[-1]["eps2_mc"] / m[0]["eps2_mc"] > 40
    assert e[-1]["eps2_ml"] / e[0]["eps2_ml"] > 8 and _band(e[-1]["eps2_ml"], 3.9, 1.6)


def test_kostenanteile_je_ebene():
    m = ev.cost_table(_a("milstein", "knick"))[-1]["shares"]
    e = ev.cost_table(_a("euler", "knick"))[-1]["shares"]
    assert m[0] > 0.45 and all(m[i] >= m[i + 1] for i in range(1, len(m) - 1))
    assert min(e) > 0.07 and max(e) < 0.25


def test_kleine_streuung_gibt_mehr_vorteil_bei_euler_grosse_weniger():
    e25 = ev.cost_table(_a("euler", "knick", sigma=0.25))[-1]["ratio"]
    e10 = ev.cost_table(_a("euler", "knick", sigma=1.0))[-1]["ratio"]
    assert _band(e25, 6.5, 1.5) and _band(e10, 4.4, 1.5) and e25 > e10


def test_milstein_bei_grosser_streuung_gewinnt_am_meisten():
    m10 = ev.cost_table(_a("milstein", "knick", sigma=1.0))[-1]["ratio"]
    m05 = ev.cost_table(_a("milstein", "knick"))[-1]["ratio"]
    assert _band(m10, 119, 1.8) and m10 > m05


def test_vorasymptotische_strecke_bei_grosser_streuung():
    a = _a("euler", "knick", sigma=1.0)
    assert a.levels[2]["var"] > a.levels[1]["var"] and all(a.levels[l]["var"] < a.levels[l - 1]["var"] for l in range(4, 9))
    assert _band(a.beta, 0.82, 1.6)


# --- gelaufene Schätzungen ------------------------------------------------------------------------------------------------------------------------

def test_gelaufene_schaetzung_milstein_knick_eng_und_locker():
    a = _a("milstein", "knick")
    eng = ev.run_experiment(Settings(ausgabe="knick", verfahren="milstein", lmax=8, eps=0.0025), a)
    assert _band(eng["ratio"], 10.4, 2.0) and eng["ml"]["rmse"] / 0.0025 < 1.25 and eng["ml"]["L_min"] >= 4 and eng["ml"]["L_max"] <= 6
    locker = ev.run_experiment(Settings(ausgabe="knick", verfahren="milstein", lmax=8, eps=0.04), a)
    assert locker["ratio"] < 0.1 and _band(locker["ml"]["cost"], 1.0e4, 1.1) and _band(locker["mc"]["cost"], 330, 1.5)


def test_gelaufene_schaetzung_euler_knick_und_sprung():
    a = _a("euler", "knick")
    ex = ev.run_experiment(Settings(ausgabe="knick", verfahren="euler", lmax=8, eps=0.0025), a)
    assert _band(ex["ratio"], 1.85, 2.0) and ex["ml"]["rmse"] / 0.0025 < 1.25
    s = _a("euler", "sprung")
    ex = ev.run_experiment(Settings(ausgabe="sprung", verfahren="euler", lmax=8, eps=0.01), s)
    assert _band(ex["ratio"], 0.34, 2.0) and ex["ratio"] < 1.0


# --- Zwei Ebenen ---------------------------------------------------------------------------------------------------------------------------------

def test_zwei_ebenen_gegen_viele():
    s = Settings(ausgabe="sprung", verfahren="euler", lmax=8, eps=0.0025)
    ex = ev.two_level_experiment(s, _a("euler", "sprung"))
    assert _band(ex["ratio_two"], 2.28, 1.5) and ex["ratio_ml"] < 1.0 < ex["ratio_two"] and ex["best"] <= 3
    s = Settings(ausgabe="knick", verfahren="euler", lmax=8, eps=0.0025)
    ex = ev.two_level_experiment(s, _a("euler", "knick"))
    assert _band(ex["ratio_two"], 2.9, 1.5) and 0.8 < ex["ratio_two"] / ex["ratio_ml"] < 1.4
    s = Settings(ausgabe="knick", verfahren="milstein", lmax=8, eps=0.0025)
    ex = ev.two_level_experiment(s, _a("milstein", "knick"))
    assert _band(ex["ratio_two"], 8.8, 1.6) and ex["ratio_ml"] > 1.3 * ex["ratio_two"] and _band(ex["ratio_ml"], 15.8, 1.6)


# --- Entzerrter Schätzer ----------------------------------------------------------------------------------------------------------------------------

def _curve(scheme, kind, s):
    a = _a(scheme, kind)
    return [ev.single_term_theory(a, s, lm)[0] for lm in range(3, 9)]


def test_ein_term_varianz_bei_milstein_beschraenkt():
    v = _curve("milstein", "knick", 1.5)
    assert _band(v[-1], 0.21, 1.5) and v[-1] < 1.3 * v[0]
    g = _curve("milstein", "glatt", 1.5)
    assert _band(g[-1], 1.17, 1.4) and _band(g[0], 1.11, 1.4) and g[-1] < 1.3 * g[0] and _band(v[0], 0.19, 1.5)


def test_ein_term_varianz_waechst_bei_euler_und_milstein_mit_sprung():
    ek, eg, ms = _curve("euler", "knick", 1.5), _curve("euler", "glatt", 2.0), _curve("milstein", "sprung", 1.5)
    assert ek[-1] > 5 * ek[0] and _band(ek[0], 0.28, 1.5) and _band(ek[-1], 2.65, 2.0)
    assert eg[-1] > 10 * eg[0] and _band(eg[0], 1.52, 1.5) and _band(eg[-1], 35.5, 2.0)
    assert ms[-1] > 3 * ms[0] and _band(ms[0], 0.53, 1.6) and _band(ms[-1], 3.66, 2.2)


def test_ein_term_lauf_bei_milstein_knick():
    ex = ev.single_term_experiment(Settings(ausgabe="knick", verfahren="milstein", lmax=8), _a("milstein", "knick"))
    assert abs(ex["run"]["z"]) < 3.5 and abs(ex["run"]["mean"] - 0.1647) < 0.004 and _band(ex["run"]["var"], 0.22, 1.4) and _band(ex["theory"]["var"], 0.21, 1.4)
    assert ex["run"]["cost"] == pytest.approx(2.84, abs=0.15)


def test_kosten_des_ein_term_schaetzers_gegen_multilevel():
    a = _a("milstein", "knick")
    work = ev.single_term_theory(a, 1.5, 8)[2]
    assert _band(work, 0.60, 1.5) and 0.7 < work / ev.cost_row(a, 0.0025)["eps2_ml"] < 1.4
    g = _a("milstein", "glatt")
    assert ev.single_term_theory(g, 1.5, 8)[2] > 2.0 * ev.cost_row(g, 0.0025)["eps2_ml"]


def test_gelaufener_fehler_ist_im_mittel_negativ_und_nutzt_das_verzerrungsbudget():
    import numpy as np
    errs = np.array([A.mlmc("milstein", "knick", 0.005, A.stream_seed(5, r), 0.125, 0.5, 1.3, n0=1000)[0] - A.truth("knick", 0.125, 0.5, 1.3) for r in range(40)])
    assert -0.95 < errs.mean() / 0.005 < -0.4 and np.sqrt((errs**2).mean()) / 0.005 < 1.1
