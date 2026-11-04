"""Randfälle: lockere und enge Toleranz, konstante Ausgabe, kleinste Stichproben, Eingabeprüfung der Spielwiese, unbekannte Namen."""

import math

import numpy as np
import pytest

import mlm_algorithm as A
import mlm_constants as C
import mlm_evaluation as ev
from mlm_evaluation import Settings
from mlm_scenario import Szenario, check_szenario, standard_szenario


def test_sehr_lockere_toleranz_verlangt_trotzdem_mindestens_ebene_eins():
    a = ev.analyse(Settings(lmax=6))
    assert ev.level_for(a, 10.0) == 1


def test_sehr_enge_toleranz_bleibt_endlich():
    a = ev.analyse(Settings(lmax=6))
    L = ev.level_for(a, 1e-9)
    assert 1 <= L <= 40
    assert math.isfinite(ev.cost_row(a, 1e-9)["ml"])


def test_konstante_ausgabe_gibt_keine_division_durch_null_und_ein_ehrliches_verdikt():
    s = Settings(ausgabe="sprung", lmax=6)
    a = ev.analyse(s, Szenario(0.0, 0.1, 2.5))              # die Schwelle wird nie überschritten
    assert ev.degenerate(a) and not np.isfinite(a.beta) and a.c == 0.0
    row = ev.cost_row(a, 0.01)
    assert row["mc"] == 0.0 and row["ml"] == 0.0 and row["ratio"] == float("inf")
    for kind, text in (ev.verdict_main(a, s), ev.verdict_levels(a), ev.verdict_cost(a)):
        assert kind == "info" and "nichts zu schätzen" in text and "nan" not in text and "inf" not in text


def test_giles_verfahren_bricht_bei_null_varianz_ab():
    est, cost, L, Nl = A.mlmc("euler", "sprung", 0.01, 3, 0.0, 0.1, 2.5, n0=100)
    assert est == 0.0 and L == 2 and cost > 0


def test_eine_probe_hat_varianz_null_ohne_fehler():
    m, v, vp, kurt, nz, mp = A.level_stats("euler", "knick", 3, 1, 5, 0.125, 0.5, 1.3)
    assert v == 0.0 and vp == 0.0 and np.isnan(kurt)


def test_rate_mit_zu_wenigen_punkten_ist_nicht_bestimmbar():
    assert np.isnan(ev.fit_rate([1.0, 0.5, 0.25]))                     # ab Ebene 3 fehlt jeder Punkt
    assert np.isnan(ev.fit_rate([1.0, 0.5, 0.25, 0.1]))                # ein Punkt
    assert ev.fit_rate([1.0, 0.5, 0.25, 0.125, 0.0625]) == pytest.approx(1.0)
    assert np.isnan(ev.fit_rate([1.0, 0.5, 0.25, 0.0, 0.0, 0.0]))      # Nullen werden übersprungen (kein log(0)), es bleibt kein Punkt
    assert ev.fit_rate([1.0, 0.5, 0.25, 0.125, 0.0625, 0.0, 0.015625]) == pytest.approx(1.0)


def test_verzerrungskonstante_ohne_passende_ebenen_ist_nan():
    assert np.isnan(ev.bias_const([{"l": 0, "mean": 1.0}, {"l": 1, "mean": 0.5}]))


def test_unbekannte_namen_werden_abgelehnt():
    with pytest.raises(ValueError):
        A.step("runge", np.ones(2), 0.5, np.ones(2), 0.1, 0.5)
    with pytest.raises(ValueError):
        A.payoff("unbekannt", np.ones(2), 1.3)
    with pytest.raises(ValueError):
        A.truth("unbekannt", 0.1, 0.5, 1.3)


def test_ein_term_schaetzer_mit_nur_ebene_null():
    mean, var, cost = A.single_term("euler", "knick", 1.5, 0, 2000, 1, 0.125, 0.5, 1.3)
    assert cost == 1.0 and mean == pytest.approx(A.e_p0("euler", "knick", 0.125, 0.5, 1.3), abs=0.02)


def test_spielwiese_eingabepruefung():
    assert check_szenario(0.125, 0.5, 1.3) == standard_szenario(0.5)
    for args, frag in (((None, 0.5, 1.3), "keine Zahl"), ((float("nan"), 0.5, 1.3), "keine Zahl"), ((float("inf"), 0.5, 1.3), "keine Zahl"), ((2.0, 0.5, 1.3), "Drift"), ((0.1, 0.01, 1.3), "Streuung"), ((0.1, 0.5, 9.0), "Schwelle")):
        with pytest.raises(ValueError, match=frag):
            check_szenario(*args)


def test_eigener_prozess_traegt_seine_parameter_im_etikett():
    s = Settings(lmax=6)
    a = ev.analyse(s, Szenario(0.3, 0.4, 1.1))
    assert "a = 0,30" in a.label and "q = 1,10" in a.label and a.sigma == 0.4
    assert a.truth == pytest.approx(A.truth("knick", 0.3, 0.4, 1.1))
    assert "a = " not in ev.analyse(s).label


def test_zahlenformate():
    assert ev.de(1234.5, 1) == "1.234,5" and ev.de(-0.5, 1) == "−0,5"
    assert ev.ke(999.0) == "999" and ev.ke(123456.0) == "1,2e+05" and ev.ke(float("nan")) == "nicht bestimmbar"
    assert ev.eps_text(0.0025) == "0,0025" and ev.eps_text(0.04) == "0,04"
    assert ev.dn(float("inf")) == "unendlich" and ev.dn(float("nan")) == "nicht bestimmbar" and ev.prozent(float("nan")) == "nicht bestimmbar"


def test_kleinste_und_groesste_einstellung_rechnen():
    for lmax in (C.LMAX_MIN, C.LMAX_MAX):
        for sigma in (C.SIGMA_MIN, C.SIGMA_MAX):
            a = ev.analyse(Settings(lmax=lmax, sigma=sigma, verfahren="milstein"))
            assert all(math.isfinite(r["ratio"]) for r in ev.cost_table(a))


def test_extrapolation_jenseits_der_gemessenen_ebenen_faellt_hoechstens_ab():
    a = ev.analyse(Settings(lmax=6))
    assert ev.level_variance(a, 6) == a.levels[6]["var"]
    assert 0 < ev.level_variance(a, 10) <= a.levels[6]["var"]
