"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset rechnet und hat das erwartete Verdikt."""

import pytest

import mlm_constants as C
import mlm_evaluation as ev
import mlm_presets as P

STANDARD = "Standardfall (Voreinstellung)"


def _settings(name):
    return ev.Settings(**C.PRESETS[name])


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_slider_bounds_and_options():
    for name, preset in C.PRESETS.items():
        assert 0 <= preset["seed"] <= C.SEED_MAX, name
        assert preset["ausgabe"] in C.AUSGABEN and preset["verfahren"] in C.VERFAHREN, name
        assert preset["sigma"] in C.SIGMA_CHOICES and preset["eps"] in C.EPS_CHOICES and preset["lmax"] in C.LMAX_CHOICES, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[STANDARD]) == ev.Settings()


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)


def test_jedes_preset_unterscheidet_sich_vom_standard():
    for name, preset in C.PRESETS.items():
        if name != STANDARD:
            assert preset != C.PRESETS[STANDARD], name


def test_standardfall_zeigt_einen_kleinen_vorteil_von_euler_mit_knick():
    s = _settings(STANDARD)
    a = ev.analyse(s)
    row = ev.cost_row(a, s.eps)
    assert 1.3 < row["ratio"] < 3.2 and 0.7 < a.beta < 1.2 and ev.verdict_main(a, s)[0] in ("info", "success")


def test_preset_milstein_laesst_multilevel_deutlich_gewinnen():
    s = _settings("Milstein: die Kette zahlt sich aus")
    a = ev.analyse(s)
    kind, text = ev.verdict_main(a, s)
    assert kind == "success" and ev.cost_row(a, s.eps)["ratio"] > 6 and a.beta > 1.6


def test_preset_sprung_mit_euler_zeigt_multilevel_teurer():
    s = _settings("Sprung mit Euler: Multilevel verliert")
    a = ev.analyse(s)
    kind, text = ev.verdict_main(a, s)
    assert kind == "warning" and ev.cost_row(a, s.eps)["ratio"] < 1.0 and "teurer als gewöhnliches Monte Carlo" in text


def test_preset_grosse_streuung_hat_ein_ansteigendes_v_auf_den_groebsten_ebenen():
    s = _settings("Große Streuung: grobe Ebenen taugen nicht")
    a = ev.analyse(s)
    assert s.sigma == 1.0 and a.levels[2]["var"] > a.levels[1]["var"]
    assert all(a.levels[l]["var"] < a.levels[l - 1]["var"] for l in range(4, a.lmax + 1))


def test_preset_lockere_toleranz_zeigt_den_pilot_als_verlust():
    s = _settings("Lockere Toleranz")
    ex = ev.run_experiment(s)
    assert ex["ratio"] < 0.2 and ev.verdict_run(ex)[0] == "warning"


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_jedes_preset_rechnet(name):
    a = ev.analyse(_settings(name))
    assert a.truth > 0 and len(a.levels) == C.PRESETS[name]["lmax"] + 1
