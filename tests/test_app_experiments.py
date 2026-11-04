"""AppTest der Experimente, der Spielwiese, des PDF-Exports und des Permalinks (Einzelheiten, die der allgemeine Rauchtest nicht trifft)."""

import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import mlm_constants as C

APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _texte(at) -> str:
    return " ".join(e.value for kind in (at.success, at.info, at.warning, at.error) for e in kind)


@pytest.mark.parametrize("key,fragment", [
    ("run_gelaufen", "Pilot 1.000 je Ebene"),
    ("run_two", "der beste Zwei-Ebenen-Schätzer"),
    ("run_single", "Varianz des Ein-Term-Schätzers bei wachsender Abschneideebene"),
])
def test_experiment_auf_klick_zeigt_sein_verdikt(key, fragment):
    at = _app()
    assert fragment not in _texte(at)                       # vor dem Klick steht noch nichts da
    at.button(key=key).click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert fragment in _texte(at)


def test_kernfrage_ebenen_und_kostenverdikt_stehen_ohne_klick_da():
    t = _texte(_app())
    assert "gewöhnliches Monte Carlo braucht die Ebene L" in t
    assert "Gekoppelt fällt die Varianz V_l der Differenz" in t
    assert "Gewöhnlich/Multilevel bei ε = 0,04:" in t


def test_experiment_bleibt_nach_einer_aenderung_der_einstellungen_sichtbar():
    at = _app()
    at.button(key="run_gelaufen").click().run(timeout=TIMEOUT)
    at.sidebar.select_slider(key="eps_select").set_value(0.01).run(timeout=TIMEOUT)
    assert not at.exception
    assert "Toleranz ε = 0,01, 30 Wiederholungen" in _texte(at)


def test_kernfrage_verdikt_wechselt_mit_dem_preset():
    at = _app()
    assert "teurer als gewöhnliches Monte Carlo" not in _texte(at)
    at.sidebar.button(key="preset_Sprung mit Euler: Multilevel verliert").click().run(timeout=TIMEOUT)
    assert "teurer als gewöhnliches Monte Carlo" in _texte(at)
    at.sidebar.button(key="preset_Milstein: die Kette zahlt sich aus").click().run(timeout=TIMEOUT)
    assert "mindestens die Hälfte" in _texte(at)


def test_jede_ausgabe_und_jedes_verfahren_laeuft_durch():
    at = _app()
    for ausgabe in C.AUSGABEN:
        for verfahren in C.VERFAHREN:
            at.sidebar.selectbox(key="ausgabe_select").set_value(ausgabe).run(timeout=TIMEOUT)
            at.sidebar.selectbox(key="verfahren_select").set_value(verfahren).run(timeout=TIMEOUT)
            assert not at.exception, (ausgabe, verfahren, [str(e) for e in at.exception])


def test_spielwiese_rechnet_mit_eigenen_werten():
    at = _app()
    assert not at.error
    at.number_input(key="eigen_sigma").set_value(1.0).run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert not at.error
    at.number_input(key="eigen_q").set_value(2.5)
    at.number_input(key="eigen_sigma").set_value(0.1).run(timeout=TIMEOUT)        # Schwelle, die nie überschritten wird: keine Streuung
    assert not at.exception, [str(e) for e in at.exception]
    assert "nichts zu schätzen" in _texte(at)


def test_pdf_wird_erst_nach_klick_erzeugt():
    at = _app()
    assert not at.get("download_button")
    at.button(key="pdf_make").click().run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    assert len(at.get("download_button")) == 1
    at.sidebar.select_slider(key="lmax_select").set_value(6).run(timeout=TIMEOUT)         # andere Einstellung: altes PDF nicht mehr anbieten
    assert not at.get("download_button")


def test_permalink_wird_begrenzt_und_eingerastet():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.query_params["s"] = "0.3"
    at.query_params["e"] = "0.003"
    at.query_params["l"] = "99"
    at.query_params["a"] = "unbekannt"        # unbekannte Ausgabe: verworfen
    at.query_params["v"] = "milstein"
    at.query_params["seed"] = "-4"
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    assert at.sidebar.select_slider(key="sigma_select").value == 0.25                        # 0,3 liegt näher an 0,25 als an 0,5
    assert at.sidebar.select_slider(key="eps_select").value == 0.0025                        # 0,003 liegt näher an 0,0025 als an 0,005
    assert at.sidebar.select_slider(key="lmax_select").value == 8                            # oberhalb des Größten: das Größte
    assert at.sidebar.selectbox(key="ausgabe_select").value == C.DEFAULT_AUSGABE
    assert at.sidebar.selectbox(key="verfahren_select").value == "milstein"
    assert at.sidebar.number_input(key="seed_input").value == 0
