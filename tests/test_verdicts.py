"""Verdikte (reine Funktionen in mlm_evaluation.py) an handgesetzten Werten: Schwellen und jede Zahl im Text."""

import dataclasses

import pytest

import mlm_evaluation as ev

S = ev.Settings(eps=0.0025, lmax=4)


def _levels(vars_, means=None):
    means = means or [0.0] + [0.04 / 2**l for l in range(1, len(vars_))]
    return tuple({"l": l, "mean": means[l], "var": v, "var_p": 0.16, "kurt": 20.0, "nonzero": 0.3, "mean_p": 0.16} for l, v in enumerate(vars_))


def _a(**kw) -> ev.Analysis:
    # gekoppelt: V_l = 0,05 · 2^(−l) ab Ebene 1 (β = 1); ungekoppelt konstant 0,32; Verzerrung c · 2^(−L) mit c = 0,04
    vs = [0.0536] + [0.05 * 0.5 ** (l - 1) for l in range(1, 5)]
    basis = ev.Analysis(label="Euler–Maruyama, Knick: Fehlbestand (S(T) − q)⁺", scheme="euler", kind="knick", drift=0.125, sigma=0.5, q=1.3, truth=0.1655, lmax=4, n=50000, e0=0.1240,
                        levels=_levels(vs), var_unc=(0.0536, 0.32, 0.32, 0.32, 0.32), c=0.04, beta=1.0, beta_unc=0.0)
    return dataclasses.replace(basis, **kw)


def test_kostenfaelle_der_rate():
    assert "β > γ" in ev.regime(2.0) and "sättigen" in ev.regime(2.0)
    assert "β ≈ γ" in ev.regime(1.0) and "Logarithmen" in ev.regime(1.0)
    assert "β < γ" in ev.regime(0.5)
    assert ev.regime(float("nan")) == "nicht bestimmbar"
    assert "β > γ" in ev.regime(1.26) and "β ≈ γ" in ev.regime(1.25) and "β ≈ γ" in ev.regime(0.8) and "β < γ" in ev.regime(0.79)   # Schwellen 0,8 und 1,25


def test_ebene_und_kosten_von_hand():
    a = _a()
    # c = 0,04: kleinste Ebene mit 0,04 · 2^(−L) ≤ 0,0025/√2 = 0,001768 ⇒ 2^L ≥ 22,6 ⇒ L = 5
    assert ev.level_for(a, 0.0025) == 5
    row = ev.cost_row(a, 0.0025)
    # V_l, C_l für l = 0 … 5: V = (0,0536; 0,05; 0,025; 0,0125; 0,00625; 0,003125 [Fortsetzung mit β = 1]), C = (1, 3, 6, 12, 24, 48)
    vl = [0.0536, 0.05, 0.025, 0.0125, 0.00625, 0.003125]
    cl = [1, 3, 6, 12, 24, 48]
    s = sum((v * c) ** 0.5 for v, c in zip(vl, cl))
    assert row["L"] == 5 and row["ml"] == pytest.approx(2 * s * s / 0.0025**2)
    assert row["mc"] == pytest.approx(2 * 0.16 / 0.0025**2 * 32)
    assert row["ratio"] == pytest.approx(row["mc"] / row["ml"]) and sum(row["shares"]) == pytest.approx(1.0)


def test_hauptverdikt_schwellen_und_zahlen():
    a = _a()
    kind, text = ev.verdict_main(a, S)
    assert kind in ("info", "success")
    for frag in ("Euler–Maruyama, Knick: Fehlbestand (S(T) − q)⁺, σ = 0,50, Toleranz ε = 0,0025", "die Ebene L = 5", "Verzerrung ≈ 0,0400·2^(−L)", "β ≈ 1,00 (β ≈ γ", "3,2e-01 (Verhältnis zur Vorebene 1,00 statt 0,50)"):
        assert frag in text, frag
    # Multilevel teurer als gewöhnlich: Warnung, wenn die gewöhnlichen Kosten sehr klein sind
    cheap = dataclasses.replace(a, levels=tuple({**r, "var_p": 1e-6} for r in a.levels))
    assert ev.verdict_main(cheap, S)[0] == "warning" and "teurer als gewöhnliches Monte Carlo" in ev.verdict_main(cheap, S)[1]
    # steile Rate: mindestens halbe Kosten
    steil = dataclasses.replace(a, levels=_levels([0.0536, 0.05, 0.0125, 0.003, 0.0008]))
    assert ev.verdict_main(steil, S)[0] == "success" and "mindestens die Hälfte" in ev.verdict_main(steil, S)[1]


def test_ebenenverdikt():
    kind, text = ev.verdict_levels(_a())
    assert kind == "info"
    for frag in ("von Ebene 3 auf 4 auf das 0,50-Fache", "Rate β ≈ 1,00, aus den Ebenen 3 bis 4", "ungekoppelt auf das 1,00-Fache (β ≈ 0,00)", "V_4 = 6,3e-03 gegen 3,2e-01", "≈ 0,0400·2^(−L)", "Größte Kurtosis der Differenz: 20", "50.000 Proben"):
        assert frag in text, frag


def test_kostenverdikt():
    kind, text = ev.verdict_cost(_a())
    assert kind == "info"
    assert "Gewöhnlich/Multilevel bei ε = 0,04:" in text and "bei ε = 0,000625:" in text and "ε²·Kosten: Multilevel" in text
    assert "Multilevel ist ab ε = " in text or "bei allen gezeigten Toleranzen billiger" in text
    teuer = dataclasses.replace(_a(), levels=tuple({**r, "var_p": 1e-6} for r in _a().levels))
    assert "noch nicht billiger" in ev.verdict_cost(teuer)[1]


def test_gelaufenes_verdikt():
    ex = {"eps": 0.005, "reps": 30, "pilot": 1000, "L_std": 3, "truth": 0.1655,
          "ml": {"rmse": 0.004, "cost": 84642.0, "L_mean": 4.1, "L_min": 4, "L_max": 5, "over": 0.1}, "mc": {"rmse": 0.0045, "cost": 89800.0, "over": 0.2}, "ratio": 1.06}
    kind, text = ev.verdict_run(ex)
    assert kind == "info"
    for frag in ("Toleranz ε = 0,005, 30 Wiederholungen", "Pilot 1.000 je Ebene", "RMSE/ε = 0,80", "84.642 Zeitschritten", "Ebenen 4 bis 5, Mittel 4,1", "10 % der Läufe", "(Ebene 3) RMSE/ε = 0,90", "89.800", "gewöhnlich/Multilevel: 1,06"):
        assert frag in text, frag
    assert ev.verdict_run({**ex, "ratio": 0.4})[0] == "warning"


def test_zwei_ebenen_verdikt():
    ex = {"eps": 0.0025, "L": 6, "wanted": 7, "capped": True, "rows": [], "best": 0, "two": 1.9e6, "mc": 4.3e6, "ml": 6.6e6, "ratio_two": 2.28, "ratio_ml": 0.65}
    kind, text = ev.verdict_two(ex)
    assert kind == "warning"
    for frag in ("Feine Ebene L = 6 (die Toleranz verlangt Ebene 7; gemessen wird höchstens bis 6)", "k = 0", "1,9e+06", "4,3e+06", "6,6e+06", "Gewöhnlich/Zwei-Ebenen = 2,28", "gewöhnlich/Multilevel = 0,65"):
        assert frag in text, frag
    assert ev.verdict_two({**ex, "capped": False, "ratio_ml": 1.5})[0] == "info"


def test_ein_term_verdikt():
    def cv(vs):
        return [{"lmax": 3 + i, "var": v, "cost": 2.0, "work": 2 * v} for i, v in enumerate(vs)]
    ex = {"curves": {1.0: cv([0.23, 0.25]), 1.5: cv([0.19, 0.21]), 2.0: cv([0.18, 0.27])}, "s_run": 1.5, "run": {"mean": 0.1647, "var": 0.221, "cost": 2.84, "n": 20000, "z": -0.3},
          "theory": {"var": 0.211, "cost": 2.84, "work": 0.6}, "truth": 0.1655, "lmax": 4, "eps": 0.005, "eps2_ml": 0.59, "eps2_mc": 5.07}
    kind, text = ev.verdict_single(ex)
    assert kind == "info"
    for frag in ("s = 1,0: 0,23 (Abschneideebene 3) → 0,25 (4)", "s = 2,0: 0,18 (Abschneideebene 3) → 0,27 (4)", "Mittel 0,1647 gegen exakt 0,1655 (−0,3 Standardfehler)", "Stichprobenvarianz 0,22", "erwartet 0,21", "0,60/ε² gegen Multilevel 0,59/ε²"):
        assert frag in text, frag
    wachsend = {**ex, "curves": {**ex["curves"], 1.5: cv([0.19, 0.5])}}
    kind, text = ev.verdict_single(wachsend)
    assert kind == "warning" and "taugt hier nicht" in text
