"""Handrechnung der Einheiten (DEMO-PLAYBOOK §1b): jede Zahl hier ist von Hand nachgerechnet, nicht aus dem Code gelesen."""

import math

import numpy as np
import pytest

import mlm_algorithm as A
from conftest import fixed

# Mini-Instanz: σ = 1, a = 0, T = 1, q = 1,5; Ebene 2 (vier Schritte, h = 1/4, √h = 1/2) mit festen Normalzahlen z = (1, 1, −1, 1).
Z = (1.0, 1.0, -1.0, 1.0)
MINI = dict(a=0.0, sigma=1.0, q=1.5)


def _pair(scheme, kind):
    p, pc = A.level_pair(scheme, kind, 2, 1, 0, zfun=fixed(Z), **MINI)
    return float(p[0]), float(pc[0])


def test_euler_bahn_von_hand():
    # fein: 1·1,5 = 1,5; ·1,5 = 2,25; ·0,5 = 1,125; ·1,5 = 1,6875. grob (Zuwächse 1 und 0): 1·2 = 2; ·1 = 2.
    assert _pair("euler", "glatt") == pytest.approx((1.6875, 2.0))
    assert _pair("euler", "knick") == pytest.approx((0.1875, 0.5))           # (1,6875 − 1,5)⁺ und (2 − 1,5)⁺
    assert _pair("euler", "sprung") == pytest.approx((1.0, 1.0))             # beide über der Schwelle: Y = 0


def test_milstein_grobe_bahn_trifft_die_feine_von_hand():
    # feine Schritte: ΔW² = ¼ = h, die Korrektur ½σ²(ΔW² − h) ist null (Euler-Werte 1,5; 2,25; 1,125; 1,6875).
    # grobe Schritte (h = ½): 1·(1 + 1 + ½(1 − ½)) = 2,25; dann 2,25·(1 + 0 + ½(0 − ½)) = 1,6875: P_1 = P_2, also Y_2 = 0.
    p, pc = _pair("milstein", "glatt")
    assert p == pytest.approx(1.6875) and pc == pytest.approx(1.6875)


def test_ein_schritt_von_hand():
    assert A.step("euler", np.array([2.0]), 0.25, np.array([0.5]), 0.1, 1.0)[0] == pytest.approx(2.0 * (1 + 0.025 + 0.5))                    # 3,05
    assert A.step("milstein", np.array([2.0]), 0.25, np.array([1.0]), 0.0, 1.0)[0] == pytest.approx(2.0 * (1 + 0 + 1.0 + 0.5 * (1.0 - 0.25)))   # 4,75


def test_kosten_je_probe():
    assert [A.cost_per_sample(l) for l in range(4)] == [1.0, 3.0, 6.0, 12.0]


def test_optimale_zuteilung_von_hand():
    # V = (1, ½, ¼), C = (1, 3, 6): √(V C) = (1; 1,2247; 1,2247), Summe 3,4495; N_l = 2 ε⁻² · 3,4495 · √(V/C) = ε⁻² (6,899; 2,816; 1,408); Kosten 2 ε⁻² · 3,4495² = 23,80 ε⁻².
    ns, cost = A.optimal_allocation([1.0, 0.5, 0.25], [1.0, 3.0, 6.0], 1.0)
    assert ns == pytest.approx([6.899, 2.816, 1.408], abs=1e-3)
    assert cost == pytest.approx(23.80, abs=1e-2) and sum(n * c for n, c in zip(ns, [1.0, 3.0, 6.0])) == pytest.approx(cost)
    assert A.optimal_allocation([1.0, 0.5, 0.25], [1.0, 3.0, 6.0], 0.1)[1] == pytest.approx(100 * cost)          # ε⁻²


def test_e_p0_glatt_von_hand():
    assert A.e_p0("euler", "glatt", 0.125, 0.5, 1.3) == pytest.approx(1.125)
    assert A.e_p0("milstein", "glatt", 0.125, 0.5, 1.3) == pytest.approx(1.125)          # E[ΔW² − h] = 0


def test_e_p0_sprung_euler_von_hand():
    # S = 1,125 + 0,5 z > 1,3 ⇔ z > 0,35: P = 1 − Φ(0,35) = Φ(−0,35)
    assert A.e_p0("euler", "sprung", 0.125, 0.5, 1.3) == pytest.approx(A.Phi(-0.35))
    assert A.Phi(-0.35) == pytest.approx(0.36317, abs=1e-5)


def test_e_p0_milstein_sprung_von_hand():
    # S = 1 + a + σ z + ½σ²(z² − 1) = 1 + 0,5 z + 0,125 z² > 1,3 ⇔ 0,125 z² + 0,5 z − 0,3 > 0 ⇔ z² + 4z − 2,4 > 0: Nullstellen z = −2 ± √(4 + 2,4) = −2 ± 2,5298
    z1, z2 = -2 - math.sqrt(6.4), -2 + math.sqrt(6.4)
    assert A.e_p0("milstein", "sprung", 0.125, 0.5, 1.3) == pytest.approx(A.Phi(z1) + 1 - A.Phi(z2))
    assert z2 == pytest.approx(0.5298, abs=1e-4)


def test_truth_von_hand():
    # a = σ²/2: μ = 0; E S = e^{σ²/2} = e^{0,125}; P(S > 1,3) = 1 − Φ(ln 1,3/0,5) = 1 − Φ(0,5247)
    assert A.truth("glatt", 0.125, 0.5, 1.3) == pytest.approx(math.exp(0.125))
    assert A.truth("sprung", 0.125, 0.5, 1.3) == pytest.approx(1 - A.Phi(math.log(1.3) / 0.5))
    assert A.truth("sprung", 0.125, 0.5, 1.3) == pytest.approx(0.2999, abs=1e-4)


def test_ein_term_wahrscheinlichkeiten_von_hand():
    p = A.single_term_probs(1.0, 2)           # (1, ½, ¼)/(7/4)
    assert p == pytest.approx([4 / 7, 2 / 7, 1 / 7]) and p.sum() == pytest.approx(1.0)


def test_strom_ist_reproduzierbar_und_verschieden():
    assert A.stream_seed(3, 4) == A.stream_seed(3, 4) and A.stream_seed(3, 4) != A.stream_seed(4, 3)
    u = A.uniforms(A.stream_seed(1), 10000)
    assert u.min() > 0 and u.max() < 1
    assert np.array_equal(A.normals(5, 100), A.normals(5, 100)) and not np.array_equal(A.normals(5, 100), A.normals(6, 100))


def test_normalzahlen_haben_mittel_null_und_varianz_eins():
    z = A.normals(A.stream_seed(2), 200000)
    assert abs(z.mean()) < 0.01 and z.var() == pytest.approx(1.0, abs=0.01)
