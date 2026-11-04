"""Kennzahlen, Verdikt-Logik und Experimente über feste Ströme (reine Funktionen, kein Streamlit).

Gemessen wird, was ein Multilevel-Schätzer gegenüber gewöhnlichem Monte Carlo spart. Kosten zählen **Zeitschritte** (nicht Laufzeit); Ziel ist ein mittlerer quadratischer Fehler ≤ ε²
(Verzerrung² ≤ ε²/2 und Varianz ≤ ε²/2). Die exakten Werte sind geschlossen bekannt (log-normaler Endbestand); die Schätzer kennen sie nicht.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import mlm_algorithm as A
import mlm_constants as C
from mlm_scenario import Szenario, standard_szenario


def de(x: float, nd: int = 2) -> str:
    """Zahl im deutschen Format: Punkt als Tausendertrenner, Komma als Dezimalzeichen, echtes Minuszeichen."""
    return f"{x:,.{nd}f}".translate(str.maketrans(",.", ".,")).replace("-", "−")


def dn(x: float, nd: int = 2) -> str:
    """Wie `de`, aber „nicht bestimmbar“ für NaN und „unendlich“ für ∞."""
    if np.isnan(x):
        return "nicht bestimmbar"
    if np.isinf(x):
        return "unendlich"
    return de(x, nd)


def de_e(x: float) -> str:
    """Wissenschaftliche Schreibweise mit Dezimalkomma, z. B. 4,2e-10."""
    return f"{x:.1e}".replace(".", ",")


def ke(x: float) -> str:
    """Kosten in Zeitschritten: ganze Zahl bis 99 999, darüber wissenschaftlich."""
    if np.isnan(x):
        return "nicht bestimmbar"
    return de(x, 0) if abs(x) < 1e5 else de_e(x)


def rf(x: float) -> str:
    """Relativer Fehler: vier Nachkommastellen unter 0,1, sonst drei; ab 10 mit zwei."""
    if np.isnan(x):
        return "nicht bestimmbar"
    if np.isinf(x):
        return "unendlich"
    return de(x, 4 if abs(x) < 0.1 else 3 if abs(x) < 10 else 2)


def prozent(x: float, nd: int = 0) -> str:
    return "nicht bestimmbar" if np.isnan(x) else f"{de(x * 100, nd)} %"


def eps_text(eps: float) -> str:
    return f"{eps:g}".replace(".", ",")


@dataclass(frozen=True)
class Settings:
    ausgabe: str = C.DEFAULT_AUSGABE
    verfahren: str = C.DEFAULT_VERFAHREN
    sigma: float = C.DEFAULT_SIGMA
    eps: float = C.DEFAULT_EPS
    lmax: int = C.DEFAULT_LMAX
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Analysis:
    label: str
    scheme: str
    kind: str
    drift: float
    sigma: float
    q: float
    truth: float
    lmax: int
    n: int                                # Proben je Ebene
    e0: float                             # exakter Erwartungswert von P_0
    levels: tuple                         # je Ebene {l, mean, var, var_p, kurt, nonzero, mean_p} (gekoppelt)
    var_unc: tuple                        # je Ebene Var(Y_l) ungekoppelt
    c: float                              # Verzerrungskonstante: |E[P_L] − exakt| ≈ c · 2^(−L)
    beta: float                           # Rate der Ebenenvarianz, gekoppelt: V_l ∝ 2^(−β l)
    beta_unc: float


def label_of(settings: Settings, sc: Szenario | None = None) -> str:
    extra = "" if sc is None or sc == standard_szenario(sc.sigma) else f", a = {de(sc.drift, 2)}, q = {de(sc.q, 2)}"
    return f"{C.VERFAHREN[settings.verfahren]}, {C.AUSGABEN[settings.ausgabe]}{extra}"


# --- Rate und Verzerrungskonstante ---------------------------------------------------------------------------------------------------------

def fit_rate(variances, first: int = C.FIT_FROM) -> float:
    """β aus der Regressionsgeraden von log₂ V_l gegen l über die Ebenen ab `first` (nur positive Varianzen); NaN, wenn weniger als zwei Punkte."""
    pts = [(l, math.log2(v)) for l, v in enumerate(variances) if l >= first and v > 0]
    if len(pts) < 2:
        return float("nan")
    return float(-np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0])


def bias_const(levels, first: int = C.FIT_FROM, last: int = 6) -> float:
    """c aus dem Median von |E[Y_l]|·2^l über die Ebenen first … min(last, größte Ebene) (schwache Ordnung 1: die Verzerrung halbiert sich je Ebene)."""
    vals = [abs(r["mean"]) * 2.0 ** r["l"] for r in levels if first <= r["l"] <= last]
    return float(np.median(vals)) if vals else float("nan")


def bias_at(a: Analysis, level: int) -> float:
    """E[P_l] − exakt aus E[P_0] (geschlossen) plus den gemessenen Ebenenmitteln."""
    return a.e0 + sum(r["mean"] for r in a.levels[1 : level + 1]) - a.truth


# --- Messung ----------------------------------------------------------------------------------------------------------------------------------

def analyse(settings: Settings, sc: Szenario | None = None) -> Analysis:
    """Ebenenstatistik der Ebenen 0 bis lmax: gekoppelt (Mittel, Varianz, Kurtosis) und ungekoppelt (Varianz); N_LEVEL Proben je Ebene."""
    sc = sc if sc is not None else standard_szenario(settings.sigma)
    levels, var_unc = [], []
    for l in range(settings.lmax + 1):
        m, v, vp, kurt, nz, mp = A.level_stats(settings.verfahren, settings.ausgabe, l, C.N_LEVEL, A.stream_seed(settings.seed, 1), sc.drift, sc.sigma, sc.q, True)
        levels.append({"l": l, "mean": m, "var": v, "var_p": vp, "kurt": kurt, "nonzero": nz, "mean_p": mp})
        var_unc.append(v if l == 0 else A.level_stats(settings.verfahren, settings.ausgabe, l, C.N_LEVEL, A.stream_seed(settings.seed, 2), sc.drift, sc.sigma, sc.q, False)[1])
    return Analysis(label_of(settings, sc), settings.verfahren, settings.ausgabe, sc.drift, sc.sigma, sc.q, A.truth(settings.ausgabe, sc.drift, sc.sigma, sc.q), settings.lmax, C.N_LEVEL,
                    A.e_p0(settings.verfahren, settings.ausgabe, sc.drift, sc.sigma, sc.q), tuple(levels), tuple(var_unc), bias_const(levels), fit_rate([r["var"] for r in levels]),
                    fit_rate(var_unc))


# --- Kosten (idealisiert) -------------------------------------------------------------------------------------------------------------------

def level_for(a: Analysis, eps: float) -> int:
    """Kleinste Ebene L ≥ 1 mit c·2^(−L) ≤ ε/√2 (Verzerrung² ≤ ε²/2)."""
    if not a.c > 0:
        return 1
    return max(1, min(40, math.ceil(math.log2(a.c / (eps / math.sqrt(2))))))


def level_variance(a: Analysis, l: int) -> float:
    """V_l gemessen bis zur größten Ebene, darüber aus der Rate β fortgesetzt (β nicht kleiner als 0)."""
    if l <= a.lmax:
        return a.levels[l]["var"]
    beta = a.beta if np.isfinite(a.beta) else 1.0
    return a.levels[a.lmax]["var"] * 2.0 ** (-max(beta, 0.0) * (l - a.lmax))


def costs_for_level(a: Analysis, eps: float, L: int) -> dict:
    """Idealisierte Kosten bei bekannten V_l und bekannter Verzerrung: gewöhnlich 2 Var(P_L) ε⁻² 2^L; Multilevel 2 ε⁻² (Σ √(V_l C_l))²; Anteile je Ebene."""
    vl = [level_variance(a, l) for l in range(L + 1)]
    cl = [A.cost_per_sample(l) for l in range(L + 1)]
    s = sum(math.sqrt(v * c) for v, c in zip(vl, cl))
    var_p = a.levels[min(L, a.lmax)]["var_p"]
    mc = 2.0 * var_p / eps**2 * 2.0**L
    ml = A.optimal_allocation(vl, cl, eps)[1]
    shares = [math.sqrt(v * c) / s if s > 0 else float("nan") for v, c in zip(vl, cl)]
    return {"eps": eps, "L": L, "mc": mc, "ml": ml, "ratio": mc / ml if ml > 0 else float("inf"), "eps2_mc": eps**2 * mc, "eps2_ml": eps**2 * ml, "shares": shares}


def cost_row(a: Analysis, eps: float) -> dict:
    return costs_for_level(a, eps, level_for(a, eps))


def cost_table(a: Analysis) -> list[dict]:
    return [cost_row(a, e) for e in C.EPS_GRID]


# --- Experiment 3: gelaufene Schätzungen ------------------------------------------------------------------------------------------------------

def run_experiment(settings: Settings, a: Analysis | None = None, sc: Szenario | None = None) -> dict:
    """Giles' adaptives Verfahren gegen gewöhnliches Monte Carlo (mit Orakelwissen über Ebene und Varianz) bei der Toleranz ε, RUN_REPS Wiederholungen mit festen Strömen."""
    sc = sc if sc is not None else standard_szenario(settings.sigma)
    a = a if a is not None else analyse(settings, sc)
    eps = settings.eps
    truth = a.truth
    L_std = level_for(a, eps)
    var_p = a.levels[min(L_std, a.lmax)]["var_p"]
    err_ml, cost_ml, levels_ml, err_mc, cost_mc = [], [], [], [], []
    for r in range(C.RUN_REPS):
        est, cost, L, _ = A.mlmc(settings.verfahren, settings.ausgabe, eps, A.stream_seed(settings.seed, 3, r, int(round(eps * 1e6))), sc.drift, sc.sigma, sc.q, n0=C.PILOT)
        err_ml.append(est - truth), cost_ml.append(cost), levels_ml.append(L)
        est2, cost2 = A.mc_run(settings.verfahren, settings.ausgabe, eps, L_std, var_p, A.stream_seed(settings.seed, 4, r, int(round(eps * 1e6))), sc.drift, sc.sigma, sc.q)
        err_mc.append(est2 - truth), cost_mc.append(cost2)
    rm = lambda e: float(np.sqrt(np.mean(np.square(e))))
    return {"eps": eps, "truth": truth, "reps": C.RUN_REPS, "pilot": C.PILOT, "L_std": L_std,
            "ml": {"rmse": rm(err_ml), "cost": float(np.mean(cost_ml)), "L_mean": float(np.mean(levels_ml)), "L_min": int(min(levels_ml)), "L_max": int(max(levels_ml)), "over": float(np.mean(np.abs(err_ml) > eps))},
            "mc": {"rmse": rm(err_mc), "cost": float(np.mean(cost_mc)), "over": float(np.mean(np.abs(err_mc) > eps))},
            "ratio": float(np.mean(cost_mc) / np.mean(cost_ml))}


# --- Experiment 4: zwei Ebenen gegen viele -----------------------------------------------------------------------------------------------------

def two_level_cost(v0: float, c0: float, v1: float, c1: float, eps: float) -> float:
    """Kosten eines Zwei-Ebenen-Schätzers (grobe Ebene mit Var v0 und Kosten c0, Differenz mit Var v1 und Kosten c1) für MSE ε²: 2 ε⁻² (√(v0 c0) + √(v1 c1))²."""
    return 2.0 / eps**2 * (math.sqrt(v0 * c0) + math.sqrt(v1 * c1)) ** 2


def two_level_experiment(settings: Settings, a: Analysis | None = None, sc: Szenario | None = None) -> dict:
    """Var(P_L − P_k) für alle groben Ebenen k < L (gekoppelt, N_PAIR Proben) und die Kosten des Zwei-Ebenen-Schätzers; L aus der Toleranz, höchstens die größte gemessene Ebene."""
    sc = sc if sc is not None else standard_szenario(settings.sigma)
    a = a if a is not None else analyse(settings, sc)
    eps = settings.eps
    wanted = level_for(a, eps)
    L = min(wanted, a.lmax)
    rows = []
    for k in range(L):
        p, pc = A.pair_general(settings.verfahren, settings.ausgabe, L, k, C.N_PAIR, A.stream_seed(settings.seed, 6, L, k), sc.drift, sc.sigma, sc.q)
        v1 = float((p - pc).var())
        v0 = a.levels[k]["var_p"]
        rows.append({"k": k, "var": v1, "cost": two_level_cost(v0, 2.0**k, v1, 2.0**L + 2.0**k, eps)})
    best = min(rows, key=lambda r: r["cost"])
    base = costs_for_level(a, eps, L)
    return {"eps": eps, "L": L, "wanted": wanted, "capped": wanted > L, "rows": rows, "best": best["k"], "two": best["cost"], "mc": base["mc"], "ml": base["ml"],
            "ratio_two": base["mc"] / best["cost"], "ratio_ml": base["ratio"]}


# --- Experiment 5: entzerrter Ein-Term-Schätzer -------------------------------------------------------------------------------------------------

def single_term_theory(a: Analysis, s: float, lmax: int) -> tuple[float, float, float]:
    """(Varianz, Kosten je Probe, Varianz·Kosten) des Ein-Term-Schätzers mit p_l ∝ 2^(−s l), abgeschnitten bei lmax, aus den gemessenen zweiten Momenten."""
    p = A.single_term_probs(s, lmax)
    m2 = [r["var"] + r["mean"] ** 2 for r in a.levels[: lmax + 1]]
    mean = sum(r["mean"] for r in a.levels[: lmax + 1])
    var = sum(m / pl for m, pl in zip(m2, p)) - mean**2
    cost = sum(pl * A.cost_per_sample(l) for l, pl in enumerate(p))
    return float(var), float(cost), float(var * cost)


def single_term_experiment(settings: Settings, a: Analysis | None = None, sc: Szenario | None = None) -> dict:
    sc = sc if sc is not None else standard_szenario(settings.sigma)
    a = a if a is not None else analyse(settings, sc)
    curves = {}
    for s in C.SINGLE_S:
        curves[s] = [{"lmax": lm, **dict(zip(("var", "cost", "work"), single_term_theory(a, s, lm)))} for lm in range(3, a.lmax + 1)]
    s_run = 1.5
    mean, var, cost = A.single_term(settings.verfahren, settings.ausgabe, s_run, a.lmax, C.N_SINGLE, A.stream_seed(settings.seed, 7), sc.drift, sc.sigma, sc.q)
    theo = single_term_theory(a, s_run, a.lmax)
    ml = cost_row(a, settings.eps)
    return {"curves": curves, "s_run": s_run, "run": {"mean": mean, "var": var, "cost": cost, "n": C.N_SINGLE, "z": (mean - a.truth) / math.sqrt(var / C.N_SINGLE)},
            "theory": {"var": theo[0], "cost": theo[1], "work": theo[2]}, "truth": a.truth, "lmax": a.lmax, "eps": settings.eps, "eps2_ml": ml["eps2_ml"], "eps2_mc": ml["eps2_mc"]}


# --- Verdikte -----------------------------------------------------------------------------------------------------------------------------

def _rate_text(beta: float) -> str:
    if not np.isfinite(beta):
        return "nicht bestimmbar"
    return de(beta, 2)


def regime(beta: float) -> str:
    """Welcher der drei Kostenfälle (Kostenrate γ = 1 je Ebene)? Schwellen mit Spielraum für das Rauschen der Regression."""
    if not np.isfinite(beta):
        return "nicht bestimmbar"
    if beta > 1.25:
        return "β > γ: die Kosten sättigen bei ε²·Kosten (Ordnung ε⁻²)"
    if beta >= 0.8:
        return "β ≈ γ: ε²·Kosten wächst langsam (Logarithmen)"
    return "β < γ: die Kosten liegen zwischen ε⁻² und gewöhnlichem Monte Carlo"


def degenerate(a: Analysis) -> bool:
    """Die Ausgabe ist auf allen gemessenen Ebenen konstant (etwa eine Schwelle, die nie überschritten wird): es gibt nichts zu schätzen."""
    return all(r["var_p"] == 0 for r in a.levels)


NICHTS_ZU_SCHAETZEN = "Die Ausgabe ist auf allen gemessenen Ebenen konstant (keine Streuung, etwa weil die Schwelle nie überschritten wird): es gibt nichts zu schätzen und kein Kostenverhältnis. Schwelle oder Streuung ändern."


def verdict_main(a: Analysis, settings: Settings) -> tuple[str, str]:
    """Kernfrage: Lohnt Multilevel bei dieser Wahl, und wie viel?"""
    if degenerate(a):
        return "info", f"{a.label}: " + NICHTS_ZU_SCHAETZEN
    row = cost_row(a, settings.eps)
    last = a.lmax
    ratio_unc = a.var_unc[last] / a.var_unc[last - 1] if a.var_unc[last - 1] > 0 else float("nan")
    ratio_c = a.levels[last]["var"] / a.levels[last - 1]["var"] if a.levels[last - 1]["var"] > 0 else float("nan")
    head = (f"{a.label}, σ = {de(a.sigma, 2)}, Toleranz ε = {eps_text(settings.eps)}: gewöhnliches Monte Carlo braucht die Ebene L = {row['L']} (Verzerrung ≈ {de(a.c, 4)}·2^(−L) ≤ ε/√2) und kostet idealisiert {ke(row['mc'])} Zeitschritte, "
            f"Multilevel {ke(row['ml'])}: gewöhnlich/Multilevel = {dn(row['ratio'], 2)}. Die Ebenenvarianz fällt mit der Rate β ≈ {_rate_text(a.beta)} ({regime(a.beta)}). "
            f"Ohne Kopplung der groben Bahn bleibt sie bei {de_e(a.var_unc[last])} (Verhältnis zur Vorebene {dn(ratio_unc, 2)} statt {dn(ratio_c, 2)}).")
    if row["ratio"] < 1.0:
        return "warning", head + " Multilevel ist hier selbst im Idealfall teurer als gewöhnliches Monte Carlo."
    if row["ratio"] >= 2.0:
        return "success", head + " Multilevel spart hier mindestens die Hälfte der Kosten."
    return "info", head + " Der Vorteil ist hier klein."


def verdict_levels(a: Analysis) -> tuple[str, str]:
    if degenerate(a):
        return "info", NICHTS_ZU_SCHAETZEN
    last = a.lmax
    r_c = a.levels[last]["var"] / a.levels[last - 1]["var"] if a.levels[last - 1]["var"] > 0 else float("nan")
    r_u = a.var_unc[last] / a.var_unc[last - 1] if a.var_unc[last - 1] > 0 else float("nan")
    kurt = max((r["kurt"] for r in a.levels[1:] if np.isfinite(r["kurt"])), default=float("nan"))
    return "info", (f"Gekoppelt fällt die Varianz V_l der Differenz von Ebene {last - 1} auf {last} auf das {de(r_c, 2)}-Fache (Rate β ≈ {_rate_text(a.beta)}, aus den Ebenen {C.FIT_FROM} bis {last}); ungekoppelt auf das {dn(r_u, 2)}-Fache (β ≈ {_rate_text(a.beta_unc)}): "
                    f"V_{last} = {de_e(a.levels[last]['var'])} gegen {de_e(a.var_unc[last])}. Die Verzerrung halbiert sich je Ebene (≈ {de(a.c, 4)}·2^(−L)). Größte Kurtosis der Differenz: {dn(kurt, 0)} (je größer, desto unsicherer die Schätzung von V_l aus {de(a.n, 0)} Proben).")


def verdict_cost(a: Analysis) -> tuple[str, str]:
    if degenerate(a):
        return "info", NICHTS_ZU_SCHAETZEN
    rows = cost_table(a)
    first, lastr = rows[0], rows[-1]
    tail = []                                          # zusammenhängend von der engsten Toleranz her, in der Multilevel billiger ist
    for r in reversed(rows):
        if r["ratio"] >= 1.0:
            tail.append(r)
        else:
            break
    if len(tail) == len(rows):
        wechsel = "Multilevel ist bei allen gezeigten Toleranzen billiger"
    elif tail:
        wechsel = f"Multilevel ist ab ε = {eps_text(tail[-1]['eps'])} und enger billiger"
    else:
        wechsel = "Multilevel ist bei der engsten Toleranz noch nicht billiger"
    return "info", (f"Gewöhnlich/Multilevel bei ε = {eps_text(first['eps'])}: {dn(first['ratio'], 2)}, bei ε = {eps_text(lastr['eps'])}: {dn(lastr['ratio'], 2)}; {wechsel}. "
                    f"ε²·Kosten: Multilevel {de(first['eps2_ml'], 2)} → {de(lastr['eps2_ml'], 2)}, gewöhnlich {de(first['eps2_mc'], 2)} → {de(lastr['eps2_mc'], 2)} (bei Sättigung bliebe es konstant).")


def verdict_run(ex: dict) -> tuple[str, str]:
    ml, mc = ex["ml"], ex["mc"]
    text = (f"Toleranz ε = {eps_text(ex['eps'])}, {ex['reps']} Wiederholungen: Multilevel (Giles, Pilot {de(ex['pilot'], 0)} je Ebene) erreicht RMSE/ε = {de(ml['rmse'] / ex['eps'], 2)} bei mittleren Kosten {ke(ml['cost'])} Zeitschritten "
            f"(Ebenen {ml['L_min']} bis {ml['L_max']}, Mittel {de(ml['L_mean'], 1)}; Fehler über ε in {prozent(ml['over'], 0)} der Läufe); gewöhnliches Monte Carlo mit Orakelwissen (Ebene {ex['L_std']}) RMSE/ε = {de(mc['rmse'] / ex['eps'], 2)}, Kosten {ke(mc['cost'])}. "
            f"Kostenverhältnis gewöhnlich/Multilevel: {dn(ex['ratio'], 2)}.")
    return ("warning" if ex["ratio"] < 1.0 else "info"), text


def verdict_two(ex: dict) -> tuple[str, str]:
    cap = f" (die Toleranz verlangt Ebene {ex['wanted']}; gemessen wird höchstens bis {ex['L']})" if ex["capped"] else ""
    text = (f"Feine Ebene L = {ex['L']}{cap}, Toleranz ε = {eps_text(ex['eps'])}: der beste Zwei-Ebenen-Schätzer (grobe Ebene k = {ex['best']}) kostet {ke(ex['two'])} Zeitschritte, gewöhnliches Monte Carlo {ke(ex['mc'])}, Multilevel {ke(ex['ml'])} "
            f"(idealisiert). Gewöhnlich/Zwei-Ebenen = {dn(ex['ratio_two'], 2)}, gewöhnlich/Multilevel = {dn(ex['ratio_ml'], 2)}.")
    return ("warning" if ex["ratio_ml"] < 1.0 <= ex["ratio_two"] else "info"), text


def verdict_single(ex: dict) -> tuple[str, str]:
    lm = ex["lmax"]
    parts = []
    for s in C.SINGLE_S:
        cv = ex["curves"][s]
        parts.append(f"s = {de(s, 1)}: {de(cv[0]['var'], 2)} (Abschneideebene {cv[0]['lmax']}) → {de(cv[-1]['var'], 2)} ({cv[-1]['lmax']})")
    run, th = ex["run"], ex["theory"]
    text = (f"Varianz des Ein-Term-Schätzers bei wachsender Abschneideebene: " + "; ".join(parts) + ". "
            f"Lauf mit s = {de(ex['s_run'], 1)}, Abschneideebene {lm}, {de(run['n'], 0)} Proben: Mittel {de(run['mean'], 4)} gegen exakt {de(ex['truth'], 4)} ({de(run['z'], 1)} Standardfehler), Stichprobenvarianz {de(run['var'], 2)} "
            f"(aus den Ebenenmomenten erwartet {de(th['var'], 2)}), Kosten je Probe {de(run['cost'], 2)} Zeitschritte. Varianz·Kosten/ε² = {de(th['work'], 2)}/ε² gegen Multilevel {de(ex['eps2_ml'], 2)}/ε² (gewöhnlich {de(ex['eps2_mc'], 2)}/ε²; Toleranz ε = {eps_text(ex['eps'])}).")
    growing = ex["curves"][1.5][-1]["var"] > 1.5 * ex["curves"][1.5][0]["var"]
    if growing:
        text += " Die Varianz wächst mit der Abschneideebene: der Schätzer taugt hier nicht, und auch die Stichprobenvarianz im Lauf ist wegen der schweren Ränder unzuverlässig."
    return ("warning" if growing else "info"), text
