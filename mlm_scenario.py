"""Das Szenario der Spielwiese: eigene Drift a, Streuung σ und Schwelle q des Bestandsprozesses dS = a S dt + σ S dW, S(0) = 1, T = 1."""

from __future__ import annotations

from dataclasses import dataclass

import mlm_constants as C


@dataclass(frozen=True)
class Szenario:
    drift: float
    sigma: float
    q: float


def standard_szenario(sigma: float) -> Szenario:
    """Die Voreinstellung: a = σ²/2, so dass S(T) = exp(σ W(T)) die log-normale Nachfrage mit μ = 0 ist (wie in Stück 11), Schwelle q = 1,3."""
    return Szenario(sigma**2 / 2, sigma, C.Q)


def check_szenario(drift, sigma, q) -> Szenario:
    """Prüft die Eingaben der Spielwiese; ValueError mit lesbarer Meldung."""
    for name, x in (("Drift", drift), ("Streuung σ", sigma), ("Schwelle q", q)):
        if x is None or x != x or x in (float("inf"), float("-inf")):
            raise ValueError(f"{name} ist keine Zahl.")
    if not C.CUSTOM_DRIFT_MIN <= drift <= C.CUSTOM_DRIFT_MAX:
        raise ValueError(f"Die Drift muss zwischen {C.CUSTOM_DRIFT_MIN} und {C.CUSTOM_DRIFT_MAX} liegen.".replace(".", ","))
    if not C.CUSTOM_SIGMA_MIN <= sigma <= C.CUSTOM_SIGMA_MAX:
        raise ValueError(f"Die Streuung σ muss zwischen {C.CUSTOM_SIGMA_MIN} und {C.CUSTOM_SIGMA_MAX} liegen.".replace(".", ","))
    if not C.CUSTOM_Q_MIN <= q <= C.CUSTOM_Q_MAX:
        raise ValueError(f"Die Schwelle q muss zwischen {C.CUSTOM_Q_MIN} und {C.CUSTOM_Q_MAX} liegen.".replace(".", ","))
    return Szenario(float(drift), float(sigma), float(q))
