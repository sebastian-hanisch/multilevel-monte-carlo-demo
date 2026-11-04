"""Gemeinsame Fixtures: die Mini-Instanz mit festen Zufallszahlen, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

STD = dict(a=0.125, sigma=0.5, q=1.3)          # Voreinstellung: a = σ²/2, so dass S(T) = exp(σ W(T)) log-normal mit μ = 0 ist


@pytest.fixture(scope="session")
def std():
    return dict(STD)


def fixed(values):
    """Zufallszahlen-Ersatz für level_pair: Schritt i liefert für alle Proben den festen Wert values[i]."""
    return lambda i, n: np.full(n, float(values[i]))
