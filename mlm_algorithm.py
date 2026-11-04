"""Kern der Demo (nur numpy und math).

Jede Regel ist eine eigene Funktion mit expliziten Ein- und Ausgaben (DEMO-PLAYBOOK §1b):
  Zufallsstrom    stream_seed, uniforms, normals          SplitMix64, zählerbasiert; Standardnormale per Box–Muller
  Ausgaben        payoff, truth, e_p0                      glatt S(T), Knick (S(T) − q)⁺, Sprung 1[S(T) > q]; exakte Werte (Black-Scholes-Form) und E[P_0] in geschlossener Form
  Ebenen          step, level_pair, pair_general           Euler–Maruyama und Milstein; feine und grobe Bahn aus denselben Zuwächsen (gekoppelt) oder aus eigenen (ungekoppelt)
  Kosten          cost_per_sample                          Zeitschritte je Probe
  Schätzer        level_stats, mlmc, mc_run, single_term   Ebenenstatistik, Giles' adaptives Verfahren, gewöhnliches Monte Carlo, entzerrter Ein-Term-Schätzer

Der Prozess: dS = a S dt + σ S dW, S(0) = 1, T = 1 (geometrische Brownsche Bewegung). Seine exakte Lösung dient nur als Orakel; die Schätzer lösen die Gleichung mit 2^l Zeitschritten (Ebene l).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

GAMMA = 0x9E3779B97F4A7C15
MASK = (1 << 64) - 1
T = 1.0


# --- Zufallsstrom ---------------------------------------------------------------------------------------------------------------------------

def mix_int(z):
    z &= MASK
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
    return (z ^ (z >> 31)) & MASK


def _mix_arr(z):
    with np.errstate(over="ignore"):
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def stream_seed(*keys):
    s = 0x1234_5678_9ABC_DEF0
    for k in keys:
        s = mix_int(s ^ mix_int(int(k) + GAMMA))
    return s


def uniforms(seed, count):
    """count gleichverteilte Zahlen in (0, 1), nie 0 oder 1."""
    idx = np.arange(1, count + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        z = _mix_arr(np.uint64(seed) + idx * np.uint64(GAMMA))
    return ((z >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / (1 << 53))


def normals(seed, count):
    """count Standardnormalzahlen (Box–Muller, zwei gleichverteilte je zwei Normale)."""
    n2 = (count + 1) // 2
    u = uniforms(seed, 2 * n2)
    r = np.sqrt(-2.0 * np.log(u[:n2]))
    z = np.concatenate([r * np.cos(2 * np.pi * u[n2:]), r * np.sin(2 * np.pi * u[n2:])])
    return z[:count]


# --- Normalverteilung und Ausgaben ----------------------------------------------------------------------------------------------------------

def Phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def payoff(kind: str, S, q: float):
    if kind == "glatt":
        return S
    if kind == "knick":
        return np.maximum(S - q, 0.0)
    if kind == "sprung":
        return (S > q).astype(float)
    raise ValueError(kind)


def truth(kind: str, a: float, sigma: float, q: float) -> float:
    """Exakter Erwartungswert von g(S(T)); ln S(T) ~ N((a − σ²/2) T, σ² T)."""
    mu = (a - sigma**2 / 2) * T
    s = sigma * math.sqrt(T)
    if kind == "glatt":
        return math.exp(mu + s**2 / 2)
    d2 = (mu - math.log(q)) / s
    d1 = d2 + s
    if kind == "knick":
        return math.exp(mu + s**2 / 2) * Phi(d1) - q * Phi(d2)
    if kind == "sprung":
        return Phi(d2)
    raise ValueError(kind)


def e_p0(scheme: str, kind: str, a: float, sigma: float, q: float) -> float:
    """Exakter Erwartungswert von P_0 (ein Zeitschritt h = T = 1) in geschlossener Form.

    Euler: S = 1 + a + σ z. Milstein: S = c0 + σ z + c z² mit c = σ²/2, c0 = 1 + a − c (quadratisch in z; Nullstellen von S − q trennen die Bereiche)."""
    if scheme == "euler":
        m = 1.0 + a
        d = (m - q) / sigma
        return {"glatt": m, "knick": (m - q) * Phi(d) + sigma * phi(d), "sprung": Phi(d)}[kind]
    c = 0.5 * sigma**2
    c0 = 1.0 + a - c
    if kind == "glatt":
        return 1.0 + a
    d = c0 - q
    disc = sigma**2 - 4.0 * c * d
    if disc <= 0.0:                        # S ≥ q für alle z
        return (1.0 + a - q) if kind == "knick" else 1.0
    z1, z2 = (-sigma - math.sqrt(disc)) / (2 * c), (-sigma + math.sqrt(disc)) / (2 * c)
    if kind == "sprung":
        return Phi(z1) + 1.0 - Phi(z2)
    lo = c * (Phi(z1) - z1 * phi(z1)) - sigma * phi(z1) + d * Phi(z1)
    hi = c * (1.0 - Phi(z2) + z2 * phi(z2)) + sigma * phi(z2) + d * (1.0 - Phi(z2))
    return lo + hi


# --- Ebenen ---------------------------------------------------------------------------------------------------------------------------------

def step(scheme: str, S, h: float, dW, a: float, sigma: float):
    """Ein Zeitschritt: Euler S (1 + a h + σ ΔW), Milstein dazu ½ σ² (ΔW² − h)."""
    if scheme == "euler":
        return S * (1.0 + a * h + sigma * dW)
    if scheme == "milstein":
        return S * (1.0 + a * h + sigma * dW + 0.5 * sigma**2 * (dW**2 - h))
    raise ValueError(scheme)


def cost_per_sample(level: int) -> float:
    """Zeitschritte je Probe der Ebene l: feine Bahn 2^l plus grobe Bahn 2^(l−1); Ebene 0 hat nur einen Schritt."""
    return 1.0 if level == 0 else 2.0**level + 2.0 ** (level - 1)


def level_pair(scheme: str, kind: str, level: int, N: int, seed: int, a: float, sigma: float, q: float, coupled: bool = True, zfun=None, zcfun=None):
    """(P_l, P_{l−1}) für N Proben; bei l = 0 ist P_{l−1} = 0.

    Die feine Bahn macht 2^l Schritte mit den Zuwächsen sqrt(h)·z_i. Gekoppelt (Voreinstellung) bekommt die grobe Bahn die Summe je zweier feiner Zuwächse,
    ungekoppelt eigene Normalzahlen. `zfun(i, N)` und `zcfun(j, N)` ersetzen die Zufallszahlen (Tests mit festen Zahlen)."""
    nf = 2**level
    hf = T / nf
    sq = math.sqrt(hf)
    if zfun is None:
        def zfun(i, n):
            return normals(stream_seed(seed, level, i), n)
    if level == 0:
        Sf = step(scheme, np.ones(N), hf, sq * zfun(0, N), a, sigma)
        return payoff(kind, Sf, q), np.zeros(N)
    if zcfun is None:
        def zcfun(j, n):
            return normals(stream_seed(seed, level, 7_000_000 + j), n)
    Sf = np.ones(N)
    Sc = np.ones(N)
    hc = 2 * hf
    for j in range(nf // 2):
        z1, z2 = zfun(2 * j, N), zfun(2 * j + 1, N)
        Sf = step(scheme, Sf, hf, sq * z1, a, sigma)
        Sf = step(scheme, Sf, hf, sq * z2, a, sigma)
        dWc = sq * (z1 + z2) if coupled else math.sqrt(hc) * zcfun(j, N)
        Sc = step(scheme, Sc, hc, dWc, a, sigma)
    return payoff(kind, Sf, q), payoff(kind, Sc, q)


def pair_general(scheme: str, kind: str, lf: int, lc: int, N: int, seed: int, a: float, sigma: float, q: float):
    """(P_lf, P_lc) mit gekoppelten Zuwächsen für beliebiges lc < lf (Zwei-Ebenen-Schätzer)."""
    r = 2 ** (lf - lc)
    hf = T / 2**lf
    sq = math.sqrt(hf)
    hc = r * hf
    Sf = np.ones(N)
    Sc = np.ones(N)
    for j in range(2**lc):
        zsum = np.zeros(N)
        for i in range(r):
            z = normals(stream_seed(seed, lf, lc, j * r + i), N)
            Sf = step(scheme, Sf, hf, sq * z, a, sigma)
            zsum += z
        Sc = step(scheme, Sc, hc, sq * zsum, a, sigma)
    return payoff(kind, Sf, q), payoff(kind, Sc, q)


@lru_cache(maxsize=4096)
def level_stats(scheme: str, kind: str, level: int, N: int, seed: int, a: float, sigma: float, q: float, coupled: bool = True) -> tuple:
    """(E[Y_l], Var Y_l, Var P_l, Kurtosis von Y_l, Anteil Y_l ≠ 0, E[P_l]) aus N Proben; Y_l = P_l − P_{l−1}."""
    p, pc = level_pair(scheme, kind, level, N, seed, a, sigma, q, coupled)
    y = p - pc
    v = float(y.var())
    kurt = float(((y - y.mean()) ** 4).mean() / v**2) if v > 0 else float("nan")
    return float(y.mean()), v, float(p.var()), kurt, float((y != 0).mean()), float(p.mean())


# --- Schätzer -------------------------------------------------------------------------------------------------------------------------------

def optimal_allocation(vl, cl, eps: float) -> tuple[list, float]:
    """Optimale Bahnenzahl je Ebene für Gesamtvarianz ≤ ε²/2: N_l = 2 ε⁻² √(V_l/C_l) Σ_k √(V_k C_k) (nicht gerundet) und die Kosten Σ N_l C_l = 2 ε⁻² (Σ √(V_l C_l))²."""
    tot = sum(math.sqrt(v * c) for v, c in zip(vl, cl))
    ns = [2.0 * eps**-2 * math.sqrt(v / c) * tot if c > 0 else 0.0 for v, c in zip(vl, cl)]
    return ns, 2.0 * tot * tot / eps**2


def level_sums(scheme: str, kind: str, level: int, n: int, seed: int, a: float, sigma: float, q: float) -> tuple[float, float]:
    """(Σ Y, Σ Y²) über n frische Proben der Ebene l (in Blöcken, damit der Speicher klein bleibt)."""
    s1 = s2 = 0.0
    done, c = 0, 0
    chunk = 200_000 if level < 5 else 50_000
    while done < n:
        m = min(chunk, n - done)
        p, pc = level_pair(scheme, kind, level, m, stream_seed(seed, level, 900 + c), a, sigma, q)
        y = p - pc
        s1 += float(y.sum())
        s2 += float((y * y).sum())
        done += m
        c += 1
    return s1, s2


def mlmc(scheme: str, kind: str, eps: float, seed: int, a: float, sigma: float, q: float, n0: int = 1000, lmax: int = 11) -> tuple[float, float, int, list]:
    """Giles' adaptives Multilevel-Verfahren (Verfeinerung M = 2, schwache Ordnung α = 1 fest): Pilot n0 je Ebene 0 bis 2, optimale Zuteilung
    N_l = 2 ε⁻² √(V_l/C_l) Σ_k √(V_k C_k), Nachziehen bis zur Zuteilung, Abbruchtest max(|Ȳ_{L−1}|/2, |Ȳ_L|) ≤ ε/√2, sonst eine Ebene dazu.
    Rückgabe: (Schätzwert, Kosten in Zeitschritten, Ebenenzahl L, Proben je Ebene)."""
    L = 2
    Nl, S1, S2 = [0] * (L + 1), [0.0] * (L + 1), [0.0] * (L + 1)
    dN = [n0] * (L + 1)
    calls = 0
    while True:
        for l in range(L + 1):
            if dN[l] > 0:
                s1, s2 = level_sums(scheme, kind, l, dN[l], stream_seed(seed, calls), a, sigma, q)
                calls += 1
                Nl[l] += dN[l]
                S1[l] += s1
                S2[l] += s2
        ml = [S1[l] / Nl[l] for l in range(L + 1)]
        Vl = [max(S2[l] / Nl[l] - ml[l] ** 2, 0.0) for l in range(L + 1)]
        Cl = [cost_per_sample(l) for l in range(L + 1)]
        Ns = [math.ceil(n) for n in optimal_allocation(Vl, Cl, eps)[0]]
        dN = [max(0, Ns[l] - Nl[l]) for l in range(L + 1)]
        if all(dN[l] <= 0.01 * Nl[l] for l in range(L + 1)):
            rem = max(abs(ml[L - 1]) / 2.0, abs(ml[L]))
            if rem > eps / math.sqrt(2) and L < lmax:
                L += 1
                Nl.append(0), S1.append(0.0), S2.append(0.0)
                dN = [0] * L + [n0]
                continue
            break
    cost = sum(Nl[l] * cost_per_sample(l) for l in range(L + 1))
    return float(sum(ml)), float(cost), L, Nl


def mc_run(scheme: str, kind: str, eps: float, level: int, var_p: float, seed: int, a: float, sigma: float, q: float) -> tuple[float, float]:
    """Gewöhnliches Monte Carlo auf Ebene `level` mit N = ⌈2 Var(P)/ε²⌉ Bahnen (Orakelwissen: Varianz und Ebene sind gegeben). Rückgabe (Schätzwert, Kosten)."""
    N = max(2, math.ceil(2 * var_p / eps**2))
    s, done, c = 0.0, 0, 0
    chunk = 100_000 if level < 5 else 20_000
    while done < N:
        m = min(chunk, N - done)
        p, _ = level_pair(scheme, kind, level, m, stream_seed(seed, 800 + c), a, sigma, q)
        s += float(p.sum())
        done += m
        c += 1
    return s / N, float(N * 2.0**level)


def single_term_probs(s: float, lmax: int) -> np.ndarray:
    p = np.array([2.0 ** (-s * l) for l in range(lmax + 1)])
    return p / p.sum()


def single_term(scheme: str, kind: str, s: float, lmax: int, n: int, seed: int, a: float, sigma: float, q: float) -> tuple[float, float, float]:
    """Entzerrter Ein-Term-Schätzer Z = Y_L / p_L mit zufälliger Ebene L ~ p_l ∝ 2^(−s l), l ≤ lmax. Rückgabe (Mittel, Stichprobenvarianz, mittlere Kosten je Probe).
    Erwartungswert: E[P_lmax] (Abschneideverzerrung nur durch lmax)."""
    p = single_term_probs(s, lmax)
    u = uniforms(stream_seed(seed, 55), n)
    levels = np.searchsorted(np.cumsum(p), u, side="right").clip(0, lmax)
    zs = []
    cost = 0.0
    for l in range(lmax + 1):
        nl = int((levels == l).sum())
        if nl == 0:
            continue
        pf, pc = level_pair(scheme, kind, l, nl, stream_seed(seed, 56, l), a, sigma, q)
        zs.append((pf - pc) / p[l])
        cost += nl * cost_per_sample(l)
    z = np.concatenate(zs)
    return float(z.mean()), float(z.var(ddof=1)), cost / n
