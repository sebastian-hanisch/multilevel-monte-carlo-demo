"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets."""

AUSGABEN = {"glatt": "glatt: Endbestand S(T)", "knick": "Knick: Fehlbestand (S(T) − q)⁺", "sprung": "Sprung: Fehlmenge P(S(T) > q)"}
DEFAULT_AUSGABE = "knick"
VERFAHREN = {"euler": "Euler–Maruyama", "milstein": "Milstein"}
DEFAULT_VERFAHREN = "euler"
Q = 1.3                       # Schwelle des Fehlbestands und der Fehlmenge
SIGMA_CHOICES = (0.25, 0.5, 1.0)
SIGMA_MIN, SIGMA_MAX, DEFAULT_SIGMA = SIGMA_CHOICES[0], SIGMA_CHOICES[-1], 0.5
EPS_CHOICES = (0.04, 0.02, 0.01, 0.005, 0.0025)               # Toleranz (Wurzel des mittleren quadratischen Fehlers) für die gelaufenen Schätzungen
EPS_MIN, EPS_MAX, DEFAULT_EPS = EPS_CHOICES[-1], EPS_CHOICES[0], 0.005
EPS_GRID = (0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625)    # Toleranzen der idealisierten Kostentabelle (Experiment 2)
LMAX_CHOICES = (6, 7, 8)                                      # größte gemessene Ebene (2^l Zeitschritte)
LMAX_MIN, LMAX_MAX, DEFAULT_LMAX = LMAX_CHOICES[0], LMAX_CHOICES[-1], 7
SEED_MAX, DEFAULT_SEED = 999999, 36

N_LEVEL = 50000               # Proben je Ebene bei der Ebenenstatistik (Experiment 1)
N_PAIR = 50000                # Proben je Paar im Zwei-Ebenen-Experiment (Experiment 4)
N_SINGLE = 20000              # Proben des entzerrten Schätzers im Lauf (Experiment 5)
RUN_REPS = 30                 # Wiederholungen der gelaufenen Schätzungen (Experiment 3)
PILOT = 1000                  # Pilotproben je Ebene in Giles' Verfahren
SINGLE_S = (1.0, 1.5, 2.0)    # Abfallexponenten der Ebenenwahrscheinlichkeiten p_l ∝ 2^(−s l)
FIT_FROM = 3                  # Ebenen ab hier gehen in die Regression der Rate und der Verzerrungskonstante ein

CUSTOM_DRIFT_MIN, CUSTOM_DRIFT_MAX = -0.5, 1.0
CUSTOM_SIGMA_MIN, CUSTOM_SIGMA_MAX = 0.1, 1.2
CUSTOM_Q_MIN, CUSTOM_Q_MAX = 0.5, 2.5

COLORS = {"gekoppelt": "#2F6B65", "ungekoppelt": "#C0392B", "bias": "#E08E0B", "mc": "#C0392B", "ml": "#2F6B65", "two": "#4c78a8", "ref": "#222222"}
COLOR_BAD = "#C0392B"
COLOR_OK = "#2F6B65"

PRESETS = {
    "Standardfall (Voreinstellung)": {"ausgabe": DEFAULT_AUSGABE, "verfahren": DEFAULT_VERFAHREN, "sigma": DEFAULT_SIGMA, "eps": DEFAULT_EPS, "lmax": DEFAULT_LMAX, "seed": DEFAULT_SEED},
    "Milstein: die Kette zahlt sich aus": {"ausgabe": "knick", "verfahren": "milstein", "sigma": DEFAULT_SIGMA, "eps": 0.0025, "lmax": DEFAULT_LMAX, "seed": DEFAULT_SEED},
    "Sprung mit Euler: Multilevel verliert": {"ausgabe": "sprung", "verfahren": "euler", "sigma": DEFAULT_SIGMA, "eps": 0.0025, "lmax": 8, "seed": DEFAULT_SEED},
    "Große Streuung: grobe Ebenen taugen nicht": {"ausgabe": "knick", "verfahren": "euler", "sigma": 1.0, "eps": DEFAULT_EPS, "lmax": 8, "seed": DEFAULT_SEED},
    "Lockere Toleranz": {"ausgabe": "knick", "verfahren": "milstein", "sigma": DEFAULT_SIGMA, "eps": 0.04, "lmax": 6, "seed": DEFAULT_SEED},
}
PRESET_HELP = {
    "Standardfall (Voreinstellung)": "Euler–Maruyama, Fehlbestand (S(T) − q)⁺, σ = 0,5, Toleranz 0,005: Multilevel ist etwas billiger als gewöhnliches Monte Carlo, aber nicht viel (Rate β = 1 gleich Kostenrate γ = 1).",
    "Milstein: die Kette zahlt sich aus": "Mit Milstein fällt die Ebenenvarianz mit 2^(−2l): der Multilevel-Schätzer ist bei Toleranz 0,0025 um ein Vielfaches billiger, und bei engerer Toleranz sättigt ε²·Kosten.",
    "Sprung mit Euler: Multilevel verliert": "Bei der Fehlmengenwahrscheinlichkeit und Euler fällt die Ebenenvarianz nur mit 2^(−l/2): Multilevel ist im Idealfall nicht billiger als gewöhnliches Monte Carlo, ein Zwei-Ebenen-Schätzer schon.",
    "Große Streuung: grobe Ebenen taugen nicht": "Mit σ = 1,0 steigt die Ebenenvarianz auf den gröbsten Ebenen erst, bevor sie fällt: die Rate gilt erst ab einer tieferen Ebene.",
    "Lockere Toleranz": "Bei Toleranz 0,04 kostet schon der Pilot von Giles' Verfahren (3 Ebenen mal 1000 Bahnen) mehr als gewöhnliches Monte Carlo insgesamt.",
}
