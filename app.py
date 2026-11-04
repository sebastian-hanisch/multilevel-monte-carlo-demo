"""Multilevel Monte Carlo: wann sich die Ebenen lohnen - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Stück 12 der Linie "Monte-Carlo-Schätzung und Varianzreduktion": Ein Erwartungswert, den man nur diskretisiert lösen kann (2^l Zeitschritte), kostet bei gewöhnlichem Monte Carlo ε⁻³. Multilevel schreibt ihn als
Summe von Differenzen aufeinanderfolgender Ebenen, die aus denselben Zufallszahlen gekoppelt werden: auf den teuren Ebenen genügen wenige Bahnen. Ob und wie viel das spart, hängt an der Kopplung, am Verfahren
(Euler oder Milstein), an der Glattheit der Ausgabe und an der Toleranz; ein Zwei-Ebenen-Schätzer und der entzerrte Ein-Term-Schätzer sind die Vergleiche. Vehikel ist ein Bestandsprozess mit exakter Lösung als Orakel.

Lauffähig mit: streamlit run app.py
"""

import dataclasses

import streamlit as st

import mlm_constants as C
import mlm_evaluation as ev
from mlm_evaluation import Settings, bias_at, de, de_e, eps_text, ke
from mlm_layout import dunkles_design, kopf_css
from mlm_pdf_export import build_pdf
from mlm_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    options,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from mlm_scenario import check_szenario
from mlm_visualization import build_cost, build_cost_norm, build_levels, build_run, build_shares, build_single, build_two

st.set_page_config(page_title="Multilevel Monte Carlo: wann sich die Ebenen lohnen – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner="Ebenen messen …")
def _analysis(settings):
    return ev.analyse(settings)


@st.cache_data(show_spinner="Schätzungen laufen lassen …")
def _run(settings):
    return ev.run_experiment(settings)


@st.cache_data(show_spinner="Zwei Ebenen messen …")
def _two(settings):
    return ev.two_level_experiment(settings)


@st.cache_data(show_spinner="Entzerrten Schätzer laufen lassen …")
def _single(settings):
    return ev.single_term_experiment(settings)


@st.cache_data(show_spinner=False)
def _custom(settings, drift, sigma, q):
    sc = check_szenario(drift, sigma, q)
    return sc, ev.analyse(settings, sc)


def _sigma_label(s):
    return f"{s:g}".replace(".", ",")


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in mlm_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🪜 Multilevel Monte Carlo: wann sich die Ebenen lohnen")
st.markdown(
    """
Wer einen **Erwartungswert** braucht, den sich nur **diskretisiert** lösen lässt (hier: ein Bestandsprozess mit 2^l Zeitschritten), zahlt bei gewöhnlichem Monte Carlo für kleine Fehler doppelt: feine Bahnen (wenig Verzerrung) **und** viele davon (wenig Streuung), insgesamt etwa ε⁻³.
**Multilevel** schreibt den feinen Wert als Summe von Differenzen aufeinanderfolgender Ebenen und schätzt jede mit eigenen Bahnen, deren grobe und feine Hälfte **dieselben Zufallszahlen** benutzen. Die Differenzen streuen wenig, also reichen auf den teuren Ebenen wenige Bahnen.
Ob das lohnt, hängt an der **Kopplung**, am **Verfahren** (Euler oder Milstein), an der **Glattheit** der Ausgabe und an der **Toleranz**: bei einem Sprung mit Euler oder bei lockerer Toleranz kann gewöhnliches Monte Carlo billiger sein.

Weiter unten: **🔬 fünf Experimente**, **🧪 Spielwiese**, **🚧 Wo die Annahmen enden** und die Expander **⚠️ Annahmen, Grenzen und bekannte Probleme** und **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Multilevel", expanded=True):
    st.markdown(
        """
Ebene l löst den Prozess dS = a S dt + σ S dW mit 2^l Zeitschritten; P_l ist die Ausgabe (Endbestand, Fehlbestand oder Fehlmenge). Gesucht ist E[P_∞] (hier exakt bekannt, die Schätzer kennen es nicht).

1. **Gewöhnlich:** wähle die Ebene L so fein, dass die Verzerrung ≈ c·2^(−L) unter ε/√2 liegt, und ziehe N ≈ 2 Var(P_L)/ε² Bahnen. Die Kosten wachsen wie ε⁻² (Bahnen) mal ε⁻¹ (Schritte je Bahn).
2. **Multilevel:** E[P_L] = E[P_0] + Σ_l E[P_l − P_(l−1)]. Jede Differenz Y_l wird aus **eigenen** Bahnen geschätzt, bei denen die grobe Bahn die **summierten Zuwächse** der feinen benutzt. Fällt die Varianz V_l = Var(Y_l) mit 2^(−βl) schneller, als die Kosten je Bahn mit 2^l wachsen (β > γ = 1), sättigt ε²·Kosten.
   Optimal sind N_l ∝ √(V_l/C_l) Bahnen auf Ebene l (C_l = 2^l + 2^(l−1) Schritte).
3. **Zwei Ebenen** (wie Stück 4): eine billige Ebene k und eine feine L, nur die eine Differenz P_L − P_k.
4. **Entzerrt** (Ein-Term): ziehe eine zufällige Ebene L mit Wahrscheinlichkeit p_l und schätze Y_L / p_L; kein Abbruchfehler, aber nur sinnvoll, wenn V_l schneller als p_l fällt und die Kosten langsamer als 1/p_l wachsen.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Prozess und Ausgabe**")
    ausgabe = st.selectbox("Ausgabe g(S(T))", options("ausgabe_select"), format_func=C.AUSGABEN.get, key="ausgabe_select",
                           help="Glatt: der Endbestand S(T). Knick: der erwartete Fehlbestand (S(T) − q)⁺ mit q = 1,3. Sprung: die Fehlmengenwahrscheinlichkeit P(S(T) > q).")
    verfahren = st.selectbox("Verfahren", options("verfahren_select"), format_func=C.VERFAHREN.get, key="verfahren_select", help="Euler–Maruyama (starke Ordnung 1/2) oder Milstein (starke Ordnung 1): bestimmt, wie schnell die Ebenenvarianz fällt.")
    sigma = st.select_slider("Streuung σ des Prozesses", options("sigma_select"), key="sigma_select", format_func=_sigma_label, help="Bei großem σ taugen die gröbsten Ebenen nicht; die Rate gilt erst ab einer tieferen Ebene.")
    st.markdown("**Toleranz und Ebenen**")
    eps = st.select_slider("Toleranz ε (Wurzel des mittleren quadratischen Fehlers)", options("eps_select"), key="eps_select", format_func=eps_text, help="Die gelaufenen Schätzungen (Experimente 3 bis 5) zielen auf diese Toleranz; die Kostentabelle zeigt alle.")
    lmax = st.select_slider("Größte gemessene Ebene", options("lmax_select"), key="lmax_select", help="Ebene l hat 2^l Zeitschritte. Mehr Ebenen kosten Rechenzeit, geben aber eine verlässlichere Rate.")
    seed = st.number_input("Zufallsstrom", *bounds("seed_input"), key="seed_input", step=1, help="Welcher feste Strom von Zufallszahlen benutzt wird (SplitMix64).")
    st.button("🎲 Neuen Strom wählen", width="stretch", on_click=randomize_seed)

sync_query_params({"ausgabe_select": ausgabe, "verfahren_select": verfahren, "sigma_select": float(sigma), "eps_select": float(eps), "lmax_select": int(lmax), "seed_input": int(seed)})

settings = Settings(str(ausgabe), str(verfahren), float(sigma), float(eps), int(lmax), int(seed))
a = _analysis(settings)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

row = ev.cost_row(a, settings.eps)
st.markdown("## 🎯 Ergebnis")
st.caption(f"{a.label}; Bestandsprozess mit a = σ²/2, q = 1,3, σ = {_sigma_label(a.sigma)}; Ebenen 0 bis {a.lmax}, {de(a.n, 0)} Bahnen je Ebene; Kosten in Zeitschritten, idealisiert (Ebenenvarianzen und Verzerrung bekannt).")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Exakter Wert", de(a.truth, 4), help="Geschlossene Formel (log-normaler Endbestand), keine Simulation.")
m2.metric("Ebene L (gewöhnlich)", str(row["L"]), delta=f"Verzerrung ≈ {de(a.c, 4)}·2^(−L)", delta_color="off", help="Kleinste Ebene mit Verzerrung ≤ ε/√2.")
m3.metric("Rate β der Ebenenvarianz", ev.dn(a.beta, 2), delta=ev.regime(a.beta).split(":")[0], delta_color="off", help="V_l ∝ 2^(−β l), Regression über die Ebenen ab 3. Kostenrate je Ebene γ = 1.")
m4.metric("Gewöhnlich / Multilevel", f"{ev.dn(row['ratio'], 2)}×", delta=f"bei ε = {eps_text(settings.eps)}", delta_color="off", help="Verhältnis der idealisierten Kosten; unter 1 ist Multilevel teurer.")
st.plotly_chart(build_levels(a), width="stretch", key="main_chart")
st.caption("Varianz der Ebenendifferenz gegen die Ebene, logarithmisch. Gekoppelt fällt sie mit einer festen Rate; ungekoppelt bleibt sie auf dem Niveau der Varianz der Ausgabe selbst und der Multilevel-Schätzer lohnt nicht. Gepunktet: Steigungen −1 und −2.")

if st.button("📄 PDF der Auswertung erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(settings, a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="multilevel.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Lohnt Multilevel hier, und wie viel spart es?")
kind, text = ev.verdict_main(a, settings)
getattr(st, kind)(text)

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Die Ebenen")
st.caption(f"Für jede Ebene {de(a.n, 0)} gekoppelte Bahnpaare (feine und grobe Bahn aus denselben Zuwächsen) und, zum Vergleich, ebenso viele mit unabhängigen Zahlen für die grobe Bahn. Gemessen: Mittel und Varianz der Differenz und die Verzerrung der Ebene gegen den exakten Wert.")
st.dataframe(
    {"Ebene l": [r["l"] for r in a.levels], "Schritte": [2 ** r["l"] for r in a.levels], "E[Y_l]": [de_e(r["mean"]) if r["l"] > 0 else "–" for r in a.levels],
     "V_l gekoppelt": [de_e(r["var"]) for r in a.levels], "V_l / V_(l−1)": [de(r["var"] / a.levels[i - 1]["var"], 2) if i > 0 and a.levels[i - 1]["var"] > 0 else "–" for i, r in enumerate(a.levels)],
     "V_l ungekoppelt": [de_e(v) if i > 0 else "–" for i, v in enumerate(a.var_unc)], "Kurtosis": [de(r["kurt"], 0) if r["l"] > 0 else "–" for r in a.levels],
     "Verzerrung E[P_l] − exakt": [de_e(bias_at(a, r["l"])) for r in a.levels]}, hide_index=True, width="stretch", key="level_table")
kind, text = ev.verdict_levels(a)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 2: Kosten gegen Toleranz")
st.caption("Idealisierte Kosten für einen mittleren quadratischen Fehler ≤ ε² (Verzerrung² ≤ ε²/2, Varianz ≤ ε²/2), aus den gemessenen Ebenenvarianzen; jenseits der gemessenen Ebenen mit der Rate fortgesetzt. Gewöhnlich: Ebene L, N ≈ 2 Var(P_L)/ε² Bahnen. Multilevel: optimale Zuteilung.")
table = ev.cost_table(a)
c1, c2 = st.columns(2)
c1.plotly_chart(build_cost(table), width="stretch", key="cost_chart")
c2.plotly_chart(build_cost_norm(table), width="stretch", key="cost_norm_chart")
st.dataframe(
    {"Toleranz ε": [eps_text(r["eps"]) for r in table], "Ebene L": [r["L"] for r in table], "gewöhnlich": [ke(r["mc"]) for r in table], "Multilevel": [ke(r["ml"]) for r in table],
     "gewöhnlich / Multilevel": [ev.dn(r["ratio"], 2) for r in table], "ε² · gewöhnlich": [de(r["eps2_mc"], 2) for r in table], "ε² · Multilevel": [de(r["eps2_ml"], 2) for r in table]},
    hide_index=True, width="stretch", key="cost_table")
st.plotly_chart(build_shares(row), width="stretch", key="shares_chart")
st.caption(f"Kostenanteil je Ebene des Multilevel-Schätzers bei ε = {eps_text(settings.eps)} (Ebene 0 bis {row['L']}): bei β = γ verteilen sich die Kosten fast gleichmäßig, bei β > γ liegen sie auf den groben Ebenen.")
kind, text = ev.verdict_cost(a)
getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 3: Gelaufene Schätzungen")
st.caption(f"Giles' adaptives Verfahren (Pilot {de(C.PILOT, 0)} Bahnen auf den Ebenen 0 bis 2, Zuteilung aus den geschätzten Varianzen, neue Ebene bei nicht bestandenem Abbruchtest) gegen gewöhnliches Monte Carlo mit Orakelwissen über Ebene und Varianz, {C.RUN_REPS} Wiederholungen bei ε = {eps_text(settings.eps)}.")
if st.button("▶ Messung starten", key="run_gelaufen"):
    st.session_state["run_done"] = True
if st.session_state.get("run_done"):
    ex3 = _run(settings)
    st.plotly_chart(build_run(ex3), width="stretch", key="run_chart")
    st.dataframe({"Verfahren": ["gewöhnlich (Orakelwissen)", "Multilevel (Giles)"], "RMSE / ε": [de(ex3["mc"]["rmse"] / ex3["eps"], 2), de(ex3["ml"]["rmse"] / ex3["eps"], 2)],
                  "mittlere Kosten": [ke(ex3["mc"]["cost"]), ke(ex3["ml"]["cost"])], "Fehler über ε": [ev.prozent(ex3["mc"]["over"], 0), ev.prozent(ex3["ml"]["over"], 0)]}, hide_index=True, width="stretch", key="run_table")
    kind, text = ev.verdict_run(ex3)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 4: Zwei Ebenen gegen viele")
st.caption("Ein Zwei-Ebenen-Schätzer nimmt eine billige Ebene k und die feine Ebene L und schätzt nur P_k und P_L − P_k (Stück 4). Gemessen wird die Varianz der einen Differenz für jedes k; die Kosten folgen aus der optimalen Zuteilung.")
if st.button("▶ Messung starten", key="run_two"):
    st.session_state["two_done"] = True
if st.session_state.get("two_done"):
    ex4 = _two(settings)
    st.plotly_chart(build_two(ex4), width="stretch", key="two_chart")
    st.dataframe({"grobe Ebene k": [r["k"] for r in ex4["rows"]], f"Var(P_{ex4['L']} − P_k)": [de_e(r["var"]) for r in ex4["rows"]], "Kosten Zwei Ebenen": [ke(r["cost"]) for r in ex4["rows"]]}, hide_index=True, width="stretch", key="two_table")
    kind, text = ev.verdict_two(ex4)
    getattr(st, kind)(text)

st.markdown("---")
st.subheader("🔬 Experiment 5: Der entzerrte Schätzer")
st.caption("Ein-Term-Schätzer: eine zufällige Ebene L mit Wahrscheinlichkeit p_l ∝ 2^(−s·l) wird gezogen und Y_L / p_L gemittelt. Kein Abbruchfehler, aber die Varianz kann mit der Abschneideebene unbeschränkt wachsen. Gezeigt: Varianz gegen die Abschneideebene für drei s (aus den gemessenen zweiten Momenten) und ein gelaufener Lauf.")
if st.button("▶ Messung starten", key="run_single"):
    st.session_state["single_done"] = True
if st.session_state.get("single_done"):
    ex5 = _single(settings)
    st.plotly_chart(build_single(ex5), width="stretch", key="single_chart")
    lms = [r["lmax"] for r in next(iter(ex5["curves"].values()))]
    st.dataframe({"Abschneideebene": lms, **{f"Varianz, s = {de(s, 1)}": [de(r["var"], 2) for r in cv] for s, cv in ex5["curves"].items()}, **{f"Varianz·Kosten, s = {de(s, 1)}": [de(r["work"], 2) for r in cv] for s, cv in ex5["curves"].items()}},
                 hide_index=True, width="stretch", key="single_table")
    kind, text = ev.verdict_single(ex5)
    getattr(st, kind)(text)

st.markdown("---")

# --- Spielwiese -----------------------------------------------------------------------------------------------------------------------------

with st.expander("🧪 Spielwiese: ein eigener Prozess"):
    st.caption("Eigene Drift a, Streuung σ und Schwelle q (S(0) = 1, T = 1). Es gelten die gewählte Ausgabe, das gewählte Verfahren und die gewählte Toleranz; die exakte Erwartung bleibt geschlossen bekannt.")
    s1, s2, s3 = st.columns(3)
    drift_in = s1.number_input("Drift a", C.CUSTOM_DRIFT_MIN, C.CUSTOM_DRIFT_MAX, 0.125, 0.025, key="eigen_drift", format="%.3f")
    sigma_in = s2.number_input("Streuung σ", C.CUSTOM_SIGMA_MIN, C.CUSTOM_SIGMA_MAX, 0.5, 0.05, key="eigen_sigma", format="%.2f")
    q_in = s3.number_input("Schwelle q", C.CUSTOM_Q_MIN, C.CUSTOM_Q_MAX, 1.3, 0.05, key="eigen_q", format="%.2f")
    try:
        csc, ca = _custom(settings, float(drift_in), float(sigma_in), float(q_in))
    except ValueError as err:
        st.error(str(err))
    else:
        st.plotly_chart(build_levels(ca), width="stretch", key="custom_levels")
        ctable = ev.cost_table(ca)
        st.plotly_chart(build_cost(ctable), width="stretch", key="custom_cost")
        kind, text = ev.verdict_main(ca, settings)
        getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Die grobe Bahn benutzt dieselben Zuwächse** | Ohne Kopplung bleibt V_l auf dem Niveau der Varianz der Ausgabe; Multilevel wird um ein Vielfaches teurer als gewöhnliches Monte Carlo (Experiment 1, Kostenrechnung). | Stück 2 |
| **Die Differenz wird schneller klein, als die Bahn teurer wird (β > γ)** | Bei Euler gilt β = γ = 1: ε²·Kosten wächst weiter, der Vorteil bleibt bei einem kleinen Faktor (Experiment 2). | – |
| **Die Ausgabe ist glatt** | Bei einem Sprung halbiert sich die Rate: Euler mit Sprung kostet im Idealfall mehr als gewöhnliches Monte Carlo (Preset »Sprung mit Euler«). | – |
| **Die Toleranz ist eng** | Bei lockerem ε kostet der Pilot mehr, als er spart (Experiment 3, Preset »Lockere Toleranz«). | – |
| **Eine Kette über viele Ebenen ist nötig** | Ein Zwei-Ebenen-Schätzer holt bei Euler fast alles, beim Sprung mehr als Multilevel (Experiment 4). | Stück 4 |
| **Der Abbruchfehler stört** | Der entzerrte Schätzer verzichtet auf die Ebenenwahl, braucht dafür β > γ; bei Euler wächst seine Varianz mit der Abschneideebene (Experiment 5). | – |
| **Die gröbsten Ebenen zeigen schon die Rate** | Bei großem σ steigt V_l erst, bevor es fällt: Regressionen über die ersten Ebenen täuschen (Preset »Große Streuung«). | – |
"""
)

with st.expander("⚠️ Annahmen, Grenzen und bekannte Probleme"):
    st.markdown(
        """
**Annahmen.** Der Prozess (geometrische Brownsche Bewegung mit a = σ²/2) ist **erfunden**, und sein Endwert ließe sich hier **direkt ziehen**; die exakte Lösung dient nur als Orakel für die Wahrheit, die Schätzer lösen mit Zeitschritten. Die Zufallszahlen kommen aus festen Strömen (SplitMix64, Box–Muller), nie aus einem einzelnen Seed.
Kosten zählen Zeitschritte, nicht Laufzeit.

**Grenzen.** Es gibt nur drei Ausgaben des Endwerts, keine pfadabhängigen (asiatisch, Barriere, Lookback), nur die Verfeinerung um den Faktor 2, kein Quasi-Monte-Carlo im Multilevel und keine adaptiven Zeitschritte. Giles' Verfahren läuft mit fest angesetzter schwacher Ordnung 1; die Rate β wird für die Kostenrechnung aus den Ebenen ab 3 geschätzt.
**Gewöhnliches Monte Carlo ist mit Orakelwissen gerechnet** (es kennt Verzerrung und Varianz) und wird dadurch begünstigt.

**Bekannte Probleme und Unsicherheiten.** Die Ebenenvarianz bei Sprung und Milstein beruht auf seltenen Ereignissen und ist mit 50.000 Bahnen je Ebene nur grob messbar (hohe Kurtosis); sie zeigt sich erst mit Millionen Bahnen sauber. Die Verzerrungskonstante wird aus den gemessenen Ebenenmitteln geschätzt und rauscht.
Bei großem σ gilt die Rate erst ab einer tieferen Ebene; die Fortsetzung jenseits der gemessenen Ebenen ist dort unsicher. Die Ein-Term-Varianzen stammen aus den gemessenen zweiten Momenten; der Lauf in Experiment 5 prüft sie an einem Fall. Der Kostensatz und Giles' Verfahren stehen im Mathematik-Expander aus dem Gedächtnis;
die Quellen (Giles 2015, Rhee und Glynn) sind nur im Abstract eingesehen. Die Oberfläche ist mit AppTest geprüft und im Browser angesehen, nicht mit echtem Mausklick auf jedem Gerät.
        """
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Prozess.** $dS=aS\,dt+\sigma S\,dW$, $S(0)=1$, $T=1$; $\ln S(T)\sim N\bigl((a-\sigma^2/2)T,\sigma^2T\bigr)$. Ausgaben $g(S)=S$ (glatt), $(S-q)^+$ (Knick), $\mathbf 1[S>q]$ (Sprung). Exakt mit $\mu=(a-\sigma^2/2)T$, $s=\sigma\sqrt T$, $d_2=(\mu-\ln q)/s$, $d_1=d_2+s$:
$\mathbb E S=e^{\mu+s^2/2}$, $\mathbb E(S-q)^+=e^{\mu+s^2/2}\Phi(d_1)-q\Phi(d_2)$, $\mathbb P(S>q)=\Phi(d_2)$.

**Verfahren.** Ebene $l$: $n=2^l$ Schritte, $h=T/n$. Euler: $S\leftarrow S(1+ah+\sigma\Delta W)$. Milstein: $S\leftarrow S\bigl(1+ah+\sigma\Delta W+\tfrac12\sigma^2(\Delta W^2-h)\bigr)$. Für beide gilt $\mathbb E S_n=(1+ah)^n$.

**Kopplung.** Feine Zuwächse $\Delta W_i=\sqrt h\,z_i$; die grobe Bahn (Schrittweite $2h$) benutzt $\sqrt h\,(z_{2j}+z_{2j+1})$. Dann ist $Y_l=P_l-P_{l-1}$ klein; mit unabhängigen Zahlen gilt $\operatorname{Var}Y_l\approx2\operatorname{Var}P$.

**Teleskopsumme.** $\mathbb E[P_L]=\mathbb E[P_0]+\sum_{l=1}^L\mathbb E[Y_l]$. Schätzer $\hat Y_l=\frac1{N_l}\sum_iY_l^{(i)}$ mit unabhängigen Bahnen je Ebene; Varianz $\sum_lV_l/N_l$.

**Optimale Zuteilung.** Kosten $C_l=2^l+2^{l-1}$ ($C_0=1$). Minimiere $\sum_lN_lC_l$ unter $\sum_lV_l/N_l\le\varepsilon^2/2$: $N_l=2\varepsilon^{-2}\sqrt{V_l/C_l}\sum_k\sqrt{V_kC_k}$, Kosten $2\varepsilon^{-2}\bigl(\sum_l\sqrt{V_lC_l}\bigr)^2$. Gewöhnlich: $N=2\operatorname{Var}(P_L)\varepsilon^{-2}$, Kosten $N\,2^L$.

**Verzerrung und Ebene.** Schwache Ordnung 1: $|\mathbb E[P_L]-\mathbb E[P]|\approx c\,2^{-L}$; $L$ ist die kleinste Ebene mit $c\,2^{-L}\le\varepsilon/\sqrt2$. Rate: $V_l\propto2^{-\beta l}$. Kostenrate je Ebene $\gamma=1$.
Dreifallunterscheidung (aus dem Gedächtnis, nicht aus dem Volltext; die Demo misst sie): $\beta>\gamma$ führt zu Kosten $\propto\varepsilon^{-2}$, $\beta=\gamma$ zu $\varepsilon^{-2}$ mal Logarithmen, $\beta<\gamma$ zu $\varepsilon^{-2-(\gamma-\beta)/\alpha}$ mit schwacher Ordnung $\alpha$.

**Adaptives Verfahren.** Pilot $n_0$ je Ebene 0 bis 2, Zuteilung aus geschätzten $V_l$, Nachziehen; Abbruchtest $\max(|\bar Y_{L-1}|/2,|\bar Y_L|)\le\varepsilon/\sqrt2$ (mit $\alpha=1$), sonst Ebene $L+1$.

**Zwei Ebenen.** Schätze $P_k$ (Varianz $V_0^{(k)}$, Kosten $2^k$) und $P_L-P_k$ (Varianz $V$, Kosten $2^L+2^k$): Kosten $2\varepsilon^{-2}\bigl(\sqrt{V_0^{(k)}2^k}+\sqrt{V(2^L+2^k)}\bigr)^2$.

**Ein-Term-Schätzer.** $L\sim p$, $Z=Y_L/p_L$: $\mathbb E Z=\sum_l\mathbb E Y_l$, $\operatorname{Var}Z=\sum_l\mathbb E[Y_l^2]/p_l-\bigl(\sum_l\mathbb EY_l\bigr)^2$, erwartete Kosten $\sum_lp_lC_l$. Mit $p_l\propto2^{-sl}$ und $\mathbb E Y_l^2\approx2^{-\beta l}$ ist die Varianz endlich genau dann, wenn $s<\beta$, die erwarteten Kosten sind endlich, wenn $s>\gamma$: es braucht $\beta>\gamma$.

**Literatur (aus dem Gedächtnis, wenn nicht anders vermerkt).** Giles 2008, „Multilevel Monte Carlo path simulation“ (Operations Research); Giles 2015, „Multilevel Monte Carlo methods“ (Acta Numerica; arXiv 1304.5472, nur Abstract eingesehen); Heinrich 2001 (Multilevel-Idee für Parameterintegrale); Rhee und Glynn 2015, „Unbiased estimation with square root convergence for SDE models“ (Operations Research; arXiv 1207.2452 unter dem Titel „A new approach to unbiased estimation for SDE's“, nur Abstract eingesehen);
Giles, Debrabant und Rößler 2013 (Multilevel mit Milstein; nicht eingesehen).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
