# Multilevel Monte Carlo: wann sich die Ebenen lohnen – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-multilevel-monte-carlo-demo.streamlit.app/)**

Zwölftes Stück der Linie **„Monte-Carlo-Schätzung und Varianzreduktion“** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“ (die Linie wird erst vollständig veröffentlicht). Nachfolger von [monte-carlo-gradient-demo](https://github.com/sebastian-hanisch/monte-carlo-gradient-demo) (Stück 11, Gradienten schätzen): Hier geht es um einen Erwartungswert, den man nur **diskretisiert** lösen kann, und um die Frage, wann der **Multilevel-Schätzer** gegenüber gewöhnlichem Monte Carlo Kosten spart. Vehikel ist die log-normale Nachfrage aus Stück 11 als **Bestandsprozess** mit drei Ausgaben in drei Glattheitsstufen (Endbestand, Fehlbestand, Fehlmengenwahrscheinlichkeit); die exakte Lösung dient nur als Orakel.

## Warum dieses Problem

Wer E[g(S(T))] für einen Prozess braucht, den nur ein Zeitschrittverfahren löst (Ebene l hat 2^l Schritte), zahlt bei gewöhnlichem Monte Carlo doppelt: feine Bahnen für wenig Verzerrung und viele Bahnen für wenig Streuung, insgesamt etwa ε⁻³. **Multilevel** (Übersicht: Giles, Acta Numerica 2015; hier nur der Abstract gelesen) schreibt den feinen Wert als Teleskopsumme von Differenzen aufeinanderfolgender Ebenen, schätzt jede mit eigenen Bahnen, deren grobe und feine Hälfte **dieselben Zuwächse** benutzen, und zieht auf den teuren Ebenen nur wenige Bahnen. Das ist die Kontrollvariaten-Kette aus den Stücken 3 und 4, rekursiv. Ob sie sich lohnt, ist eine Frage der Rate, mit der die Ebenenvarianz fällt, gegen die Rate, mit der die Kosten wachsen. Die Demo misst sie, statt den bekannten Satz zu behaupten.

## Befunde und Korrekturen gegenüber dem Plan

Vor dem Bau wurden die Hypothesen der Gliederung an Messreihen geprüft (Vormessung im Website-Repo, `docs/planung/monte-carlo-planung/vormessung/`). Was sich geändert hat oder beim Bau dazukam:

- **„Kosten O(ε⁻²) statt O(ε⁻³) bei geeigneter Varianzabnahme“ gilt bei Milstein mit glatter oder geknickter Ausgabe, nicht mit Sprung.** Bei Milstein steigt ε²·Kosten nur noch mild (Fehlbestand: 0,32 bei ε = 0,04, 0,72 bei ε = 0,000625), gewöhnliches Monte Carlo wächst von 0,53 auf 41; bei ε = 0,000625 ist Multilevel 57-mal billiger. Bei Euler (Rate β gleich der Kostenrate 1) wächst ε²·Kosten weiter (0,28 → 3,9), und der Vorteil bleibt bei den gemessenen Toleranzen zwischen 1,1 und 5,3.
- **Mit einem Sprung verliert Euler-Multilevel im gezeigten Bereich:** bei Fehlmengenwahrscheinlichkeit und Euler fällt die Ebenenvarianz nur mit der Rate 0,46, und die Kosten liegen im Idealfall bei Toleranzen von 0,04 bis 0,0025 über denen von gewöhnlichem Monte Carlo (Verhältnis 0,51 bis 0,65); bei ε = 0,000625 herrscht Gleichstand (1,01).
- **Ohne Kopplung ist Multilevel nutzlos:** bleibt die grobe Bahn unabhängig, bleibt die Varianz der Differenz bei etwa dem Doppelten der Varianz der Ausgabe (0,745 beim Endbestand, 0,331 beim Fehlbestand, 0,422 bei der Fehlmenge) und fällt nicht mehr (Rate 0).
- **Bei lockerer Toleranz gewinnt gewöhnliches Monte Carlo:** schon der Pilot von Giles' Verfahren (1000 Bahnen auf drei Ebenen) kostet 10⁴ Schritte; bei ε = 0,04 braucht gewöhnliches Monte Carlo 330. Bei Milstein liegt das Kostenverhältnis der gelaufenen Schätzungen ab ε = 0,01 über 1 (Fehlbestand: 10,4 bei ε = 0,0025), bei Euler erst um ε = 0,005 (Fehlbestand: 1,85 bei ε = 0,0025).
- **Ein Zwei-Ebenen-Schätzer (wie in Stück 4) holt bei Euler fast alles** (Fehlbestand, ε = 0,0025: 2,9-fach billiger als gewöhnlich, Multilevel 2,7-fach) **und schlägt Multilevel beim Sprung** (2,3-fach billiger gegen 0,65). Erst bei Milstein zahlt sich die Kette über viele Ebenen aus (Multilevel 15,8-fach, Zwei Ebenen 8,8-fach).
- **Der entzerrte Ein-Term-Schätzer funktioniert nur, wenn die Ebenenvarianz schneller fällt als die Kosten wachsen:** bei Milstein mit Fehlbestand bleibt seine Varianz bei Abschneideebene 3 bis 8 beschränkt (0,21 bei s = 1,5), bei Euler wächst sie (Fehlbestand 0,28 → 2,65) und bei Milstein mit Sprung ebenfalls. Seine Kosten für Fehler ε liegen bei Milstein mit Fehlbestand bei 0,60/ε², gleich denen von Multilevel; beim Endbestand mehr als doppelt so hoch.
- **Bei großer Streuung gilt die Rate erst ab einer tieferen Ebene (Euler):** bei σ = 1 steigt die Ebenenvarianz von Ebene 1 auf 2, bevor sie fällt (vorasymptotische Strecke); Milstein gewinnt dort am meisten (119-fach).
- **Bei Sprung und Milstein ist die Rate aus 50 000 Bahnen nur grob messbar:** die Ebenenvarianz hängt dort an seltenen Ereignissen (Rate 0,79); in einer Kontrollmessung mit 4 · 10⁶ Bahnen liegen die letzten Ebenenschritte bei etwa 0,9, die Regression über die Ebenen 3 bis 7 bei etwa 0,8.
- **Gelaufene Multilevel-Schätzungen schöpfen das Verzerrungsbudget aus:** der Abbruchtest erlaubt eine Abbruchverzerrung bis ε/√2; der Fehler ist deshalb im Mittel negativ (etwa −0,7 ε bei Milstein, Fehlbestand, ε = 0,005) und der RMSE liegt zwischen 0,2 und etwa 1,1 ε (bei lockerem ε darunter).
- **Kein scipy zur Laufzeit:** die Normalverteilungsfunktion läuft über `math.erf`, Standardnormale per Box–Muller.

## Modell

Bestandsprozess dS = a S dt + σ S dW, S(0) = 1, T = 1, mit a = σ²/2, so dass S(T) = exp(σ W(T)) dieselbe log-normale Nachfrage ist wie in Stück 11 (μ = 0, σ = 0,5, Schwelle q = 1,3). Ausgaben: **glatt** S(T) (exakt 1,13315), **Knick** (S(T) − q)⁺ (Fehlbestand, exakt 0,16554), **Sprung** 1[S(T) > q] (Fehlmengenwahrscheinlichkeit, exakt 0,29989). Verfahren: **Euler–Maruyama** S ← S (1 + a h + σ ΔW) und **Milstein** S ← S (1 + a h + σ ΔW + ½ σ² (ΔW² − h)). Ebene l: 2^l Schritte; Kosten je Bahnpaar 2^l + 2^(l−1) Zeitschritte (Ebene 0: 1). Die exakte Lösung ist nur das Orakel; sie ließe sich direkt ziehen, deshalb ist das Beispiel ein Lehrbeispiel und kein Anwendungsfall.

## Methodik

Je Ebene 50 000 gekoppelte Bahnpaare (und ebenso viele ungekoppelte) aus festen SplitMix64-Strömen, Box–Muller. Gemessen: Mittel und Varianz der Differenz Y_l = P_l − P_(l−1), Kurtosis, die Verzerrung E[P_l] − exakt (aus dem in geschlossener Form berechneten E[P_0] plus den Ebenenmitteln). Rate β aus der Regression von log₂ V_l gegen l über die Ebenen ab 3; Verzerrungskonstante c aus dem Median von |E[Y_l]|·2^l über die Ebenen 3 bis 6. **Idealisierte Kosten** für einen mittleren quadratischen Fehler ≤ ε² (Verzerrung² ≤ ε²/2, Varianz ≤ ε²/2): gewöhnlich 2 Var(P_L) ε⁻² 2^L mit der kleinsten Ebene L, für die c·2^(−L) ≤ ε/√2 gilt; Multilevel 2 ε⁻² (Σ √(V_l C_l))² mit der optimalen Zuteilung N_l ∝ √(V_l/C_l); jenseits der gemessenen Ebenen wird V_l mit der Rate fortgesetzt. **Gelaufene Schätzungen:** Giles' adaptives Verfahren (Pilot 1000 Bahnen auf den Ebenen 0 bis 2, Zuteilung aus den geschätzten Varianzen, Abbruchtest mit fest angesetzter schwacher Ordnung 1) gegen gewöhnliches Monte Carlo **mit Orakelwissen** über Ebene und Varianz, 30 Wiederholungen. **Zwei Ebenen:** Var(P_L − P_k) für jede grobe Ebene k gemessen, Kosten 2 ε⁻² (√(V_0 2^k) + √(V (2^L + 2^k)))². **Ein-Term-Schätzer:** zufällige Ebene mit p_l ∝ 2^(−s l), Z = Y_L/p_L; Varianz aus den gemessenen zweiten Momenten und ein gelaufener Lauf mit 20 000 Proben.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: Strom 36, σ = 0,5, Ebenen 0 bis 8 (50 000 Bahnen je Ebene), `tests/test_claims.py` rechnet sie nach. Kosten in Zeitschritten, idealisiert.

| | Rate β gekoppelt | Rate β ungekoppelt | Verzerrung ≈ c·2^(−L) | gewöhnlich/Multilevel bei ε = 0,04 | bei ε = 0,000625 |
|---|---|---|---|---|---|
| Euler, glatt | 0,97 | −0,03 | c = 0,0066 | 1,14 | 2,52 |
| Euler, Knick | 0,91 | −0,07 | c = 0,0231 | 1,29 | 5,27 |
| Euler, Sprung | 0,46 | 0,01 | c = 0,0913 | 0,56 | 1,01 |
| Milstein, glatt | 1,96 | −0,02 | c = 0,0089 | 1,62 | 16,58 |
| Milstein, Knick | 1,92 | −0,03 | c = 0,0305 | 1,64 | 57,3 |
| Milstein, Sprung | 0,79 | 0,00 | c = 0,0067 | 1,36 | 2,61 |

- **Sprung bei ε = 0,000625:** die Werte 1,01 (Euler) und 2,61 (Milstein) hängen an der gemessenen Rate (0,46 und 0,79); bei Milstein ist sie aus 50 000 Bahnen nur grob bekannt (siehe Befunde).
- **Varianzverhältnis je Ebene** (Ebene 8 gegen 7): Euler glatt 0,50, Milstein glatt 0,26; ungekoppelt etwa 1,00.
- **ε²·Kosten, Milstein, Fehlbestand:** Multilevel 0,32 → 0,72 (steigt nur noch mild), gewöhnlich 0,53 → 41,3; Euler, Fehlbestand: Multilevel 0,28 → 3,9.
- **Kostenanteil je Ebene bei ε = 0,000625:** Milstein, Fehlbestand: 55 % auf Ebene 0 und danach fallend; Euler, Fehlbestand: 7 Ebenen mit je 10 bis 17 %.
- **Streuung σ, Fehlbestand, ε = 0,000625:** Euler σ = 0,25: 6,5-fach, σ = 1,0: 4,4-fach; Milstein σ = 1,0: 119-fach (σ = 0,5: 57-fach). Rate bei σ = 1,0 (Euler) 0,82, Ebenenvarianz steigt von Ebene 1 auf 2.
- **Gelaufene Schätzungen,** Kostenverhältnis gewöhnlich/Multilevel: Milstein, Fehlbestand, ε = 0,0025: 10,4 (RMSE/ε 0,91), ε = 0,04: 0,03; Euler, Fehlbestand, ε = 0,0025: 1,85; Euler, Sprung, ε = 0,01: 0,34.
- **Zwei Ebenen, ε = 0,0025,** Verhältnis gewöhnlich/Zwei-Ebenen gegen gewöhnlich/Multilevel: Euler, Sprung 2,28 gegen 0,65 (bestes k = 0); Euler, Fehlbestand 2,92 gegen 2,66; Milstein, Fehlbestand 8,8 gegen 15,8.
- **Ein-Term-Varianz bei Abschneideebene 3 → 8:** Milstein, Fehlbestand, s = 1,5: 0,19 → 0,21; Milstein, Endbestand, s = 1,5: 1,11 → 1,17; Euler, Fehlbestand, s = 1,5: 0,28 → 2,65; Euler, Endbestand, s = 2: 1,52 → 35,5; Milstein, Sprung, s = 1,5: 0,53 → 3,66. Lauf (Milstein, Fehlbestand): Mittel 0,1647 gegen exakt 0,1655, Stichprobenvarianz 0,221 (erwartet 0,211), Kosten 2,84 Schritte je Probe.

## Ehrliche Grenzen

- **Nur eine geometrische Brownsche Bewegung,** deren Endwert sich direkt ziehen ließe; sie dient als Orakel. **Keine pfadabhängigen Ausgaben** (asiatisch, Barriere, Lookback), obwohl Giles' Standardbeispiele dort liegen; dafür bräuchte die Demo eine Referenzlösung.
- **Gewöhnliches Monte Carlo ist mit Orakelwissen gerechnet** (es kennt Verzerrung und Varianz) und wird dadurch begünstigt; Giles' Verfahren zahlt dagegen für den Pilot.
- **Kosten sind Zeitschritte,** nicht Laufzeit; nur die Verfeinerung M = 2; schwache Ordnung fest 1 im Abbruchtest; kein Quasi-Monte-Carlo im Multilevel, keine adaptiven Zeitschritte.
- **Die Ebenenvarianz bei Sprung und Milstein** beruht auf seltenen Ereignissen und ist mit 50 000 Bahnen nur grob gemessen (hohe Kurtosis); die Verzerrungskonstante schwankt um etwa 30 % (die Bänder der Tests sind dort weit).
- **Bei großem σ** gilt die Rate erst ab einer tieferen Ebene; die Fortsetzung jenseits der gemessenen Ebenen ist dort unsicher.
- **Der Kostensatz (drei Fälle β gegen γ) und Giles' Verfahren stehen aus dem Gedächtnis im Mathematik-Expander;** Giles (2015) und Rhee und Glynn (2015) sind nur im Abstract gelesen. Die Ein-Term-Varianzen stammen aus den gemessenen zweiten Momenten; bei wachsender Varianz ist die Stichprobenvarianz im Lauf selbst unzuverlässig.
- **Literaturangaben im Mathematik-Expander** stammen aus dem Gedächtnis.

## Verifikation

- **Kern** (`tests/test_algorithm.py`): eine Euler-Bahn mit vier Schritten und festen Normalzahlen von Hand (fein 1,6875, grob 2); die Milstein-Bahn, bei der die grobe Bahn die feine trifft (Y = 0); Kosten je Probe; optimale Zuteilung von Hand (V = (1, ½, ¼), C = (1, 3, 6): 6,899 / 2,816 / 1,408 und 23,80 ε⁻²); E[P_0] für Euler und Milstein von Hand; exakte Werte; Ein-Term-Wahrscheinlichkeiten; Zufallsstrom.
- **Orakel** (`tests/test_oracle_mlm.py`): **exakter Wert gegen Quadratur** (zwölf Parameterfälle), **E[P_0] gegen Quadratur** mit Nullstellen aus Wurzelsuche (dreißig Fälle inklusive Milstein mit quadratischer Auszahlung), Normalzahlen gegen `scipy.stats` (Kolmogorov–Smirnov), **E[S_n] = (1 + a h)^n** für beide Verfahren, **Ebenenmittel gegen Gauß–Hermite-Produktintegrale** (Ebenen 1 und 2), Teleskopsumme, **Kopplung** (die grobe Bahn gleicht einer von Hand aus den summierten Zuwächsen gebauten), Zwei-Ebenen-Paare gegen Nachbarpaare, **jede zufällige Störung der optimalen Zuteilung ist nicht billiger**.
- **Eigenschaften** (`tests/test_properties.py`): Rate Euler 1 und Milstein 2, ungekoppelt konstant und etwa das Doppelte der Varianz der Ausgabe, Verzerrung halbiert sich (und gleicht der geschlossenen Form beim Endbestand), Ebene und Kosten wachsen mit enger werdender Toleranz, Kostenanteile summieren sich zu 1, ungekoppelt viel teurer als gewöhnlich, Reproduzierbarkeit, gelaufener Schätzer trifft die Toleranz, Ein-Term-Schätzer unverzerrt bei Milstein und mit wachsender Varianz bei Euler, Zwei-Ebenen-Varianz fällt zur feinen Ebene hin.
- **Randfälle** (`tests/test_edge_cases.py`): sehr lockere und sehr enge Toleranz, konstante Ausgabe (ehrliches Verdikt statt Division durch null), Null-Varianz in Giles' Verfahren, eine Probe, Regression mit zu wenigen Punkten, unbekannte Namen, Eingabeprüfung der Spielwiese, Zahlenformate, Extrapolation.
- **Verdikte** (`tests/test_verdicts.py`): jede Schwelle und jede genannte Zahl an von Hand gesetzten Werten, die drei Kostenfälle der Rate, Ebene und Kosten von Hand.
- **Presets, Layout, Oberfläche:** Presets vollständig und jedes mit dem erwarteten Verdikt; Schnellstart oben, fester Kopf, aktives Preset hervorgehoben; AppTest für jeden Button, jede Auswahl jedes Reglers, die Experimente, die Spielwiese, den PDF-Export und den Permalink.
- **Laufzeit-Importe, Footer, Claims** (`tests/test_claims.py`).

Zahl der Tests: siehe `pytest --collect-only -q tests`.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `mlm_algorithm.py` | Kern: Strom, Ausgaben, Euler und Milstein, gekoppelte Ebenenpaare, Ebenenstatistik, Giles' Verfahren, Ein-Term-Schätzer |
| `mlm_constants.py` | Regler-Grenzen, feste Annahmen, Farben, Presets |
| `mlm_evaluation.py` | Kennzahlen, Kostenrechnung, Verdikt-Logik, Experimente über feste Ströme |
| `mlm_layout.py` | Fester Kopf (Stylesheet) |
| `mlm_pdf_export.py` | PDF der aktuellen Einstellung |
| `mlm_presets.py` | Permalink (Begrenzen), Presets (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben), Zufallsstrom-Knopf |
| `mlm_scenario.py` | Szenario der Spielwiese (Drift, σ, Schwelle) und seine Prüfung |
| `mlm_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `requirements-dev.txt` | Laufzeit plus pytest und scipy |
| `requirements.txt` | Laufzeit-Bibliotheken |
| `tests/` | Kern von Hand, Orakel, Eigenschaften, Randfälle, Verdikte, Presets, Layout, AppTest, Footer, Laufzeit-Importe, Claims |

## Bewusst nicht umgesetzt

| Annahme | Verweis |
|---|---|
| Zwei Modelle mit festem Budgetanteil | [mehrere-modelle-demo](https://github.com/sebastian-hanisch/mehrere-modelle-demo) (Stück 4) |
| Kopplung als Handwerk (gemeinsame Zufallszahlen) | [gekoppelte-ziehungen-demo](https://github.com/sebastian-hanisch/gekoppelte-ziehungen-demo) (Stück 2) |
| „Multilevel Splitting“ für seltene Ereignisse (anderer Mechanismus, gleicher Name) | [splitting-demo](https://github.com/sebastian-hanisch/splitting-demo) (Warteschlangen) |
| Wert der Information beim Bestand | [newsvendor-evpi-demo](https://github.com/sebastian-hanisch/newsvendor-evpi-demo) |
| Geschachtelte Erwartungen (innere Stichprobe als Ebene) | Stück 13 der Linie |
| Pfadabhängige Ausgaben, Brownsche Brücke, adaptive Zeitschritte, Quasi-Monte-Carlo im Multilevel | nicht gebaut |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [Monte-Carlo-Schätzung: vom Schätzer zur Wahl](https://sebastianhanisch.net/konzepte-monte-carlo.html).
