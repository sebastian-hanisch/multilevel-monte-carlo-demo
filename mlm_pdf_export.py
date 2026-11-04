"""PDF-Export der aktuellen Einstellung (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Sonderzeichen werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import mlm_constants as C
from mlm_evaluation import Analysis, Settings, cost_table, de, eps_text, ke, verdict_main

_REPLACE = {"→": "->", "–": "-", "—": "-", "−": "-", "≤": "<=", "≥": ">=", "·": "*", "€": "EUR", "„": '"', "“": '"', "²": "^2", "³": "3", "×": "x", "√": "Wurzel", "≈": "~",
            "σ": "sigma", "μ": "mu", "ρ": "rho", "α": "alpha", "β": "beta", "γ": "gamma", "θ": "theta", "τ": "tau", "ε": "eps", "∞": "unendlich", "Σ": "Summe", "δ": "delta", "Δ": "Delta",
            "½": "1/2", "̂": "", "ᵢ": "_i", "∫": "Integral", "⁺": "+", "′": "'", "⁻": "-", "¹": "1"}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(settings: Settings, a: Analysis) -> bytes:
    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Multilevel Monte Carlo: wann sich die Ebenen lohnen"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"Bestandsprozess dS = a S dt + sigma S dW, S(0) = 1, T = 1; a = {de(a.drift, 3)}, sigma = {de(a.sigma, 2)}, Schwelle q = {de(a.q, 2)}. {a.label}. Ebenen 0 bis {a.lmax}, "
                            f"{de(a.n, 0)} Proben je Ebene, Strom {settings.seed}. Alles simuliert, die exakte Erwartung ist {de(a.truth, 5)}. Kosten in Zeitschritten, idealisiert (bekannte Ebenenvarianzen und Verzerrung)."),
                   new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    heads = ("Toleranz", "Ebene L", "gewoehnlich", "Multilevel", "Verhaeltnis")
    widths = (30, 25, 45, 45, 35)
    pdf.set_font("Helvetica", "B", 9)
    for h, w in zip(heads, widths):
        pdf.cell(w, 6, _s(h), border=1)
    pdf.ln()
    pdf.set_font("Helvetica", "", 9)
    for r in cost_table(a):
        for text, w in zip((eps_text(r["eps"]), str(r["L"]), ke(r["mc"]), ke(r["ml"]), de(r["ratio"], 2)), widths):
            pdf.cell(w, 6, _s(text), border=1)
        pdf.ln()
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(verdict_main(a, settings)[1]), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Quelle: multilevel-monte-carlo-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
