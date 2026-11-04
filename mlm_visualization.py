"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import mlm_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), plot_bgcolor="rgba(0,0,0,0)")
    fig.update_yaxes(exponentformat="power")
    return lock_axes(fig)


def build_levels(a):
    """Varianz der Ebenendifferenz gegen die Ebene (logarithmisch): gekoppelt, ungekoppelt, dazu |E[Y_l]| und Steigungen −1 und −2 als Hilfslinien."""
    ls = [r["l"] for r in a.levels]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ls, y=[r["var"] if r["var"] > 0 else None for r in a.levels], mode="lines+markers", name="V_l gekoppelt", line=dict(color=C.COLORS["gekoppelt"])))
    fig.add_trace(go.Scatter(x=ls[1:], y=[v if v > 0 else None for v in a.var_unc[1:]], mode="lines+markers", name="V_l ungekoppelt", line=dict(color=C.COLORS["ungekoppelt"], dash="dash")))
    fig.add_trace(go.Scatter(x=ls[1:], y=[abs(r["mean"]) if r["mean"] != 0 else None for r in a.levels[1:]], mode="lines+markers", name="|E[Y_l]|", line=dict(color=C.COLORS["bias"], dash="dot")))
    base = a.levels[1]["var"] if a.levels[1]["var"] > 0 else 1.0
    for slope, name in ((1, "Steigung −1 (β = 1)"), (2, "Steigung −2 (β = 2)")):
        fig.add_trace(go.Scatter(x=ls[1:], y=[base * 2.0 ** (-slope * (l - 1)) for l in ls[1:]], mode="lines", name=name, line=dict(color=C.COLORS["ref"], dash="dot", width=1)))
    fig.update_xaxes(title_text="Ebene l (2^l Zeitschritte)", tickvals=ls)
    fig.update_yaxes(title_text="Varianz bzw. Mittel der Differenz", type="log")
    return _base(fig, 380)


def build_cost(rows):
    """Idealisierte Kosten (Zeitschritte) gegen die Toleranz ε, doppelt logarithmisch."""
    fig = go.Figure()
    xs = [r["eps"] for r in rows]
    fig.add_trace(go.Scatter(x=xs, y=[r["mc"] for r in rows], mode="lines+markers", name="gewöhnlich", line=dict(color=C.COLORS["mc"])))
    fig.add_trace(go.Scatter(x=xs, y=[r["ml"] for r in rows], mode="lines+markers", name="Multilevel", line=dict(color=C.COLORS["ml"])))
    fig.update_xaxes(title_text="Toleranz ε", type="log", autorange="reversed", tickvals=xs, ticktext=[f"{x:g}".replace(".", ",") for x in xs])
    fig.update_yaxes(title_text="Kosten (Zeitschritte)", type="log")
    return _base(fig, 340)


def build_cost_norm(rows):
    """ε² · Kosten gegen ε: bei Sättigung (Multilevel mit β > γ) waagerecht, gewöhnlich wächst mit 1/ε."""
    fig = go.Figure()
    xs = [r["eps"] for r in rows]
    fig.add_trace(go.Scatter(x=xs, y=[r["eps2_mc"] for r in rows], mode="lines+markers", name="gewöhnlich", line=dict(color=C.COLORS["mc"])))
    fig.add_trace(go.Scatter(x=xs, y=[r["eps2_ml"] for r in rows], mode="lines+markers", name="Multilevel", line=dict(color=C.COLORS["ml"])))
    fig.update_xaxes(title_text="Toleranz ε", type="log", autorange="reversed", tickvals=xs, ticktext=[f"{x:g}".replace(".", ",") for x in xs])
    fig.update_yaxes(title_text="ε² · Kosten", type="log")
    return _base(fig, 340)


def build_shares(row):
    """Kostenanteil je Ebene des Multilevel-Schätzers bei der gewählten Toleranz."""
    fig = go.Figure(go.Bar(x=list(range(len(row["shares"]))), y=[100 * s for s in row["shares"]], marker_color=C.COLORS["ml"], name="Kostenanteil"))
    fig.update_xaxes(title_text="Ebene l", tickvals=list(range(len(row["shares"]))))
    fig.update_yaxes(title_text="Anteil an den Kosten (%)", range=[0, 100])
    fig.update_layout(showlegend=False)
    return _base(fig, 260)


def build_run(ex):
    """Mittlere Kosten der gelaufenen Schätzungen: gewöhnlich (Orakel) gegen Multilevel."""
    fig = go.Figure(go.Bar(x=["gewöhnlich (Orakelwissen)", "Multilevel (Giles)"], y=[ex["mc"]["cost"], ex["ml"]["cost"]], marker_color=[C.COLORS["mc"], C.COLORS["ml"]],
                           text=[f"RMSE/ε = {ex['mc']['rmse'] / ex['eps']:.2f}".replace(".", ","), f"RMSE/ε = {ex['ml']['rmse'] / ex['eps']:.2f}".replace(".", ",")], textposition="outside"))
    fig.update_yaxes(title_text="mittlere Kosten (Zeitschritte)", type="log")
    fig.update_layout(showlegend=False)
    return _base(fig, 320)


def build_two(ex):
    """Kosten des Zwei-Ebenen-Schätzers gegen die grobe Ebene k, mit gewöhnlich und Multilevel als waagerechten Linien."""
    ks = [r["k"] for r in ex["rows"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ks, y=[r["cost"] for r in ex["rows"]], mode="lines+markers", name="Zwei Ebenen (k, L)", line=dict(color=C.COLORS["two"])))
    fig.add_hline(y=ex["mc"], line=dict(color=C.COLORS["mc"], dash="dash"), annotation_text="gewöhnlich", annotation_position="top right")
    fig.add_hline(y=ex["ml"], line=dict(color=C.COLORS["ml"], dash="dash"), annotation_text="Multilevel", annotation_position="bottom right")
    fig.update_xaxes(title_text=f"grobe Ebene k (feine Ebene L = {ex['L']})", tickvals=ks)
    fig.update_yaxes(title_text="Kosten (Zeitschritte)", type="log")
    return _base(fig, 340)


def build_single(ex):
    """Varianz des entzerrten Ein-Term-Schätzers gegen die Abschneideebene für drei Abfallexponenten."""
    fig = go.Figure()
    palette = {1.0: C.COLORS["gekoppelt"], 1.5: C.COLORS["two"], 2.0: C.COLORS["bias"]}
    for s, cv in ex["curves"].items():
        fig.add_trace(go.Scatter(x=[r["lmax"] for r in cv], y=[r["var"] if np.isfinite(r["var"]) and r["var"] > 0 else None for r in cv], mode="lines+markers",
                                 name=f"p_l ∝ 2^(−{s:g} l)".replace(".", ","), line=dict(color=palette[s])))
    fig.update_xaxes(title_text="Abschneideebene", tickvals=[r["lmax"] for r in next(iter(ex["curves"].values()))])
    fig.update_yaxes(title_text="Varianz je Probe", type="log")
    return _base(fig, 340)
