"""
Graphiques Plotly réutilisables — thème sombre cohérent avec la carte et le reste de l'UI.
Chaque fonction retourne une figure Plotly prête à être affichée avec st.plotly_chart().
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DARK_BG    = "#0F1117"
PANEL_BG   = "#1A1D27"
TEXT_COLOR = "#FAFAFA"
GRID_COLOR = "#2E3250"
ACCENT     = "#3498DB"
RISK_COLORS = {"Faible": "#2ECC71", "Modéré": "#F39C12", "Élevé": "#E74C3C"}

FEATURE_LABELS = {
    "t2m_c": "Température (°C)",
    "rh_pct": "Humidité relative (%)",
    "tp_mm": "Précipitations (mm)",
    "ndvi": "NDVI (végétation)",
    "ndwi": "NDWI (eau)",
    "pfpr": "Prévalence Pf (PfPR)",
    "itn_use": "Usage moustiquaires (ITN)",
    "itn_access": "Accès moustiquaires (ITN)",
    "population": "Population",
}


def _base_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        title=title,
        height=height,
        plot_bgcolor=PANEL_BG,
        paper_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR),
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR),
    )
    return fig


# ── 1. Série temporelle observé vs prédit ───────────────────────────────────
def plot_timeseries(history: pd.DataFrame, title: str = "Cas observés vs prédits") -> go.Figure:
    h = history.copy()
    h["date"] = pd.to_datetime(h["date"])
    h = h.sort_values("date")

    fig = go.Figure()
    obs = h.dropna(subset=["cases_observed"]) if "cases_observed" in h else pd.DataFrame()
    pred = h.dropna(subset=["cases_predicted"]) if "cases_predicted" in h else pd.DataFrame()

    if not obs.empty:
        fig.add_trace(go.Scatter(
            x=obs["date"], y=obs["cases_observed"], name="Cas observés",
            mode="lines", line=dict(color=ACCENT, width=2),
        ))
    if not pred.empty:
        fig.add_trace(go.Scatter(
            x=pred["date"], y=pred["cases_predicted"], name="Cas prédits",
            mode="lines+markers", line=dict(color="#E74C3C", width=2, dash="dot"),
        ))
        # Zone de confiance simulée (±15%) — illustrative tant qu'aucun intervalle n'est calibré
        upper = pred["cases_predicted"] * 1.15
        lower = pred["cases_predicted"] * 0.85
        fig.add_trace(go.Scatter(
            x=pd.concat([pred["date"], pred["date"][::-1]]),
            y=pd.concat([upper, lower[::-1]]),
            fill="toself", fillcolor="rgba(231,76,60,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="Intervalle (P10-P90, indicatif)",
            hoverinfo="skip",
        ))

    return _base_layout(fig, title)


# ── 2. Précipitations vs cas (overlay) ──────────────────────────────────────
def plot_rainfall_overlay(history: pd.DataFrame, rainfall: pd.DataFrame) -> go.Figure:
    """rainfall: DataFrame avec colonnes date, tp_mm — observé."""
    h = history.dropna(subset=["cases_observed"]).copy()
    h["date"] = pd.to_datetime(h["date"])
    r = rainfall.copy()
    r["date"] = pd.to_datetime(r["date"])

    fig = go.Figure()
    fig.add_trace(go.Bar(x=r["date"], y=r["tp_mm"], name="Précipitations (mm)",
                         marker_color="#3498DB", opacity=0.55, yaxis="y2"))
    fig.add_trace(go.Scatter(x=h["date"], y=h["cases_observed"], name="Taux d'incidence (/1000 hab.)",
                             mode="lines", line=dict(color="#E74C3C", width=2)))

    fig.update_layout(
        yaxis=dict(title="Taux d'incidence", gridcolor=GRID_COLOR),
        yaxis2=dict(title="Précipitations (mm)", overlaying="y", side="right", showgrid=False),
    )
    return _base_layout(fig, "Précipitations vs incidence — effet décalé pluie → paludisme")


# ── 3. Heatmap risque par mois/année ─────────────────────────────────────────
def plot_risk_heatmap(history: pd.DataFrame, title: str = "Risque par mois (2 dernières années)") -> go.Figure:
    h = history.dropna(subset=["cases_observed"]).copy()
    h["date"] = pd.to_datetime(h["date"])
    h = h[h["date"] >= h["date"].max() - pd.DateOffset(years=2)]
    h["year"] = h["date"].dt.year
    h["month"] = h["date"].dt.month

    pivot = h.pivot_table(index="month", columns="year", values="cases_observed", aggfunc="mean")
    pivot = pivot.reindex(range(1, 13))
    months_fr = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]

    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[str(c) for c in pivot.columns], y=months_fr,
        colorscale=[[0, "#2ECC71"], [0.5, "#F39C12"], [1, "#E74C3C"]],
        colorbar=dict(title="Incidence"),
    ))
    return _base_layout(fig, title, height=380)


# ── 4. Radar — district vs moyenne nationale ─────────────────────────────────
def plot_radar(district_values: dict, national_values: dict, district_name: str) -> go.Figure:
    feats = list(district_values.keys())
    labels = [FEATURE_LABELS.get(f, f) for f in feats]

    def _norm(d):
        return [
            (d[f] - min(d[f], national_values[f]) ) / (max(abs(d[f]), abs(national_values[f]), 1e-9))
            if max(abs(d[f]), abs(national_values[f])) else 0
            for f in feats
        ]

    # Normalisation simple par feature (par rapport au max des deux séries) pour rendre le radar lisible
    d_norm, n_norm = [], []
    for f in feats:
        m = max(abs(district_values[f]), abs(national_values[f]), 1e-9)
        d_norm.append(district_values[f] / m)
        n_norm.append(national_values[f] / m)

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=n_norm + n_norm[:1], theta=labels + labels[:1],
                                  name="Moyenne nationale", fill="toself",
                                  line=dict(color="#7F8C9A"), opacity=0.5))
    fig.add_trace(go.Scatterpolar(r=d_norm + d_norm[:1], theta=labels + labels[:1],
                                  name=district_name, fill="toself",
                                  line=dict(color=ACCENT)))
    fig.update_layout(polar=dict(bgcolor=PANEL_BG, radialaxis=dict(visible=True, gridcolor=GRID_COLOR)))
    return _base_layout(fig, f"{district_name} vs moyenne nationale", height=440)


# ── 5. Contributions des features (proxy SHAP) ──────────────────────────────
def plot_shap_bars(top_features: list[dict], title: str = "Facteurs les plus déterminants") -> go.Figure:
    if not top_features:
        fig = go.Figure()
        fig.add_annotation(text="Pas de données d'explicabilité disponibles", showarrow=False,
                           font=dict(color=TEXT_COLOR))
        return _base_layout(fig, title, height=320)

    df = pd.DataFrame(top_features).sort_values("contribution")
    df["label"] = df["feature"].map(lambda f: FEATURE_LABELS.get(f, f))

    fig = go.Figure(go.Bar(
        x=df["contribution"], y=df["label"], orientation="h",
        marker_color=ACCENT,
        text=[f"valeur: {v:.3g}" for v in df["value"]],
        textposition="outside",
    ))
    fig.update_layout(xaxis_title="Contribution à la prédiction (|valeur SHAP|)")
    return _base_layout(fig, title, height=320)


# ── 6. Boxplot saisonnier ────────────────────────────────────────────────────
def plot_seasonal_boxplot(df: pd.DataFrame, title: str = "Saisonnalité du risque (par mois)") -> go.Figure:
    d = df.copy()
    d["week_predicted"] = pd.to_datetime(d["week_predicted"]) if "week_predicted" in d else pd.to_datetime(d["date"])
    d["mois"] = d["week_predicted"].dt.month
    months_fr = ["Jan", "Fév", "Mar", "Avr", "Mai", "Juin", "Juil", "Août", "Sep", "Oct", "Nov", "Déc"]
    d["mois_label"] = d["mois"].map(lambda m: months_fr[m - 1])

    fig = px.box(d, x="mois_label", y="risk_score", category_orders={"mois_label": months_fr},
                 color_discrete_sequence=[ACCENT])
    fig.update_layout(xaxis_title="Mois", yaxis_title="Score de risque")
    return _base_layout(fig, title)


# ── 7. Top districts à risque ────────────────────────────────────────────────
def plot_top_districts(df: pd.DataFrame, n: int = 10, title: str = "Top districts les plus à risque") -> go.Figure:
    d = df.sort_values("risk_score", ascending=False).head(n).sort_values("risk_score")
    colors = [RISK_COLORS.get(lvl, ACCENT) for lvl in d["risk_level"]]
    fig = go.Figure(go.Bar(
        x=d["risk_score"], y=d["district_name"], orientation="h",
        marker_color=colors,
        text=[f"{s:.2f}" for s in d["risk_score"]], textposition="outside",
    ))
    fig.update_layout(xaxis_title="Score de risque", xaxis_range=[0, 1.05])
    return _base_layout(fig, title, height=420)


# ── 7bis. Cas prédits par district (barres verticales, vue tableau de bord) ──
def plot_predicted_cases_bar(df: pd.DataFrame, n: int = 12,
                              title: str = "Cas prédits par district (taux /1000 hab.)") -> go.Figure:
    d = df.sort_values("cases_predicted", ascending=False).head(n)
    colors = [RISK_COLORS.get(lvl, ACCENT) for lvl in d["risk_level"]]
    fig = go.Figure(go.Bar(
        x=d["district_name"], y=d["cases_predicted"],
        marker_color=colors,
        text=[f"{v:.0f}" for v in d["cases_predicted"]], textposition="outside",
    ))
    fig.update_layout(xaxis_title="", yaxis_title="Taux d'incidence prédit (/1000 hab.)",
                      xaxis=dict(tickangle=-45))
    return _base_layout(fig, title, height=460)


# ── 7ter. Évolution nationale prédite (agrégat multi-districts) ─────────────
def plot_national_trend(df: pd.DataFrame, title: str = "Évolution de l'incidence nationale prédite") -> go.Figure:
    g = df.groupby("week_predicted")["cases_predicted"].mean().reset_index()
    fig = go.Figure(go.Scatter(
        x=g["week_predicted"], y=g["cases_predicted"],
        mode="lines+markers", line=dict(color=ACCENT, width=3),
        marker=dict(size=8), fill="tozeroy", fillcolor="rgba(52,152,219,0.15)",
    ))
    fig.update_layout(xaxis_title="Mois prédit", yaxis_title="Taux d'incidence moyen national (/1000 hab.)")
    return _base_layout(fig, title)


def plot_risk_distribution_over_time(df: pd.DataFrame,
                                       title: str = "Répartition des districts par niveau de risque") -> go.Figure:
    g = df.groupby(["week_predicted", "risk_level"]).size().reset_index(name="n")
    fig = go.Figure()
    for level in ["Faible", "Modéré", "Élevé"]:
        sub = g[g["risk_level"] == level]
        fig.add_trace(go.Bar(x=sub["week_predicted"], y=sub["n"], name=level,
                             marker_color=RISK_COLORS[level]))
    fig.update_layout(barmode="stack", xaxis_title="Mois prédit", yaxis_title="Nombre de districts")
    return _base_layout(fig, title)


def plot_cumulative_national(df: pd.DataFrame,
                              title: str = "Cumul prédit de l'incidence nationale") -> go.Figure:
    g = df.groupby("week_predicted")["cases_predicted"].mean().reset_index()
    g["cumule"] = g["cases_predicted"].cumsum()
    fig = go.Figure(go.Scatter(
        x=g["week_predicted"], y=g["cumule"],
        mode="lines+markers", line=dict(color="#9B59B6", width=3),
        marker=dict(size=8), fill="tozeroy", fillcolor="rgba(155,89,182,0.15)",
    ))
    fig.update_layout(xaxis_title="Mois prédit", yaxis_title="Cumul du taux d'incidence moyen (/1000 hab.)")
    return _base_layout(fig, title)


# ── 8. Scatter observé vs prédit (page modèle) ───────────────────────────────
def plot_scatter_obs_pred(df: pd.DataFrame, title: str = "Cas observés vs prédits (jeu de test)") -> go.Figure:
    fig = px.scatter(df, x="observed", y="predicted", opacity=0.55,
                     color_discrete_sequence=[ACCENT])
    lo = float(min(df["observed"].min(), df["predicted"].min()))
    hi = float(max(df["observed"].max(), df["predicted"].max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                             line=dict(color="#E74C3C", dash="dash"), name="y = x (référence)"))
    fig.update_layout(xaxis_title="Observé (taux d'incidence /1000 hab.)", yaxis_title="Prédit")
    return _base_layout(fig, title)


# ── 9. Courbes de loss ───────────────────────────────────────────────────────
def plot_training_curves(history: pd.DataFrame, title: str = "Courbes d'apprentissage") -> go.Figure:
    fig = go.Figure()
    if "loss" in history:
        fig.add_trace(go.Scatter(x=history.index, y=history["loss"], name="Train loss",
                                 line=dict(color=ACCENT)))
    if "val_loss" in history:
        fig.add_trace(go.Scatter(x=history.index, y=history["val_loss"], name="Validation loss",
                                 line=dict(color="#E74C3C", dash="dot")))
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss (MSE, échelle normalisée)")
    return _base_layout(fig, title)


# ── 10. Importance globale des features ─────────────────────────────────────
def plot_global_feature_importance(df: pd.DataFrame, title: str = "Importance globale des features") -> go.Figure:
    agg = (df.groupby("feature")["contribution"].mean()
             .sort_values()
             .rename(index=lambda f: FEATURE_LABELS.get(f, f)))
    fig = go.Figure(go.Bar(x=agg.values, y=agg.index, orientation="h", marker_color=ACCENT))
    fig.update_layout(xaxis_title="Contribution moyenne (|valeur SHAP|)")
    return _base_layout(fig, title, height=380)
