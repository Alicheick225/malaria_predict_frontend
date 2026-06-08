"""Tendances nationales — comparaison multi-districts, heatmap, saisonnalité, top districts."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.charts import (
    ACCENT,
    GRID_COLOR,
    PANEL_BG,
    RISK_COLORS,
    TEXT_COLOR,
    plot_seasonal_boxplot,
    plot_top_districts,
)
from components.styles import ALERT_HIGH, PRIMARY, inject_css
from utils.api_client import cached_district_history, cached_districts_geojson, cached_districts_table

st.set_page_config(layout="wide", page_title="Tendances — MalariaWatch CI", page_icon="🦟")
inject_css()

horizon = st.session_state.get("horizon", 6)
table   = cached_districts_table(horizon)

st.title("Tendances nationales")
st.caption(
    "Vue d'ensemble du risque épidémique à l'échelle des 33 districts sanitaires de Côte d'Ivoire. "
    f"Le modèle LSTM prédit les {horizon} prochains mois à partir des données historiques 2010-2024."
)

names = sorted(table["district_name"].unique().tolist())

# ── Palette de couleurs pour les districts ─────────────────────────────────────
PALETTE = [
    "#3498DB", "#2ECC71", "#E74C3C", "#F39C12",
    "#9B59B6", "#1ABC9C", "#E67E22", "#D35400",
    "#2980B9", "#27AE60",
]

# ══════════════════════════════════════════════════════════════════════════════
# 1. Comparaison multi-districts (historique + prédictions)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Comment l'incidence prédit-elle d'évoluer d'un district à l'autre ?")
st.markdown(
    '<div class="section-intro">'
    "Le modèle LSTM génère des prévisions pour les prochains mois à partir de l'historique climatique "
    "et épidémiologique. Ci-dessous, les <strong>lignes pleines</strong> représentent les données "
    "observées, les <strong>lignes pointillées</strong> les prévisions du modèle, "
    "et la <strong>zone grisée</strong> l'intervalle de confiance indicatif (± 15 %)."
    "</div>",
    unsafe_allow_html=True,
)

col_sel, col_range = st.columns([2, 1])
with col_sel:
    selection = st.multiselect(
        "Districts à comparer", names, default=names[:4],
        help="Sélectionnez jusqu'à 8 districts pour un rendu lisible.",
    )
with col_range:
    year_range = st.slider(
        "Période affichée", min_value=2020, max_value=2027, value=(2023, 2027),
        help="Glissez pour zoomer sur la période souhaitée.",
    )

if selection:
    today_ts = pd.Timestamp.now().normalize()
    fig = go.Figure()

    for idx, name in enumerate(selection[:8]):
        row = table[table["district_name"] == name]
        if row.empty:
            continue
        did   = row["district_id"].iloc[0]
        hist  = cached_district_history(did)
        hist["date"] = pd.to_datetime(hist["date"])

        # Filtre sur la période choisie
        hist = hist[
            (hist["date"].dt.year >= year_range[0]) &
            (hist["date"].dt.year <= year_range[1])
        ]

        color = PALETTE[idx % len(PALETTE)]
        obs   = hist.dropna(subset=["cases_observed"]).sort_values("date")
        pred  = hist.dropna(subset=["cases_predicted"]).sort_values("date")

        # Données observées — trait plein
        if not obs.empty:
            fig.add_trace(go.Scatter(
                x=obs["date"], y=obs["cases_observed"],
                name=f"{name} (observé)",
                mode="lines",
                line=dict(color=color, width=2),
                legendgroup=name,
            ))

        # Prévisions — trait pointillé + bande de confiance
        if not pred.empty:
            upper = pred["cases_predicted"] * 1.15
            lower = pred["cases_predicted"] * 0.85
            # Bande de confiance (remplie, semi-transparente)
            r_int = int(color[1:3], 16)
            g_int = int(color[3:5], 16)
            b_int = int(color[5:7], 16)
            fig.add_trace(go.Scatter(
                x=pd.concat([pred["date"], pred["date"][::-1]]),
                y=pd.concat([upper, lower[::-1]]),
                fill="toself",
                fillcolor=f"rgba({r_int},{g_int},{b_int},0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{name} (IC ±15%)",
                legendgroup=name,
                showlegend=False,
                hoverinfo="skip",
            ))
            # Ligne pointillée des prédictions
            fig.add_trace(go.Scatter(
                x=pred["date"], y=pred["cases_predicted"],
                name=f"{name} (prédit)",
                mode="lines+markers",
                line=dict(color=color, width=2, dash="dot"),
                marker=dict(size=5),
                legendgroup=name,
            ))

    # Ligne verticale : début des prédictions
    fig.add_vline(
        x=today_ts,
        line_dash="dash",
        line_color="#A0A8C0",
        line_width=1.5,
        annotation_text="Début des prédictions",
        annotation_position="top right",
        annotation_font_color="#A0A8C0",
        annotation_font_size=11,
    )

    fig.update_layout(
        height=480,
        plot_bgcolor=PANEL_BG,
        paper_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR, title="Période"),
        yaxis=dict(title="Taux d'incidence (/1 000 hab.)", gridcolor=GRID_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "Les prévisions au-delà de la ligne pointillée utilisent une climatologie saisonnière "
        "comme proxy pour les données climatiques futures (limite documentée du modèle)."
    )
else:
    st.info("Sélectionnez au moins un district à comparer.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 2. Heatmap nationale — tous districts × 12 derniers mois
# ══════════════════════════════════════════════════════════════════════════════
st.subheader(
    "Évolution de l'incidence du paludisme par district et par période "
    "(cas pour 1 000 habitants)"
)
st.markdown(
    '<div class="section-intro">'
    "Chaque ligne représente un district sanitaire, chaque colonne un mois. "
    "La couleur indique le taux d'incidence observé : "
    "<span style='color:#2ECC71'>vert = faible</span>, "
    "<span style='color:#F39C12'>orange = modéré</span>, "
    "<span style='color:#E74C3C'>rouge = élevé</span>. "
    "Les districts sont triés de haut en bas par incidence moyenne décroissante. "
    "Cette vue met en évidence la saisonnalité et les disparités géographiques persistantes."
    "</div>",
    unsafe_allow_html=True,
)

geojson = cached_districts_geojson(horizon)
rows    = [f["properties"] for f in geojson["features"]]
gdf     = pd.DataFrame(rows)

heat_rows = []
for _, r in gdf.iterrows():
    hist = cached_district_history(r["district_id"])
    hist["date"] = pd.to_datetime(hist["date"])
    obs  = hist.dropna(subset=["cases_observed"]).sort_values("date").tail(12)
    for _, h in obs.iterrows():
        heat_rows.append({
            "district": r["district_name"],
            "date":     h["date"],
            "incidence": h["cases_observed"],
        })

heat_df = pd.DataFrame(heat_rows)
if not heat_df.empty:
    pivot = heat_df.pivot_table(
        index="district", columns="date", values="incidence", aggfunc="mean"
    )
    pivot = pivot.reindex(pivot.mean(axis=1).sort_values(ascending=False).index)

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[d.strftime("%b %Y") for d in pivot.columns],
        y=pivot.index.tolist(),
        colorscale=[[0, "#2ECC71"], [0.5, "#F39C12"], [1, ALERT_HIGH]],
        colorbar=dict(
            title=dict(text="Incidence<br>(/1 000 hab.)", side="right"),
            tickfont=dict(color=TEXT_COLOR),
        ),
        hovertemplate="<b>%{y}</b><br>Période : %{x}<br>Incidence : %{z:.1f} /1000 hab.<extra></extra>",
    ))
    fig_heat.update_layout(
        height=760,
        plot_bgcolor=PANEL_BG,
        paper_bgcolor=PANEL_BG,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(title="Période (mois)", tickangle=-30, gridcolor=GRID_COLOR),
        yaxis=dict(title="District sanitaire", gridcolor=GRID_COLOR),
        margin=dict(l=160, r=20, t=30, b=80),
    )
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})
    st.caption(
        "Retenir : les districts du Centre-Ouest et de l'Ouest forestier affichent "
        "systématiquement les incidences les plus élevées, avec des pics marqués en "
        "mai-juin et octobre-novembre (saisons des pluies)."
    )
else:
    st.info("Pas assez de données pour générer la carte de chaleur nationale.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 3. Saisonnalité nationale
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Quelle est la saisonnalité du risque épidémique à l'échelle nationale ?")
st.caption(
    "Distribution du score de risque prédit par mois calendaire — "
    "pics attendus en avril-juillet et septembre-novembre (saisons des pluies en Côte d'Ivoire)."
)

preds_full = pd.DataFrame()
if not table.empty:
    parts = []
    for d, n in zip(table["district_id"], table["district_name"]):
        tmp = cached_district_history(d)
        if not tmp.empty:
            tmp = tmp.assign(district_name=n)
            parts.append(tmp)
    if parts:
        preds_full = pd.concat(parts, ignore_index=True)

if not preds_full.empty:
    fc = preds_full.dropna(subset=["cases_predicted"]).copy()
    fc["week_predicted"] = pd.to_datetime(fc["date"])
    fc["risk_score"]     = fc["cases_predicted"] / fc["cases_predicted"].max()
    st.plotly_chart(
        plot_seasonal_boxplot(fc),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.caption(
        "Chaque boîte représente la distribution du score de risque sur tous les districts "
        "pour ce mois calendaire. La médiane (trait central) et les quartiles (boîte) "
        "indiquent l'étendue de la variabilité inter-districts."
    )
else:
    st.info("Données de prévision insuffisantes pour le graphique de saisonnalité.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# 4. Top 10 districts à risque (cette période)
# ══════════════════════════════════════════════════════════════════════════════
st.subheader("Quels sont les 10 districts les plus à risque cette période ?")
if not gdf.empty and "risk_score" in gdf.columns:
    st.plotly_chart(
        plot_top_districts(gdf.dropna(subset=["risk_score"]), n=10),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.caption(
        "Score de risque normalisé de 0 à 1. Un score > 0.66 déclenche une alerte "
        "rouge ; entre 0.33 et 0.66, une alerte orange (modérée)."
    )
else:
    st.info("Scores de risque indisponibles.")
