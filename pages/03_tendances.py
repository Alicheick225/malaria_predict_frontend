"""Tendances nationales — comparaison multi-districts, heatmap, saisonnalité, top districts, animation temporelle."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from components.charts import ACCENT, GRID_COLOR, PANEL_BG, TEXT_COLOR, plot_seasonal_boxplot, plot_top_districts
from utils.api_client import cached_districts_geojson, cached_districts_table, cached_district_history

st.set_page_config(layout="wide", page_title="Tendances — MalariaWatch CI", page_icon="📈")
st.markdown("<style>.stApp { background-color: #1E1E2E; }</style>", unsafe_allow_html=True)

horizon = st.session_state.get("horizon", 6)
table = cached_districts_table(horizon)

st.title("📈 Tendances nationales")
st.caption("Vue d'ensemble du risque épidémique à l'échelle des 33 districts sanitaires de Côte d'Ivoire.")

names = sorted(table["district_name"].unique().tolist())

# ── 1. Comparaison multi-districts ───────────────────────────────────────────
st.subheader("Comparaison de l'évolution entre districts")
selection = st.multiselect("Districts à comparer", names, default=names[:3])

if selection:
    fig = go.Figure()
    for name in selection:
        row = table[table["district_name"] == name]
        if row.empty:
            continue
        did = row["district_id"].iloc[0]
        hist = cached_district_history(did)
        hist["date"] = pd.to_datetime(hist["date"])
        obs = hist.dropna(subset=["cases_observed"]).sort_values("date").tail(36)
        fig.add_trace(go.Scatter(x=obs["date"], y=obs["cases_observed"], name=name, mode="lines"))
    fig.update_layout(
        height=440, plot_bgcolor=PANEL_BG, paper_bgcolor=PANEL_BG, font=dict(color=TEXT_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR), yaxis=dict(title="Taux d'incidence (/1000 hab.)", gridcolor=GRID_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        title="Incidence observée — 3 dernières années",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sélectionnez au moins un district à comparer.")

st.divider()

# ── 2. Heatmap nationale (districts x mois) ──────────────────────────────────
st.subheader("Carte de chaleur nationale — risque par district (12 derniers mois)")
geojson = cached_districts_geojson(horizon)
rows = [f["properties"] for f in geojson["features"]]
gdf = pd.DataFrame(rows)

heat_rows = []
for _, r in gdf.iterrows():
    hist = cached_district_history(r["district_id"])
    hist["date"] = pd.to_datetime(hist["date"])
    obs = hist.dropna(subset=["cases_observed"]).sort_values("date").tail(12)
    for _, h in obs.iterrows():
        heat_rows.append({"district": r["district_name"], "date": h["date"], "incidence": h["cases_observed"]})

heat_df = pd.DataFrame(heat_rows)
if not heat_df.empty:
    pivot = heat_df.pivot_table(index="district", columns="date", values="incidence", aggfunc="mean")
    pivot = pivot.reindex(pivot.mean(axis=1).sort_values(ascending=False).index)
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[d.strftime("%Y-%m") for d in pivot.columns], y=pivot.index,
        colorscale=[[0, "#2ECC71"], [0.5, "#F39C12"], [1, "#E74C3C"]],
        colorbar=dict(title="Incidence"),
    ))
    fig.update_layout(height=720, plot_bgcolor=PANEL_BG, paper_bgcolor=PANEL_BG, font=dict(color=TEXT_COLOR),
                      margin=dict(l=140, r=20, t=30, b=40))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Pas assez de données pour la carte de chaleur nationale.")

st.divider()

# ── 3. Saisonnalité ───────────────────────────────────────────────────────────
st.subheader("Saisonnalité du risque épidémique")
st.caption("Distribution du score de risque prédit par mois calendaire — pics attendus avril-juillet "
           "et septembre-novembre (saisons des pluies).")
preds_full = pd.concat([
    cached_district_history(d).assign(district_name=n) for d, n in
    zip(table["district_id"], table["district_name"])
], ignore_index=True) if not table.empty else pd.DataFrame()

if not preds_full.empty:
    fc = preds_full.dropna(subset=["cases_predicted"]).copy()
    fc["week_predicted"] = pd.to_datetime(fc["date"])
    fc["risk_score"] = fc["cases_predicted"] / fc["cases_predicted"].max()
    st.plotly_chart(plot_seasonal_boxplot(fc), use_container_width=True)
else:
    st.info("Données de prévision insuffisantes pour le graphique de saisonnalité.")

st.divider()

# ── 4. Top 10 districts à risque ──────────────────────────────────────────────
st.subheader("Top 10 districts les plus à risque cette période")
if not gdf.empty and "risk_score" in gdf:
    st.plotly_chart(plot_top_districts(gdf.dropna(subset=["risk_score"]), n=10), use_container_width=True)
else:
    st.info("Scores de risque indisponibles.")

st.divider()

# ── 5. Carte choroplèthe animée ───────────────────────────────────────────────
st.subheader("Évolution du risque — animation temporelle")
st.caption(f"Glissez le curseur pour visualiser l'évolution du score de risque prédit sur l'horizon de {horizon} mois.")

anim_rows = []
table_geo = pd.DataFrame(rows)
for _, r in table_geo.iterrows():
    hist = cached_district_history(r["district_id"])
    hist["date"] = pd.to_datetime(hist["date"])
    fc = hist.dropna(subset=["cases_predicted"]).sort_values("date")
    for _, h in fc.iterrows():
        anim_rows.append({
            "district_name": r["district_name"],
            "date": h["date"].strftime("%Y-%m"),
            "risk_level": h["risk_level"],
            "cases_predicted": h["cases_predicted"],
        })
anim_df = pd.DataFrame(anim_rows)
if not anim_df.empty:
    risk_order = {"Faible": 0, "Modéré": 1, "Élevé": 2}
    anim_df["risk_rank"] = anim_df["risk_level"].map(risk_order)
    fig = px.bar(
        anim_df.sort_values("date"), x="district_name", y="risk_rank", color="risk_level",
        animation_frame="date",
        category_orders={"risk_level": ["Faible", "Modéré", "Élevé"]},
        color_discrete_map={"Faible": "#2ECC71", "Modéré": "#F39C12", "Élevé": "#E74C3C"},
    )
    fig.update_layout(
        height=480, plot_bgcolor=PANEL_BG, paper_bgcolor=PANEL_BG, font=dict(color=TEXT_COLOR),
        yaxis=dict(title="Niveau de risque", tickvals=[0, 1, 2], ticktext=["Faible", "Modéré", "Élevé"]),
        xaxis=dict(title="District", tickangle=-45),
        margin=dict(b=120),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Pas de données de prévision animables pour le moment.")
