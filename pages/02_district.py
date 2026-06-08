"""Détail d'un district — KPIs, historique observé/prédit, corrélation pluie/cas, radar, SHAP."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import (
    plot_radar,
    plot_rainfall_overlay,
    plot_risk_heatmap,
    plot_shap_bars,
    plot_timeseries,
)
from components.kpi_cards import render_kpi_card, render_risk_badge
from utils.api_client import DISTRICT_NAMES, cached_district_detail, cached_districts_table

st.set_page_config(layout="wide", page_title="Détail district — MalariaWatch CI", page_icon="📍")
st.markdown("<style>.stApp { background-color: #1E1E2E; }</style>", unsafe_allow_html=True)

horizon = st.session_state.get("horizon", 6)
table = cached_districts_table(horizon)

st.title("📍 Détail par district")

names = sorted(table["district_name"].unique().tolist()) if not table.empty else DISTRICT_NAMES
selected_name = st.selectbox("Sélectionner un district", names, index=0)

row = table[table["district_name"] == selected_name]
district_id = row["district_id"].iloc[0] if not row.empty else f"MOCK.{names.index(selected_name)+1}"

detail = cached_district_detail(district_id)
history = pd.DataFrame(detail.get("history", []))
feature_history = pd.DataFrame(detail.get("feature_history", []))

# ── En-tête ───────────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 1])
with c1:
    st.subheader(f"{detail['district_name']}")
    st.caption(f"Région : {detail.get('region', 'N/A')}  ·  Population estimée : "
               f"{detail.get('population', 0):,.0f} habitants".replace(",", " "))
with c2:
    if detail.get("risk_level"):
        render_risk_badge(detail["risk_level"])

st.divider()

# ── KPIs ──────────────────────────────────────────────────────────────────────
hist_obs = history.dropna(subset=["cases_observed"]) if "cases_observed" in history else pd.DataFrame()
hist_pred = history.dropna(subset=["cases_predicted"]) if "cases_predicted" in history else pd.DataFrame()

current_pred = hist_pred.iloc[0]["cases_predicted"] if not hist_pred.empty else None
last_obs = hist_obs.iloc[-1]["cases_observed"] if not hist_obs.empty else None
prev_obs = hist_obs.iloc[-2]["cases_observed"] if len(hist_obs) > 1 else None
variation = ((last_obs - prev_obs) / prev_obs * 100) if (last_obs and prev_obs) else None

trend_4 = None
if len(hist_obs) >= 4:
    trend_4 = "▲ hausse" if hist_obs.iloc[-1]["cases_observed"] > hist_obs.iloc[-4]["cases_observed"] else "▼ baisse"

k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi_card("Score de risque", f"{detail.get('risk_score', 0):.2f}", color="#3498DB")
with k2:
    render_kpi_card("Cas prédits (mois suivant)", f"{current_pred:.1f}" if current_pred is not None else "—", color="#9B59B6")
with k3:
    render_kpi_card("Variation vs mois précédent",
                    f"{variation:+.1f} %" if variation is not None else "—",
                    color="#E67E22")
with k4:
    render_kpi_card("Tendance (4 derniers mois)", trend_4 or "—", color="#1ABC9C")

st.divider()

# ── 1. Série temporelle observé vs prédit ────────────────────────────────────
st.subheader("Évolution de l'incidence — observée et prédite")
st.caption("Trait plein = taux d'incidence observé (pour 1000 hab.) · pointillé = prévision du modèle, "
           "avec une bande indicative (± 15 %, non calibrée statistiquement).")
st.plotly_chart(plot_timeseries(history), use_container_width=True)

# ── 2. Précipitations vs cas ──────────────────────────────────────────────────
st.subheader("Corrélation précipitations / incidence")
st.caption("Met en évidence le décalage typique entre un pic de précipitations et la "
           "remontée de l'incidence quelques semaines plus tard (cycle de reproduction du moustique).")
if not feature_history.empty:
    st.plotly_chart(plot_rainfall_overlay(history, feature_history[["date", "tp_mm"]]), use_container_width=True)
else:
    st.info("Historique climatique indisponible pour ce district.")

# ── 3. Heatmap risque ─────────────────────────────────────────────────────────
st.subheader("Carte de chaleur de l'incidence (24 derniers mois)")
if not hist_obs.empty:
    st.plotly_chart(plot_risk_heatmap(history), use_container_width=True)
else:
    st.info("Historique insuffisant pour générer la carte de chaleur.")

# ── 4. Radar comparatif ───────────────────────────────────────────────────────
st.subheader("Profil du district vs moyenne nationale")
st.caption("Comparaison normalisée sur 6 dimensions clés : précipitations, NDVI, NDWI, "
           "température, population et couverture en moustiquaires (usage ITN).")
d_means = detail.get("feature_means", {})
n_means = detail.get("national_feature_means", {})
radar_features = ["tp_mm", "ndvi", "ndwi", "t2m_c", "population", "itn_use"]
if d_means and n_means:
    d_sub = {f: d_means.get(f, 0) for f in radar_features}
    n_sub = {f: n_means.get(f, 0) for f in radar_features}
    st.plotly_chart(plot_radar(d_sub, n_sub, detail["district_name"]), use_container_width=True)
else:
    st.info("Profil comparatif indisponible pour ce district.")

# ── 5. SHAP — facteurs explicatifs ────────────────────────────────────────────
st.subheader("Facteurs explicatifs de la prédiction (SHAP)")
st.plotly_chart(plot_shap_bars(detail.get("top_features", [])), use_container_width=True)
st.caption(
    "Les contributions sont calculées par attribution différentiable "
    "(`shap.GradientExplainer`, approximation de SHAP adaptée aux réseaux récurrents) : "
    "elles indiquent quelles variables d'entrée pèsent le plus dans la prédiction du mois suivant."
)
