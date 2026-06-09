"""Détail d'un district — série temporelle, comparaison radar, attributions SHAP."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import (
    plot_radar,
    plot_shap_bars,
    plot_timeseries,
)
from components.kpi_cards import render_kpi_card, render_risk_badge
from components.styles import ACCENT, ALERT_MED, inject_css
from utils.api_client import DISTRICT_NAMES, cached_district_detail, cached_districts_table

st.set_page_config(layout="wide", page_title="Détail district — MalariaWatch CI", page_icon="🦟")
inject_css()

horizon = st.session_state.get("horizon", 6)
table   = cached_districts_table(horizon)

st.title("Détail par district sanitaire")

names = sorted(table["district_name"].unique().tolist()) if not table.empty else DISTRICT_NAMES

# Pré-sélectionner le district cliqué sur la carte d'accueil (si disponible)
pre_selected = st.session_state.get("selected_district")
default_idx = names.index(pre_selected) if pre_selected in names else 0

selected_name = st.selectbox("Sélectionner un district", names, index=default_idx)

row         = table[table["district_name"] == selected_name]
district_id = row["district_id"].iloc[0] if not row.empty else f"MOCK.{names.index(selected_name)+1}"

detail          = cached_district_detail(district_id)
history         = pd.DataFrame(detail.get("history", []))

# ── En-tête ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 1])
with c1:
    st.subheader(detail["district_name"])
    st.caption(
        f"Région : {detail.get('region', 'N/A')}  "
        f"·  Population estimée : {detail.get('population', 0):,.0f} habitants".replace(",", " ")
    )
with c2:
    if detail.get("risk_level"):
        render_risk_badge(detail["risk_level"])

st.divider()

# ── KPIs ───────────────────────────────────────────────────────────────────────
hist_obs  = history.dropna(subset=["cases_observed"])  if "cases_observed"  in history else pd.DataFrame()
hist_pred = history.dropna(subset=["cases_predicted"]) if "cases_predicted" in history else pd.DataFrame()

current_pred = hist_pred.iloc[0]["cases_predicted"] if not hist_pred.empty else None
last_obs     = hist_obs.iloc[-1]["cases_observed"]  if not hist_obs.empty else None
prev_obs     = hist_obs.iloc[-2]["cases_observed"]  if len(hist_obs) > 1  else None
variation    = ((last_obs - prev_obs) / prev_obs * 100) if (last_obs and prev_obs) else None

trend_4 = None
if len(hist_obs) >= 4:
    trend_4 = "hausse" if hist_obs.iloc[-1]["cases_observed"] > hist_obs.iloc[-4]["cases_observed"] else "baisse"

k1, k2, k3, k4 = st.columns(4)
with k1:
    render_kpi_card("Score de risque", f"{detail.get('risk_score', 0):.2f}", color=ACCENT)
with k2:
    render_kpi_card(
        "Cas prédits (mois suivant)",
        f"{current_pred:.1f}" if current_pred is not None else "—",
        color="#9B59B6",
    )
with k3:
    render_kpi_card(
        "Variation vs mois précédent",
        f"{variation:+.1f} %" if variation is not None else "—",
        color=ALERT_MED,
    )
with k4:
    render_kpi_card("Tendance (4 derniers mois)", trend_4 or "—", color="#1ABC9C")

st.divider()

# ── 1. Série temporelle observé vs prédit ──────────────────────────────────────
st.subheader("L'incidence est-elle en hausse, en baisse, ou au pic dans ce district ?")
st.caption(
    "Trait plein = taux d'incidence observé (pour 1 000 hab.) — "
    "pointillé = prévision du modèle pour les prochains mois, bande grisée = intervalle indicatif (±15 %, non calibré statistiquement)."
)
st.plotly_chart(plot_timeseries(history), use_container_width=True)
st.caption(
    "À retenir : deux mois consécutifs de hausse avec un score > 0,66 déclenchent l'alerte rouge — "
    "c'est le signal qui doit motiver une mobilisation préventive immédiate."
)

st.divider()

# ── 2. Radar comparatif ────────────────────────────────────────────────────────
st.subheader("Ce district cumule-t-il davantage de facteurs de risque que la moyenne nationale ?")
st.caption(
    "Comparaison normalisée sur 6 dimensions : précipitations, végétation (NDVI), eau de surface (NDWI), "
    "température, population et couverture en moustiquaires (ITN). "
    "Une surface bleue plus grande que la grise indique un risque environnemental supérieur à la moyenne."
)
d_means        = detail.get("feature_means", {})
n_means        = detail.get("national_feature_means", {})
radar_features = ["tp_mm", "ndvi", "ndwi", "t2m_c", "population", "itn_use"]
if d_means and n_means:
    d_sub = {f: d_means.get(f, 0) for f in radar_features}
    n_sub = {f: n_means.get(f, 0) for f in radar_features}
    st.plotly_chart(
        plot_radar(d_sub, n_sub, detail["district_name"]),
        use_container_width=True,
    )
    st.caption(
        "À retenir : un district peut avoir un score de risque élevé même avec peu de pluie "
        "si sa couverture ITN est faible ou sa densité de population forte — "
        "le radar permet d'identifier le levier d'action prioritaire."
    )
else:
    st.info("Profil comparatif indisponible pour ce district.")

st.divider()

# ── 3. SHAP — facteurs explicatifs ─────────────────────────────────────────────
st.subheader("Quels facteurs ont le plus influencé la prédiction pour ce district ce mois-ci ?")
st.plotly_chart(plot_shap_bars(detail.get("top_features", [])), use_container_width=True)
st.caption(
    "Contributions calculées par Gradient × Input (approximation de SHAP pour réseaux récurrents). "
    "À retenir : si les précipitations ou le NDWI dominent, l'exposition environnementale est le moteur principal du risque — "
    "si l'ITN domine, c'est la faible protection des populations qui l'amplifie. "
    "Ces attributions sont spécifiques à ce district et à cette prédiction, pas des importances globales."
)
