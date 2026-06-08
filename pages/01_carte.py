"""Carte interactive — choroplèthe des 33 districts sanitaires colorée par niveau de risque prédit."""
from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium

from components.map_component import build_choropleth_map
from utils.api_client import RISK_LEVELS, cached_districts_geojson, cached_districts_table

st.set_page_config(layout="wide", page_title="Carte — MalariaWatch CI", page_icon="🗺️")
st.markdown("<style>.stApp { background-color: #1E1E2E; }</style>", unsafe_allow_html=True)

horizon = st.session_state.get("horizon", 6)

st.title("🗺️ Carte interactive du risque épidémique")
st.caption(
    f"Niveau de risque prédit pour le mois suivant, par district sanitaire — horizon de {horizon} mois "
    "(ajustable dans le menu latéral de la page d'accueil)."
)

# ── Filtres ───────────────────────────────────────────────────────────────────
table = cached_districts_table(horizon)
regions = sorted(table["region"].dropna().unique().tolist()) if "region" in table else []

with st.sidebar:
    st.markdown("### Filtres")
    risk_filter = st.multiselect("Niveau de risque", RISK_LEVELS, default=RISK_LEVELS)
    region_filter = st.multiselect("Région administrative", regions, default=[])
    st.divider()
    st.markdown("### Légende")
    for level, color in [("Faible", "#2ECC71"), ("Modéré", "#F39C12"), ("Élevé", "#E74C3C")]:
        st.markdown(f'<span style="color:{color};">●</span> {level}', unsafe_allow_html=True)

geojson = cached_districts_geojson(horizon)
fmap = build_choropleth_map(
    geojson, horizon,
    risk_filter=risk_filter or None,
    region_filter=region_filter or None,
)

map_state = st_folium(fmap, use_container_width=True, height=620, returned_objects=["last_object_clicked_tooltip"])

st.divider()

# ── Tableau récapitulatif ─────────────────────────────────────────────────────
st.subheader("Récapitulatif par district")
display_cols = ["district_name", "region", "risk_level", "risk_score", "cases_predicted"]
display = table[[c for c in display_cols if c in table.columns]].rename(columns={
    "district_name": "District", "region": "Région", "risk_level": "Niveau de risque",
    "risk_score": "Score de risque", "cases_predicted": "Cas prédits (taux /1000 hab.)",
})
if risk_filter:
    display = display[display["Niveau de risque"].isin(risk_filter)]
if region_filter:
    display = display[display["Région"].isin(region_filter)]

st.dataframe(
    display.sort_values("Score de risque", ascending=False),
    use_container_width=True, hide_index=True,
    column_config={"Score de risque": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")},
)

st.info(
    "💡 Survolez un district sur la carte pour afficher son score de risque, ou cliquez "
    "dessus pour voir son nom — puis ouvrez la page **Détail district** pour explorer "
    "son historique complet et les facteurs explicatifs de sa prédiction."
)
