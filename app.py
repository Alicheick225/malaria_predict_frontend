"""
MalariaWatch CI — tableau de bord principal de surveillance épidémique.
"""
from __future__ import annotations

import streamlit as st
from streamlit_folium import st_folium

from components.kpi_cards import render_kpi_card
from components.map_component import build_choropleth_map
from components.styles import ACCENT, ALERT_HIGH, ALERT_LOW, ALERT_MED, inject_css
from utils.api_client import (
    APIClient,
    cached_districts_geojson,
    cached_districts_table,
    cached_latest_predictions,
    cached_summary,
)

st.set_page_config(
    layout="wide",
    page_title="MalariaWatch CI",
    page_icon="🦟",
    initial_sidebar_state="expanded",
)
inject_css()

client = APIClient()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## MalariaWatch CI")
    st.caption("Prévision spatio-temporelle du paludisme — Côte d'Ivoire")

    backend_up = client.is_backend_up()
    dot_color  = ALERT_LOW if backend_up else ALERT_HIGH
    dot_label  = "API connectée" if backend_up else "API indisponible — mode simulé"
    st.markdown(
        f'<span class="status-dot" style="background:{dot_color};"></span>{dot_label}',
        unsafe_allow_html=True,
    )
    if not backend_up:
        st.info(
            "Le backend FastAPI ne répond pas : l'interface affiche des données "
            "simulées pour rester pleinement navigable."
        )

    st.divider()
    horizon = st.select_slider(
        "Horizon de prévision (mois)", options=[4, 6, 8], value=6, key="horizon"
    )
    st.caption("Granularité des données : **mensuelle** (2010-2024, 33 districts sanitaires).")

    st.divider()
    st.page_link("app.py",               label="Accueil")
    st.page_link("pages/02_district.py", label="Détail district")
    st.page_link("pages/03_tendances.py",label="Tendances nationales")
    st.page_link("pages/04_modele.py",   label="Performance du modèle")

    st.divider()
    st.markdown("##### Pipeline de prédiction")
    if st.button("Rafraîchir les prédictions", type="primary", use_container_width=True):
        with st.spinner("Exécution du pipeline d'inférence — cela peut prendre une minute..."):
            result = client.run_pipeline(horizon=8)
        if result.get("status") == "ok":
            st.success(
                f"{result['nb_districts_processed']} districts traités le {result['timestamp']}."
            )
            st.cache_data.clear()
        else:
            st.error(f"Échec : {result.get('error', 'voir les logs du backend')}")

# ── Bandeau d'en-tête ──────────────────────────────────────────────────────────
summary = cached_summary(horizon)
st.markdown(
    f"""
    <div class="mw-header">
        <div>
            <h1>🦟 Surveillance &amp; Prédiction du Paludisme — Côte d'Ivoire</h1>
            <p>Plateforme de prévision épidémique par Intelligence Artificielle (LSTM) &middot;
               33 districts sanitaires &middot; horizon de prévision : {horizon} mois</p>
        </div>
        <div style="text-align:right;color:#A0A8C0;font-size:0.84rem;">
            Dernière mise à jour<br>
            <b style="color:#FAFAFA;font-size:1rem;">{summary['last_update']}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Données ────────────────────────────────────────────────────────────────────
predictions  = cached_latest_predictions(horizon)
geojson      = cached_districts_geojson(horizon)
table        = cached_districts_table(horizon)
first_month  = sorted(predictions["week_predicted"].unique())[0]
current      = predictions[predictions["week_predicted"] == first_month].copy()
national_avg = float(current["cases_predicted"].mean())

# ── Rangée de KPIs ─────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_kpi_card("Districts surveillés", "33", color=ACCENT)
with k2:
    render_kpi_card(
        "Risque élevé", str(summary["nb_districts_high"]),
        delta=f"{summary['trend_vs_previous_week']:+.1f}% vs mois préc.",
        color=ALERT_HIGH,
    )
with k3:
    render_kpi_card("Risque modéré", str(summary["nb_districts_moderate"]), color=ALERT_MED)
with k4:
    render_kpi_card("Risque faible", str(summary["nb_districts_low"]), color=ALERT_LOW)
with k5:
    render_kpi_card(
        "Incidence moy. nationale", f"{national_avg:.0f}",
        delta="taux prédit /1000 hab.", color="#9B59B6",
    )

st.caption(
    f"Prochain mois prédit : **{first_month.date() if hasattr(first_month, 'date') else first_month}** "
    f"— {len(current)} districts couverts."
)

st.divider()

# ── Carte choroplèthe ──────────────────────────────────────────────────────────
st.subheader("Quels districts sont les plus à risque ce mois-ci ?")
st.caption(
    "Survolez un district pour afficher son score de risque et son taux d'incidence prédit. "
    "Utilisez la page **Carte interactive** pour filtrer par région ou niveau de risque."
)

fmap = build_choropleth_map(geojson, horizon)
st_folium(
    fmap,
    use_container_width=True,
    height=560,
    key="home_map",
    returned_objects=["last_object_clicked_tooltip"],
)

st.divider()

# ── Tableau récapitulatif ──────────────────────────────────────────────────────
st.subheader("Récapitulatif par district")

display_cols = ["district_name", "region", "risk_level", "risk_score", "cases_predicted"]
display = (
    table[[c for c in display_cols if c in table.columns]]
    .rename(columns={
        "district_name":   "District",
        "region":          "Région",
        "risk_level":      "Niveau de risque",
        "risk_score":      "Score de risque",
        "cases_predicted": "Cas prédits (taux /1000 hab.)",
    })
)

st.dataframe(
    display.sort_values("Score de risque", ascending=False),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Score de risque": st.column_config.ProgressColumn(
            min_value=0, max_value=1, format="%.2f"
        )
    },
)

st.info(
    "Cliquez sur un district dans la carte ci-dessus, puis ouvrez la page **Détail district** "
    "pour explorer son historique complet et les facteurs explicatifs de sa prédiction. "
    "Utilisez **Tendances nationales** pour comparer les trajectoires de plusieurs districts."
)
