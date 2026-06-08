"""
MalariaWatch CI — application Streamlit de visualisation des prédictions du
modèle LSTM de prévision spatio-temporelle du paludisme en Côte d'Ivoire.

Page d'accueil : tableau de bord de surveillance épidémique — bandeau d'en-tête,
indicateurs clés (KPIs), carte interactive à bulles, classement des districts,
palmarès des districts les plus à risque et tendances nationales prédites.
Mise en page inspirée des dashboards de surveillance sanitaire (cf. plateformes
ministérielles de suivi épidémique), adaptée aux prédictions du paludisme en CI.
"""
from __future__ import annotations

from streamlit_folium import st_folium
import streamlit as st

from components.charts import (
    plot_cumulative_national,
    plot_national_trend,
    plot_predicted_cases_bar,
    plot_risk_distribution_over_time,
)
from components.kpi_cards import render_kpi_card
from components.map_component import RISK_COLORS, build_bubble_map
from utils.api_client import APIClient, cached_districts_geojson, cached_latest_predictions, cached_summary

st.set_page_config(layout="wide", page_title="MalariaWatch CI", page_icon="🦟", initial_sidebar_state="expanded")

# ── Thème sombre + bandeau d'en-tête ─────────────────────────────────────────
st.markdown(
    """
    <style>
    .stApp { background-color: #1E1E2E; }
    section[data-testid="stSidebar"] { background-color: #16161F; }
    .mw-header {
        background: linear-gradient(90deg, #C0392B 0%, #922B21 100%);
        border-radius: 12px; padding: 18px 26px; margin-bottom: 18px;
        display: flex; align-items: center; justify-content: space-between;
    }
    .mw-header h1 { color: #FFFFFF; font-size: 1.6rem; margin: 0; }
    .mw-header p { color: #F5D5D0; margin: 2px 0 0 0; font-size: 0.9rem; }
    .mw-rank-row {
        display: flex; align-items: center; gap: 10px; padding: 6px 4px;
        border-bottom: 1px solid #2E2E40; font-size: 0.92rem; color: #E0E0EA;
    }
    .mw-rank-num { font-weight: 700; min-width: 38px; text-align: right; color: #F0F0F5; }
    </style>
    """,
    unsafe_allow_html=True,
)

client = APIClient()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🦟 MalariaWatch CI")
    st.caption("Prévision spatio-temporelle du paludisme — Côte d'Ivoire")

    backend_up = client.is_backend_up()
    status_color = "#2ECC71" if backend_up else "#E74C3C"
    status_label = "API connectée" if backend_up else "API indisponible — mode simulé"
    st.markdown(
        f'<span style="color:{status_color};">●</span> {status_label}',
        unsafe_allow_html=True,
    )
    if not backend_up:
        st.info("Le backend FastAPI ne répond pas : l'interface affiche des données "
                "simulées pour rester pleinement navigable.")

    st.divider()
    horizon = st.select_slider("Horizon de prévision (mois)", options=[4, 6, 8], value=6,
                               key="horizon")
    st.caption("Granularité réelle des données : **mensuelle** (2010-2022, 33 districts sanitaires).")

    st.divider()
    st.page_link("app.py", label="Accueil", icon="🏠")
    st.page_link("pages/01_carte.py", label="Carte interactive", icon="🗺️")
    st.page_link("pages/02_district.py", label="Détail district", icon="📍")
    st.page_link("pages/03_tendances.py", label="Tendances nationales", icon="📈")
    st.page_link("pages/04_modele.py", label="Performance du modèle", icon="🧠")

    st.divider()
    st.markdown("##### Pipeline de prédiction")
    if st.button("🔄 Rafraîchir les prédictions", type="primary", use_container_width=True):
        with st.spinner("Exécution du pipeline d'inférence — cela peut prendre une minute..."):
            result = client.run_pipeline(horizon=8)
        if result.get("status") == "ok":
            st.success(f"{result['nb_districts_processed']} districts traités le {result['timestamp']}.")
            st.cache_data.clear()
        else:
            st.error(f"Échec : {result.get('error', 'voir les logs du backend')}")

# ── Bandeau d'en-tête ─────────────────────────────────────────────────────────
summary = cached_summary(horizon)
st.markdown(
    f"""
    <div class="mw-header">
        <div>
            <h1>🦟 Surveillance &amp; Prédiction du Paludisme — Côte d'Ivoire</h1>
            <p>Plateforme de prévision épidémique par Intelligence Artificielle (LSTM) ·
               33 districts sanitaires · horizon de prévision : {horizon} mois</p>
        </div>
        <div style="text-align:right;color:#F5D5D0;font-size:0.85rem;">
            Dernière mise à jour des prédictions<br><b style="color:#FFFFFF;font-size:1.05rem;">{summary['last_update']}</b>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Données ───────────────────────────────────────────────────────────────────
predictions = cached_latest_predictions(horizon)
geojson = cached_districts_geojson(horizon)
first_month = sorted(predictions["week_predicted"].unique())[0]
current = predictions[predictions["week_predicted"] == first_month].copy()
national_avg = float(current["cases_predicted"].mean())

# ── Rangée de KPIs ────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    render_kpi_card("Districts surveillés", "33", color="#3498DB", icon="🏥")
with k2:
    render_kpi_card("Risque élevé", str(summary["nb_districts_high"]),
                    delta=f"{summary['trend_vs_previous_week']:+.1f}% vs mois préc.", color="#E74C3C", icon="🛑")
with k3:
    render_kpi_card("Risque modéré", str(summary["nb_districts_moderate"]), color="#F39C12", icon="⚠️")
with k4:
    render_kpi_card("Risque faible", str(summary["nb_districts_low"]), color="#2ECC71", icon="✅")
with k5:
    render_kpi_card("Incidence moy. nationale", f"{national_avg:.0f}",
                    delta="taux prédit /1000 hab.", color="#9B59B6", icon="📊")

st.caption(f"Prochain mois prédit : **{first_month.date() if hasattr(first_month, 'date') else first_month}** "
           f"— {len(current)} districts couverts.")

st.divider()

# ── Corps principal : classement | carte à bulles | palmarès ─────────────────
col_rank, col_map, col_bar = st.columns([1.1, 2.2, 1.3])

with col_rank:
    st.markdown("##### 📋 Classement (taux d'incidence prédit)")
    ranked = current.sort_values("cases_predicted", ascending=False).reset_index(drop=True)
    rows_html = ""
    for i, r in ranked.head(15).iterrows():
        color = RISK_COLORS.get(r["risk_level"], "#7F8C9A")
        rows_html += (
            f'<div class="mw-rank-row">'
            f'<span class="mw-rank-num">{r["cases_predicted"]:.0f}</span>'
            f'<span style="color:{color};">●</span>'
            f'<span>{r["district_name"]}</span>'
            f'</div>'
        )
    st.markdown(f'<div style="max-height:560px;overflow-y:auto;">{rows_html}</div>', unsafe_allow_html=True)

with col_map:
    st.markdown("##### 🗺️ Carte du risque épidémique prédit")
    st.caption("Cliquez sur un district pour afficher son score de risque et son taux d'incidence prédit.")
    bubble_map = build_bubble_map(geojson, value_field="cases_predicted")
    st_folium(bubble_map, use_container_width=True, height=560, key="home_map",
              returned_objects=["last_object_clicked_tooltip"])

with col_bar:
    st.markdown("##### 🏆 Districts les plus à risque")
    st.plotly_chart(plot_predicted_cases_bar(current, n=10), use_container_width=True, config={"displayModeBar": False})

st.divider()

# ── Tendances nationales prédites (onglets) ──────────────────────────────────
st.markdown("##### 📈 Évolution nationale prédite")
tab1, tab2, tab3 = st.tabs(["Évolution de l'incidence", "Répartition par niveau de risque", "Cumul prédit"])
with tab1:
    st.plotly_chart(plot_national_trend(predictions), use_container_width=True, config={"displayModeBar": False})
with tab2:
    st.plotly_chart(plot_risk_distribution_over_time(predictions), use_container_width=True, config={"displayModeBar": False})
with tab3:
    st.plotly_chart(plot_cumulative_national(predictions), use_container_width=True, config={"displayModeBar": False})

st.markdown(
    "👉 Utilisez le menu latéral pour explorer la **carte interactive** en plein écran, le **détail par "
    "district**, les **tendances nationales** ou la **performance du modèle**."
)
