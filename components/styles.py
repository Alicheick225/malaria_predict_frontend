"""Constantes de couleur et injection CSS globale — partagées entre toutes les pages."""
from __future__ import annotations

import streamlit as st

# ── Palette principale ─────────────────────────────────────────────────────────
PRIMARY    = "#1F4E79"   # bleu IDSI — titres, header banner, boutons
ACCENT     = "#3498DB"   # bleu clair — graphiques, liens actifs
ALERT_HIGH = "#E74C3C"   # rouge — risque élevé
ALERT_MED  = "#F39C12"   # orange — risque modéré
ALERT_LOW  = "#2ECC71"   # vert — risque faible

# ── Fond et texte ──────────────────────────────────────────────────────────────
APP_BG      = "#0F1117"  # fond Streamlit dark natif
SIDEBAR_BG  = "#090C10"  # fond sidebar
PANEL_BG    = "#1A1D27"  # fond cards / graphiques
CARD_BG     = "#1E2130"  # fond KPI cards
TEXT_COLOR  = "#FAFAFA"  # texte principal
TEXT_MUTED  = "#A0A8C0"  # texte secondaire / captions
GRID_COLOR  = "#2E3250"  # grilles graphiques

RISK_COLORS = {"Faible": ALERT_LOW, "Modéré": ALERT_MED, "Élevé": ALERT_HIGH}

_CSS = f"""
<style>
/* ── Réduction du padding haut ── */
.block-container {{
    padding-top: 1.2rem !important;
    padding-bottom: 1.5rem !important;
    max-width: 1280px;
}}
/* ── Sidebar : conserve un fond foncé quelle que soit la thème ── */
section[data-testid="stSidebar"] {{
    background-color: {SIDEBAR_BG} !important;
}}
section[data-testid="stSidebar"] * {{
    color: {TEXT_COLOR} !important;
}}
/* ── Onglets ── */
.stTabs [data-baseweb="tab-list"] {{
    border-radius: 8px;
    gap: 2px;
}}
.stTabs [aria-selected="true"] {{
    background-color: {PRIMARY}33;
    border-bottom: 2px solid {ACCENT};
}}
/* ── DataFrames ── */
.stDataFrame thead tr th {{
    background-color: {PRIMARY} !important;
    color: {TEXT_COLOR} !important;
}}
/* ── Boutons primaires ── */
.stButton > button[kind="primary"] {{
    background-color: {PRIMARY} !important;
    color: {TEXT_COLOR} !important;
    border: none;
    border-radius: 6px;
}}
.stButton > button[kind="primary"]:hover {{
    background-color: {ACCENT} !important;
}}
/* ── Badges de risque HTML ── */
.badge-high {{
    background: {ALERT_HIGH}28;
    color: {ALERT_HIGH};
    border: 1px solid {ALERT_HIGH};
    border-radius: 999px;
    padding: 3px 14px;
    font-weight: 600;
    font-size: 0.88rem;
    display: inline-block;
}}
.badge-med {{
    background: {ALERT_MED}28;
    color: {ALERT_MED};
    border: 1px solid {ALERT_MED};
    border-radius: 999px;
    padding: 3px 14px;
    font-weight: 600;
    font-size: 0.88rem;
    display: inline-block;
}}
.badge-low {{
    background: {ALERT_LOW}28;
    color: {ALERT_LOW};
    border: 1px solid {ALERT_LOW};
    border-radius: 999px;
    padding: 3px 14px;
    font-weight: 600;
    font-size: 0.88rem;
    display: inline-block;
}}
/* ── Classement rows : hérite de la couleur de texte du thème ── */
.mw-rank-row {{
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 4px;
    border-bottom: 1px solid {GRID_COLOR};
    font-size: 0.92rem;
}}
.mw-rank-num {{
    font-weight: 700;
    min-width: 52px;
    text-align: right;
    font-variant-numeric: tabular-nums;
    color: {ACCENT};
}}
/* ── Header banner : fond sombre propre, toujours blanc ── */
.mw-header {{
    background: linear-gradient(90deg, {PRIMARY} 0%, #163a5c 100%);
    border-radius: 10px;
    padding: 16px 24px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-left: 4px solid {ACCENT};
}}
.mw-header h1 {{
    color: {TEXT_COLOR} !important;
    font-size: 1.45rem;
    margin: 0;
}}
.mw-header p {{
    color: #c8d8e8;
    margin: 4px 0 0 0;
    font-size: 0.87rem;
}}
/* ── Status indicator ── */
.status-dot {{
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    margin-right: 6px;
    vertical-align: middle;
}}
/* ── Section intro text : hérite du thème, légèrement atténuée ── */
.section-intro {{
    font-size: 0.9rem;
    line-height: 1.6;
    margin-bottom: 10px;
    border-left: 3px solid {ACCENT};
    padding-left: 10px;
    opacity: 0.82;
}}
/* ── Alerte box ── */
.alert-box {{
    border: 1px solid {GRID_COLOR};
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
}}
</style>
"""


def inject_css() -> None:
    """Injecter le CSS global dans la page courante (à appeler une fois par page, après set_page_config)."""
    st.markdown(_CSS, unsafe_allow_html=True)


def risk_badge_html(risk_level: str) -> str:
    """Retourne le HTML d'un badge de risque (Faible/Modéré/Élevé)."""
    classes = {"Faible": "badge-low", "Modéré": "badge-med", "Élevé": "badge-high"}
    cls = classes.get(risk_level, "badge-low")
    return f'<span class="{cls}">Risque {risk_level}</span>'
