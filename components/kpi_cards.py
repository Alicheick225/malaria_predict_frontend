"""Cartes KPI réutilisables — affichage stylisé de métriques clés."""
from __future__ import annotations

import streamlit as st

RISK_COLORS = {"Faible": "#2ECC71", "Modéré": "#F39C12", "Élevé": "#E74C3C"}
_CARD_BG    = "#1E2130"
_TEXT_MUTED = "#A0A8C0"


def render_kpi_card(label: str, value: str, delta: str | None = None,
                    color: str = "#3498DB", icon: str | None = None):
    """Affiche une carte métrique stylisée façon tableau de bord."""
    delta_html = ""
    if delta is not None:
        arrow = "▲" if delta.strip().startswith("+") else ("▼" if delta.strip().startswith("-") else "▬")
        delta_color = "#E74C3C" if delta.strip().startswith("+") else "#2ECC71"
        delta_html = (
            f'<div style="color:{delta_color};font-size:0.83rem;margin-top:5px;">'
            f'{arrow} {delta}</div>'
        )

    st.markdown(
        f"""
        <div style="background:{_CARD_BG};border-top:3px solid {color};border-radius:10px;
                    padding:14px 16px;margin-bottom:8px;text-align:center;">
            <div style="color:{_TEXT_MUTED};font-size:0.76rem;text-transform:uppercase;
                        letter-spacing:0.06em;">{label}</div>
            <div style="color:{color};font-size:2rem;font-weight:800;margin-top:5px;">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_badge(risk_level: str):
    """Badge HTML coloré selon le niveau de risque (sans emoji)."""
    color = RISK_COLORS.get(risk_level, "#3498DB")
    classes = {"Faible": "badge-low", "Modéré": "badge-med", "Élevé": "badge-high"}
    cls = classes.get(risk_level, "badge-low")
    st.markdown(
        f'<span class="{cls}">Risque {risk_level}</span>',
        unsafe_allow_html=True,
    )
