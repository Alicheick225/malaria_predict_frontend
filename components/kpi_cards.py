"""Cartes KPI réutilisables — affichage stylisé de métriques clés."""
from __future__ import annotations

import streamlit as st

RISK_COLORS = {"Faible": "#2ECC71", "Modéré": "#F39C12", "Élevé": "#E74C3C"}


def render_kpi_card(label: str, value: str, delta: str | None = None, color: str = "#3498DB",
                     icon: str | None = None):
    """Affiche une carte métrique stylisée façon tableau de bord (icône + grand chiffre coloré)."""
    delta_html = ""
    if delta is not None:
        arrow = "▲" if delta.strip().startswith("+") else ("▼" if delta.strip().startswith("-") else "▬")
        delta_color = "#E74C3C" if delta.strip().startswith("+") else "#2ECC71"
        delta_html = f'<div style="color:{delta_color};font-size:0.85rem;margin-top:6px;">{arrow} {delta}</div>'

    icon_html = f'<span style="font-size:1.6rem;margin-right:10px;">{icon}</span>' if icon else ""

    st.markdown(
        f"""
        <div style="background:#262637;border-top:3px solid {color};border-radius:10px;
                    padding:16px 18px;margin-bottom:8px;text-align:center;height:100%;">
            <div style="color:#A0A0B8;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.05em;">{label}</div>
            <div style="color:{color};font-size:2.1rem;font-weight:800;margin-top:6px;">{icon_html}{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_risk_badge(risk_level: str):
    color = RISK_COLORS.get(risk_level, "#3498DB")
    st.markdown(
        f"""
        <span style="background:{color}22;color:{color};border:1px solid {color};
                     border-radius:999px;padding:4px 14px;font-weight:600;font-size:0.9rem;">
            ● Risque {risk_level}
        </span>
        """,
        unsafe_allow_html=True,
    )
