"""Cartes Folium des districts sanitaires de Côte d'Ivoire — réutilisables entre pages."""
from __future__ import annotations

import folium
import plotly.graph_objects as go
from shapely.geometry import shape

RISK_COLORS = {"Faible": "#2ECC71", "Modéré": "#F39C12", "Élevé": "#E74C3C", None: "#7F8C9A"}
CI_CENTER = [7.54, -5.55]


def _style_factory(field: str = "risk_level"):
    def _style(feature):
        level = feature["properties"].get(field)
        return {
            "fillColor": RISK_COLORS.get(level, "#7F8C9A"),
            "color": "#1E1E2E",
            "weight": 1,
            "fillOpacity": 0.75,
        }
    return _style


def _highlight(feature):
    return {"weight": 3, "color": "#F0F0F5", "fillOpacity": 0.9}


def build_choropleth_map(geojson: dict, horizon: int, risk_filter: list[str] | None = None,
                          region_filter: list[str] | None = None) -> folium.Map:
    """Construit une carte Folium choroplèthe colorée selon le niveau de risque prédit."""
    fmap = folium.Map(location=CI_CENTER, zoom_start=7, tiles="CartoDB dark_matter", control_scale=True)

    features = geojson.get("features", [])
    if risk_filter:
        features = [f for f in features if f["properties"].get("risk_level") in risk_filter]
    if region_filter:
        features = [f for f in features if f["properties"].get("region") in region_filter]
    filtered = {"type": "FeatureCollection", "features": features}

    tooltip = folium.GeoJsonTooltip(
        fields=["district_name", "risk_level", "risk_score", "cases_predicted"],
        aliases=["District :", "Niveau de risque :", "Score :", "Cas prédits (taux /1000 hab.) :"],
        localize=True, sticky=True,
        style="background-color:#262637;color:#F0F0F5;border:1px solid #3A3A50;border-radius:6px;padding:8px;",
    )

    folium.GeoJson(
        filtered,
        name=f"Risque prédit (horizon {horizon} mois)",
        style_function=_style_factory("risk_level"),
        highlight_function=_highlight,
        tooltip=tooltip,
    ).add_to(fmap)

    _add_legend(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)
    try:
        from folium.plugins import Fullscreen
        Fullscreen(position="topleft").add_to(fmap)
    except ImportError:
        pass
    return fmap


def build_bubble_map(geojson: dict, value_field: str = "cases_predicted",
                       risk_filter: list[str] | None = None) -> folium.Map:
    """
    Carte à bulles : un cercle par district, centré sur son centroïde géographique,
    coloré selon le niveau de risque prédit et dimensionné selon la valeur prédite
    (taux d'incidence). Rendu inspiré des tableaux de bord de surveillance épidémique
    (cercles superposés à un fond de carte sombre, cliquables pour le détail).
    """
    fmap = folium.Map(location=CI_CENTER, zoom_start=7, tiles="CartoDB dark_matter", control_scale=True)

    features = geojson.get("features", [])
    if risk_filter:
        features = [f for f in features if f["properties"].get("risk_level") in risk_filter]

    values = [f["properties"].get(value_field, 0) or 0 for f in features]
    vmin, vmax = (min(values), max(values)) if values else (0, 1)
    span = (vmax - vmin) or 1

    for feature in features:
        props = feature["properties"]
        try:
            centroid = shape(feature["geometry"]).centroid
        except Exception:
            continue
        level = props.get("risk_level")
        value = props.get(value_field, 0) or 0
        radius = 8 + 22 * ((value - vmin) / span)         # rayon proportionnel à la valeur prédite
        color = RISK_COLORS.get(level, "#7F8C9A")

        popup_html = (
            f'<div style="font-family:sans-serif;min-width:170px;">'
            f'<b style="font-size:1rem;">{props.get("district_name", "?")}</b><br>'
            f'<span style="color:{color};">●</span> Risque {level}<br>'
            f'Score : {props.get("risk_score", 0):.2f}<br>'
            f'Cas prédits (taux /1000 hab.) : {value:.1f}'
            f'</div>'
        )
        folium.CircleMarker(
            location=[centroid.y, centroid.x],
            radius=radius,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.55,
            tooltip=folium.Tooltip(
                f'<b>{props.get("district_name", "?")}</b> — {level} ({value:.1f})',
                style="background-color:#262637;color:#F0F0F5;border:1px solid #3A3A50;border-radius:6px;padding:6px 10px;",
            ),
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(fmap)

    _add_legend(fmap)
    try:
        from folium.plugins import Fullscreen
        Fullscreen(position="topleft").add_to(fmap)
    except ImportError:
        pass
    return fmap


def _add_legend(fmap: folium.Map):
    items = "".join(
        f'<i style="background:{color};width:14px;height:14px;display:inline-block;'
        f'margin-right:6px;border-radius:3px;"></i>{level}<br>'
        for level, color in [("Faible", RISK_COLORS["Faible"]), ("Modéré", RISK_COLORS["Modéré"]), ("Élevé", RISK_COLORS["Élevé"])]
    )
    legend_html = f"""
    <div style="position: fixed; bottom: 30px; right: 30px; z-index: 9999;
                background: #262637; color: #E0E0EA; padding: 12px 16px;
                border-radius: 8px; border: 1px solid #3A3A50; font-size: 13px;">
        <b>Niveau de risque</b><br>{items}
    </div>"""
    fmap.get_root().html.add_child(folium.Element(legend_html))


def build_sparkline(history_points: list[dict]) -> go.Figure:
    """Mini-graphique d'évolution récente pour les popups au clic."""
    import pandas as pd
    df = pd.DataFrame(history_points)
    if df.empty:
        return go.Figure()
    df["date"] = pd.to_datetime(df["date"])
    fig = go.Figure(go.Scatter(x=df["date"], y=df.get("cases_observed", df.get("cases_predicted")),
                               mode="lines", line=dict(color="#3498DB", width=2)))
    fig.update_layout(
        height=120, width=240, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="#262637", paper_bgcolor="#262637",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        showlegend=False,
    )
    return fig
