"""Client HTTP vers le backend FastAPI MalariaWatch CI — avec cache et repli sur données simulées."""
from __future__ import annotations

import os
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st

def _resolve_backend_url() -> str:
    """
    Résout l'URL du backend : Streamlit Cloud expose les secrets via `st.secrets`
    (et non `os.environ`), tandis que Docker Compose / l'exécution locale passent
    par une variable d'environnement classique. On essaie les deux, dans cet ordre,
    avec un repli sur le backend local pour le développement.
    """
    try:
        return st.secrets["BACKEND_URL"]
    except Exception:
        return os.environ.get("BACKEND_URL", "http://localhost:8000")


BACKEND_URL = _resolve_backend_url()
API = f"{BACKEND_URL}/api/v1"
TIMEOUT = 30  # le backend Render (tier gratuit) peut connaître un "cold start" de plusieurs dizaines de secondes

DISTRICT_NAMES = [
    "Abidjan", "Agnéby-Tiassa", "Bafing", "Bagoué", "Bounkani", "Bélier", "Béré",
    "Cavally", "Folon", "Gbeke", "Gbôkle", "Gontougo", "Grands Ponts", "Guémon",
    "Gôh", "Hambol", "Haut-Sassandra", "Iffou", "Indénié-Djuablin", "Kabadougou",
    "La Mé", "Lôh-Djiboua", "Marahoué", "Moronou", "N'zi", "Nawa", "Poro",
    "San-Pédro", "Sud Comoé", "Tchologo", "Tonkpi", "Worodougou", "Yamoussoukro",
]
RISK_LEVELS = ["Faible", "Modéré", "Élevé"]
FEATURES = ["t2m_c", "rh_pct", "tp_mm", "ndvi", "ndwi", "pfpr", "itn_use", "itn_access", "population"]


class APIClient:
    """Encapsule les appels au backend ; bascule silencieusement sur des données simulées si indisponible."""

    def __init__(self, base_url: str = BACKEND_URL):
        self.base_url = base_url

    # ── Bas niveau ─────────────────────────────────────────────────────────────
    def _get(self, path: str, params: dict | None = None):
        try:
            r = requests.get(f"{API}{path}", params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json(), None
        except Exception as e:
            return None, str(e)

    def is_backend_up(self) -> bool:
        data, err = self._get("/health")
        return err is None and data is not None

    # ── Endpoints ──────────────────────────────────────────────────────────────
    def get_summary(self, horizon: int = 6) -> dict:
        data, err = self._get("/predictions/summary", {"horizon": horizon})
        if err:
            return _mock_summary()
        return data

    def get_latest_predictions(self, horizon: int = 6) -> pd.DataFrame:
        data, err = self._get("/predictions/latest", {"horizon": horizon})
        if err:
            return _mock_latest_predictions(horizon)
        return pd.DataFrame(data)

    def get_district_history(self, district_id: str, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
        params = {k: v for k, v in {"start_date": start_date, "end_date": end_date}.items() if v}
        data, err = self._get(f"/predictions/{district_id}/history", params)
        if err:
            return _mock_history(district_id)
        return pd.DataFrame(data)

    def get_districts_geojson(self, horizon: int = 6) -> dict:
        data, err = self._get("/districts", {"horizon": horizon})
        if err:
            return _mock_districts_geojson()
        return {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": d["geometry"], "properties": {k: v for k, v in d.items() if k != "geometry"}}
                for d in data
            ],
        }

    def get_districts_table(self, horizon: int = 6) -> pd.DataFrame:
        geojson = self.get_districts_geojson(horizon)
        return pd.DataFrame([f["properties"] for f in geojson["features"]])

    def get_district_detail(self, district_id: str) -> dict:
        data, err = self._get(f"/districts/{district_id}")
        if err:
            return _mock_district_detail(district_id)
        return data

    def run_pipeline(self, horizon: int = 8) -> dict:
        try:
            r = requests.post(f"{API}/predictions/run", params={"horizon": horizon}, timeout=120)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            return {"status": "echec", "error": str(e)}


# ── Cache Streamlit (1h) ─────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def cached_summary(horizon: int = 6) -> dict:
    return APIClient().get_summary(horizon)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_latest_predictions(horizon: int = 6) -> pd.DataFrame:
    return APIClient().get_latest_predictions(horizon)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_districts_geojson(horizon: int = 6) -> dict:
    return APIClient().get_districts_geojson(horizon)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_districts_table(horizon: int = 6) -> pd.DataFrame:
    return APIClient().get_districts_table(horizon)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_district_history(district_id: str) -> pd.DataFrame:
    return APIClient().get_district_history(district_id)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_district_detail(district_id: str) -> dict:
    return APIClient().get_district_detail(district_id)


# ── Repli : données simulées (si le backend est inaccessible) ────────────────
def _seasonal_factor(month: int) -> float:
    """Pics simulés avril-juillet et septembre-novembre (saisons des pluies en CI)."""
    return 1.6 if month in (4, 5, 6, 7, 9, 10, 11) else 0.8


def _mock_latest_predictions(horizon: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    today = date.today().replace(day=1)
    for i, name in enumerate(DISTRICT_NAMES):
        base = rng.uniform(40, 220)
        for h in range(1, horizon + 1):
            week = (today + pd.DateOffset(months=h)).date()
            val = max(0, base * _seasonal_factor(week.month) * rng.uniform(0.85, 1.15))
            score = float(np.clip(val / 350, 0, 1))
            level = RISK_LEVELS[0] if score < 0.33 else (RISK_LEVELS[1] if score < 0.66 else RISK_LEVELS[2])
            rows.append({
                "district_id": f"MOCK.{i+1}",
                "district_name": name,
                "risk_score": round(score, 4),
                "risk_level": level,
                "cases_predicted": round(val, 2),
                "week_predicted": week,
            })
    df = pd.DataFrame(rows)
    df["week_predicted"] = pd.to_datetime(df["week_predicted"])
    return df


def _mock_summary() -> dict:
    df = _mock_latest_predictions(6)
    first = sorted(df["week_predicted"].unique())[0]
    cur = df[df["week_predicted"] == first]
    counts = cur["risk_level"].value_counts()
    return {
        "nb_districts_high": int(counts.get("Élevé", 0)),
        "nb_districts_moderate": int(counts.get("Modéré", 0)),
        "nb_districts_low": int(counts.get("Faible", 0)),
        "trend_vs_previous_week": float(np.random.default_rng(3).uniform(-8, 8)),
        "last_update": date.today().isoformat(),
    }


def _mock_history(district_id: str) -> pd.DataFrame:
    rng = np.random.default_rng(abs(hash(district_id)) % (2**32))
    base = rng.uniform(60, 200)
    dates = pd.date_range(end=date.today().replace(day=1) - timedelta(days=1), periods=52, freq="MS")
    rows = []
    for d in dates:
        obs = max(0, base * _seasonal_factor(d.month) * rng.uniform(0.8, 1.2))
        rows.append({"date": d.date(), "cases_observed": round(obs, 2), "cases_predicted": None, "risk_level": None})
    today = date.today().replace(day=1)
    for h in range(1, 9):
        week = (today + pd.DateOffset(months=h)).date()
        pred = max(0, base * _seasonal_factor(week.month) * rng.uniform(0.85, 1.15))
        score = float(np.clip(pred / 350, 0, 1))
        level = RISK_LEVELS[0] if score < 0.33 else (RISK_LEVELS[1] if score < 0.66 else RISK_LEVELS[2])
        rows.append({"date": week, "cases_observed": None, "cases_predicted": round(pred, 2), "risk_level": level})
    return pd.DataFrame(rows)


def _mock_districts_geojson() -> dict:
    """Sans géométries réelles, on place des points fictifs en grille (utilisé seulement si le backend est down)."""
    rng = np.random.default_rng(11)
    features = []
    for i, name in enumerate(DISTRICT_NAMES):
        lon = -8.0 + (i % 7) * 0.9
        lat = 5.0 + (i // 7) * 0.9
        score = float(rng.uniform(0, 1))
        level = RISK_LEVELS[0] if score < 0.33 else (RISK_LEVELS[1] if score < 0.66 else RISK_LEVELS[2])
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[lon, lat], [lon + 0.8, lat], [lon + 0.8, lat + 0.8], [lon, lat + 0.8], [lon, lat]]],
            },
            "properties": {
                "district_id": f"MOCK.{i+1}",
                "district_name": name,
                "region": "Région simulée",
                "risk_score": round(score, 4),
                "risk_level": level,
                "cases_predicted": round(score * 300, 2),
            },
        })
    return {"type": "FeatureCollection", "features": features}


def _mock_district_detail(district_id: str) -> dict:
    hist = _mock_history(district_id)
    rng = np.random.default_rng(abs(hash(district_id)) % (2**32))
    name = DISTRICT_NAMES[abs(hash(district_id)) % len(DISTRICT_NAMES)]
    last_pred = hist[hist["cases_predicted"].notna()].iloc[0]

    obs_dates = hist[hist["cases_observed"].notna()]["date"]
    feature_history = []
    for d in obs_dates:
        feature_history.append({
            "date": d,
            "t2m_c": float(rng.uniform(24, 32)),
            "rh_pct": float(rng.uniform(30, 90)),
            "tp_mm": max(0.0, float(rng.normal(3, 4))),
            "ndvi": float(rng.uniform(0.2, 0.7)),
            "ndwi": float(rng.uniform(-0.1, 0.3)),
            "pfpr": float(rng.uniform(0.1, 0.6)),
            "itn_use": float(rng.uniform(0.1, 0.7)),
            "itn_access": float(rng.uniform(0.1, 0.8)),
            "population": float(rng.uniform(150_000, 4_500_000)),
        })
    means = pd.DataFrame(feature_history)[FEATURES].mean().to_dict() if feature_history else {}
    national_means = {f: means.get(f, 0) * float(rng.uniform(0.85, 1.15)) for f in FEATURES}

    return {
        "district_id": district_id,
        "district_name": name,
        "region": "Région simulée",
        "population": float(rng.uniform(150_000, 4_500_000)),
        "risk_score": float(np.clip(last_pred["cases_predicted"] / 350, 0, 1)),
        "risk_level": last_pred["risk_level"],
        "history": hist.to_dict(orient="records"),
        "feature_history": feature_history,
        "feature_means": means,
        "national_feature_means": national_means,
        "top_features": [
            {"feature": f, "value": float(rng.uniform(0, 1)), "contribution": float(rng.uniform(0.05, 1))}
            for f in rng.choice(FEATURES, size=5, replace=False)
        ],
    }
