"""Performance du modèle — comparaison des horizons, scatter, courbes d'apprentissage, importance globale, à propos."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from components.charts import plot_global_feature_importance, plot_scatter_obs_pred, plot_training_curves

st.set_page_config(layout="wide", page_title="Modèle — MalariaWatch CI", page_icon="🧠")
st.markdown("<style>.stApp { background-color: #1E1E2E; }</style>", unsafe_allow_html=True)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")
# Le frontend lit directement les artefacts d'entraînement partagés via le volume backend/ml/saved_model
SAVED_DIR = Path(__file__).resolve().parent.parent.parent / "backend" / "ml" / "saved_model"

st.title("🧠 Performance du modèle LSTM")


@st.cache_data(ttl=3600, show_spinner=False)
def _load_artifact(name: str) -> pd.DataFrame | None:
    path = SAVED_DIR / name
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(ttl=3600, show_spinner=False)
def _load_metadata() -> dict | None:
    import json
    path = SAVED_DIR / "metadata.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


metadata = _load_metadata()
horizon_metrics = _load_artifact("horizon_metrics.csv")
test_predictions = _load_artifact("test_predictions.csv")
training_history = _load_artifact("training_history.csv")
contributions = _load_artifact("feature_contributions.csv")
if contributions is None:
    contributions = _load_artifact(str(Path("..") / "data" / "predictions" / "feature_contributions.csv"))

if metadata is None:
    st.warning(
        "Aucun modèle entraîné détecté. Lancez `python backend/ml/train.py` puis "
        "`python backend/ml/evaluate.py` pour générer les artefacts de performance."
    )

# ── 1. Tableau comparatif des horizons ───────────────────────────────────────
st.subheader("Comparaison des horizons de prévision")
st.caption(
    "Métriques de la stratégie de prévision RÉCURSIVE (celle utilisée en production) "
    "sur le jeu de test 2021-2022 — jamais vu pendant l'entraînement. Plus l'horizon "
    "s'éloigne, plus l'incertitude sur les variables exogènes futures (climatologie "
    "saisonnière) dégrade naturellement la précision."
)
if horizon_metrics is not None:
    sub = horizon_metrics[horizon_metrics["horizon"].isin([4, 6, 8])].copy()
    sub["horizon"] = sub["horizon"].astype(str) + " mois"
    sub = sub.rename(columns={"horizon": "Horizon", "rmse": "RMSE", "mae": "MAE", "r2": "R²", "n": "Nb. observations"})
    st.dataframe(sub.set_index("Horizon"), use_container_width=True)
else:
    st.info("Tableau indisponible — exécutez `python backend/ml/evaluate.py` pour le générer.")

st.divider()

# ── 2. Scatter observé vs prédit ──────────────────────────────────────────────
st.subheader("Cas observés vs prédits (jeu de test, prévision à 1 mois)")
if test_predictions is not None:
    st.plotly_chart(plot_scatter_obs_pred(test_predictions), use_container_width=True)
    if metadata:
        m = metadata["test_metrics"]
        st.caption(f"RMSE = {m['rmse']:.2f}  ·  MAE = {m['mae']:.2f}  ·  R² = {m['r2']:.3f}  "
                   f"(période de test : {metadata['test_period'][0]} → {metadata['test_period'][1]})")
else:
    st.info("Scatter indisponible — entraînez le modèle pour générer `test_predictions.csv`.")

st.divider()

# ── 3. Courbes d'apprentissage ────────────────────────────────────────────────
st.subheader("Courbes de loss — entraînement vs validation")
if training_history is not None:
    st.plotly_chart(plot_training_curves(training_history.set_index("epoch")), use_container_width=True)
else:
    st.info("Historique d'entraînement indisponible.")

st.divider()

# ── 4. Importance globale des features ────────────────────────────────────────
st.subheader("Importance globale des features (moyenne nationale)")
st.caption("Moyenne, sur l'ensemble des districts, des contributions SHAP (Gradient × Input) "
           "à la prédiction du mois suivant.")
if contributions is not None and not contributions.empty:
    st.plotly_chart(plot_global_feature_importance(contributions), use_container_width=True)
else:
    st.info("Contributions indisponibles — déclenchez le pipeline d'inférence "
            "(page d'accueil → *Rafraîchir les prédictions*) pour les générer.")

st.divider()

# ── 5. À propos du modèle ─────────────────────────────────────────────────────
st.subheader("À propos du modèle")
if metadata:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""
            **Architecture**
            - Type : LSTM empilé (2 couches récurrentes + tête dense)
            - `LSTM(64) → Dropout(0.2) → LSTM(32) → Dropout(0.2) → Dense(16, relu) → Dense(1, sigmoid)`
            - Fenêtre d'entrée : **{metadata['seq_len']} mois** glissants
            - Variable cible : `{metadata['target']}` (taux d'incidence pour 1000 habitants)
            - Version : `{metadata['version']}` — entraîné le {metadata['trained_at']}
            """
        )
    with c2:
        st.markdown(
            f"""
            **Données d'entraînement**
            - {metadata['n_districts']} districts sanitaires × granularité mensuelle
            - Sources : Malaria Atlas Project (PfPR, incidence, ITN), ERA5/Copernicus
              (température, humidité, précipitations), MODIS (NDVI, NDWI), WorldPop
              (population), GADM (contours administratifs)
            - Découpage **chronologique** anti-fuite :
                - Train : {metadata['train_period'][0]} → {metadata['train_period'][1]} ({metadata['n_train']} séquences)
                - Validation : {metadata['val_period'][0]} → {metadata['val_period'][1]} ({metadata['n_val']} séquences)
                - Test : {metadata['test_period'][0]} → {metadata['test_period'][1]} ({metadata['n_test']} séquences)
            """
        )
    st.markdown(
        "**Features utilisées :** " + ", ".join(f"`{f}`" for f in metadata["features"])
    )
else:
    st.info("Métadonnées du modèle indisponibles.")
