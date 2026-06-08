"""Performance du modèle — comparaison des horizons, scatter, courbes d'apprentissage, importance globale, à propos."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from components.charts import plot_global_feature_importance, plot_scatter_obs_pred, plot_training_curves
from utils.api_client import cached_model_info

st.set_page_config(layout="wide", page_title="Modèle — MalariaWatch CI", page_icon="🧠")
st.markdown("<style>.stApp { background-color: #1E1E2E; }</style>", unsafe_allow_html=True)

st.title("🧠 Performance du modèle LSTM")


def _as_dataframe(records: list[dict] | None) -> pd.DataFrame | None:
    return pd.DataFrame(records) if records else None


# Le frontend est déployé séparément du backend (Streamlit Cloud / Render) : il n'a
# donc pas accès au système de fichiers de `backend/ml/saved_model`. Ces artefacts
# transitent par l'API (`/api/v1/model/info`) plutôt que d'être lus directement sur disque.
info = cached_model_info() or {}
metadata = info.get("metadata")
horizon_metrics = _as_dataframe(info.get("horizon_metrics"))
test_predictions = _as_dataframe(info.get("test_predictions"))
training_history = _as_dataframe(info.get("training_history"))
contributions = _as_dataframe(info.get("feature_contributions"))

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
