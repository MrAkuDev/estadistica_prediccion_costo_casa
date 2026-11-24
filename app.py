"""
Punto de entrada de la aplicación Streamlit:
Predicción del Valor de Propiedades Inmobiliarias (Regresión Lineal - scikit-learn).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_loader import artifacts_exist, load_metrics, load_model, load_sample
from src.tabs.eda import render_eda_tab
from src.tabs.models import render_models_tab
from src.tabs.sidebar import render_sidebar
from src.tabs.simulator import render_simulator_tab

st.set_page_config(
    page_title="Predicción de Precio de Casas | Regresión Lineal",
    page_icon="🏠",
    layout="wide",
)


def main() -> None:
    if not artifacts_exist():
        st.error(
            "No se encontraron los artefactos del modelo. Ejecuta `python train_model.py` primero."
        )
        st.stop()

    model = load_model()
    metrics = load_metrics()
    sample_df = load_sample()
    states_df = pd.DataFrame(metrics["states"])
    lr_metrics = next(
        m for m in metrics["model_comparison"] if m["category"] == "Estadístico (Principal)"
    )

    render_sidebar(metrics, lr_metrics)

    st.title("Predicción del Valor de Propiedades Inmobiliarias")
    st.markdown(
        "Dashboard interactivo y simulador de precios basado en el dataset de "
        "propiedades residenciales (`realtor-data.zip.csv`) y un modelo de "
        "**Regresión Lineal Múltiple** entrenado con `scikit-learn`."
    )

    tab_sim, tab_eda, tab_models = st.tabs(
        [
            "🧮 Simulador de Precio",
            "📊 Análisis Exploratorio (EDA)",
            "🤖 Evaluación y Comparación de Modelos",
        ]
    )

    with tab_sim:
        render_simulator_tab(model, metrics, states_df)

    with tab_eda:
        render_eda_tab(metrics, sample_df, states_df)

    with tab_models:
        render_models_tab(metrics, sample_df)


if __name__ == "__main__":
    main()
