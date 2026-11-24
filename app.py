"""
Punto de entrada de la aplicación Streamlit:
Estudio Estadístico y Valuador de Viviendas (Regresión Lineal - scikit-learn).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_loader import artifacts_exist, load_metrics, load_model, load_sample
from src.styles import inject_styles, render_hero
from src.tabs.eda import render_eda_tab
from src.tabs.models import render_models_tab
from src.tabs.simulator import render_simulator_tab

st.set_page_config(
    page_title="Valuador Inmobiliario | Regresión Lineal",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def main() -> None:
    inject_styles()

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

    render_hero(
        train_rows=metrics["train_rows"],
        num_states=metrics["data_summary"]["num_states"],
        r2_score=lr_metrics["r2"],
    )

    section = st.segmented_control(
        "Navegación",
        options=[
            "Valuador de Vivienda",
            "Exploración del Mercado",
            "Evaluación del Modelo",
        ],
        default="Valuador de Vivienda",
        label_visibility="collapsed",
        width="stretch",
    ) or "Valuador de Vivienda"

    st.write("")

    if section == "Valuador de Vivienda":
        render_simulator_tab(model, metrics, sample_df, states_df)
    elif section == "Exploración del Mercado":
        render_eda_tab(metrics, sample_df, states_df)
    else:
        render_models_tab(metrics, sample_df)


if __name__ == "__main__":
    main()
