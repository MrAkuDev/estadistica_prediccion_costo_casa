"""Componente de barra lateral con resumen del modelo principal."""

from __future__ import annotations

import streamlit as st


def render_sidebar(metrics: dict, lr_metrics: dict) -> None:
    data_summary = metrics["data_summary"]
    with st.sidebar:
        st.title("🏠 Proyecto Inmobiliario")
        st.caption(
            "Predicción del valor de propiedades residenciales mediante "
            "**Regresión Lineal (`scikit-learn`)** y análisis estadístico."
        )
        st.divider()
        st.subheader("Resumen del Modelo Principal")
        st.metric("Algoritmo", "LinearRegression")
        st.metric("R² (Prueba)", f"{lr_metrics['r2']:.4f}")
        st.metric("RMSE (Prueba)", f"${lr_metrics['rmse']:,.0f}")
        st.metric("MAE (Prueba)", f"${lr_metrics['mae']:,.0f}")
        st.divider()
        st.caption(
            f"Entrenado con **{metrics['train_rows']:,}** propiedades y "
            f"evaluado con **{metrics['test_rows']:,}** propiedades "
            f"({data_summary['num_states']} estados de EE. UU.)."
        )
