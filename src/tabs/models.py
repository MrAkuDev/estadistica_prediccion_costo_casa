"""Sección 3: Evaluación y Comparación de Modelos."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.charts import plot_actual_vs_predicted, plot_model_comparison
from src.styles import render_kpi_card


def render_models_tab(metrics: dict, sample_df: pd.DataFrame) -> None:
    lr_info = metrics["linear_regression_full"]
    model_comp_df = pd.DataFrame(metrics["model_comparison"])
    lr_row = next(
        m for m in metrics["model_comparison"] if m["category"] == "Estadístico (Principal)"
    )
    best_row = max(metrics["model_comparison"], key=lambda m: m["r2"])

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        render_kpi_card("Modelo del Valuador", "Reg. Lineal", "LinearRegression (sklearn)")
    with c2:
        render_kpi_card("R² Regresión Lineal", f"{lr_row['r2']:.4f}", f"RMSE: ${lr_row['rmse']:,.0f}")
    with c3:
        render_kpi_card("Mejor R² (ML)", f"{best_row['r2']:.4f}", best_row["name"])
    with c4:
        render_kpi_card("Conjunto de Prueba", f"{metrics['test_rows']:,}", "20% de los datos limpios")

    st.markdown(
        '<div class="section-heading">Comparativa Visual de Modelos</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        fig = plot_model_comparison(metrics["model_comparison"])
        st.pyplot(fig)
        plt.close(fig)

    st.markdown(
        '<div class="section-heading">Tabla de Métricas y Ajuste Lineal</div>',
        unsafe_allow_html=True,
    )
    col_tbl, col_scat = st.columns([1.15, 1.0], gap="large")

    with col_tbl:
        comp_display = model_comp_df[["name", "category", "r2", "rmse", "mae", "mse"]].copy()
        comp_display.columns = ["Modelo", "Tipo", "R²", "RMSE (USD)", "MAE (USD)", "MSE"]
        st.dataframe(
            comp_display.style.format(
                {
                    "R²": "{:.4f}",
                    "RMSE (USD)": "${:,.0f}",
                    "MAE (USD)": "${:,.0f}",
                    "MSE": "{:,.0f}",
                }
            ),
            width="stretch",
            hide_index=True,
        )

        st.markdown("**Coeficientes de la Regresión Lineal Múltiple**")
        coef_rows = [
            {
                "Variable": f"Intercepto (base: {lr_info['base_state']})",
                "Coeficiente (USD)": lr_info["intercept"],
            },
            {
                "Variable": "Superficie (por ft²)",
                "Coeficiente (USD)": lr_info["numeric_coefficients"]["house_size"],
            },
            {
                "Variable": "Baños (bath)",
                "Coeficiente (USD)": lr_info["numeric_coefficients"]["bath"],
            },
            {
                "Variable": "Habitaciones (bed)*",
                "Coeficiente (USD)": lr_info["numeric_coefficients"]["bed"],
            },
            {
                "Variable": "Terreno (por acre)",
                "Coeficiente (USD)": lr_info["numeric_coefficients"]["acre_lot"],
            },
        ]
        st.dataframe(
            pd.DataFrame(coef_rows).style.format({"Coeficiente (USD)": "${:,.2f}"}),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "*El signo negativo parcial de `bed` refleja multicolinealidad al mantener constante "
            "la superficie total y los baños."
        )

    with col_scat:
        with st.container(border=True):
            st.markdown("**Precio Real vs. Predicho (Regresión Lineal)**")
            fig = plot_actual_vs_predicted(sample_df)
            st.pyplot(fig)
            plt.close(fig)
