"""Sección 3: Evaluación y diagnóstico del modelo de Regresión Lineal."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.charts import plot_actual_vs_predicted, plot_residuals_distribution
from src.styles import render_kpi_card


def render_models_tab(metrics: dict, sample_df: pd.DataFrame) -> None:
    lr_info = metrics["linear_regression_full"]
    lr_rows = [
        m for m in metrics["model_comparison"] if "Regresión Lineal" in m["name"]
    ]
    lr_main = next(
        m for m in lr_rows if m["category"] == "Estadístico (Principal)"
    )

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        render_kpi_card("Coeficiente R²", f"{lr_main['r2']:.4f}", "LinearRegression (+ Estado)")
    with c2:
        render_kpi_card("RMSE (Prueba)", f"${lr_main['rmse']:,.0f}", "Error cuadrático medio (raíz)")
    with c3:
        render_kpi_card("MAE (Prueba)", f"${lr_main['mae']:,.0f}", "Error absoluto medio")
    with c4:
        render_kpi_card("Conjunto de Prueba", f"{metrics['test_rows']:,}", "20% de los datos limpios")

    st.markdown(
        '<div class="section-heading">Diagnóstico de la Regresión Lineal</div>',
        unsafe_allow_html=True,
    )
    col_scat, col_res = st.columns(2, gap="large")
    with col_scat:
        with st.container(border=True):
            st.markdown("**Precio Real vs. Predicho**")
            fig = plot_actual_vs_predicted(sample_df)
            st.pyplot(fig)
            plt.close(fig)

    with col_res:
        with st.container(border=True):
            st.markdown("**Distribución de Residuos (Real − Predicho)**")
            fig = plot_residuals_distribution(sample_df)
            st.pyplot(fig)
            plt.close(fig)

    st.markdown(
        '<div class="section-heading">Métricas y Coeficientes del Modelo</div>',
        unsafe_allow_html=True,
    )
    col_m, col_c = st.columns(2, gap="large")

    with col_m:
        st.markdown("**Impacto de incluir ubicación geográfica (`state`)**")
        comp_df = pd.DataFrame(lr_rows)[["name", "r2", "rmse", "mae", "mse"]].copy()
        comp_df.columns = ["Especificación Lineal", "R²", "RMSE (USD)", "MAE (USD)", "MSE"]
        st.dataframe(
            comp_df.style.format(
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
        st.caption(
            "Incorporar la variable geográfica `state` en la Regresión Lineal eleva R² "
            "de 0.2836 a 0.5275 y reduce el RMSE en más de $52,900 USD."
        )

    with col_c:
        st.markdown("**Coeficientes de Variables Numéricas (Modelo Principal)**")
        coef_rows = [
            {
                "Variable": f"Intercepto (base: {lr_info['base_state']})",
                "Coeficiente (USD)": lr_info["intercept"],
            },
            {
                "Variable": "Superficie (house_size, por ft²)",
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
                "Variable": "Terreno (acre_lot, por acre)",
                "Coeficiente (USD)": lr_info["numeric_coefficients"]["acre_lot"],
            },
        ]
        st.dataframe(
            pd.DataFrame(coef_rows).style.format({"Coeficiente (USD)": "${:,.2f}"}),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "*El signo negativo parcial de `bed` refleja multicolinealidad al mantener fija "
            "la superficie construida y el número de baños."
        )
