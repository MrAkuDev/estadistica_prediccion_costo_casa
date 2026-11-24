"""Pestaña 3: Evaluación y Comparación de Modelos."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.charts import plot_actual_vs_predicted, plot_model_comparison


def render_models_tab(metrics: dict, sample_df: pd.DataFrame) -> None:
    lr_info = metrics["linear_regression_full"]
    model_comp_df = pd.DataFrame(metrics["model_comparison"])

    st.subheader("Evaluación y Comparación de Modelos Predictivos")
    st.markdown(
        "Se comparan modelos estadísticos lineales frente a modelos de Machine Learning "
        "evaluados sobre el mismo conjunto de prueba (20% de los datos limpios)."
    )

    comp_display = model_comp_df[["name", "category", "r2", "rmse", "mae", "mse"]].copy()
    comp_display.columns = ["Modelo", "Tipo", "R²", "RMSE (USD)", "MAE (USD)", "MSE"]
    st.dataframe(
        comp_display.style.format(
            {
                "R²": "{:.4f}",
                "RMSE (USD)": "${:,.2f}",
                "MAE (USD)": "${:,.2f}",
                "MSE": "{:,.0f}",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    col_m1, col_m2 = st.columns(2, gap="large")
    with col_m1:
        st.markdown("#### Comparación de R² y RMSE")
        fig = plot_model_comparison(metrics["model_comparison"])
        st.pyplot(fig)
        plt.close(fig)

    with col_m2:
        st.markdown("#### Precio Real vs. Predicho (Regresión Lineal)")
        fig = plot_actual_vs_predicted(sample_df)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Coeficientes de la Regresión Lineal Múltiple (Variables Numéricas)")
    coef_rows = [
        {
            "Variable": f"Intercepto (Estado base: {lr_info['base_state']})",
            "Coeficiente (USD)": lr_info["intercept"],
            "Interpretación": "Valor base del modelo en el estado de referencia.",
        },
        {
            "Variable": "Superficie (house_size, por ft²)",
            "Coeficiente (USD)": lr_info["numeric_coefficients"]["house_size"],
            "Interpretación": "Incremento promedio en USD por cada pie cuadrado adicional.",
        },
        {
            "Variable": "Baños (bath)",
            "Coeficiente (USD)": lr_info["numeric_coefficients"]["bath"],
            "Interpretación": "Incremento promedio en USD por cada baño adicional.",
        },
        {
            "Variable": "Habitaciones (bed)",
            "Coeficiente (USD)": lr_info["numeric_coefficients"]["bed"],
            "Interpretación": "Efecto parcial manteniendo fija la superficie y los baños (multicolinealidad).",
        },
        {
            "Variable": "Terreno (acre_lot, por acre)",
            "Coeficiente (USD)": lr_info["numeric_coefficients"]["acre_lot"],
            "Interpretación": "Incremento promedio en USD por cada acre adicional de terreno.",
        },
    ]
    st.dataframe(
        pd.DataFrame(coef_rows).style.format({"Coeficiente (USD)": "${:,.2f}"}),
        width="stretch",
        hide_index=True,
    )
