"""Sección 2: Análisis Exploratorio de Datos (EDA)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.charts import (
    plot_correlation_heatmap,
    plot_top_states,
    plot_variable_distribution,
)
from src.styles import render_kpi_card


def render_eda_tab(
    metrics: dict, sample_df: pd.DataFrame, states_df: pd.DataFrame
) -> None:
    data_summary = metrics["data_summary"]
    state_list = sorted(states_df["state"].tolist())

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        render_kpi_card("Registros Crudos", f"{data_summary['raw_rows']:,}", "Dataset original")
    with k2:
        render_kpi_card(
            "Registros Limpios",
            f"{data_summary['clean_rows']:,}",
            f"{data_summary['retained_pct']}% conservado",
        )
    with k3:
        render_kpi_card(
            "Precio Mediano",
            f"${metrics['descriptive_stats']['price']['median']:,.0f}",
            "USD (residencial)",
        )
    with k4:
        render_kpi_card(
            "Superficie Mediana",
            f"{metrics['descriptive_stats']['house_size']['median']:,.0f} ft²",
            "≈ 166 m² construidos",
        )

    st.markdown(
        '<div class="section-heading">Estadísticas Descriptivas</div>',
        unsafe_allow_html=True,
    )
    desc_df = pd.DataFrame(metrics["descriptive_stats"]).T[
        ["count", "mean", "std", "min", "median", "max"]
    ]
    desc_df.index = [
        "Precio (USD)",
        "Habitaciones (bed)",
        "Baños (bath)",
        "Terreno (acres)",
        "Superficie (ft²)",
    ]
    desc_df.columns = ["Conteo", "Media", "Desv. Est.", "Mín", "Mediana", "Máx"]
    st.dataframe(desc_df.style.format("{:,.2f}"), width="stretch")

    st.markdown(
        '<div class="section-heading">Correlaciones y Distribuciones</div>',
        unsafe_allow_html=True,
    )
    col_corr, col_dist = st.columns(2, gap="large")

    with col_corr:
        with st.container(border=True):
            st.markdown("**Matriz de Correlación de Pearson**")
            corr_df = pd.DataFrame(metrics["correlation_matrix"])
            fig = plot_correlation_heatmap(corr_df)
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "Superficie (r = 0.50) y baños (r = 0.50) lideran la correlación lineal con el precio."
            )

    with col_dist:
        with st.container(border=True):
            st.markdown("**Histograma por Variable**")
            var_map = {
                "Precio (USD)": "price",
                "Superficie construida (ft²)": "house_size",
                "Habitaciones (bed)": "bed",
                "Baños (bath)": "bath",
                "Tamaño del terreno (acres)": "acre_lot",
            }
            chosen_label = st.selectbox("Variable a inspeccionar", list(var_map.keys()))
            chosen_col = var_map[chosen_label]

            fig = plot_variable_distribution(
                sample_df[chosen_col],
                label=chosen_label,
                is_discrete=chosen_col in ("bed", "bath"),
            )
            st.pyplot(fig)
            plt.close(fig)

    st.markdown(
        '<div class="section-heading">Panorama Regional (Top 12 Estados)</div>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        fig = plot_top_states(states_df, top_n=12)
        st.pyplot(fig)
        plt.close(fig)

    with st.expander("Consultar muestra de registros limpios (8,000 propiedades)"):
        filter_state = st.multiselect("Filtrar por estado", options=state_list, default=[])
        view_df = (
            sample_df[sample_df["state"].isin(filter_state)]
            if filter_state
            else sample_df
        )
        st.dataframe(view_df.head(200), width="stretch")
