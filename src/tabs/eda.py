"""Pestaña 2: Análisis Exploratorio de Datos (EDA)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.charts import (
    plot_correlation_heatmap,
    plot_top_states,
    plot_variable_distribution,
)


def render_eda_tab(
    metrics: dict, sample_df: pd.DataFrame, states_df: pd.DataFrame
) -> None:
    data_summary = metrics["data_summary"]
    state_list = sorted(states_df["state"].tolist())

    st.subheader("Análisis Exploratorio de Datos (EDA)")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Registros Originales", f"{data_summary['raw_rows']:,}")
    k2.metric(
        "Registros Limpios",
        f"{data_summary['clean_rows']:,}",
        f"{data_summary['retained_pct']}% retenido",
    )
    k3.metric(
        "Precio Mediano",
        f"${metrics['descriptive_stats']['price']['median']:,.0f}",
    )
    k4.metric(
        "Superficie Mediana",
        f"{metrics['descriptive_stats']['house_size']['median']:,.0f} ft²",
    )

    st.markdown("#### Estadísticas Descriptivas (Dataset Limpio Completo)")
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
    desc_df.columns = [
        "Conteo",
        "Media",
        "Desv. Est.",
        "Mín",
        "Mediana",
        "Máx",
    ]
    st.dataframe(desc_df.style.format("{:,.2f}"), width="stretch")

    col_corr, col_dist = st.columns(2, gap="large")

    with col_corr:
        st.markdown("#### Matriz de Correlación de Pearson")
        corr_df = pd.DataFrame(metrics["correlation_matrix"])
        fig = plot_correlation_heatmap(corr_df)
        st.pyplot(fig)
        plt.close(fig)
        st.info(
            "**Hallazgo clave:** `house_size` (r = 0.50) y `bath` (r = 0.50) "
            "son las variables numéricas con mayor correlación positiva con el precio."
        )

    with col_dist:
        st.markdown("#### Distribución Interactiva de Variables")
        var_map = {
            "Precio (USD)": "price",
            "Superficie construida (ft²)": "house_size",
            "Habitaciones (bed)": "bed",
            "Baños (bath)": "bath",
            "Tamaño del terreno (acres)": "acre_lot",
        }
        chosen_label = st.selectbox("Selecciona una variable", list(var_map.keys()))
        chosen_col = var_map[chosen_label]

        fig = plot_variable_distribution(
            sample_df[chosen_col],
            label=chosen_label,
            is_discrete=chosen_col in ("bed", "bath"),
        )
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Top 15 Estados con Mayor Precio Mediano")
    fig = plot_top_states(states_df, top_n=15)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("Explorar muestra del dataset limpio (8,000 registros)"):
        filter_state = st.multiselect("Filtrar por estado", options=state_list, default=[])
        view_df = (
            sample_df[sample_df["state"].isin(filter_state)]
            if filter_state
            else sample_df
        )
        st.dataframe(view_df.head(200), width="stretch")
