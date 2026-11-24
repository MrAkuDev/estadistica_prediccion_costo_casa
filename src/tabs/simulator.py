"""Sección 1: Valuador interactivo de propiedades (Regresión Lineal)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.charts import plot_price_breakdown, plot_state_price_position
from src.config import SQFT_PER_SQM, SQM_PER_ACRE
from src.styles import render_kpi_card


def render_simulator_tab(
    model, metrics: dict, sample_df: pd.DataFrame, states_df: pd.DataFrame
) -> None:
    state_list = sorted(states_df["state"].tolist())
    lr_info = metrics["linear_regression_full"]

    st.markdown(
        '<div class="section-heading">Parámetros de la Propiedad a Valuar</div>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        r1c1, r1c2, r1c3 = st.columns([1.2, 1.0, 1.3], gap="medium")
        with r1c1:
            default_state_idx = state_list.index("Texas") if "Texas" in state_list else 0
            selected_state = st.selectbox(
                "Estado (Mercado regional)",
                options=state_list,
                index=default_state_idx,
            )
        with r1c2:
            unit = st.segmented_control(
                "Unidad de superficie",
                options=["m²", "ft²"],
                default="m²",
            ) or "m²"
        with r1c3:
            if unit == "m²":
                house_size_sqm = st.slider(
                    "Superficie construida (m²)",
                    min_value=40,
                    max_value=740,
                    value=180,
                    step=5,
                )
                house_size_sqft = house_size_sqm * SQFT_PER_SQM
                st.caption(f"Equivalente: {house_size_sqft:,.0f} ft²")
            else:
                house_size_sqft = float(
                    st.slider(
                        "Superficie construida (ft²)",
                        min_value=400,
                        max_value=8000,
                        value=1950,
                        step=50,
                    )
                )
                house_size_sqm = house_size_sqft / SQFT_PER_SQM
                st.caption(f"Equivalente: {house_size_sqm:,.1f} m²")

        r2c1, r2c2, r2c3 = st.columns([1.0, 1.0, 1.5], gap="medium")
        with r2c1:
            bed = st.number_input("Habitaciones", min_value=1, max_value=10, value=3, step=1)
        with r2c2:
            bath = st.number_input("Baños", min_value=1, max_value=8, value=2, step=1)
        with r2c3:
            acre_lot = st.slider(
                "Terreno (acres)",
                min_value=0.05,
                max_value=10.0,
                value=0.25,
                step=0.05,
            )
            st.caption(f"Equivalente: {acre_lot * SQM_PER_ACRE:,.0f} m² de lote")

    # Predicción con LinearRegression de scikit-learn
    input_df = pd.DataFrame(
        [
            {
                "bed": float(bed),
                "bath": float(bath),
                "acre_lot": float(acre_lot),
                "house_size": float(house_size_sqft),
                "state": selected_state,
            }
        ]
    )
    raw_pred = float(model.predict(input_df)[0])
    predicted_price = max(raw_pred, 0.0)

    state_row = states_df[states_df["state"] == selected_state].iloc[0]
    state_median = float(state_row["median_price"])
    pct_vs_median = (
        ((predicted_price - state_median) / state_median) * 100.0
        if state_median > 0
        else 0.0
    )
    badge_class = "valuation-badge-pos" if pct_vs_median >= 0 else "valuation-badge-neg"

    # Tarjeta destacada de valuación + 3 KPIs secundarios
    col_val, col_kpis = st.columns([1.25, 1.35], gap="medium")
    with col_val:
        st.markdown(
            f"""
            <div class="valuation-card">
                <div class="valuation-label">Valor Estimado · Regresión Lineal</div>
                <div class="valuation-price">${predicted_price:,.0f} USD</div>
                <span class="{badge_class}">
                    {pct_vs_median:+.1f}% respecto a la mediana de {selected_state}
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if raw_pred < 0:
            st.warning(
                "Combinación extrema de variables con resultado lineal negativo; acotado a $0 USD."
            )

    with col_kpis:
        k1, k2, k3 = st.columns(3, gap="small")
        with k1:
            render_kpi_card("Precio / m²", f"${predicted_price / house_size_sqm:,.0f}", "USD por m²")
        with k2:
            render_kpi_card("Precio / ft²", f"${predicted_price / house_size_sqft:,.0f}", "USD por ft²")
        with k3:
            render_kpi_card(f"Mediana {selected_state}", f"${state_median:,.0f}", f"{int(state_row['count']):,} registros")

    # Dos gráficas complementarias: desglose de ecuación + posición en el mercado estatal
    st.markdown(
        '<div class="section-heading">Explicación de la Valuación</div>',
        unsafe_allow_html=True,
    )
    c_left, c_right = st.columns(2, gap="large")

    with c_left:
        with st.container(border=True):
            st.markdown("**Desglose por término de la ecuación lineal**")
            num_coefs = lr_info["numeric_coefficients"]
            state_coef = lr_info["state_coefficients"].get(selected_state, 0.0)
            contributions = {
                "Intercepto base": lr_info["intercept"],
                f"Superficie ({house_size_sqft:,.0f} ft²)": num_coefs["house_size"] * house_size_sqft,
                f"Baños ({bath})": num_coefs["bath"] * bath,
                f"Habitaciones ({bed})": num_coefs["bed"] * bed,
                f"Terreno ({acre_lot:.2f} acres)": num_coefs["acre_lot"] * acre_lot,
                f"Estado ({selected_state})": state_coef,
            }
            fig = plot_price_breakdown(contributions)
            st.pyplot(fig)
            plt.close(fig)

    with c_right:
        with st.container(border=True):
            st.markdown("**Posición frente al mercado estatal**")
            fig = plot_state_price_position(
                sample_df, selected_state, predicted_price, state_median
            )
            st.pyplot(fig)
            plt.close(fig)
