"""Pestaña 1: Simulador interactivo de precios basado en Regresión Lineal."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.charts import plot_price_breakdown
from src.config import SQFT_PER_SQM, SQM_PER_ACRE


def render_simulator_tab(model, metrics: dict, states_df: pd.DataFrame) -> None:
    state_list = sorted(states_df["state"].tolist())
    lr_info = metrics["linear_regression_full"]

    st.subheader("Simulador Interactivo de Precio (Regresión Lineal)")
    st.markdown(
        "Ingresa las características de la vivienda para estimar su valor de mercado "
        "utilizando el modelo `LinearRegression` de `scikit-learn`."
    )

    col_form, col_result = st.columns([1.05, 1.15], gap="large")

    with col_form:
        with st.container(border=True):
            st.markdown("#### Características de la Propiedad")

            default_state_idx = state_list.index("Texas") if "Texas" in state_list else 0
            selected_state = st.selectbox(
                "Ubicación (Estado)",
                options=state_list,
                index=default_state_idx,
                help="El estado captura diferencias regionales del mercado inmobiliario.",
            )

            unit = st.radio(
                "Unidad de superficie construida",
                options=["Metros cuadrados (m²)", "Pies cuadrados (ft²)"],
                horizontal=True,
            )

            if unit == "Metros cuadrados (m²)":
                house_size_sqm = st.slider(
                    "Superficie construida (m²)",
                    min_value=40,
                    max_value=740,
                    value=180,
                    step=5,
                )
                house_size_sqft = house_size_sqm * SQFT_PER_SQM
                st.caption(f"Equivalente a **{house_size_sqft:,.0f} ft²**")
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
                st.caption(f"Equivalente a **{house_size_sqm:,.1f} m²**")

            c1, c2 = st.columns(2)
            with c1:
                bed = st.number_input(
                    "Habitaciones (bed)",
                    min_value=1,
                    max_value=10,
                    value=3,
                    step=1,
                )
            with c2:
                bath = st.number_input(
                    "Baños (bath)",
                    min_value=1,
                    max_value=8,
                    value=2,
                    step=1,
                )

            acre_lot = st.slider(
                "Tamaño del terreno (acres)",
                min_value=0.05,
                max_value=10.0,
                value=0.25,
                step=0.05,
            )
            st.caption(f"Equivalente a **{acre_lot * SQM_PER_ACRE:,.0f} m²** de terreno")

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

    with col_result:
        with st.container(border=True):
            st.markdown("#### Resultado de la Estimación")
            st.metric(
                label="Precio Estimado de la Vivienda (USD)",
                value=f"${predicted_price:,.0f}",
                delta=f"{pct_vs_median:+.1f}% vs. mediana de {selected_state}",
            )
            if raw_pred < 0:
                st.warning(
                    "La combinación extrema de variables produjo un valor lineal negativo; "
                    "se acota a $0 USD."
                )

            m1, m2, m3 = st.columns(3)
            m1.metric("Precio / m²", f"${predicted_price / house_size_sqm:,.0f}")
            m2.metric("Precio / ft²", f"${predicted_price / house_size_sqft:,.0f}")
            m3.metric(f"Mediana {selected_state}", f"${state_median:,.0f}")

            st.divider()
            st.markdown("##### Desglose de la Ecuación Lineal")
            num_coefs = lr_info["numeric_coefficients"]
            state_coef = lr_info["state_coefficients"].get(selected_state, 0.0)
            contributions = {
                "Intercepto base": lr_info["intercept"],
                f"Superficie ({house_size_sqft:,.0f} ft²)": num_coefs["house_size"] * house_size_sqft,
                f"Baños ({bath})": num_coefs["bath"] * bath,
                f"Habitaciones ({bed})": num_coefs["bed"] * bed,
                f"Terreno ({acre_lot:.2f} acres)": num_coefs["acre_lot"] * acre_lot,
                f"Ajuste por Estado ({selected_state})": state_coef,
            }

            fig = plot_price_breakdown(contributions)
            st.pyplot(fig)
            plt.close(fig)
