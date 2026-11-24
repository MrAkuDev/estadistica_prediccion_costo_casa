"""Estilos CSS personalizados y componentes visuales editoriales para Streamlit."""

from __future__ import annotations

import streamlit as st

CUSTOM_CSS = """
<style>
/* Ocultar sidebar y menú por defecto para una navegación superior limpia */
[data-testid="stSidebar"], [data-testid="collapsedControl"] {
    display: none;
}

.block-container {
    padding-top: 1.6rem;
    padding-bottom: 2.5rem;
    max-width: 1180px;
}

/* Encabezado editorial principal */
.hero-header {
    background: linear-gradient(120deg, #1B4332 0%, #2D6A4F 65%, #40916C 100%);
    color: #F6F4EF;
    padding: 1.8rem 2.2rem;
    border-radius: 16px;
    margin-bottom: 1.4rem;
    box-shadow: 0 10px 26px rgba(27, 67, 50, 0.14);
}
.hero-kicker {
    font-size: 0.78rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #D8F3DC;
    font-weight: 600;
    margin-bottom: 0.35rem;
}
.hero-title {
    font-size: 2.05rem;
    font-weight: 800;
    margin: 0 0 0.45rem 0;
    line-height: 1.15;
    color: #FFFFFF;
}
.hero-subtitle {
    font-size: 0.98rem;
    color: #E8F5E9;
    margin: 0;
    max-width: 780px;
}

/* Tarjeta principal de valuación */
.valuation-card {
    background: #FFFFFF;
    border: 1px solid #DDD6C7;
    border-left: 6px solid #1B4332;
    border-radius: 14px;
    padding: 1.5rem 1.8rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 4px 14px rgba(30, 36, 33, 0.05);
}
.valuation-label {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #5C6B63;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.valuation-price {
    font-size: 2.75rem;
    font-weight: 800;
    color: #1B4332;
    line-height: 1.05;
    margin: 0;
}
.valuation-badge-pos {
    display: inline-block;
    margin-top: 0.55rem;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.84rem;
    font-weight: 600;
    background: #E6F4EA;
    color: #1B4332;
}
.valuation-badge-neg {
    display: inline-block;
    margin-top: 0.55rem;
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    font-size: 0.84rem;
    font-weight: 600;
    background: #FDECE8;
    color: #A63A1E;
}

/* Tarjetas KPI compactas */
.kpi-card {
    background: #FFFFFF;
    border: 1px solid #DDD6C7;
    border-radius: 12px;
    padding: 1rem 1.15rem;
    height: 100%;
}
.kpi-label {
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #647069;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.kpi-value {
    font-size: 1.55rem;
    font-weight: 800;
    color: #1E2421;
    margin: 0;
}
.kpi-sub {
    font-size: 0.82rem;
    color: #52796F;
    margin-top: 0.2rem;
}

/* Títulos de sección */
.section-heading {
    font-size: 1.25rem;
    font-weight: 700;
    color: #1B4332;
    border-bottom: 2px solid #DDD6C7;
    padding-bottom: 0.35rem;
    margin-top: 0.6rem;
    margin-bottom: 1rem;
}
</style>
"""


def inject_styles() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_hero(train_rows: int, num_states: int, r2_score: float) -> None:
    st.markdown(
        f"""
        <div class="hero-header">
            <div class="hero-kicker">Valuación Inmobiliaria · Regresión Lineal (Scikit-Learn)</div>
            <h1 class="hero-title">Estudio Estadístico y Valuador de Viviendas</h1>
            <p class="hero-subtitle">
                Modelo entrenado con <strong>{train_rows:,}</strong> propiedades en
                <strong>{num_states}</strong> mercados estatales de EE. UU.
                (R² = <strong>{r2_score:.4f}</strong>).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, sub: str = "") -> None:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )
