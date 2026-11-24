"""Carga en caché del modelo entrenado, métricas y muestra de datos."""

from __future__ import annotations

import json

import joblib
import pandas as pd
import streamlit as st

from src.config import METRICS_PATH, MODEL_PATH, SAMPLE_PATH


def artifacts_exist() -> bool:
    return MODEL_PATH.exists() and METRICS_PATH.exists() and SAMPLE_PATH.exists()


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)
