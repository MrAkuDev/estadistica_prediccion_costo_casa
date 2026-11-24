"""Configuración de rutas y constantes de conversión de unidades."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "linear_regression_model.joblib"
METRICS_PATH = BASE_DIR / "models" / "metrics_and_eda.json"
SAMPLE_PATH = BASE_DIR / "data" / "sample_realtor_clean.csv"

SQFT_PER_SQM = 10.7639104
SQM_PER_ACRE = 4046.85642
