"""
Script de entrenamiento, análisis exploratorio y generación de artefactos
para el proyecto: Predicción del Valor de Propiedades Inmobiliarias.

Genera:
- models/linear_regression_model.joblib  (Modelo principal de Regresión Lineal de scikit-learn)
- models/metrics_and_eda.json            (Estadísticas, correlaciones, métricas y coeficientes)
- data/sample_realtor_clean.csv          (Muestra representativa ligera para Streamlit)
- informe/figuras/*.png                  (Gráficas en alta resolución para el informe en LaTeX)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_RAW = BASE_DIR / "realtor-data.zip.csv"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
FIGURES_DIR = BASE_DIR / "informe" / "figuras"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 10,
})


def load_and_clean_data(path: Path) -> tuple[pd.DataFrame, dict]:
    print(f"Cargando dataset desde {path}...")
    df_raw = pd.read_csv(path)
    raw_rows, raw_cols = df_raw.shape
    missing_raw = df_raw.isna().sum().to_dict()

    cols = ["price", "bed", "bath", "acre_lot", "house_size", "state", "status"]
    df = df_raw[cols].dropna().copy()
    after_dropna = len(df)

    # Filtrado de valores atípicos extremos y errores de captura para viviendas residenciales
    df_clean = df[
        (df["price"] >= 30_000)
        & (df["price"] <= 2_000_000)
        & (df["bed"] >= 1)
        & (df["bed"] <= 10)
        & (df["bath"] >= 1)
        & (df["bath"] <= 8)
        & (df["acre_lot"] > 0)
        & (df["acre_lot"] <= 20)
        & (df["house_size"] >= 400)
        & (df["house_size"] <= 8_000)
    ].copy()

    # Filtrar estados con al menos 100 registros para estabilidad estadística
    state_counts = df_clean["state"].value_counts()
    valid_states = state_counts[state_counts >= 100].index
    df_clean = df_clean[df_clean["state"].isin(valid_states)].copy()

    summary = {
        "raw_rows": int(raw_rows),
        "raw_cols": int(raw_cols),
        "after_dropna_rows": int(after_dropna),
        "clean_rows": int(len(df_clean)),
        "retained_pct": round(100.0 * len(df_clean) / raw_rows, 2),
        "num_states": int(df_clean["state"].nunique()),
        "missing_raw": {k: int(v) for k, v in missing_raw.items()},
        "status_counts_clean": {
            str(k): int(v) for k, v in df_clean["status"].value_counts().items()
        },
    }
    print(f"Filas originales: {raw_rows:,} -> Filas limpias: {len(df_clean):,} ({summary['retained_pct']}%)")
    return df_clean, summary


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "mse": round(mse, 2),
        "rmse": round(rmse, 2),
        "mae": round(mae, 2),
        "r2": round(r2, 4),
    }


def generate_figures(
    df_clean: pd.DataFrame,
    corr_df: pd.DataFrame,
    state_stats: pd.DataFrame,
    model_results: list[dict],
    y_test_sample: np.ndarray,
    y_pred_sample: np.ndarray,
) -> None:
    print("Generando figuras para el informe en LaTeX (paleta editorial)...")

    forest = "#1B4332"
    sage = "#40916C"
    mint = "#52B788"
    terracotta = "#C85A32"
    sand = "#D4A373"
    charcoal = "#2D3A3A"
    slate = "#95A5A6"

    def style_ax(ax) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CFC8B8")
        ax.spines["bottom"].set_color("#CFC8B8")
        ax.tick_params(colors=charcoal)

    # 1. Distribuciones de variables numéricas
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    plot_specs = [
        ("price", "Precio (USD)", forest, 1_000, "Miles de USD"),
        ("house_size", "Superficie construida (ft²)", sage, 1, "Pies cuadrados (ft²)"),
        ("bed", "Habitaciones (bed)", mint, 1, "Número de habitaciones"),
        ("bath", "Baños (bath)", sand, 1, "Número de baños"),
        ("acre_lot", "Tamaño del terreno (acres)", terracotta, 1, "Acres"),
    ]
    sample_plot = df_clean.sample(n=min(100_000, len(df_clean)), random_state=42)
    for ax, (col, title, color, div, xlabel) in zip(axes.flat, plot_specs):
        vals = sample_plot[col] / div
        if col in ("bed", "bath"):
            bins = np.arange(vals.min() - 0.5, vals.max() + 1.5, 1)
            ax.hist(vals, bins=bins, color=color, alpha=0.86, edgecolor="white")
        else:
            ax.hist(vals, bins=40, color=color, alpha=0.86, edgecolor="white")
        ax.axvline(vals.median(), color=terracotta if color != terracotta else forest, linestyle="--", linewidth=1.5, label=f"Mediana: {vals.median():.2f}")
        ax.set_title(title, fontweight="bold", color=forest)
        ax.set_xlabel(xlabel, color=charcoal)
        ax.set_ylabel("Frecuencia", color=charcoal)
        style_ax(ax)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.legend(frameon=False, fontsize=8.5)

    ax_last = axes.flat[5]
    log_price = np.log10(sample_plot["price"])
    ax_last.hist(log_price, bins=40, color=charcoal, alpha=0.82, edgecolor="white")
    ax_last.set_title("Distribución de log10(Precio)", fontweight="bold", color=forest)
    ax_last.set_xlabel("log10(Precio en USD)", color=charcoal)
    ax_last.set_ylabel("Frecuencia", color=charcoal)
    style_ax(ax_last)
    ax_last.grid(axis="y", alpha=0.25, linestyle="--")

    fig.suptitle("Distribución de Variables Principales del Dataset Limpio", fontsize=13, fontweight="bold", color=forest, y=0.99)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "distribuciones.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Matriz de correlación (escala BrBG igual que en Streamlit)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    labels = ["Precio", "Habitaciones", "Baños", "Terreno (acres)", "Superficie (ft²)"]
    mat = corr_df.values
    im = ax.imshow(mat, cmap="BrBG", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10, color=charcoal)
    ax.set_yticklabels(labels, fontsize=10, color=charcoal)
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(False)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mat[i, j]
            text_color = "white" if abs(val) > 0.55 else "#1E2421"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=10.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlación de Pearson (r)", color=charcoal)
    ax.set_title("Matriz de Correlación de Variables Numéricas", fontweight="bold", color=forest, pad=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "correlacion.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Top 15 estados por precio mediano (gráfico tipo lollipop igual que en Streamlit)
    top15 = state_stats.sort_values("median_price", ascending=False).head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    y_pos = np.arange(len(top15))
    vals = top15["median_price"] / 1_000
    ax.hlines(y=y_pos, xmin=0, xmax=vals, color=sage, linewidth=2.6, alpha=0.88)
    ax.scatter(vals, y_pos, color=forest, s=75, zorder=3)
    for y, v in zip(y_pos, vals):
        ax.text(v + 14, y, f"${v:,.0f}k", va="center", fontsize=8.5, fontweight="bold", color=charcoal)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top15["state"])
    ax.set_xlabel("Precio Mediano (Miles de USD)", color=charcoal)
    ax.set_title("Top 15 Estados con Mayor Precio Mediano de Vivienda", fontweight="bold", color=forest)
    ax.set_xlim(0, vals.max() * 1.16)
    style_ax(ax)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "top_estados.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Real vs Predicho y Distribución de Residuos (paleta bosque y terracota)
    residuals = (y_test_sample - y_pred_sample) / 1_000
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    ax1.scatter(y_test_sample / 1_000, y_pred_sample / 1_000, alpha=0.18, s=9, color=sage, edgecolors="none")
    lims = [0, 2000]
    ax1.plot(lims, lims, "--", color=terracotta, linewidth=1.8, label="Predicción perfecta (y = x)")
    ax1.set_xlim(lims)
    ax1.set_ylim([-100, 2000])
    ax1.set_xlabel("Precio Real (Miles de USD)", color=charcoal)
    ax1.set_ylabel("Precio Predicho (Miles de USD)", color=charcoal)
    ax1.set_title("Precio Real vs. Predicho (Regresión Lineal)", fontweight="bold", color=forest)
    style_ax(ax1)
    ax1.grid(alpha=0.25, linestyle="--")
    ax1.legend(frameon=False, fontsize=8.5)

    ax2.hist(residuals, bins=50, range=(-800, 800), color=sand, alpha=0.88, edgecolor="white")
    ax2.axvline(0, color=terracotta, linestyle="--", linewidth=1.6, label="Residuo = 0")
    ax2.set_xlabel("Residuo: Real − Predicho (Miles de USD)", color=charcoal)
    ax2.set_ylabel("Frecuencia", color=charcoal)
    ax2.set_title("Distribución de Residuos (Regresión Lineal)", fontweight="bold", color=forest)
    style_ax(ax2)
    ax2.grid(axis="y", alpha=0.25, linestyle="--")
    ax2.legend(frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "real_vs_predicho.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df_clean, data_summary = load_and_clean_data(DATA_RAW)

    # Estadísticas descriptivas
    num_cols = ["price", "bed", "bath", "acre_lot", "house_size"]
    desc = df_clean[num_cols].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc_dict = {
        col: {
            "count": int(row["count"]),
            "mean": round(float(row["mean"]), 2),
            "std": round(float(row["std"]), 2),
            "min": round(float(row["min"]), 2),
            "p25": round(float(row["25%"]), 2),
            "median": round(float(row["50%"]), 2),
            "p75": round(float(row["75%"]), 2),
            "max": round(float(row["max"]), 2),
        }
        for col, row in desc.iterrows()
    }

    # Matriz de correlación
    corr_df = df_clean[num_cols].corr()
    corr_dict = {
        row_name: {col_name: round(float(val), 4) for col_name, val in row.items()}
        for row_name, row in corr_df.iterrows()
    }

    # Estadísticas por estado
    state_stats = (
        df_clean.groupby("state")
        .agg(
            count=("price", "count"),
            mean_price=("price", "mean"),
            median_price=("price", "median"),
            median_house_size=("house_size", "median"),
        )
        .reset_index()
        .sort_values("state")
    )

    # División Train / Test (80% / 20%) sobre el conjunto limpio completo
    features_num = ["bed", "bath", "acre_lot", "house_size"]
    features_full = ["bed", "bath", "acre_lot", "house_size", "state"]
    X = df_clean[features_full]
    y = df_clean["price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Entrenamiento: {len(X_train):,} registros | Prueba: {len(X_test):,} registros")

    model_results: list[dict] = []

    # Modelo 1: Regresión Lineal (solo variables numéricas)
    print("Entrenando Modelo 1: Regresión Lineal (solo numéricas)...")
    lr_num = LinearRegression()
    lr_num.fit(X_train[features_num], y_train)
    pred_lr_num = lr_num.predict(X_test[features_num])
    m1 = evaluate_predictions(y_test, pred_lr_num)
    model_results.append({
        "name": "Regresión Lineal (Solo Numéricas)",
        "short_name": "Reg. Lineal\n(Numéricas)",
        "category": "Estadístico",
        **m1,
    })

    # Modelo 2 (Principal para el Simulador): Regresión Lineal (Numéricas + Estado)
    print("Entrenando Modelo 2: Regresión Lineal (Numéricas + Estado)...")
    pre_ohe_drop = ColumnTransformer(
        transformers=[
            ("num", "passthrough", features_num),
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), ["state"]),
        ]
    )
    lr_full = Pipeline([
        ("preprocessor", pre_ohe_drop),
        ("regressor", LinearRegression()),
    ])
    lr_full.fit(X_train, y_train)
    pred_lr_full = lr_full.predict(X_test)
    m2 = evaluate_predictions(y_test, pred_lr_full)
    model_results.append({
        "name": "Regresión Lineal (Numéricas + Estado)",
        "short_name": "Reg. Lineal\n(+ Estado)",
        "category": "Estadístico (Principal)",
        **m2,
    })

    # Extraer coeficientes de la Regresión Lineal Principal
    ohe = lr_full.named_steps["preprocessor"].named_transformers_["cat"]
    reg = lr_full.named_steps["regressor"]
    base_state = str(ohe.categories_[0][0])
    state_categories = [str(s) for s in ohe.categories_[0][1:]]

    num_coefs_full = {
        feat: round(float(coef), 4)
        for feat, coef in zip(features_num, reg.coef_[: len(features_num)])
    }
    state_coefs = {base_state: 0.0}
    for st, coef in zip(state_categories, reg.coef_[len(features_num) :]):
        state_coefs[st] = round(float(coef), 2)

    num_coefs_simple = {
        feat: round(float(coef), 4)
        for feat, coef in zip(features_num, lr_num.coef_)
    }

    # Guardar modelo principal de Regresión Lineal (scikit-learn)
    model_path = MODELS_DIR / "linear_regression_model.joblib"
    joblib.dump(lr_full, model_path, compress=3)
    print(f"Modelo guardado en {model_path} ({model_path.stat().st_size / 1024:.1f} KB)")

    # Guardar muestra ligera representativa para Streamlit (8,000 filas)
    sample_df = df_clean.sample(n=8_000, random_state=42).copy()
    sample_df["predicted_price"] = np.clip(lr_full.predict(sample_df[features_full]), 0, None).round(2)
    sample_path = DATA_DIR / "sample_realtor_clean.csv"
    sample_df.to_csv(sample_path, index=False)
    print(f"Muestra ligera guardada en {sample_path} ({sample_path.stat().st_size / 1024:.1f} KB)")

    # Generar figuras para el informe LaTeX
    rng = np.random.default_rng(42)
    idx_sample = rng.choice(len(y_test), size=min(15_000, len(y_test)), replace=False)
    generate_figures(
        df_clean=df_clean,
        corr_df=corr_df,
        state_stats=state_stats,
        model_results=model_results,
        y_test_sample=y_test[idx_sample],
        y_pred_sample=pred_lr_full[idx_sample],
    )

    # Guardar JSON con todas las métricas y estadísticas
    artifact = {
        "data_summary": data_summary,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "descriptive_stats": desc_dict,
        "correlation_matrix": corr_dict,
        "model_comparison": model_results,
        "linear_regression_simple": {
            "intercept": round(float(lr_num.intercept_), 2),
            "coefficients": num_coefs_simple,
        },
        "linear_regression_full": {
            "intercept": round(float(reg.intercept_), 2),
            "base_state": base_state,
            "numeric_coefficients": num_coefs_full,
            "state_coefficients": state_coefs,
        },
        "states": state_stats.to_dict(orient="records"),
    }
    json_path = MODELS_DIR / "metrics_and_eda.json"
    json_path.write_text(json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Métricas y EDA guardados en {json_path}")

    print("\n--- RESUMEN DE MODELOS ---")
    for m in model_results:
        print(f"{m['name']:42s} | R2={m['r2']:.4f} | RMSE=${m['rmse']:>10,.2f} | MAE=${m['mae']:>10,.2f}")


if __name__ == "__main__":
    main()
