"""Funciones de visualización con Matplotlib para la aplicación Streamlit."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def plot_price_breakdown(contributions: dict[str, float]) -> Figure:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    labels = list(contributions.keys())[::-1]
    vals = [contributions[k] / 1_000 for k in labels]
    bar_colors = ["#10b981" if v >= 0 else "#ef4444" for v in vals]

    ax.barh(labels, vals, color=bar_colors, alpha=0.88)
    ax.axvline(0, color="#334155", linewidth=1)
    ax.set_xlabel("Contribución al Precio (Miles de USD)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> Figure:
    labels_es = ["Precio", "Habitaciones", "Baños", "Terreno", "Superficie"]
    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=130)
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(len(labels_es)))
    ax.set_yticks(range(len(labels_es)))
    ax.set_xticklabels(labels_es, rotation=30, ha="right", fontsize=9.5)
    ax.set_yticklabels(labels_es, fontsize=9.5)
    ax.set_xticks(np.arange(-0.5, len(labels_es), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels_es), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(labels_es)):
        for j in range(len(labels_es)):
            val = corr_df.values[i, j]
            color = "white" if abs(val) > 0.55 else "#111827"
            ax.text(
                j,
                i,
                f"{val:.2f}",
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
                fontsize=9.5,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    fig.tight_layout()
    return fig


def plot_variable_distribution(
    values: pd.Series, label: str, is_discrete: bool = False
) -> Figure:
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    bins = (
        np.arange(values.min() - 0.5, values.max() + 1.5, 1)
        if is_discrete
        else 35
    )
    ax.hist(values, bins=bins, color="#2563eb", alpha=0.82, edgecolor="white")
    ax.axvline(
        values.median(),
        color="#dc2626",
        linestyle="--",
        label=f"Mediana: {values.median():,.2f}",
    )
    ax.set_xlabel(label)
    ax.set_ylabel("Frecuencia (muestra de 8,000)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_top_states(states_df: pd.DataFrame, top_n: int = 15) -> Figure:
    top_df = states_df.sort_values("median_price", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    ax.barh(top_df["state"], top_df["median_price"] / 1_000, color="#0ea5e9", alpha=0.9)
    ax.set_xlabel("Precio Mediano (Miles de USD)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    return fig


def plot_model_comparison(model_comparison: list[dict]) -> Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 5.4))
    short_names = [m["short_name"].replace("\n", " ") for m in model_comparison]
    r2_vals = [m["r2"] for m in model_comparison]
    rmse_vals = [m["rmse"] / 1_000 for m in model_comparison]
    colors = ["#94a3b8", "#2563eb", "#0ea5e9", "#f59e0b", "#10b981"]

    ax1.barh(short_names[::-1], r2_vals[::-1], color=colors[: len(r2_vals)][::-1])
    ax1.set_xlabel("R² (Mayor es mejor)")
    ax1.set_xlim(0, max(r2_vals) * 1.18)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(axis="x", alpha=0.25)

    ax2.barh(short_names[::-1], rmse_vals[::-1], color=colors[: len(rmse_vals)][::-1])
    ax2.set_xlabel("RMSE en Miles de USD (Menor es mejor)")
    ax2.set_xlim(0, max(rmse_vals) * 1.18)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(axis="x", alpha=0.25)

    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(sample_df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    ax.scatter(
        sample_df["price"] / 1_000,
        sample_df["predicted_price"] / 1_000,
        alpha=0.18,
        s=10,
        color="#2563eb",
        edgecolors="none",
    )
    ax.plot([0, 2000], [0, 2000], "--", color="#dc2626", linewidth=1.5, label="y = x (Ideal)")
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)
    ax.set_xlabel("Precio Real (Miles de USD)")
    ax.set_ylabel("Precio Predicho (Miles de USD)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig
