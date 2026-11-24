"""Funciones de visualización con Matplotlib con paleta editorial (bosque y terracota)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

FOREST = "#1B4332"
SAGE = "#40916C"
MINT = "#74C69D"
TERRACOTTA = "#C85A32"
SAND = "#D4A373"
CHARCOAL = "#2D3A3A"
BG_CARD = "#FFFFFF"


def _style_axis(ax) -> None:
    ax.set_facecolor(BG_CARD)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CFC8B8")
    ax.spines["bottom"].set_color("#CFC8B8")
    ax.tick_params(colors=CHARCOAL, labelsize=9)


def plot_price_breakdown(contributions: dict[str, float]) -> Figure:
    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=BG_CARD)
    labels = list(contributions.keys())[::-1]
    vals = [contributions[k] / 1_000 for k in labels]
    bar_colors = [SAGE if v >= 0 else TERRACOTTA for v in vals]

    bars = ax.barh(labels, vals, color=bar_colors, height=0.62, alpha=0.92)
    ax.axvline(0, color=CHARCOAL, linewidth=1.1)
    for bar, v in zip(bars, vals):
        offset = 4 if v >= 0 else -4
        ha = "left" if v >= 0 else "right"
        ax.text(
            v + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{v:+,.1f}k",
            va="center",
            ha=ha,
            fontsize=8.5,
            fontweight="bold",
            color=CHARCOAL,
        )
    ax.set_xlabel("Impacto en el Precio (Miles de USD)", color=CHARCOAL, fontsize=9.5)
    _style_axis(ax)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    ax.margins(x=0.18)
    fig.tight_layout()
    return fig


def plot_state_price_position(
    sample_df: pd.DataFrame, state: str, predicted_price: float, state_median: float
) -> Figure:
    state_prices = sample_df.loc[sample_df["state"] == state, "price"] / 1_000
    if len(state_prices) < 15:
        state_prices = sample_df["price"] / 1_000
        subtitle = "Referencia nacional (muestra estatal pequeña)"
    else:
        subtitle = f"Distribución muestral en {state}"

    fig, ax = plt.subplots(figsize=(6.4, 3.6), facecolor=BG_CARD)
    ax.hist(state_prices, bins=28, color=SAND, alpha=0.65, edgecolor=BG_CARD)
    ax.axvline(
        state_median / 1_000,
        color=CHARCOAL,
        linestyle="--",
        linewidth=1.5,
        label=f"Mediana estatal: ${state_median / 1_000:,.0f}k",
    )
    ax.axvline(
        predicted_price / 1_000,
        color=FOREST,
        linewidth=2.4,
        label=f"Tu simulación: ${predicted_price / 1_000:,.0f}k",
    )
    ax.set_title(subtitle, fontsize=10, fontweight="bold", color=FOREST, loc="left")
    ax.set_xlabel("Precio (Miles de USD)", color=CHARCOAL, fontsize=9.5)
    ax.set_ylabel("Propiedades", color=CHARCOAL, fontsize=9.5)
    _style_axis(ax)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(corr_df: pd.DataFrame) -> Figure:
    labels_es = ["Precio", "Habitaciones", "Baños", "Terreno", "Superficie"]
    fig, ax = plt.subplots(figsize=(6.6, 5.4), dpi=130, facecolor=BG_CARD)
    im = ax.imshow(corr_df.values, cmap="BrBG", vmin=-1, vmax=1, aspect="equal")

    ax.set_xticks(range(len(labels_es)))
    ax.set_yticks(range(len(labels_es)))
    ax.set_xticklabels(labels_es, rotation=30, ha="right", fontsize=9.5, color=CHARCOAL)
    ax.set_yticklabels(labels_es, fontsize=9.5, color=CHARCOAL)
    ax.set_xticks(np.arange(-0.5, len(labels_es), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels_es), 1), minor=True)
    ax.grid(which="minor", color=BG_CARD, linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(len(labels_es)):
        for j in range(len(labels_es)):
            val = corr_df.values[i, j]
            color = "white" if abs(val) > 0.55 else "#1E2421"
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
    cbar.ax.tick_params(labelsize=9, colors=CHARCOAL)
    fig.tight_layout()
    return fig


def plot_variable_distribution(
    values: pd.Series, label: str, is_discrete: bool = False
) -> Figure:
    fig, ax = plt.subplots(figsize=(5.8, 4.1), facecolor=BG_CARD)
    bins = (
        np.arange(values.min() - 0.5, values.max() + 1.5, 1)
        if is_discrete
        else 32
    )
    ax.hist(values, bins=bins, color=SAGE, alpha=0.85, edgecolor=BG_CARD)
    ax.axvline(
        values.median(),
        color=TERRACOTTA,
        linestyle="--",
        linewidth=1.8,
        label=f"Mediana: {values.median():,.2f}",
    )
    ax.set_xlabel(label, color=CHARCOAL, fontsize=9.5)
    ax.set_ylabel("Frecuencia (muestra de 8,000)", color=CHARCOAL, fontsize=9.5)
    _style_axis(ax)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def plot_top_states(states_df: pd.DataFrame, top_n: int = 12) -> Figure:
    top_df = states_df.sort_values("median_price", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.5, 4.4), facecolor=BG_CARD)
    y_pos = np.arange(len(top_df))
    vals = top_df["median_price"] / 1_000

    # Gráfico tipo lollipop para diferenciarse de barras genéricas
    ax.hlines(y=y_pos, xmin=0, xmax=vals, color=SAGE, linewidth=2.6, alpha=0.85)
    ax.scatter(vals, y_pos, color=FOREST, s=75, zorder=3)
    for y, v in zip(y_pos, vals):
        ax.text(v + 14, y, f"${v:,.0f}k", va="center", fontsize=8.5, fontweight="bold", color=CHARCOAL)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df["state"])
    ax.set_xlabel("Precio Mediano (Miles de USD)", color=CHARCOAL, fontsize=9.5)
    ax.set_xlim(0, vals.max() * 1.16)
    _style_axis(ax)
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    return fig


def plot_model_comparison(model_comparison: list[dict]) -> Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.0), facecolor=BG_CARD)
    short_names = [m["short_name"].replace("\n", " ") for m in model_comparison]
    r2_vals = [m["r2"] for m in model_comparison]
    rmse_vals = [m["rmse"] / 1_000 for m in model_comparison]
    colors = ["#95A5A6", FOREST, SAGE, SAND, TERRACOTTA]

    y = np.arange(len(short_names))
    b1 = ax1.barh(y, r2_vals[::-1], color=colors[: len(r2_vals)][::-1], height=0.58)
    ax1.set_yticks(y)
    ax1.set_yticklabels(short_names[::-1])
    ax1.set_xlabel("Coeficiente R² (Mayor es mejor)", color=CHARCOAL)
    ax1.set_xlim(0, max(r2_vals) * 1.2)
    for bar, v in zip(b1, r2_vals[::-1]):
        ax1.text(v + 0.012, bar.get_y() + bar.get_height() / 2, f"{v:.4f}", va="center", fontsize=8.5, fontweight="bold", color=CHARCOAL)
    _style_axis(ax1)
    ax1.grid(axis="x", alpha=0.25, linestyle="--")

    b2 = ax2.barh(y, rmse_vals[::-1], color=colors[: len(rmse_vals)][::-1], height=0.58)
    ax2.set_yticks(y)
    ax2.set_yticklabels([""] * len(short_names))
    ax2.set_xlabel("RMSE en Miles de USD (Menor es mejor)", color=CHARCOAL)
    ax2.set_xlim(0, max(rmse_vals) * 1.24)
    for bar, v in zip(b2, rmse_vals[::-1]):
        ax2.text(v + 4, bar.get_y() + bar.get_height() / 2, f"${v:,.1f}k", va="center", fontsize=8.5, fontweight="bold", color=CHARCOAL)
    _style_axis(ax2)
    ax2.grid(axis="x", alpha=0.25, linestyle="--")

    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(sample_df: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(6.4, 4.4), facecolor=BG_CARD)
    ax.scatter(
        sample_df["price"] / 1_000,
        sample_df["predicted_price"] / 1_000,
        alpha=0.20,
        s=12,
        color=SAGE,
        edgecolors="none",
    )
    ax.plot([0, 2000], [0, 2000], "--", color=TERRACOTTA, linewidth=1.8, label="Referencia y = x")
    ax.set_xlim(0, 2000)
    ax.set_ylim(0, 2000)
    ax.set_xlabel("Precio Real (Miles de USD)", color=CHARCOAL)
    ax.set_ylabel("Precio Predicho (Miles de USD)", color=CHARCOAL)
    _style_axis(ax)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig
