import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def mi_matrix_per_horizon(mi_df: pd.DataFrame, top_n: int = 20, *, source: str = ""):
   
    _cmap = LinearSegmentedColormap.from_list(
        "mi_vibrant", ["#0a0a1a", "#2e1a6e", "#7b2d8b", "#d63a6e", "#f4845f"], N=256
    )

    horizons = list(mi_df.columns)
    n_h = len(horizons)
    mi_norm = 1 - np.exp(-mi_df)  # normalise to [0,1)

    # For each horizon, pick top-N features by normalised MI (descending).
    values = np.zeros((n_h, top_n), dtype=float)
    labels = np.empty((n_h, top_n), dtype=object)
    for i, h in enumerate(horizons):
        s = mi_norm[h].dropna().sort_values(ascending=False).head(top_n)
        values[i, :len(s)] = s.values
        labels[i, :len(s)] = [f"{name}\n{val:.2f}" for name, val in s.items()]

    rank_cols = [f"#{r+1}" for r in range(top_n)]
    horizon_labels = [f"h{int(h[1:])}  ({int(h[1:]) * OUTPUT_RESOLUTION}min)" for h in horizons]

    data = pd.DataFrame(values, index=horizon_labels, columns=rank_cols)
    annot = pd.DataFrame(labels, index=horizon_labels, columns=rank_cols)

    cell_w, cell_h = 1.6, 0.42
    fig_w = max(12, top_n * cell_w + 3)
    fig_h = max(8, n_h * cell_h + 2)

    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor("#1e1e1e")
        ax.set_facecolor("#1e1e1e")

        sns.heatmap(
            data, ax=ax, cmap=_cmap, vmin=0, vmax=1,
            annot=annot.values, fmt="",
            annot_kws={"size": 9, "color": "white"},
            linewidths=0.2, linecolor="#2e2e2e",
            cbar_kws={"label": "MI  (1 − e⁻ᴹᴵ)", "shrink": 0.6, "pad": 0.01},
        )
        ax.xaxis.set_ticks_position("top")
        ax.xaxis.set_label_position("top")
        ax.tick_params(axis="x", labelsize=10, colors="white", rotation=0)
        ax.tick_params(axis="y", labelsize=10, colors="white", rotation=0)
        ax.set_xlabel("Feature rank", color="white", fontsize=10, labelpad=6)
        ax.set_ylabel("Forecast horizon", color="white", fontsize=10)

        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(colors="white")
        cbar.ax.yaxis.label.set_color("white")

        suffix = f" — {source}" if source else ""
        fig.suptitle(
            f"Top {top_n} features by MI — per horizon ({n_h} horizons × {OUTPUT_RESOLUTION}min){suffix}",
            color="white", fontsize=14, y=0.995,
        )
        plt.tight_layout()
        plt.show()


