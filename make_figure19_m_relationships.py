import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

DATA_FILE = Path("HTMDEC_Y2_db (1).xlsx")
OUT_PDF = Path("Figures/M_plots.pdf")
OUT_PNG = Path("Figures/M_plots.png")

ITERATIONS = ["BBA", "BBB", "BBC"]
ITER_COLORS = {
    "BBA": "#c2410c",
    "BBB": "#1d4ed8",
    "BBC": "#047857",
}
SIGMA_COLORS = {
    "BBA": "#c27f57",
    "BBB": "#6f99da",
    "BBC": "#68a58a",
}
TEXT_COLOR = "#111827"
GRID_COLOR = "#e5e7eb"

PANELS = [
    {
        "x": "Rate Sensitivity Exponent (M)",
        "y": "Depth of Penetration (mm) FE_Sim",
        "xlabel": "Rate Sensitivity Exponent (M)",
        "ylabel": "DoP (mm)",
        "title": "(a) Positive M-DoP Correlation",
    },
    {
        "x": "Rate Sensitivity Exponent (M)",
        "y": "Yield Strength (MPa)",
        "xlabel": "Rate Sensitivity Exponent (M)",
        "ylabel": "YS (MPa)",
        "title": "(b) Negative M-YS Correlation",
    },
    {
        "x": "Rate Sensitivity Exponent (M)",
        "y": "UTS_True (Mpa)",
        "xlabel": "Rate Sensitivity Exponent (M)",
        "ylabel": "UTS (MPa)",
        "title": "(c) Weak Negative M-UTS Correlation",
    },
]


def main():
    cols = ["Iteration", "Alloy Name", "XRD Phase"] + sorted({p["x"] for p in PANELS} | {p["y"] for p in PANELS})
    df = pd.read_excel(DATA_FILE, usecols=lambda c: c in cols)
    df = df[df["Iteration"].isin(ITERATIONS)].copy()
    df["alloy_num"] = df["Alloy Name"].str[-2:].astype(int)
    df = df[~((df["Iteration"] == "BBA") & (df["alloy_num"] > 16))].copy()
    df["sigma"] = df["XRD Phase"].fillna("").astype(str).str.contains(r"σ|sigma", case=False, regex=True)

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.9))

    for ax, panel in zip(axes, PANELS):
        panel_df = df[["Iteration", "sigma", panel["x"], panel["y"]]].dropna().copy()

        for it in ITERATIONS:
            sub = panel_df[panel_df["Iteration"] == it]
            non_sigma = sub[~sub["sigma"]]
            sigma = sub[sub["sigma"]]

            ax.scatter(
                non_sigma[panel["x"]],
                non_sigma[panel["y"]],
                s=64,
                color=ITER_COLORS[it],
                edgecolors='none',
                alpha=1.0,
                zorder=3,
            )
            if not sigma.empty:
                ax.scatter(
                    sigma[panel["x"]],
                    sigma[panel["y"]],
                    s=64,
                    color=SIGMA_COLORS[it],
                    edgecolors='none',
                    alpha=0.95,
                    zorder=4,
                )

        ax.set_title(panel["title"], fontsize=19, color=TEXT_COLOR, pad=8)
        ax.set_xlabel(panel["xlabel"], fontsize=18, color=TEXT_COLOR)
        ax.set_ylabel(panel["ylabel"], fontsize=18, color=TEXT_COLOR)
        ax.tick_params(axis='both', labelsize=15, colors=TEXT_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#9ca3af')
        ax.spines['bottom'].set_color('#9ca3af')

    legend_handles = [
        Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=ITER_COLORS['BBA'], markeredgecolor='none', markersize=8, label='BBA'),
        Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=ITER_COLORS['BBB'], markeredgecolor='none', markersize=8, label='BBB'),
        Line2D([0], [0], marker='o', linestyle='None', markerfacecolor=ITER_COLORS['BBC'], markeredgecolor='none', markersize=8, label='BBC'),
        Line2D([0], [0], marker='o', linestyle='None', markerfacecolor='#9ca3af', markeredgecolor='none', markersize=8, label='XRD sigma phase presence'),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.96), fontsize=14)
    fig.suptitle('Rate Sensitivity Relationships Across Successive Alloy Design Iterations', fontsize=22, color=TEXT_COLOR, y=0.995)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.79, wspace=0.22)
    fig.savefig(OUT_PDF, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {OUT_PDF}")
    print(f"wrote {OUT_PNG}")


if __name__ == '__main__':
    main()
