"""
Two-panel figure showing V's dual role in Campaign 2:
(top)    V vs YS scatter with regression + 95% CI on the feasible set
(bottom) Infeasibility rate per V bin over the full 48-alloy Campaign 2

Vertical band at V = 24 at.% marks where the three strongest alloys and
the three sigma failures share the same compositional point.

Data: HTMDEC_Y2_db.xlsx (Year 2, Iterations BBA/BBB/BBC).
Feasibility: XRD Phase == 'FCC' (single-phase FCC) is feasible; all else
(FCC+sigma or no XRD due to processing failure) is infeasible.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_ROOT / "data" / "HTMDEC_Y2_db.xlsx"
OUT_PDF = PROJECT_ROOT / "paper" / "fig_19_v_ys_feasibility.pdf"
OUT_PNG = PROJECT_ROOT / "paper" / "fig_19_v_ys_feasibility.png"

CAMPAIGN2_ITERS = {"BBA", "BBB", "BBC"}
# Excluded from the 48-alloy Campaign 2 set (paper IGSN range TMAMAK00097-00144).
# BBA17-19 (IGSN 195-197) were later remakes that don't appear in the paper's
# per-iteration analyses.
EXCLUDE_ALLOYS = {"BBA17", "BBA18", "BBA19"}

V_BINS = [-2, 2, 6, 10, 14, 18, 22, 26]
V_BIN_CENTERS = [0, 4, 8, 12, 16, 20, 24]

HIGHLIGHT_V = 24.0
HIGHLIGHT_HALFWIDTH = 1.5

FEAS_COLOR = "#0f766e"
INFEAS_COLOR = "#7f1d1d"
FIT_COLOR = "#0f766e"
CI_ALPHA = 0.18
HIGHLIGHT_COLOR = "#facc15"
HIGHLIGHT_ALPHA = 0.18
GRID_COLOR = "#e5e7eb"
TEXT_COLOR = "#111827"


def compute_regression(x: np.ndarray, y: np.ndarray):
    result = stats.linregress(x, y)
    xline = np.linspace(x.min(), x.max(), 100)
    yline = result.intercept + result.slope * xline
    n = len(x)
    x_mean = x.mean()
    sxx = np.sum((x - x_mean) ** 2)
    residuals = y - (result.intercept + result.slope * x)
    s_err = np.sqrt(np.sum(residuals**2) / (n - 2))
    tval = stats.t.ppf(0.975, n - 2)
    se_line = s_err * np.sqrt(1.0 / n + (xline - x_mean) ** 2 / sxx)
    ci = tval * se_line
    return result, xline, yline, ci


def infeasibility_by_bin(v_values: np.ndarray, feasible_mask: np.ndarray):
    counts = np.zeros(len(V_BIN_CENTERS), dtype=int)
    fails = np.zeros(len(V_BIN_CENTERS), dtype=int)
    for i, (lo, hi) in enumerate(zip(V_BINS[:-1], V_BINS[1:])):
        in_bin = (v_values > lo) & (v_values <= hi)
        counts[i] = in_bin.sum()
        fails[i] = (in_bin & ~feasible_mask).sum()
    rate = np.where(counts > 0, fails / counts, np.nan)
    return counts, fails, rate


def main():
    df = pd.read_excel(DATA_FILE)
    df = df[df["Year"] == 2].copy()
    df = df[df["Iteration"].isin(CAMPAIGN2_ITERS)].copy()
    df = df[~df["Alloy Name"].isin(EXCLUDE_ALLOYS)].copy()

    v = df["V"].astype(float).to_numpy()
    ys = df["Yield Strength (MPa)"].to_numpy(dtype=float)
    phase = df["XRD Phase"].astype(str).str.strip().str.upper()
    feasible = (phase == "FCC").to_numpy()

    n_total = len(df)
    n_feasible = feasible.sum()
    n_infeasible = (~feasible).sum()

    print(f"Campaign 2 total: {n_total}")
    print(f"Feasible (XRD Phase = FCC): {n_feasible}")
    print(f"Infeasible (sigma / processing failure): {n_infeasible}")

    ys_meas = ~np.isnan(ys)
    x_reg = v[feasible & ys_meas]
    y_reg = ys[feasible & ys_meas]
    print(f"Feasible with measurable YS: n = {len(x_reg)}")

    x_inf_meas = v[~feasible & ys_meas]
    y_inf_meas = ys[~feasible & ys_meas]
    print(f"Infeasible with measurable YS: n = {len(x_inf_meas)}")

    result, xline, yline, ci = compute_regression(x_reg, y_reg)
    print(f"Regression on feasible set: r = {result.rvalue:.3f}, "
          f"P = {result.pvalue:.2e}, slope = {result.slope:.1f} MPa/at%")

    r_full, p_full = stats.pearsonr(v[ys_meas], ys[ys_meas])
    print(f"Regression on all with measurable YS: r = {r_full:.3f}, "
          f"P = {p_full:.2e}")

    r_infeas, p_infeas = stats.pearsonr(v, (~feasible).astype(float))
    print(f"V vs infeasibility (all {n_total}): r = {r_infeas:.3f}, "
          f"P = {p_infeas:.2e}")

    counts, fails, rate = infeasibility_by_bin(v, feasible)
    print("Bin V (at.%) | count | fails | rate")
    for c, ct, ft, rt in zip(V_BIN_CENTERS, counts, fails, rate):
        print(f"  {c:>4d}       | {ct:>5d} | {ft:>5d} | {rt:.2f}")

    # Sized for a single elsarticle column (~3.35 in wide). Vertical layout
    # keeps the y-axis text and per-bar counts legible at final print size.
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(3.5, 4.6),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0], "hspace": 0.08},
    )
    plt.rcParams.update({"font.size": 8})

    for ax in (ax_top, ax_bot):
        ax.axvspan(
            HIGHLIGHT_V - HIGHLIGHT_HALFWIDTH,
            HIGHLIGHT_V + HIGHLIGHT_HALFWIDTH,
            color=HIGHLIGHT_COLOR, alpha=HIGHLIGHT_ALPHA, zorder=0,
        )

    ax_top.scatter(
        x_reg, y_reg,
        s=56, c=FEAS_COLOR, edgecolors="white", linewidths=0.6,
        label=f"Feasible (n={len(x_reg)})", zorder=3,
    )
    if len(x_inf_meas):
        ax_top.scatter(
            x_inf_meas, y_inf_meas,
            s=64, marker="X", c=INFEAS_COLOR, edgecolors="white", linewidths=0.6,
            label=f"Infeasible w/ YS (n={len(x_inf_meas)})", zorder=3,
        )
    ax_top.plot(xline, yline, color=FIT_COLOR, lw=1.6, zorder=2)
    ax_top.fill_between(xline, yline - ci, yline + ci, color=FIT_COLOR, alpha=CI_ALPHA, zorder=1)

    r_annot = (
        f"Feasible set:\n"
        f"  r = {result.rvalue:.2f}\n"
        f"  P = {result.pvalue:.1e}\n"
        f"  n = {len(x_reg)}"
    )
    ax_top.text(
        0.03, 0.97, r_annot,
        transform=ax_top.transAxes, va="top", ha="left",
        fontsize=9, color=TEXT_COLOR,
        bbox=dict(facecolor="white", edgecolor=GRID_COLOR, boxstyle="round,pad=0.3"),
    )

    ax_top.set_ylabel("Yield strength (MPa)", color=TEXT_COLOR)
    ax_top.grid(True, color=GRID_COLOR, lw=0.5, alpha=0.9)
    ax_top.set_axisbelow(True)
    ax_top.legend(loc="lower right", fontsize=8, frameon=False)

    ax_bot.bar(
        V_BIN_CENTERS, rate,
        width=3.4, color=INFEAS_COLOR, edgecolor="white", linewidth=0.6,
        zorder=3,
    )
    for cx, ct, ft, rt in zip(V_BIN_CENTERS, counts, fails, rate):
        if ct == 0:
            continue
        ax_bot.text(
            cx, rt + 0.04,
            f"{ft}/{ct}",
            ha="center", va="bottom",
            fontsize=8, color=TEXT_COLOR,
        )

    ax_bot.text(
        0.03, 0.95,
        f"All {n_total} alloys:\n  r(V,infeas) = {r_infeas:.2f}\n  P = {p_infeas:.1e}",
        transform=ax_bot.transAxes, va="top", ha="left",
        fontsize=9, color=TEXT_COLOR,
        bbox=dict(facecolor="white", edgecolor=GRID_COLOR, boxstyle="round,pad=0.3"),
    )

    ax_bot.set_ylim(0, 1.15)
    ax_bot.set_ylabel("Infeasibility rate", color=TEXT_COLOR)
    ax_bot.set_xlabel("V content (at.%)", color=TEXT_COLOR)
    ax_bot.grid(True, axis="y", color=GRID_COLOR, lw=0.5, alpha=0.9)
    ax_bot.set_axisbelow(True)

    ax_top.set_xlim(-2, 26)
    ax_bot.set_xticks(V_BIN_CENTERS)

    ax_top.annotate(
        "V = 24 at.%",
        xy=(HIGHLIGHT_V, 1.02),
        xycoords=("data", "axes fraction"),
        ha="center", va="bottom",
        fontsize=9, color=TEXT_COLOR,
    )

    fig.tight_layout()

    OUT_PDF.parent.mkdir(exist_ok=True)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    print(f"\nSaved: {OUT_PDF}")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
