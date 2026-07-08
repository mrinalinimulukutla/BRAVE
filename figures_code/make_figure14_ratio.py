import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / 'data' / 'HTMDEC_Y2_db.xlsx'
OUT_PDF = ROOT / 'paper' / 'fig_13_dyn_qs_ratio.pdf'
OUT_PNG = ROOT / 'paper' / 'fig_13_dyn_qs_ratio.png'

ITERATIONS = ['BBA', 'BBB', 'BBC']
PANEL_TITLES = {
    'BBA': '(a) BBA',
    'BBB': '(b) BBB',
    'BBC': '(c) BBC',
}
PANEL_COLORS = {
    'BBA': '#fdba74',
    'BBB': '#93c5fd',
    'BBC': '#86efac',
}
MISSING_COLOR = '#d1d5db'
TEXT_COLOR = '#111827'
GRID_COLOR = '#e5e7eb'

cols = ['Iteration', 'Alloy Name', 'XRD Phase', 'Avg HDYN/HQS', 'Std, Hdyn/Hqs']
df = pd.read_excel(DATA_FILE, usecols=lambda c: c in cols)
df = df[df['Iteration'].isin(ITERATIONS)].copy()
df['alloy_num'] = df['Alloy Name'].str[-2:].astype(int)
df = df.sort_values(['Iteration', 'alloy_num'])

fig, axes = plt.subplots(1, 3, figsize=(16, 6.0), sharey=True)
max_y = 0

for ax, iteration in zip(axes, ITERATIONS):
    sub = df[df['Iteration'] == iteration].copy()
    if iteration == 'BBA':
        sub = sub[sub['alloy_num'] <= 16].copy()
    x = np.arange(len(sub))

    ratio = sub['Avg HDYN/HQS'].to_numpy(dtype=float)
    ratio_sd = sub['Std, Hdyn/Hqs'].to_numpy(dtype=float)
    sigma_mask = sub['XRD Phase'].fillna('').astype(str).str.contains('σ|sigma', case=False, regex=True).to_numpy()
    valid = ~np.isnan(ratio)

    ax.bar(
        x[valid],
        ratio[valid],
        0.62,
        yerr=ratio_sd[valid],
        color=PANEL_COLORS[iteration],
        edgecolor='black',
        linewidth=0.8,
        capsize=2,
        zorder=3,
    )

    missing_mask = np.isnan(ratio)
    if missing_mask.any():
        ax.scatter(
            x[missing_mask],
            np.full(missing_mask.sum(), 1.005),
            marker='x',
            s=42,
            color=MISSING_COLOR,
            linewidths=1.8,
            zorder=5,
        )

    for xi, is_sigma, val, err in zip(x, sigma_mask, ratio, ratio_sd):
        if not is_sigma:
            continue
        if np.isnan(val):
            y = 1.01
        else:
            y = val + (0.0 if np.isnan(err) else err)
        ax.text(xi, y + 0.012, r'$\sigma$', ha='center', va='bottom', fontsize=13, color='black', zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels(sub['Alloy Name'], rotation=90, fontsize=15)
    ax.text(0.5, -0.34, PANEL_TITLES[iteration], transform=ax.transAxes, ha='center', va='top', fontsize=19, color=TEXT_COLOR)
    ax.set_xlim(-0.65, len(sub) - 0.35)
    ax.grid(axis='y', color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#9ca3af')
    ax.spines['bottom'].set_color('#9ca3af')
    ax.tick_params(axis='y', labelsize=15, colors=TEXT_COLOR)
    ax.tick_params(axis='x', colors=TEXT_COLOR)

    local_max = np.nanmax(ratio) if np.any(valid) else 1.0
    max_y = max(max_y, local_max)

fig.suptitle('Evolution of Dynamic-to-Quasi-Static Hardness Ratio Across Successive Alloy Design Iterations', fontsize=22, color=TEXT_COLOR, y=0.975)
axes[0].set_ylabel(r'$H_{dyn}/H_{qs}$', fontsize=18, color=TEXT_COLOR)
for ax in axes:
    ax.set_ylim(1.0, max_y + 0.12)

legend_handles = [
    Line2D([0], [0], marker=r'$\sigma$', color='black', linestyle='None', markersize=12, label='XRD-confirmed sigma phase presence'),
    Line2D([0], [0], marker='x', color=MISSING_COLOR, linestyle='None', markersize=8, markeredgewidth=1.8, label='Processing failure'),
]
fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.935), fontsize=14, columnspacing=1.5, handletextpad=0.6)
fig.subplots_adjust(left=0.06, right=0.995, bottom=0.29, top=0.80, wspace=0.005)
fig.savefig(OUT_PDF, bbox_inches='tight', pad_inches=0.02)
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', pad_inches=0.02)
print(f'wrote {OUT_PDF}')
print(f'wrote {OUT_PNG}')
