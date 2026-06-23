import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / 'data' / 'HTMDEC_Y2_db.xlsx'
OUT_PDF = ROOT / 'paper' / 'Figures' / 'Hdyn_Hqs_1_02.pdf'
OUT_PNG = ROOT / 'paper' / 'Figures' / 'Hdyn_Hqs_1_02.png'

ITERATIONS = ['BBA', 'BBB', 'BBC']
PANEL_TITLES = {
    'BBA': '(a) BBA',
    'BBB': '(b) BBB',
    'BBC': '(c) BBC',
}

PANEL_COLORS = {
    'BBA': ('#fdba74', '#c2410c'),
    'BBB': ('#93c5fd', '#1d4ed8'),
    'BBC': ('#86efac', '#047857'),
}
MISSING_COLOR = '#d1d5db'
TEXT_COLOR = '#111827'

cols = [
    'Iteration', 'Alloy Name', 'XRD Phase',
    'Avg Hdyn (GPa) HSR', 'SD, Avg Hdyn',
    'Avg Hqs (GPa) HSR', 'SD, Avg Hqs',
]

df = pd.read_excel(DATA_FILE, usecols=cols)
df = df[df['Iteration'].isin(ITERATIONS)].copy()
df['alloy_num'] = df['Alloy Name'].str[-2:].astype(int)
df = df.sort_values(['Iteration', 'alloy_num'])

# Narrower source figure so it appears larger when scaled to 2-column width.
fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), sharey=True)
bar_width = 0.38
max_y = 0

for ax, iteration in zip(axes, ITERATIONS):
    sub = df[df['Iteration'] == iteration].copy()
    if iteration == 'BBA':
        sub = sub[sub['alloy_num'] <= 16].copy()
    x = np.arange(len(sub))

    dyn = sub['Avg Hdyn (GPa) HSR'].to_numpy(dtype=float)
    dyn_sd = sub['SD, Avg Hdyn'].to_numpy(dtype=float)
    qs = sub['Avg Hqs (GPa) HSR'].to_numpy(dtype=float)
    qs_sd = sub['SD, Avg Hqs'].to_numpy(dtype=float)
    sigma_mask = sub['XRD Phase'].fillna('').astype(str).str.contains('σ|sigma', case=False, regex=True).to_numpy()

    dyn_mask = ~np.isnan(dyn)
    qs_mask = ~np.isnan(qs)

    ax.bar(
        x[dyn_mask] - bar_width / 2,
        dyn[dyn_mask],
        bar_width,
        yerr=dyn_sd[dyn_mask],
        color=PANEL_COLORS[iteration][0],
        edgecolor='black',
        linewidth=0.8,
        capsize=2,
        label='Dynamic Hardness',
        zorder=3,
    )
    ax.bar(
        x[qs_mask] + bar_width / 2,
        qs[qs_mask],
        bar_width,
        yerr=qs_sd[qs_mask],
        color=PANEL_COLORS[iteration][1],
        edgecolor='black',
        linewidth=0.8,
        capsize=2,
        label='Quasi-Static Hardness',
        zorder=3,
    )

    missing_mask = np.isnan(dyn) & np.isnan(qs)
    if missing_mask.any():
        ax.scatter(
            x[missing_mask],
            np.full(missing_mask.sum(), 0.08),
            marker='x',
            s=42,
            color=MISSING_COLOR,
            linewidths=1.8,
            zorder=5,
        )

    for xi, is_sigma, dval, qval in zip(x, sigma_mask, dyn, qs):
        if not is_sigma:
            continue
        ymax = np.nanmax([dval, qval]) if not (np.isnan(dval) and np.isnan(qval)) else 0
        ax.text(xi, ymax + 0.20, r'$\sigma$', ha='center', va='bottom', fontsize=13, color='black', zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels(sub['Alloy Name'], rotation=90, fontsize=15)
    ax.text(0.5, -0.34, PANEL_TITLES[iteration], transform=ax.transAxes, ha='center', va='top', fontsize=19, color=TEXT_COLOR)
    ax.set_xlim(-0.65, len(sub) - 0.35)
    ax.grid(axis='y', color='#e5e7eb', linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#9ca3af')
    ax.spines['bottom'].set_color('#9ca3af')
    ax.tick_params(axis='y', labelsize=15, colors=TEXT_COLOR)
    ax.tick_params(axis='x', colors=TEXT_COLOR)
    ax.legend(loc=('upper left' if iteration == 'BBB' else 'upper right'), frameon=False, fontsize=13)

    local_max = np.nanmax(np.concatenate([dyn[~np.isnan(dyn)], qs[~np.isnan(qs)]]))
    max_y = max(max_y, local_max)

fig.suptitle('Evolution of Dynamic and Quasi-Static Hardness Across Successive Alloy Design Iterations', fontsize=22, color=TEXT_COLOR, y=0.975)
axes[0].set_ylabel('Hardness (GPa)', fontsize=18, color=TEXT_COLOR)
for ax in axes:
    ax.set_ylim(0, max_y + 0.60)

legend_handles = [
    Line2D([0], [0], marker=r'$\sigma$', color='black', linestyle='None', markersize=12, label='XRD-confirmed sigma phase presence'),
    Line2D([0], [0], marker='x', color=MISSING_COLOR, linestyle='None', markersize=8, markeredgewidth=1.8, label='Processing failure'),
]
fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.935), fontsize=14, columnspacing=1.5, handletextpad=0.6)

# Explicit spacing control removes the panel gaps more aggressively than tight_layout.
fig.subplots_adjust(left=0.055, right=0.995, bottom=0.29, top=0.80, wspace=0.005)
fig.savefig(OUT_PDF, bbox_inches='tight', pad_inches=0.02)
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', pad_inches=0.02)
print(f'wrote {OUT_PDF}')
print(f'wrote {OUT_PNG}')
