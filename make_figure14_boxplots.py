import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

DATA_FILE = Path('HTMDEC_Y2_db.xlsx')
OUT_PDF = Path('Figures/box_plot.pdf')
OUT_PNG = Path('Figures/box_plot.png')

ITERATIONS = ['BBA', 'BBB', 'BBC']
ITER_COLORS = {
    'BBA': '#fdba74',
    'BBB': '#93c5fd',
    'BBC': '#86efac',
}
POINT_COLORS = {
    'BBA': '#c2410c',
    'BBB': '#1d4ed8',
    'BBC': '#047857',
}
TEXT_COLOR = '#111827'
GRID_COLOR = '#e5e7eb'

PROPS = [
    ('Yield Strength (MPa)', 'YS (MPa)'),
    ('UTS/YS', 'UTS/YS'),
    ('Elong_T (%)', 'Strain at UTS (%)'),
    ('Avg Hdyn (GPa) HSR', r'$H_{\mathrm{dyn}}$ (GPa)'),
    ('Avg Hqs (GPa) HSR', r'$H_{\mathrm{qs}}$ (GPa)'),
    ('Avg HDYN/HQS', r'$H_{\mathrm{dyn}}/H_{\mathrm{qs}}$'),
    ('Depth of Penetration (mm) FE_Sim', 'DoP (mm)'),
]

cols = ['Iteration', 'Alloy Name'] + [c for c, _ in PROPS]
df = pd.read_excel(DATA_FILE, usecols=lambda c: c in cols)
df = df[df['Iteration'].isin(ITERATIONS)].copy()
df['alloy_num'] = df['Alloy Name'].str[-2:].astype(int)
df = df[~((df['Iteration'] == 'BBA') & (df['alloy_num'] > 16))].copy()

fig, axes = plt.subplots(2, 4, figsize=(16, 8.2))
axes = axes.flatten()

for ax, (col, label) in zip(axes, PROPS):
    data = [df.loc[df['Iteration'] == it, col].dropna().to_numpy(dtype=float) for it in ITERATIONS]
    positions = np.arange(1, 4)

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color='black', linewidth=1.6),
        boxprops=dict(linewidth=1.2, color='black'),
        whiskerprops=dict(linewidth=1.2, color='black'),
        capprops=dict(linewidth=1.2, color='black'),
        flierprops=dict(marker='o', markersize=3.5, markerfacecolor='white', markeredgecolor='black', alpha=0.9),
    )

    for patch, it in zip(bp['boxes'], ITERATIONS):
        patch.set_facecolor(ITER_COLORS[it])
        patch.set_alpha(0.82)

    # light jittered points for readability
    for pos, it, vals in zip(positions, ITERATIONS, data):
        if len(vals) == 0:
            continue
        jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.array([0.0])
        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=12,
            color=POINT_COLORS[it],
            edgecolors='none',
            alpha=0.75,
            zorder=3,
        )

    ax.set_title(label, fontsize=19, color=TEXT_COLOR, pad=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(ITERATIONS, fontsize=15)
    ax.tick_params(axis='y', labelsize=15, colors=TEXT_COLOR)
    ax.tick_params(axis='x', colors=TEXT_COLOR)
    ax.grid(axis='y', color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#9ca3af')
    ax.spines['bottom'].set_color('#9ca3af')

# use the 8th panel for the iteration legend
legend_ax = axes[-1]
legend_ax.axis('off')

fig.suptitle('Evolution of Mechanical Properties and Simulated Ballistic Performance Across Iterations', fontsize=22, color=TEXT_COLOR, y=0.98)
legend_handles = [Patch(facecolor=ITER_COLORS[it], edgecolor='black', label=it, alpha=0.82) for it in ITERATIONS]
legend_ax.legend(
    legend_handles,
    ITERATIONS,
    loc='center',
    ncol=1,
    frameon=False,
    fontsize=18,
    handlelength=1.8,
    handletextpad=0.8,
    borderaxespad=0.0,
)
fig.subplots_adjust(left=0.06, right=0.99, bottom=0.08, top=0.83, wspace=0.22, hspace=0.35)
fig.savefig(OUT_PDF, bbox_inches='tight', pad_inches=0.02)
fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', pad_inches=0.02)
print(f'wrote {OUT_PDF}')
print(f'wrote {OUT_PNG}')
