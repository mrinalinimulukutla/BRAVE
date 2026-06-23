import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / 'data' / 'HTMDEC_Y2_db.xlsx'
OUT_PDF = ROOT / 'paper' / 'Figures' / 'pair_plots.pdf'
OUT_PNG = ROOT / 'paper' / 'Figures' / 'pair_plots.png'

ITERATIONS = ['BBA', 'BBB', 'BBC']
ITER_COLORS = {
    'BBA': '#c2410c',
    'BBB': '#1d4ed8',
    'BBC': '#047857',
}
SIGMA_COLORS = {
    'BBA': '#c27f57',
    'BBB': '#6f99da',
    'BBC': '#68a58a',
}
TEXT_COLOR = '#111827'
GRID_COLOR = '#e5e7eb'
PARETO_COLOR = '#4b5563'
SIGMA_PARETO_COLOR = '#9ca3af'

PANELS = [
    {
        'x': 'Yield Strength (MPa)',
        'y': 'UTS_True (Mpa)',
        'xlabel': 'YS (MPa)',
        'ylabel': 'UTS (MPa)',
        'title': '(a) YS–UTS Synergy',
        'maximize': (True, True),
    },
    {
        'x': 'UTS/YS',
        'y': 'Elong_T (%)',
        'xlabel': 'UTS/YS',
        'ylabel': 'Strain at UTS (%)',
        'title': '(b) UTS/YS–Ductility Synergy',
        'maximize': (True, True),
    },
    {
        'x': 'Yield Strength (MPa)',
        'y': 'Elong_T (%)',
        'xlabel': 'YS (MPa)',
        'ylabel': 'Strain at UTS (%)',
        'title': '(c) Strength–Ductility Trade-off',
        'maximize': (True, True),
    },
    {
        'x': 'UTS_True (Mpa)',
        'y': 'Elong_T (%)',
        'xlabel': 'UTS (MPa)',
        'ylabel': 'Strain at UTS (%)',
        'title': '(d) UTS–Ductility Synergy',
        'maximize': (True, True),
    },
]


def pareto_front(df, xcol, ycol, maximize=(True, True)):
    cols = ['Alloy Name', xcol, ycol]
    sub = df[cols].dropna().copy()
    pts = sub[[xcol, ycol]].to_numpy(dtype=float)
    if len(pts) == 0:
        return sub.iloc[0:0].copy()
    sx = 1.0 if maximize[0] else -1.0
    sy = 1.0 if maximize[1] else -1.0
    transformed = np.column_stack([sx * pts[:, 0], sy * pts[:, 1]])
    keep = np.ones(len(transformed), dtype=bool)
    for i in range(len(transformed)):
        if not keep[i]:
            continue
        dominates = np.all(transformed >= transformed[i], axis=1) & np.any(transformed > transformed[i], axis=1)
        dominates[i] = False
        if dominates.any():
            keep[i] = False
    front = sub.loc[keep].copy()
    order = np.argsort(front[xcol].to_numpy(dtype=float))
    return front.iloc[order].copy()


def main():
    cols = ['Iteration', 'Alloy Name', 'XRD Phase'] + sorted({p['x'] for p in PANELS} | {p['y'] for p in PANELS})
    df = pd.read_excel(DATA_FILE, usecols=lambda c: c in cols)
    df = df[df['Iteration'].isin(ITERATIONS)].copy()
    df['alloy_num'] = df['Alloy Name'].str[-2:].astype(int)
    df = df[~((df['Iteration'] == 'BBA') & (df['alloy_num'] > 16))].copy()
    df = df.sort_values(['Iteration', 'alloy_num'])
    df['sigma'] = df['XRD Phase'].fillna('').astype(str).str.contains('σ|sigma', case=False, regex=True)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.5))
    axes = axes.flatten()

    for panel_idx, (ax, panel) in enumerate(zip(axes, PANELS)):
        panel_df = df[['Iteration', 'Alloy Name', 'sigma', panel['x'], panel['y']]].dropna().copy()

        feasible_df = panel_df[~panel_df['sigma']].copy()
        sigma_df = panel_df[panel_df['sigma']].copy()

        for it in ITERATIONS:
            sub = panel_df[panel_df['Iteration'] == it]
            non_sigma = sub[~sub['sigma']]
            sigma = sub[sub['sigma']]
            ax.scatter(
                non_sigma[panel['x']],
                non_sigma[panel['y']],
                s=64,
                color=ITER_COLORS[it],
                edgecolors='none',
                alpha=1.0,
                label=it,
                zorder=3,
            )
            if not sigma.empty:
                ax.scatter(
                    sigma[panel['x']],
                    sigma[panel['y']],
                    s=64,
                    color=SIGMA_COLORS[it],
                    edgecolors='none',
                    alpha=0.9,
                    zorder=4,
                )

        front = pareto_front(feasible_df, panel['x'], panel['y'], maximize=panel['maximize'])
        if len(front) >= 2:
            ax.plot(front[panel['x']], front[panel['y']], linestyle='--', linewidth=2.4, color=PARETO_COLOR, zorder=2)

        all_front = pareto_front(panel_df, panel['x'], panel['y'], maximize=panel['maximize'])
        if len(all_front) >= 2:
            ax.plot(all_front[panel['x']], all_front[panel['y']], linestyle='--', linewidth=2.4, color=SIGMA_PARETO_COLOR, zorder=1)

        annotated = set()
        xspan = panel_df[panel['x']].max() - panel_df[panel['x']].min()
        yspan = panel_df[panel['y']].max() - panel_df[panel['y']].min()
        dx_scale = 0.016 if panel_idx == 1 else 0.012
        dy_scale = 0.026 if panel_idx == 1 else 0.018
        base_dx = dx_scale * xspan if xspan > 0 else 0.0
        base_dy = dy_scale * yspan if yspan > 0 else 0.0
        label_boxes = []
        offset_pattern = [
            (1.2, 1.2, 'left', 'bottom'),
            (1.2, -1.4, 'left', 'top'),
            (-1.2, 1.2, 'right', 'bottom'),
            (-1.2, -1.4, 'right', 'top'),
            (2.1, 0.35, 'left', 'center'),
            (-2.1, 0.35, 'right', 'center'),
            (0.25, 2.3, 'center', 'bottom'),
            (0.25, -2.3, 'center', 'top'),
            (2.0, 1.5, 'left', 'bottom'),
            (-2.0, 1.5, 'right', 'bottom'),
            (2.0, -1.7, 'left', 'top'),
            (-2.0, -1.7, 'right', 'top'),
        ]
        front_union = pd.concat([all_front, front], ignore_index=True).drop_duplicates(subset=['Alloy Name'])
        if panel_idx == 1 and not front_union.empty:
            stacked = front_union.sort_values(panel['y']).reset_index(drop=True)
            x_anchor = panel_df[panel['x']].max() + 2.8 * base_dx
            y_min = panel_df[panel['y']].min() + 0.08 * yspan
            y_max = panel_df[panel['y']].max() - 0.08 * yspan
            y_positions = np.linspace(y_min, y_max, len(stacked))
            elbow_x = x_anchor - 1.0 * base_dx
            for (_, row), y_text in zip(stacked.iterrows(), y_positions):
                alloy = row['Alloy Name']
                annotated.add(alloy)
                x0 = row[panel['x']]
                y0 = row[panel['y']]
                ax.plot([x0, x0], [y0, y_text], color='#d1d5db', linewidth=0.9, linestyle=(0, (2.2, 2.2)), zorder=2)
                ax.plot([x0, elbow_x], [y_text, y_text], color='#d1d5db', linewidth=0.9, linestyle=(0, (2.2, 2.2)), zorder=2)
                ax.text(
                    x_anchor,
                    y_text,
                    alloy,
                    fontsize=8.8,
                    color=TEXT_COLOR,
                    ha='left',
                    va='center',
                    zorder=5,
                    bbox=dict(boxstyle='round,pad=0.10', facecolor='white', edgecolor='none', alpha=0.9),
                )
            ax.set_xlim(panel_df[panel['x']].min() - 0.04 * xspan, x_anchor + 3.2 * base_dx)
        else:
            for front_df in (all_front, front):
                for _, row in front_df.iterrows():
                    alloy = row['Alloy Name']
                    if alloy in annotated:
                        continue
                    annotated.add(alloy)
                    for mx, my, ha, va in offset_pattern:
                        x = row[panel['x']] + mx * base_dx
                        y = row[panel['y']] + my * base_dy
                        if all(abs(x - px) > (3.8 if panel_idx == 1 else 3.2) * base_dx or abs(y - py) > (3.2 if panel_idx == 1 else 2.6) * base_dy for px, py in label_boxes):
                            label_boxes.append((x, y))
                            ax.text(
                                x,
                                y,
                                alloy,
                                fontsize=9.0,
                                color=TEXT_COLOR,
                                ha=ha,
                                va=va,
                                zorder=5,
                                bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.8),
                            )
                            break
                    else:
                        x = row[panel['x']] + base_dx
                        y = row[panel['y']] + base_dy
                        label_boxes.append((x, y))
                        ax.text(
                            x,
                            y,
                            alloy,
                            fontsize=9.0,
                            color=TEXT_COLOR,
                            ha='left',
                            va='bottom',
                            zorder=5,
                            bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.8),
                        )

        ax.set_title(panel['title'], fontsize=19, color=TEXT_COLOR, pad=8)
        ax.set_xlabel(panel['xlabel'], fontsize=18, color=TEXT_COLOR)
        ax.set_ylabel(panel['ylabel'], fontsize=18, color=TEXT_COLOR)
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
        Line2D([0], [0], color=PARETO_COLOR, linestyle='--', linewidth=2.4, label='Feasible Pareto front'),
        Line2D([0], [0], color=SIGMA_PARETO_COLOR, linestyle='--', linewidth=2.4, label='All-alloy Pareto front'),
    ]
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98), fontsize=14)
    fig.suptitle('Property-Pair Evolution Across Successive Alloy Design Iterations', fontsize=22, color=TEXT_COLOR, y=0.995)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.87, wspace=0.24, hspace=0.28)
    fig.savefig(OUT_PDF, bbox_inches='tight', pad_inches=0.02)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print(f'wrote {OUT_PDF}')
    print(f'wrote {OUT_PNG}')


if __name__ == '__main__':
    main()
