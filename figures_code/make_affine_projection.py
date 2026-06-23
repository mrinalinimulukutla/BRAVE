from pathlib import Path
import math
import re

import matplotlib.pyplot as plt
import numpy as np


ELEMENT_ORDER = ["Al", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
TABLE_ORDER   = ["Al", "Co", "Cr", "Cu", "Fe", "Mn", "Ni", "V"]

# ── colour palette shared with Figures 14 & 15 ──────────────────────────────
ITER_COLORS = {
    "BBA": "#fdba74",
    "BBB": "#93c5fd",
    "BBC": "#86efac",
}
TEXT_COLOR  = "#111827"
POLY_COLOR  = "#374151"
RAY_COLOR   = "#d1d5db"

ROOT = Path(__file__).resolve().parents[1]
TEX_SOURCE = ROOT / "paper" / "03_results.tex"
OUT_PDF = ROOT / "paper" / "Figures" / "alloy_affine_projection.pdf"
OUT_PNG = ROOT / "paper" / "Figures" / "alloy_affine_projection.png"


def clean_cell(cell: str) -> str:
    cell = re.sub(r"\\SetCell\{[^}]*\}", "", cell)
    cell = re.sub(r"\\textbf\{([^}]*)\}", r"\1", cell)
    cell = cell.replace("$\\cdot$", "")
    cell = cell.replace("$", "")
    cell = cell.replace("\\mathbf{", "").replace("}", "")
    return cell.strip()


def parse_alloys(tex_path: str):
    lines = Path(tex_path).read_text().splitlines()
    rows = []
    for line in lines:
        if not any(tag in line for tag in ("BBA", "BBB", "BBC")):
            continue
        if "&" not in line or "\\\\" not in line:
            continue
        if "Avg EHVI" in line or "Pareto" in line:
            continue
        stripped = clean_cell(line)
        cells = [clean_cell(part) for part in stripped.split("&")]
        name = cells[0]
        if not re.match(r"^(BBA|BBB|BBC)\d+$", name):
            continue
        target_values = [float(cells[i]) for i in range(1, 9)]
        target = dict(zip(TABLE_ORDER, target_values))
        comp = np.array([target[el] for el in ELEMENT_ORDER], dtype=float) / 100.0
        rows.append((name, comp))
    return rows


def affine_project(comp, vertices):
    return comp @ vertices


def main():
    alloys = parse_alloys(TEX_SOURCE)
    n = len(ELEMENT_ORDER)
    angles   = np.linspace(math.pi / 2, math.pi / 2 + 2 * math.pi, n, endpoint=False)
    vertices = np.c_[np.cos(angles), np.sin(angles)]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    # outer polygon
    poly = np.vstack([vertices, vertices[0]])
    ax.plot(poly[:, 0], poly[:, 1], color=POLY_COLOR, lw=1.8)

    # element labels only — no radial rays
    for el, vertex in zip(ELEMENT_ORDER, vertices):
        label_pos = 1.13 * vertex
        ax.text(
            label_pos[0], label_pos[1], el,
            ha="center", va="center",
            fontsize=13, fontweight="bold", color=TEXT_COLOR,
        )

    # alloy scatter — filled with iteration colour, black edge (matches Fig 14 bars)
    for iteration, color in ITER_COLORS.items():
        pts = []
        for name, comp in alloys:
            if name.startswith(iteration):
                pts.append(affine_project(comp, vertices))
        if not pts:
            continue
        pts = np.array(pts)
        ax.scatter(
            pts[:, 0], pts[:, 1],
            s=60,
            c=color,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.92,
            label=iteration,
            zorder=3,
        )

    ax.scatter([0], [0], s=14, c="#6b7280", alpha=0.5, zorder=0)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=3,
        frameon=False,
        fontsize=13,
        handletextpad=0.5,
        columnspacing=1.0,
        labelcolor=TEXT_COLOR,
    )

    fig.tight_layout()
    fig.savefig(OUT_PDF, bbox_inches="tight", transparent=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", transparent=True)
    print(f"saved {OUT_PDF}")
    print(f"saved {OUT_PNG}")


if __name__ == "__main__":
    main()
