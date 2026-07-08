# figures_code — figure provenance

This directory holds everything used to produce the manuscript figures, plus a
complete map of where every figure comes from. The final figure files live flat
in [`../paper/`](../paper/) with `fig_NN` names; the manuscript is the submitted
single-column, line-numbered `paper/manuscript.tex`.

## Contents

- `make_*.py` — data-driven figure scripts. Each reads `../data/HTMDEC_Y2_db.xlsx`
  (or, for the affine projection, the composition table in `../paper/manuscript.tex`)
  and writes its `fig_NN` PDF + PNG into `../paper/`. Run all of them with
  `make figures` from the repo root, or individually from any directory.
- `C2_Visualizations.ipynb` — notebook that produces the correlation matrix / SHAP /
  corrSHAP panels (`fig_15`) and the SI property pair plot; reads `../data/HTMDEC_Y2_db.xlsx`.
- `prompts/` — text prompts used to generate the schematic/AI figures (`fig_01`, and
  the graphical abstract).
- `ai_generated/` — source renders and prompts for the BRAVE architecture schematic
  (`fig_07`).

## Figure → source map (main text)

| Figure | File (`paper/`) | Produced by |
|---|---|---|
| 1  | `fig_01_kkt.png` | Schematic (AI-assisted) — `prompts/PROMPT_kkt_*.md` |
| 2  | `fig_02_birdshot.pdf` | Schematic — BIRDSHOT framework (reproduced from Hastings et al.) |
| 3  | `fig_03_workflow.pdf` | Schematic / photo collage — experimental workflow |
| 4  | `fig_04_cutting.pdf` | Schematic — specimen sectioning geometry |
| 5  | `fig_05_eds.pdf` | EDS target-vs-measured parity plot (analysis; no standalone script) |
| 6  | `fig_06_shpb.pdf` | Raw SHPB dynamic stress–strain curves (instrument data) |
| 7  | `fig_07_brave.pdf` | Schematic — BRAVE data-flow architecture — `ai_generated/` |
| 8  | `fig_08_affine.pdf` | **`make_affine_projection.py`** |
| 9  | `fig_09_phase.pdf` | XRD diffraction patterns (instrument data) |
| 10 | `fig_10_bse_sem.pdf` | BSE-SEM / EBSD micrographs + grain-size histograms (microscopy) |
| 11 | `fig_11_tension.pdf` | Quasi-static tensile stress–strain curves (instrument data) |
| 12 | `fig_12_hdyn.pdf` | **`make_figure13_hardness_bars.py`** |
| 13 | `fig_13_dyn_qs_ratio.pdf` | **`make_figure14_ratio.py`** |
| 14 | `fig_14_boxplot.pdf` | **`make_figure14_boxplots.py`** |
| 15 | `fig_15_correlation_matrix.pdf` | **`C2_Visualizations.ipynb`** (correlation + SHAP + corrSHAP) |
| 16 | `fig_16_pair_plots.pdf` | **`make_figure17_pair_plots.py`** |
| 17 | `fig_17_dop_vs_strengths.pdf` | DoP-vs-YS/UTS scatter (analysis; no standalone script) |
| 18 | `fig_18_m_plots.pdf` | **`make_figure19_m_relationships.py`** |
| 19 | `fig_19_v_ys_feasibility.pdf` | **`make_figure_V_YS_feasibility.py`** |
| 20 | `fig_20_benchmark.pdf` | Appendix B MOBBO benchmark study (computational; not scripted here) |

**Data-driven and reproducible from `data/`:** figures 8, 12, 13, 14, 15, 16, 18, 19.
The remainder are schematics, microscopy, raw-instrument plots, or the separate
Appendix B benchmark, and are not regenerated from the database. Figures 5 and 17
are data-derived but were produced during analysis without a standalone script.

## Supplementary figures

The Supplementary Information source is `../paper/supplementary_information.tex`
(compiled `../paper/supplementary_information.pdf`), with its figures in
`../paper/Sup_figures/`:

| SI figure | File (`paper/Sup_figures/`) | Produced by |
|---|---|---|
| S1 | `fig_si_01_birdshot_workflow.pdf` | Schematic — integrated workflow |
| S2 | `fig_si_02_selection_pipeline.pdf` | Schematic — Iteration-0 selection pipeline |
| S3 | `fig_si_03_composition_space.pdf` | Combinatorial design-space size (analytic) |
| S4 | `fig_si_04_fcc_filtering.pdf` | CALPHAD FCC-filtering funnel (analysis) |
| S5 | `fig_si_05_perturbation.pdf` | Local FCC-stability perturbation analysis |
| S6 | `fig_si_06_subsystem_counts.pdf` | Feasible-count-per-subsystem bar (analysis) |
| S7 | `fig_si_07_bba_suggestions.pdf` | Iteration-0 k-medoids selection (analysis) |
| S8 | `fig_si_08_file_structure.pdf` | Schematic — data architecture |
| S9 | `fig_si_09_property_pairplot.pdf` | **`C2_Visualizations.ipynb`** (property pair plot) |

Of these, S9 is regenerated from the database via the notebook; the rest are
schematics or analysis plots produced during the campaign.
