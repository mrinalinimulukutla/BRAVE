PY ?= python3
FIG := figures_code
export MPLBACKEND := Agg

.PHONY: help figures paper clean-pyc

help:
	@echo "make figures   Regenerate the data-driven figures from data/HTMDEC_Y2_db.xlsx into paper/"
	@echo "make paper      Build paper/manuscript.pdf (requires latexmk + a full TeX distribution)"
	@echo "make clean-pyc  Remove Python bytecode caches"

# Regenerate every script-produced figure from data/HTMDEC_Y2_db.xlsx. Scripts
# resolve paths from the repo root, so they can be invoked from anywhere, and
# write directly into paper/ using the fig_NN naming scheme. See
# figures_code/README.md for the full figure-to-source map (including the
# notebook-generated and non-code figures).
figures:
	$(PY) $(FIG)/make_figure13_hardness_bars.py       # fig_12_hdyn
	$(PY) $(FIG)/make_figure14_ratio.py               # fig_13_dyn_qs_ratio
	$(PY) $(FIG)/make_figure14_boxplots.py            # fig_14_boxplot
	$(PY) $(FIG)/make_figure17_pair_plots.py          # fig_16_pair_plots
	$(PY) $(FIG)/make_figure19_m_relationships.py     # fig_18_m_plots
	$(PY) $(FIG)/make_figure_V_YS_feasibility.py      # fig_19_v_ys_feasibility
	$(PY) $(FIG)/make_affine_projection.py            # fig_08_affine

paper:
	cd paper && latexmk -pdf manuscript.tex
	cd paper && latexmk -pdf supplementary_information.tex

clean-pyc:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
