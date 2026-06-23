PY ?= python3
FIG := figures_code
export MPLBACKEND := Agg

.PHONY: help figures paper clean-pyc

help:
	@echo "make figures   Regenerate all data-driven figures from data/HTMDEC_Y2_db.xlsx into paper/Figures/"
	@echo "make paper      Build paper/main.pdf (requires latexmk + a TeX distribution)"
	@echo "make clean-pyc  Remove Python bytecode caches"

# Regenerate every script-produced figure. Scripts resolve paths from the repo
# root, so they can be invoked from anywhere.
figures:
	$(PY) $(FIG)/make_figure13_hardness_bars.py
	$(PY) $(FIG)/make_figure14_boxplots.py
	$(PY) $(FIG)/make_figure14_ratio.py
	$(PY) $(FIG)/make_figure17_pair_plots.py
	$(PY) $(FIG)/make_figure19_m_relationships.py
	$(PY) $(FIG)/make_affine_projection.py

paper:
	cd paper && latexmk -pdf main.tex

clean-pyc:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
