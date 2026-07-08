# BRAVE

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21253553.svg)](https://doi.org/10.5281/zenodo.21253553)

Representative code, data, and manuscript for HTMDEC Campaign 2 — a Bayesian batch
optimization framework for high-throughput alloy discovery in the
Al–V–Cr–Mn–Fe–Co–Ni–Cu compositional space across five mechanical objectives.

## Repository layout

```
data/          Master experimental database
code/          Analysis & modeling code (BBO, feasibility, Thermo-Calc, HEACalculator)
figures_code/  Scripts, notebook, and prompts that generate the figures from data/
paper/         Submitted manuscript, supplement, highlights, and final figures
```

- **`data/HTMDEC_Y2_db.xlsx`** — master experimental database used to train the
  CatBoost priors and to generate the figures. Composition columns are `Al`, `Co`,
  `Cr`, `Cu`, `Fe`, `Mn`, `Ni`, `V`; property columns consumed by the priors and
  figure scripts include `Yield Strength (MPa)`, `UTS_True (Mpa)`, `UTS/YS`,
  `Elong_T (%)`, `Avg Hdyn (GPa) HSR`, `Avg Hqs (GPa) HSR`, `Avg HDYN/HQS`,
  `Depth of Penetration (mm) FE_Sim`, `Rate Sensitivity Exponent (M)`, alongside
  `Iteration`, `Alloy Name`, and `XRD Phase`.
- **`code/htmdec_y2_tc_property_gen.py`** — Thermo-Calc property generation
  (liquidus / solidus / freeze-in equilibrium) for candidate compositions.
  Requires a licensed Thermo-Calc installation and `tc_python`.
- **`code/bbo_python/`** — batch Bayesian optimization loop. `main.py` is the entry
  point; the other modules implement the Gaussian-process fusion (`gpModel.py`,
  `reificationFusion.py`), the EHVI acquisition function and Pareto/hypervolume
  utilities (`multiobjective.py`, `acquisitionFunc.py`), the CatBoost prior models
  (`priors.py`, `prior_eval.py`, `helper.py`), and the Peierls–Nabarro
  yield-strength prior (`HT_FCC_YS.py`, `YS_pb.py`).
- **`code/Probability_calculations/`** — Gaussian-process classifier that scores
  points outside the strict feasible space, producing `probs.csv` and
  `infeasibles.csv` consumed by `code/bbo_python/main.py`.
- **`code/HEACalculator/`** — high-entropy-alloy thermodynamic feature library
  imported by `code/bbo_python/helper.py`. Third-party, GPLv3 — see
  [THIRD_PARTY.md](THIRD_PARTY.md).
- **`figures_code/`** — everything used to produce the figures: the data-driven
  scripts (`make_*.py`, which read `data/HTMDEC_Y2_db.xlsx` and write `fig_NN`
  PDFs/PNGs into `paper/`), `C2_Visualizations.ipynb` (correlation / SHAP /
  corrSHAP panels → `fig_15`), and the `prompts/` + `ai_generated/` provenance for
  the schematic figures. The complete figure-to-source map for **all** main and
  supplementary figures is in [figures_code/README.md](figures_code/README.md).
- **`paper/`** — the submission bundle, flat: `manuscript.tex` (the submitted
  **single-column, 12 pt, line-numbered** review version, self-contained with an
  inlined bibliography) and its compiled `manuscript_final.pdf`; the Supplementary
  Information source `supplementary_information.tex` (two-column; cites
  `references.bib`; figures in `paper/Sup_figures/`) and its compiled
  `supplementary_information.pdf`; `highlights.pdf`;
  `graphical_abstract.{png,pdf}`; and the final main-text figures
  `fig_01_kkt` … `fig_20_benchmark`. Build both PDFs with `make paper`.

## Element order

All composition arrays use the fixed order: **Al, V, Cr, Mn, Fe, Co, Ni, Cu**.

## Design objectives

The BBO loop optimizes five objectives jointly (1–4 maximized, 5 minimized): yield
strength, UTS/YS, uniform elongation, dynamic/quasi-static hardness ratio, and
ballistic penetration depth.

## Setup

```
pip install -r requirements.txt
```

Versions are pinned to the environment used to produce the published results.
`tc_python` is proprietary (Thermo-Calc) and must be installed separately following
the vendor instructions; it is only required to run
`code/htmdec_y2_tc_property_gen.py`.

## Reproducing the figures

The data-driven paper figures are regenerated from `data/HTMDEC_Y2_db.xlsx`:

```
make figures
```

This runs the data-driven scripts in `figures_code/` and writes the `fig_NN`
PDFs/PNGs into `paper/` (see [figures_code/README.md](figures_code/README.md) for
which figures are script-, notebook-, or non-code-generated). The scripts resolve
their input and output paths relative to the repository root, so they can also be
run individually from any directory, e.g.:

```
python figures_code/make_figure14_boxplots.py
```

To rebuild the manuscript (requires `latexmk` and a full TeX distribution — the
single-column class uses Helvetica, so `texlive-fontsrecommended` / `psnfss` must
be installed):

```
make paper
```

## Running the BBO pipeline

The BBO scripts use flat sibling imports (e.g. `from gpModel import gp_model`) and
read input CSVs from the current directory, so run them from inside their own
folder. Put `code/` on `PYTHONPATH` so that `from HEACalculator import HEACalculator`
resolves:

```
cd code/bbo_python && PYTHONPATH=.. python main.py
cd code/Probability_calculations && PYTHONPATH=.. python main.py
```

### Pipeline order

1. **Composition generation & screening (Thermo-Calc).**
   `code/htmdec_y2_tc_property_gen.py` reads a composition file (e.g.
   `htmdecyear2_n8_d25_subset_n8.csv`) and writes per-batch property CSVs into
   `CalcFiles/`. The resulting feasible and full composition spaces are saved as
   `feasibles.csv` and `all_space.csv`.
2. **Probabilistic feasibility (`code/Probability_calculations/main.py`).** Trains a
   GP classifier on `feasibles.csv` + `tested_alloys.csv` and scores `all_space.csv`,
   producing `infeasibles.csv` and `probs.csv`.
3. **Prior training (`code/bbo_python/priors.py`).** Fits CatBoost models for the YS /
   UTS-to-YS / elongation priors from `data/HTMDEC_Y2_db.xlsx`.
4. **Prior evaluation (`code/bbo_python/prior_eval.py`).** Queries the CatBoost priors
   at the design points `x_test.csv` and saves `YS_prior.csv`, `EUTS_prior.csv`,
   `UTStoYS_prior.csv` (and `YS_pb_prior.csv` from the Peierls–Nabarro model).
5. **Batch BO (`code/bbo_python/main.py`).** Fuses ground-truth and information-source
   GPs across the five objectives, evaluates the constraint-aware EHVI acquisition
   function, and selects a batch of `Batch_size` candidates via k-medoid clustering.
   Outputs `x_query.csv`, `all_candidates.csv`, `all_improvements.csv`.

## Data availability

Per-iteration intermediate files (`o{1..5}_GT_y.csv`, `lhp.csv`, `tested_alloys.csv`,
prior test CSVs, etc.) are part of the campaign workflow and are available on request.
The master database `data/HTMDEC_Y2_db.xlsx` and the code in this repository are
sufficient to reproduce the modeling framework and all data-driven figures.

## Citation

If you use this software or data, please cite the archived release
(DOI [10.5281/zenodo.21253553](https://doi.org/10.5281/zenodo.21253553); see
[CITATION.cff](CITATION.cff)) together with the accompanying BRAVE publication.

## License

The HTMDEC Campaign 2 code (`code/bbo_python/`, `code/Probability_calculations/`,
`code/htmdec_y2_tc_property_gen.py`) and the figure scripts in `figures_code/` are
released under the MIT license — see [LICENSE](LICENSE).

The bundled `code/HEACalculator/` package is third-party code distributed under the
GNU General Public License v3. See [THIRD_PARTY.md](THIRD_PARTY.md).
