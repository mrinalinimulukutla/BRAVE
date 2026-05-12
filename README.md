# BRAVE

Representative code for HTMDEC Campaign 2 — Bayesian batch optimization framework for high-throughput alloy discovery in the Al–V–Cr–Mn–Fe–Co–Ni–Cu compositional space across five mechanical objectives.


## Repository layout

- `htmdec_y2_tc_property_gen.py` — Thermo-Calc property generation (liquidus / solidus / freeze-in equilibrium) for candidate compositions. Requires a licensed Thermo-Calc installation and `tc_python`.
- `bbo_python/` — batch Bayesian optimization loop. `main.py` is the entry point; the other modules implement the Gaussian-process fusion (`gpModel.py`, `reificationFusion.py`), the EHVI acquisition function and Pareto/hypervolume utilities (`multiobjective.py`, `acquisitionFunc.py`), the CatBoost prior models (`priors.py`, `prior_eval.py`, `helper.py`), and the Peierls–Nabarro yield-strength prior (`HT_FCC_YS.py`, `YS_pb.py`).
- `Probability_calculations/` — Gaussian-process classifier that scores points outside the strict feasible space, producing `probs.csv` and `infeasibles.csv` consumed by `bbo_python/main.py`.
- `HEACalculator/` — high-entropy-alloy thermodynamic feature library imported by `bbo_python/helper.py` (CatBoost feature generation). Bundled from [github.com/dogusariturk/HEACalculator](https://github.com/dogusariturk/HEACalculator) and used here as a library; the original GUI assets (`.ui`, resources, icons) are not redistributed.
- `HTMDEC_Y2_db.xlsx` — master experimental database used to train the CatBoost priors. Required columns: `Al`, `V`, `Cr`, `Mn`, `Fe`, `Co`, `Ni`, `Cu`, `YS (MPa)`, `UTS / YS`, `EL (%)`.
- `C2_Visualizations.ipynb` — exploratory plots over `HTMDEC_Y2_db.xlsx` (correlation heatmap, multi-objective SHAP beeswarm + corrSHAP bar plot, property pair plot, and per-property box plots) for iterations BBA / BBB / BBC.

## Element order

All composition arrays use the fixed order: **Al, V, Cr, Mn, Fe, Co, Ni, Cu**.

## Design objectives

The BBO loop optimizes five objectives jointly (1–4 maximized, 5 minimized): yield strength, UTS/YS, uniform elongation, dynamic/quasi-static hardness ratio, and ballistic penetration depth.

## Dependencies

```
pip install -r requirements.txt
```

`tc_python` is proprietary (Thermo-Calc) and must be installed separately following the vendor instructions; it is only required to run `htmdec_y2_tc_property_gen.py`.

The BBO scripts use flat sibling imports (e.g. `from gpModel import gp_model`) and read input CSVs from the current directory, so run them from inside their own folder. Put the repository root on `PYTHONPATH` so that `from HEACalculator import HEACalculator` resolves:

```
cd bbo_python && PYTHONPATH=.. python main.py
cd Probability_calculations && PYTHONPATH=.. python main.py
```

## Pipeline order

1. **Composition generation & screening (Thermo-Calc).** `htmdec_y2_tc_property_gen.py` reads a composition file (e.g. `htmdecyear2_n8_d25_subset_n8.csv`) and writes per-batch property CSVs into `CalcFiles/`. The resulting feasible and full composition spaces are saved as `feasibles.csv` and `all_space.csv`.
2. **Probabilistic feasibility (`Probability_calculations/main.py`).** Trains a GP classifier on `feasibles.csv` + `tested_alloys.csv` and scores `all_space.csv`, producing `infeasibles.csv` and `probs.csv`.
3. **Prior training (`bbo_python/priors.py`).** Fits CatBoost models for the YS / UTS-to-YS / elongation priors from `HTMDEC_Y2_db.xlsx`.
4. **Prior evaluation (`bbo_python/prior_eval.py`).** Queries the CatBoost priors at the design points `x_test.csv` and saves `YS_prior.csv`, `EUTS_prior.csv`, `UTStoYS_prior.csv` (and `YS_pb_prior.csv` from the Peierls–Nabarro model).
5. **Batch BO (`bbo_python/main.py`).** Fuses ground-truth and information-source GPs across the five objectives, evaluates the constraint-aware EHVI acquisition function, and selects a batch of `Batch_size` candidates via k-medoid clustering. Outputs `x_query.csv`, `all_candidates.csv`, `all_improvements.csv`.

## Data availability

Per-iteration intermediate files (`o{1..5}_GT_y.csv`, `lhp.csv`, `tested_alloys.csv`, prior test CSVs, etc.) are part of the campaign workflow and are available on request. The master database `HTMDEC_Y2_db.xlsx` and the representative code in this repository are sufficient to reproduce the modeling framework.

## Citation

If you use this code, please cite the accompanying BRAVE publication.

## License

The HTMDEC Campaign 2 code (`bbo_python/`, `Probability_calculations/`, `htmdec_y2_tc_property_gen.py`) is released under the MIT license — see [LICENSE](LICENSE).

The bundled `HEACalculator/` package is third-party code by Doguhan Sariturk, distributed under the GNU General Public License v3 (see the header of each `HEACalculator/**/*.py` file). Redistribution of any combined work that links against `HEACalculator/` must comply with the GPLv3.
