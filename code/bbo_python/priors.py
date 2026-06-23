# -*- coding: utf-8 -*-


from pathlib import Path

from helper import FeatureGenerator
import pandas as pd
import catboost as cb
import numpy as np

# Resolve the master database from the repository root so this runs from any
# working directory (the figure scripts use the same pattern).
DB_FILE = Path(__file__).resolve().parents[2] / "data" / "HTMDEC_Y2_db.xlsx"

df = pd.read_excel(DB_FILE)
df = df[(df["YS (MPa)"].notna()) & (df["YS (MPa)"] != 0)]

df_final = df.copy()
df_final = df_final[["Al", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]].astype(float)
analyzer = FeatureGenerator(df_final)
comp_df = analyzer.generate_composition_formula()
comp_df = analyzer.generate_features_ys()
X_train = comp_df
y_train = df["YS (MPa)"]
dtrain = cb.Pool(X_train, label=y_train)
params = {
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "logging_level": "Silent",
    "custom_metric": ["RMSE", "MAE", "R2"],
}
ys_model = cb.CatBoostRegressor(**params)
ys_model.fit(dtrain)


df = pd.read_excel(DB_FILE)
df = df[(df["UTS / YS"].notna()) & (df["UTS / YS"] != 0)]
df_final = df.copy()
df_final = df_final[["Al", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]].astype(float)
analyzer = FeatureGenerator(df_final)
comp_df = analyzer.generate_composition_formula()
comp_df = analyzer.generate_features_uts_ys()
X_train = comp_df
y_train = df["UTS / YS"]
dtrain = cb.Pool(X_train, label=y_train)
params = {
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "logging_level": "Silent",
    "custom_metric": ["RMSE", "MAE", "R2"],
}
uts_ys_model = cb.CatBoostRegressor(**params)
uts_ys_model.fit(dtrain)

df = pd.read_excel(DB_FILE)
df = df[(df["EL (%)"].notna()) & (df["EL (%)"] != 0)]
df_final = df.copy()
df_final = df_final[["Al", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]].astype(float)
analyzer = FeatureGenerator(df_final)
comp_df = analyzer.generate_composition_formula()
comp_df = analyzer.generate_features_el()
X_train = comp_df
y_train = df["EL (%)"]
dtrain = cb.Pool(X_train, label=y_train)
params = {
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "logging_level": "Silent",
    "custom_metric": ["RMSE", "MAE", "R2"],
}
el_model = cb.CatBoostRegressor(**params)
el_model.fit(dtrain)


#data = np.array(
#    [
#        [20, 20, 20, 5, 10, 10, 10, 5],
#        [30, 10, 20, 5, 10, 10, 10, 5],
#        [30, 10, 20, 5, 10, 10, 10, 5],
#    ]  # ["Al", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
#)


def YS_prior(data):
    # Column names
    columns = ["Al", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
    # Creating the DataFrame
    query_df = pd.DataFrame(data, columns=columns)

    analyzer = FeatureGenerator(query_df)
    comp_df_ys = analyzer.generate_composition_formula()
    comp_df_ys = analyzer.generate_features_ys()
    # print(ys_model.predict(comp_df_ys))
    return ys_model.predict(comp_df_ys)


def UTStoYS_prior(data):
    # Column names
    columns = ["Al", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
    # Creating the DataFrame
    query_df = pd.DataFrame(data, columns=columns)

    analyzer = FeatureGenerator(query_df)
    comp_df_uts_ys = analyzer.generate_composition_formula()
    comp_df_uts_ys = analyzer.generate_features_uts_ys()
    # print(uts_ys_model.predict(comp_df_uts_ys))
    return uts_ys_model.predict(comp_df_uts_ys)


def EUTS_prior(data):
    # Column names
    columns = ["Al", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu"]
    # Creating the DataFrame
    query_df = pd.DataFrame(data, columns=columns)

    analyzer = FeatureGenerator(query_df)
    comp_df_el = analyzer.generate_composition_formula()
    comp_df_el = analyzer.generate_features_el()
    # print(el_model.predict(comp_df_el) / 100)
    return el_model.predict(comp_df_el) / 100


# YS_prior(data)
# UTStoYS_prior(data)
# EUTS_prior(data)
