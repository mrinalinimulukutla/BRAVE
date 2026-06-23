# -------------------------------------
# Code by Sina Zadeh
# Nobember 2023
# https://www.sina.science/
# -------------------------------------

import pandas as pd
from CBFV import composition
from functools import reduce
from HEACalculator import HEACalculator


class DataHandler:
    def __init__(self, data):
        self.df = pd.DataFrame(data)

    def generate_dataframe(self):
        return self.df


class FeatureGenerator:
    def __init__(self, df):
        self.df = df.copy()
        self.ext_df = df.copy()
        self.columns_range = self.df.columns
        self.base_features_generated = False

    @staticmethod
    def stringify(x):
        return str(float(x)) if pd.notnull(x) and x != 0 else ""

    @staticmethod
    def is_almost_zero(num):
        return abs(num) < 1e-9

    @staticmethod
    def greatest_common_divisor(a, b):
        return (
            a
            if FeatureGenerator.is_almost_zero(b)
            else FeatureGenerator.greatest_common_divisor(b, a % b)
        )

    def gcd_of_array(self, array):
        return reduce(self.greatest_common_divisor, array)

    def correct_ratios(self, arr_values):
        gcd = self.gcd_of_array(arr_values)
        return [round(a / gcd) for a in arr_values]

    def to_formula_string(self, dct):
        corrected_vals = self.correct_ratios(list(dct.values()))
        elements = [
            element for element, value in zip(dct.keys(), corrected_vals) if value != 0
        ]
        values = [value for value in corrected_vals if value != 0]
        return "".join(
            [f"{element}{value}" for element, value in zip(elements, values)]
        )

    def generate_composition_formula(self):
        self.df["composition"] = (
            self.df[self.columns_range]
            .apply(lambda x: x.map(self.stringify))
            .apply(
                lambda row: "".join(
                    [f"{k}{v}" for k, v in row.items() if v != "" and v != "0.0"]
                ),
                axis=1,
            )
        )

        dict_alloy_compositions = self.df[self.columns_range].to_dict("split")
        formula_list = [
            self.to_formula_string(
                dict(zip(dict_alloy_compositions["columns"], row_data))
            )
            for row_data in dict_alloy_compositions["data"]
        ]
        self.df["formula"] = formula_list
        return self.df

    def generate_features_ys(self):
        # Generate alloy system
        self.df["Alloy_System"] = (
            self.df.formula.replace("\d+", "", regex=True)
            .replace("AlNiTiZr", "NiTiZrAl")
            .replace("NiRhTi", "NiTiRh")
            .replace("CuNiPdTi", "NiTiPdCu")
            .replace("NiPdPtTi", "NiTiPdPt")
            .replace("NbNiTiZr", "NiTiNbZr")
            .replace("HfNiTiZr", "NiTiHfZr")
            .replace("NiPdTaTi", "NiTiPdTa")
            .replace("CuNiTi", "NiTiCu")
            .replace("NiPtTi", "NiTiPt")
            .replace("HfNiTi", "NiTiHf")
            .replace("AuCuNiTi", "NiTiCuAu")
            .replace("AuNiTi", "NiTiAu")
            .replace("BNiPdTi", "NiTiPdB")
            .replace("BNiTiZr", "NiTiZrB")
            .replace("CoCuNiTi", "NiTiCuCo")
            .replace("CoInMnNi", "NiMnCoIn")
            .replace("CoNiPdTi", "NiTiPdCo")
            .replace("CuFeHfNiTi", "NiTiHfFeCu")
            .replace("CuHfNiTi", "NiTiHfCu")
            .replace("CuHfNiTiZr", "NiTiHfZrCu")
            .replace("CuNbNiTi", "NiTiCuNb")
            .replace("CuNiTiZr", "NiTiCuZr")
            .replace("HfNiPdTi", "NiTiPdHf")
            .replace("NbNiTi", "NiTiNb")
            .replace("NiPdScTi", "NiTiPdSc")
            .replace("NiPdTi", "NiTiPd")
            .replace("NiTaTi", "NiTiTa")
            .replace("HfNiTaTi", "NiTiHfTa")
            .replace("NiReTi", "NiTiRe")
            .replace("NiSiTi", "NiTiSi")
            .replace("NiSnTi", "NiTiSn")
            .replace("NiSbTi", "NiTiSb")
            .replace("NiScTi", "NiTiSc")
            .replace("NiTeTi", "NiTiTe")
            .replace("NiPbTi", "NiTiPb")
            .replace("NiPrTi", "NiTiPr")
            .replace("CuNiSiTi", "NiTiCuSi")
            .replace("HfNiSnTi", "NiTiHfSn")
            .replace("NiPbTiZr", "NiTiPbZr")
            .replace("CuHfNiPbTi", "NiTiHfCuPb")
            .replace("CoNiPbTi", "NiTiCoPb")
            .replace("CuNiPbTiZr", "NiTiCuZrPb")
            .replace("CuHfNiPbTiZr", "NiTiCuHfPbZr")
        )
        self.df["niti_base"] = self.df.Alloy_System.apply(
            lambda x: "True" if "NiTi" in x else "False"
        )
        # temp_df = self.df.copy()

        # Generate HEA related features
        # lst = []
        # for alloy in temp_df["composition"]:
        #     lst.append(HEACalculator(alloy, csv=True).get_csv_list())
        # headers = [
        #     "Formula",
        #     "Density",
        #     "Delta",
        #     "Omega",
        #     "Gamma",
        #     "lambda",
        #     "VEC",
        #     "Mixing Enthalpy",
        #     "Mixing Entropy",
        #     "Melting Temperature",
        # ]
        # hea_features_df = pd.DataFrame(lst, columns=headers).iloc[:, 1:]

        # Generate compositional features
        self.df["target"] = 0

        # X_jarvis, _, _, _ = composition.generate_features(
        #     self.df, elem_prop="jarvis", sum_feat=True
        # )
        # X_magpie, _, _, _ = composition.generate_features(
        #     self.df, elem_prop="magpie", sum_feat=True
        # )
        # X_oliynyk, _, _, _ = composition.generate_features(
        #     self.df, elem_prop="oliynyk", sum_feat=True
        # )
        X_mat2vec, _, _, _ = composition.generate_features(
            self.df, elem_prop="mat2vec", sum_feat=True
        )
        # X_onehot, _, _, _ = composition.generate_features(
        #     self.df, elem_prop="onehot", sum_feat=True
        # )
        self.df = pd.concat(
            [
                self.ext_df,
                # X_jarvis.add_prefix("jarvis_"),
                # X_magpie.add_prefix("magpie_"),
                # X_oliynyk.add_prefix("oliynyk_"),
                X_mat2vec.add_prefix("mat2vec_"),
                # X_onehot.add_prefix("onehot_"),
                # hea_features_df.add_prefix("hea_"),
            ],
            axis=1,
        )
        self.df = self.df[["mat2vec_sum_121", "mat2vec_dev_86"]]

        return self.df

    def generate_features_uts_ys(self):
        # Generate alloy system
        self.df["Alloy_System"] = (
            self.df.formula.replace("\d+", "", regex=True)
            .replace("AlNiTiZr", "NiTiZrAl")
            .replace("NiRhTi", "NiTiRh")
            .replace("CuNiPdTi", "NiTiPdCu")
            .replace("NiPdPtTi", "NiTiPdPt")
            .replace("NbNiTiZr", "NiTiNbZr")
            .replace("HfNiTiZr", "NiTiHfZr")
            .replace("NiPdTaTi", "NiTiPdTa")
            .replace("CuNiTi", "NiTiCu")
            .replace("NiPtTi", "NiTiPt")
            .replace("HfNiTi", "NiTiHf")
            .replace("AuCuNiTi", "NiTiCuAu")
            .replace("AuNiTi", "NiTiAu")
            .replace("BNiPdTi", "NiTiPdB")
            .replace("BNiTiZr", "NiTiZrB")
            .replace("CoCuNiTi", "NiTiCuCo")
            .replace("CoInMnNi", "NiMnCoIn")
            .replace("CoNiPdTi", "NiTiPdCo")
            .replace("CuFeHfNiTi", "NiTiHfFeCu")
            .replace("CuHfNiTi", "NiTiHfCu")
            .replace("CuHfNiTiZr", "NiTiHfZrCu")
            .replace("CuNbNiTi", "NiTiCuNb")
            .replace("CuNiTiZr", "NiTiCuZr")
            .replace("HfNiPdTi", "NiTiPdHf")
            .replace("NbNiTi", "NiTiNb")
            .replace("NiPdScTi", "NiTiPdSc")
            .replace("NiPdTi", "NiTiPd")
            .replace("NiTaTi", "NiTiTa")
            .replace("HfNiTaTi", "NiTiHfTa")
            .replace("NiReTi", "NiTiRe")
            .replace("NiSiTi", "NiTiSi")
            .replace("NiSnTi", "NiTiSn")
            .replace("NiSbTi", "NiTiSb")
            .replace("NiScTi", "NiTiSc")
            .replace("NiTeTi", "NiTiTe")
            .replace("NiPbTi", "NiTiPb")
            .replace("NiPrTi", "NiTiPr")
            .replace("CuNiSiTi", "NiTiCuSi")
            .replace("HfNiSnTi", "NiTiHfSn")
            .replace("NiPbTiZr", "NiTiPbZr")
            .replace("CuHfNiPbTi", "NiTiHfCuPb")
            .replace("CoNiPbTi", "NiTiCoPb")
            .replace("CuNiPbTiZr", "NiTiCuZrPb")
            .replace("CuHfNiPbTiZr", "NiTiCuHfPbZr")
        )
        self.df["niti_base"] = self.df.Alloy_System.apply(
            lambda x: "True" if "NiTi" in x else "False"
        )
        temp_df = self.df.copy()

        # # Generate HEA related features
        # lst = []
        # for alloy in temp_df["composition"]:
        #     lst.append(HEACalculator(alloy, csv=True).get_csv_list())
        # headers = [
        #     "Formula",
        #     "Density",
        #     "Delta",
        #     "Omega",
        #     "Gamma",
        #     "lambda",
        #     "VEC",
        #     "Mixing Enthalpy",
        #     "Mixing Entropy",
        #     "Melting Temperature",
        # ]
        # hea_features_df = pd.DataFrame(lst, columns=headers).iloc[:, 1:]

        # Generate compositional features
        self.df["target"] = 0

        X_jarvis, _, _, _ = composition.generate_features(
            self.df, elem_prop="jarvis", sum_feat=True
        )
        X_magpie, _, _, _ = composition.generate_features(
            self.df, elem_prop="magpie", sum_feat=True
        )
        X_oliynyk, _, _, _ = composition.generate_features(
            self.df, elem_prop="oliynyk", sum_feat=True
        )
        X_mat2vec, _, _, _ = composition.generate_features(
            self.df, elem_prop="mat2vec", sum_feat=True
        )
        # X_onehot, _, _, _ = composition.generate_features(
        #     self.df, elem_prop="onehot", sum_feat=True
        # )
        self.df = pd.concat(
            [
                self.ext_df,
                X_jarvis.add_prefix("jarvis_"),
                X_magpie.add_prefix("magpie_"),
                X_oliynyk.add_prefix("oliynyk_"),
                X_mat2vec.add_prefix("mat2vec_"),
                # X_onehot.add_prefix("onehot_"),
                # hea_features_df.add_prefix("hea_"),
            ],
            axis=1,
        )
        self.df = self.df[
            [
                "jarvis_sum_bp_mult_atom_mass",
                "jarvis_sum_polzbl_subs_first_ion_en",
                "jarvis_dev_hfus_divi_atom_rad",
                "jarvis_dev_mol_vol_add_therm_cond",
                "jarvis_dev_mol_vol_mult_atom_mass",
                "jarvis_dev_mp_mult_atom_rad",
                "magpie_dev_NdValence",
                "oliynyk_avg_specific_heat_(J/g_K)_",
                "oliynyk_dev_Number_of_unfilled_d_valence_electrons",
                "oliynyk_dev_Cohesive_energy",
                "mat2vec_sum_5",
                "mat2vec_dev_41",
                "mat2vec_dev_96",
            ]
        ]

        return self.df

    def generate_features_el(self):
        # Generate alloy system
        self.df["Alloy_System"] = (
            self.df.formula.replace("\d+", "", regex=True)
            .replace("AlNiTiZr", "NiTiZrAl")
            .replace("NiRhTi", "NiTiRh")
            .replace("CuNiPdTi", "NiTiPdCu")
            .replace("NiPdPtTi", "NiTiPdPt")
            .replace("NbNiTiZr", "NiTiNbZr")
            .replace("HfNiTiZr", "NiTiHfZr")
            .replace("NiPdTaTi", "NiTiPdTa")
            .replace("CuNiTi", "NiTiCu")
            .replace("NiPtTi", "NiTiPt")
            .replace("HfNiTi", "NiTiHf")
            .replace("AuCuNiTi", "NiTiCuAu")
            .replace("AuNiTi", "NiTiAu")
            .replace("BNiPdTi", "NiTiPdB")
            .replace("BNiTiZr", "NiTiZrB")
            .replace("CoCuNiTi", "NiTiCuCo")
            .replace("CoInMnNi", "NiMnCoIn")
            .replace("CoNiPdTi", "NiTiPdCo")
            .replace("CuFeHfNiTi", "NiTiHfFeCu")
            .replace("CuHfNiTi", "NiTiHfCu")
            .replace("CuHfNiTiZr", "NiTiHfZrCu")
            .replace("CuNbNiTi", "NiTiCuNb")
            .replace("CuNiTiZr", "NiTiCuZr")
            .replace("HfNiPdTi", "NiTiPdHf")
            .replace("NbNiTi", "NiTiNb")
            .replace("NiPdScTi", "NiTiPdSc")
            .replace("NiPdTi", "NiTiPd")
            .replace("NiTaTi", "NiTiTa")
            .replace("HfNiTaTi", "NiTiHfTa")
            .replace("NiReTi", "NiTiRe")
            .replace("NiSiTi", "NiTiSi")
            .replace("NiSnTi", "NiTiSn")
            .replace("NiSbTi", "NiTiSb")
            .replace("NiScTi", "NiTiSc")
            .replace("NiTeTi", "NiTiTe")
            .replace("NiPbTi", "NiTiPb")
            .replace("NiPrTi", "NiTiPr")
            .replace("CuNiSiTi", "NiTiCuSi")
            .replace("HfNiSnTi", "NiTiHfSn")
            .replace("NiPbTiZr", "NiTiPbZr")
            .replace("CuHfNiPbTi", "NiTiHfCuPb")
            .replace("CoNiPbTi", "NiTiCoPb")
            .replace("CuNiPbTiZr", "NiTiCuZrPb")
            .replace("CuHfNiPbTiZr", "NiTiCuHfPbZr")
        )
        self.df["niti_base"] = self.df.Alloy_System.apply(
            lambda x: "True" if "NiTi" in x else "False"
        )
        temp_df = self.df.copy()

        # Generate HEA related features
        lst = []
        for alloy in temp_df["composition"]:
            lst.append(HEACalculator(alloy, csv=True).get_csv_list())
        headers = [
            "Formula",
            "Density",
            "Delta",
            "Omega",
            "Gamma",
            "lambda",
            "VEC",
            "Mixing Enthalpy",
            "Mixing Entropy",
            "Melting Temperature",
        ]
        hea_features_df = pd.DataFrame(lst, columns=headers).iloc[:, 1:]

        # Generate compositional features
        self.df["target"] = 0

        X_jarvis, _, _, _ = composition.generate_features(
            self.df, elem_prop="jarvis", sum_feat=True
        )
        X_magpie, _, _, _ = composition.generate_features(
            self.df, elem_prop="magpie", sum_feat=True
        )
        X_oliynyk, _, _, _ = composition.generate_features(
            self.df, elem_prop="oliynyk", sum_feat=True
        )
        X_mat2vec, _, _, _ = composition.generate_features(
            self.df, elem_prop="mat2vec", sum_feat=True
        )
        X_onehot, _, _, _ = composition.generate_features(
            self.df, elem_prop="onehot", sum_feat=True
        )
        self.df = pd.concat(
            [
                self.ext_df,
                X_jarvis.add_prefix("jarvis_"),
                X_magpie.add_prefix("magpie_"),
                X_oliynyk.add_prefix("oliynyk_"),
                X_mat2vec.add_prefix("mat2vec_"),
                X_onehot.add_prefix("onehot_"),
                hea_features_df.add_prefix("hea_"),
            ],
            axis=1,
        )
        self.df = self.df[
            [
                "jarvis_dev_X_subs_mol_vol",
                "jarvis_mode_bp_divi_first_ion_en",
                "jarvis_mode_elec_aff_mult_mp",
                "jarvis_mode_first_ion_en_mult_elec_aff",
                "jarvis_mode_first_ion_en_mult_mol_vol",
                "jarvis_mode_hfus_subs_polzbl",
                "jarvis_mode_polzbl_mult_mol_vol",
                "jarvis_mode_voro_coord_divi_bp",
                "jarvis_mode_voro_coord_subs_first_ion_en",
                "magpie_dev_MendeleevNumber",
                "oliynyk_avg_specific_heat_(J/g_K)_",
                "oliynyk_dev_Miracle_Radius_[pm]",
                "oliynyk_dev_Density_(g/mL)",
                "oliynyk_dev_heat_of_vaporization_(kJ/mol)_",
                "mat2vec_sum_133",
                "mat2vec_dev_134",
                "mat2vec_dev_156",
                "mat2vec_dev_185",
                "mat2vec_max_126",
                "mat2vec_max_168",
                "mat2vec_max_186",
                "mat2vec_min_16",
                "onehot_dev_24",
                "hea_Delta",
            ]
        ]

        return self.df


def print_cv_summary(cv_data, err_type):
    if err_type == "MultiRMSE":
        best_value = cv_data["test-MultiRMSE-mean"].min()
        best_iter = cv_data["test-MultiRMSE-mean"].values.argmin()
        print(
            "Best validation MultiRMSE score : {:.4f}±{:.4f} on step {}".format(
                best_value, cv_data["test-MultiRMSE-std"][best_iter], best_iter
            )
        )
    if err_type == "RMSE":
        best_value = cv_data["test-RMSE-mean"].min()
        best_iter = cv_data["test-RMSE-mean"].values.argmin()
        print(
            "Best validation RMSE score : {:.4f}±{:.4f} on step {}".format(
                best_value, cv_data["test-RMSE-std"][best_iter], best_iter
            )
        )
    if err_type == "MAE":
        best_value = cv_data["test-MAE-mean"].min()
        best_iter = cv_data["test-MAE-mean"].values.argmin()
        print(
            "Best validation MAE score : {:.4f}±{:.4f} on step {}".format(
                best_value, cv_data["test-MAE-std"][best_iter], best_iter
            )
        )
    if err_type == "R2":
        best_value = cv_data["test-R2-mean"].max()
        best_iter = cv_data["test-R2-mean"].values.argmax()
        print(
            "Best validation R2 score : {:.4f}±{:.4f} on step {}".format(
                best_value, cv_data["test-R2-std"][best_iter], best_iter
            )
        )
