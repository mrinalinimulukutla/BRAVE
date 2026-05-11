import numpy as np
import pandas as pd
from tc_python import *
from itertools import compress
from tc_python import server
import concurrent.futures
import os.path as path
import os, time

def Property(param):
    indices = param["INDICES"]
    comp_df = param["COMP"]
    elements = param["ACT_EL"]
    prev_active_el = []
    with TCPython() as session:

        eq_calculation = None
        prop_calc = None
        
        for i in indices:
            comp = comp_df.loc[i][elements]
            active_el = list(compress(elements, list(comp > 0)))
            comp = np.array(comp_df.loc[i][active_el])
            
            if active_el != prev_active_el:
                prev_active_el = active_el

                eq_calculation = (
                    session.select_database_and_elements('TCHEA6', active_el)
                            .get_system()
                            .with_property_model_calculation("Liquidus and Solidus Temperature")
                            .set_argument('upperTemperatureLimit', 2200)
                            .set_composition_unit(CompositionUnit.MOLE_FRACTION))
                
                prop_calc = (
                    session.select_database_and_elements('TCHEA6', active_el)
                            .get_system()
                            .with_property_model_calculation('Equilibrium with Freeze-in Temperature')
                            .set_composition_unit(CompositionUnit.MOLE_FRACTION))
            
            for j in range(len(active_el) - 1):
                eq_calculation = eq_calculation.set_composition(active_el[j], comp[j])
                prop_calc = prop_calc.set_composition(active_el[j], comp[j])
            
            try:
                result = eq_calculation.calculate()
                prop_result = prop_calc.set_argument('Freeze-in-temperature', 298) \
                                      .set_argument('Minimization strategy', 'Global minimization only') \
                                      .set_temperature(298) \
                                      .calculate()
                comp_df.at[i, 'PROP LT (K)'] = result.get_value_of('Liquidus temperature')
                comp_df.at[i, 'PROP ST (K)'] = result.get_value_of('Solidus temperature')
            except Exception as e:
                print(f'Error in calculation for index {i}: {e}')
            
        comp_df.to_csv(f'CalcFiles/PROP_OUT_{param["INDICES"][0]}.csv', index=False)
        print(f'Completed: PROP_OUT_{param["INDICES"][0]}.csv')
    
    return 'Complete'

if __name__ == '__main__':
    
    tic = time.time()
    filename = 'htmdecyear2_n8_d25_subset_n8'
    results_df = pd.read_csv(filename+'.csv')
    elements =	['Al', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu']

    if not path.exists("CalcFiles"):
        os.mkdir("CalcFiles")

    indices = results_df.index
    prev_active_el = []
    parameters = []
    count = 0
    
    cpurows = int(results_df.shape[0] / 20) + 1 # number of rows per cpu

    for i in indices:
        comp = results_df.loc[i][elements]
        active_el = list(compress(elements, list(comp > 0)))

        if (active_el != prev_active_el) or (count == cpurows):
            try:
                new_calc_dict["COMP"] = results_df.loc[new_calc_dict["INDICES"]]
                new_calc_dict["ACT_EL"] = prev_active_el
                if not os.path.exists(
                        "CalcFiles/{}-Results-Set-{}".format('EQUIL', new_calc_dict["INDICES"][0])):
                    parameters.append(new_calc_dict)
                new_calc_dict = {"INDICES": [],
                                  "COMP": [],
                                  "ACT_EL": []}
            except Exception as e:
                new_calc_dict = {"INDICES": [],
                                  "COMP": [],
                                  "ACT_EL": []}
            count = 0

        new_calc_dict["INDICES"].append(i)
        prev_active_el = active_el
        count += 1

    # add the last calculation set
    new_calc_dict["COMP"] = results_df.loc[new_calc_dict["INDICES"]]
    new_calc_dict["ACT_EL"] = prev_active_el
    if not os.path.exists("CalcFiles/{}-Results-Set-{}".format('PROP', new_calc_dict["INDICES"][0])):
        parameters.append(new_calc_dict)

    completed_calculations = []

    with concurrent.futures.ProcessPoolExecutor(20) as executor:
        for result_from_process in zip(parameters, executor.map(Property, parameters)):
            # params can be used to identify the process and its parameters
            params, results = result_from_process
            if results == "Calculation Completed":
                completed_calculations.append('Completed')
                
    toc = time.time()
    time_to_evaluate = round(toc - tic, 3)
    print('\ntime to evaluate: ',time_to_evaluate,'\n')