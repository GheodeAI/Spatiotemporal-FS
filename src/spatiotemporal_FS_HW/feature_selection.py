from PyCROSL.CRO_SL import *
from PyCROSL.AbsObjectiveFunc import *
from PyCROSL.SubstrateReal import *
from PyCROSL.SubstrateInt import *

from .optimization import MLprediction

def main():
    objfunc = MLPrediction(3*pred_dataframe.shape[1])

    params = {
        "popSize": 100,
        "rho": 0.6,
        "Fb": 0.98,
        "Fd": 0.2,
        "Pd": 0.8,
        "k": 3,
        "K": 20,
        "group_subs": True,

        "stop_cond": "Neval",
        "time_limit": 4000.0,
        "Ngen": 10000,
        "Neval": 15000,
        "fit_target": 1000,

        "verbose": True,
        "v_timer": 1,
        "Njobs": 1,

        "dynamic": True,
        "dyn_method": "success",
        "dyn_metric": "avg",
        "dyn_steps": 10,
        "prob_amp": 0.01,

        # "prob_file": "prob_history_"+filename+".csv",
        # "popul_file": "last_population"+filename+".csv",
        # "history_file": "fit_history_"+filename+".csv",
        "solution_file": "best_solution_"+filename+".csv",
        # "indiv_file": "indiv_hisotry_"+filename+".csv",
    }

    operators = [
        SubstrateInt("BLXalpha", {"F":0.8}),
        SubstrateInt("Multipoint"),
        SubstrateInt("HS", {"F": 0.7, "Cr":0.8,"Par":0.2}),
        SubstrateInt("Xor"),
    ]

    cro_alg = CRO_SL(objfunc, operators, params)

    solution, obj_value = cro_alg.optimize()

    solution.tofile(path_output+solution_file, sep=',')


