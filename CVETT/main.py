# -*- coding: utf-8 -*-

import argparse
import hashlib
import os
import pickle
import numpy as np
import torch

from common import load_data, load_env, set_seed
from config import ERA5_NC_PATH_TRAIN, ERA5_NC_PATH_TEST, OUTPUT_DIR
from experiment_B import run_esperimento_B, run_esperimento_B_wind
from utsp import run_esperimento_B_UTSP
from wind_perturbation import load_wind_field

CACHE_PATH = f"/home/atorre/UTSP/unione/git/UTSP/CVETT/RISULTATI_{os.environ['TESI_N_NODES']}/pkl/res_B_cached.pkl"

def main():
    parser = argparse.ArgumentParser(description="Esegue Esperimento B e le varianti UTSP.")
    parser.add_argument(
        "--only",
        choices=["B", "B_UTSP_LS"],
        default="B_UTSP_LS",
        help=(
            "B = solo esperimento B; "
            "B_UTSP_LS = B + heatmap UTSP + local search."
        ),
    )
    args = parser.parse_args()

    set_seed()
        
    env = load_env()
    nodes, coords, base_dist, E, root = load_data()
    wind_train = load_wind_field(ERA5_NC_PATH_TRAIN)
    wind_test = load_wind_field(ERA5_NC_PATH_TEST)
    
    u = wind_train["u100"][0]
    v = wind_train["v100"][0]
    speed = np.sqrt(u**2 + v**2)
    
    print(f"[WIND TRAIN] istanti temporali: {wind_train['n_times']}")
    print(f"[WIND TRAIN] griglia: {len(wind_train['lats'])} lat x {len(wind_train['lons'])} lon")
    print(f"[WIND TRAIN] velocita media: {speed.mean():.2f} m/s | max: {speed.max():.2f} m/s")
    
    print(f"[WIND TEST] istanti temporali: {wind_test['n_times']}")
    print(f"[WIND TEST] griglia: {len(wind_test['lats'])} lat x {len(wind_test['lons'])} lon")
    risultati = {}

    if os.path.exists(CACHE_PATH):
        print(f"Carico res_B da file .pkl: {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            risultati["B"] = pickle.load(f)
    else:
        print("File .pkl non trovato. Eseguo esperimento B...")
        risultati["B"] = run_esperimento_B_wind(nodes, coords, base_dist, E, root, env, wind_train)
    

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    with open(CACHE_PATH, "wb") as f:
    	pickle.dump(risultati["B"], f)

    print(f"res_B salvato in: {CACHE_PATH}")
    
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(risultati["B"], f)
    print(f"res_B salvato in: {CACHE_PATH}")

    # NOTA: tensore diagnostico per analizzare la variabilità delle perturbazioni
    # ERA5 tra scenari (causa possibile del gap nullo STO/EEV). Usa i risultati
    # già calcolati in res_B, nessuna rigenerazione scenari.
    from scenarios import scenario_dist_tensor
    T, scenario_ids = scenario_dist_tensor(risultati["B"]["results"], nodes)
    np.save(os.path.join(OUTPUT_DIR, "pkl", "scenario_dist_tensor.npy"), T)
    print(f"[DIAG] tensore scenari salvato: shape={T.shape} -> "
          f"{os.path.join(OUTPUT_DIR, 'pkl', 'scenario_dist_tensor.npy')}")

    if args.only == "B":
        return risultati
    
        
    if args.only == "B":
        return risultati

    if args.only == "B_UTSP_LS":
        mode = "local_search"
    else:
        mode = "local_search"

    risultati["B_UTSP"] = run_esperimento_B_UTSP(
        nodes, coords, base_dist, E, root, env,
        res_B=risultati["B"],
        mode=mode,
        scenario_kwargs={ "wind_train": wind_train, "wind_test": wind_test,},
        exp_name="espB_wind_UTSP_LS",
    )

    return risultati

 

if __name__ == "__main__":
    main()
