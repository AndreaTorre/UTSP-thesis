# -*- coding: utf-8 -*-
import argparse
import os
import pickle

import numpy as np

from common import load_data, load_env, set_seed
from config import ERA5_NC_PATH_TRAIN, ERA5_NC_PATH_TEST, OUTPUT_DIR, TEST_SCENARIO_CACHE_DIR
from experiment_B import run_esperimento_B_wind
from utsp import run_esperimento_B_UTSP
from wind_perturbation import load_wind_field

# NOTA: era un path assoluto costruito con os.environ['TESI_N_NODES'] (KeyError
# se non esportata). Ora allineato a PERT/main.py: la cache di Esperimento B non
# dipende dal batch size ne' dalla sottocartella di test, quindi vive in
# RISULTATI_N/pkl ed e' condivisa da tutte le combinazioni.
CACHE_PATH = os.path.join(TEST_SCENARIO_CACHE_DIR, "res_B_cached.pkl")


def main():
    parser = argparse.ArgumentParser(description="Esegue Esperimento B e le varianti UTSP.")
    parser.add_argument(
        "--only",
        choices=["B", "B_UTSP_LS"],
        default="B_UTSP_LS",
        help="B = solo esperimento B; B_UTSP_LS = B + heatmap UTSP + local search.",
    )
    parser.add_argument(
        "--diag-tensor",
        action="store_true",
        help="Salva scenario_dist_tensor.npy per analizzare la variabilita' delle perturbazioni.",
    )
    args = parser.parse_args()

    set_seed()
    env = load_env()
    nodes, coords, base_dist, E, root = load_data()

    wind_train = load_wind_field(ERA5_NC_PATH_TRAIN)
    wind_test = load_wind_field(ERA5_NC_PATH_TEST)
    speed = np.hypot(wind_train["u100"][0], wind_train["v100"][0])
    print(f"[WIND TRAIN] istanti: {wind_train['n_times']} | "
          f"griglia: {len(wind_train['lats'])}x{len(wind_train['lons'])} | "
          f"velocita media: {speed.mean():.2f} m/s | max: {speed.max():.2f} m/s")
    print(f"[WIND TEST]  istanti: {wind_test['n_times']} | "
          f"griglia: {len(wind_test['lats'])}x{len(wind_test['lons'])}")

    risultati = {}

    # NOTA: il dump sta DENTRO l'else. Prima era fuori e ripetuto due volte,
    # quindi la cache veniva riscritta anche subito dopo averla letta.
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

    if args.diag_tensor:
        from scenarios import scenario_dist_tensor
        T, _ = scenario_dist_tensor(risultati["B"]["results"], nodes)
        dest = os.path.join(OUTPUT_DIR, "pkl", "scenario_dist_tensor.npy")
        np.save(dest, T)
        print(f"[DIAG] tensore scenari salvato: shape={T.shape} -> {dest}")

    if args.only == "B":
        return risultati

    risultati["B_UTSP"] = run_esperimento_B_UTSP(
        nodes, coords, base_dist, E, root, env,
        res_B=risultati["B"],
        mode="local_search",
        scenario_kwargs={"wind_train": wind_train, "wind_test": wind_test},
        exp_name="espB_wind_UTSP_LS",
    )
    return risultati


if __name__ == "__main__":
    main()
