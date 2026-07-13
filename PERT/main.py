# -*- coding: utf-8 -*-
import argparse
import pickle
from common import load_data, load_env, set_seed
from experiment_B import run_esperimento_B
import torch  
from utsp import run_esperimento_B_UTSP
import os
import hashlib



from config import TEST_SCENARIO_CACHE_DIR

# NOTA: fisso su TEST_SCENARIO_CACHE_DIR (= RISULTATI_N/pkl), non su OUTPUT_DIR:
# Experiment B (STO/EEV/PI) non dipende da UTSP_BATCH_SIZE né dalla sottocartella
# di test, quindi va condiviso tra tutti i BATCH_X e tutte le combinazioni
# IS/DIM. Con OUTPUT_DIR ogni combinazione avrebbe rifatto STO/EEV da zero
# (STO_TIME_LIMIT arriva a 12h su 40 nodi).
CACHE_PATH = os.path.join(TEST_SCENARIO_CACHE_DIR, "res_B_cached.pkl")
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
    
    risultati = {}

    if os.path.exists(CACHE_PATH):
        print(f"Carico res_B da file .pkl: {CACHE_PATH}")
        with open(CACHE_PATH, "rb") as f:
            risultati["B"] = pickle.load(f)
    else:
        print("File .pkl non trovato. Eseguo esperimento B...")
        risultati["B"] = run_esperimento_B(nodes, coords, base_dist, E, root, env)
    
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "wb") as f:
            pickle.dump(risultati["B"], f)
    
        print(f"res_B salvato in: {CACHE_PATH}")
    
        
    if args.only == "B":
        return risultati

    if args.only == "B_UTSP_LS":
        mode = "local_search"
    else:
        mode = "local_search"

    risultati["B_UTSP"] = run_esperimento_B_UTSP(
        nodes, coords, base_dist, E, root, env, res_B=risultati["B"],
        mode=mode)

    return risultati

 

if __name__ == "__main__":
    main()
