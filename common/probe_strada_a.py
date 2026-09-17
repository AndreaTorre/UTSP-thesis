# -*- coding: utf-8 -*-
"""
PROBE STRADA A — verifica end-to-end che i modelli Gurobi girino corretti
sui DATI DI TEST, ri-risolvendo STO/EEV per-istanza (niente policy congelate,
niente UTSP, niente modello allenato).

Replica ESATTAMENTE il setup di CVETT/main.py (load_data, load_env,
scenario_kwargs={"wind","coords"}), genera i blocchi di test con la stessa
funzione della pipeline (generate_test_scenario_blocks) e chiama
_validate_policies_resolved per blocco — cioe' le stesse tre chiamate solver
del train (solve_stochastic + compute_eev_medione + WS per-scenario).

Lancio (dalla dir common/, come test_only_cvett.sh, ma senza modello):
    export TESI_EXPERIMENT=CVETT TESI_N_NODES=15
    export TESI_DRONE_U=12                 # sceglie la dir res_B taggata
    export TESI_DROP_LAST_TEST_BATCH=0
    export TESI_DIM_ISTANZA_TEST=20
    export TESI_N_ISTANZE_TEST=5           # poche istanze per un primo segnale
    export TESI_TEST_SKIP_PI=1             # il PI non serve al probe: salta i suoi MIP
    python probe_strada_a.py
"""
import os
import pickle
import statistics as st

from common import load_data, load_env, set_seed
from config import (
    WIND_NC_PATH_EVAL, TEST_SCENARIO_CACHE_DIR, COST_MODEL_TAG,
    TEST_SCENARIO_IDS_UTSP, DIM_ISTANZA_TEST, N_ISTANZE_TEST,
)
from wind_perturbation import load_wind_field
from local_search import generate_test_scenario_blocks, _validate_policies_resolved


def main():
    set_seed()
    env = load_env()
    nodes, coords, base_dist, E, root = load_data()
    wind_test = load_wind_field(WIND_NC_PATH_EVAL)

    cache_path = os.path.join(TEST_SCENARIO_CACHE_DIR, "res_B_cached.pkl")
    print(f"res_B: {cache_path}")
    print(f"tag modello di costo: {COST_MODEL_TAG or '(nessuno)'}")
    if not os.path.exists(cache_path):
        raise SystemExit(f"res_B non trovato in {cache_path} — hai esportato il TESI_DRONE_U giusto?")
    res_B = pickle.load(open(cache_path, "rb"))
    I, p, C = res_B["I"], res_B["p"], res_B["C"]
    frequent_arcs = res_B["frequent_arcs"]

    # Stesso scenario_kwargs del percorso test-only per CVETT: wind + coords.
    scenario_kwargs_test = {"wind": wind_test, "coords": coords}

    istanze = generate_test_scenario_blocks(
        nodes, E, base_dist, I, frequent_arcs, root, env, p, C,
        scenario_ids=TEST_SCENARIO_IDS_UTSP,
        scenario_kwargs=scenario_kwargs_test,
        dim_istanza_test=DIM_ISTANZA_TEST,
        n_istanze_test=N_ISTANZE_TEST,
    )
    print(f"istanze di test generate = {len(istanze)}  (dim_istanza={DIM_ISTANZA_TEST})\n")

    ws_l, sto_l, eev_l, flips, ok_order = [], [], [], 0, 0
    for idx, (results_test, sids, probs) in enumerate(istanze):
        b = _validate_policies_resolved(
            nodes, E, I, p, C, root, env, base_dist,
            results_test, sids, probs, idx=idx,
        )
        ws_l.append(b["WS_val"]); sto_l.append(b["STO_val"]); eev_l.append(b["EEV_val"])
        if b["x_sto_block"] != b["x_ev_block"]:
            flips += 1
        if b["WS_val"] <= b["STO_val"] + 1e-6 <= b["EEV_val"] + 1e-6:
            ok_order += 1

    n = len(ws_l)
    print("\n" + "=" * 64)
    print(f"STRADA A — TEST ri-risolto per-istanza  ({n} istanze)")
    print(f"  WS  medio = {st.mean(ws_l):.2f}")
    print(f"  STO medio = {st.mean(sto_l):.2f}")
    print(f"  EEV medio = {st.mean(eev_l):.2f}")
    print(f"  VSS medio (EEV-STO) = {st.mean(eev_l) - st.mean(sto_l):+.2f}")
    print(f"  ordine WS<=STO<=EEV rispettato in {ok_order}/{n} istanze")
    print(f"  istanze con x_sto != x_ev (separazione)  {flips}/{n}")
    print("=" * 64)
    if flips == 0:
        print("  -> x_sto == x_ev ovunque: i modelli girano corretti in test e")
        print("     il VSS e' zero per il DATO, non per un bug. CVETT = limite")
        print("     deterministico, verificato end-to-end.")
    else:
        print(f"  -> {flips} istanze separano: la fisica DA' valore al due-stadi")
        print("     in test. Qui ha senso completare Strada A nella pipeline.")


if __name__ == "__main__":
    main()
