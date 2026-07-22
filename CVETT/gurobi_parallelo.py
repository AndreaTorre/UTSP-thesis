# -*- coding: utf-8 -*-
"""
Esecuzione parallela dell'esperimento B — ramo CVETT (vento ERA5).
Identico a PERT/gurobi_parallelo.py tranne per:
  1. import di WIND_NC_PATH, DRONE_U, load_wind_field
  2. fase_setup: carica wind field, lo passa a find_frequent_arcs e generate_scenarios
  3. fase_validate: carica wind field, lo passa a validate_policies

Uso:
    python gurobi_parallelo.py setup
    python gurobi_parallelo.py pi
    python gurobi_parallelo.py eev
    python gurobi_parallelo.py sto
    python gurobi_parallelo.py assemble
    python gurobi_parallelo.py validate
"""
import argparse
import functools
import os
import pickle
import sys
import time

print = functools.partial(print, flush=True)

# ── Import del progetto ──────────────────────────────────────────
from common import load_data, load_env, set_seed
from config import (
    SCENARIO_IDS, K_MEDOID_NODES, PRENOTAZIONE_FRAC, PENALTY_FRAC,
    N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, FINAL_SCENARIO_SEED,
    N_VALIDATION_SCENARIOS,
    WIND_NC_PATH,   # percorso del file NetCDF del vento
    DRONE_U,        # ground speed nominale del drone in m/s
)
from tsp_utils import base_cost_undirected
from gurobi_models import (
    build_I_from_medoid_outgoing_nodes,
    solve_exact_tsp,
    solve_stochastic,
)
from scenarios import (
    find_frequent_arcs,
    generate_scenarios,
)
from evaluation import (
    compute_eev_medione,
    compute_random_edge_usage_stats,
    print_and_save_summary,
    validate_policies,
)
from wind_perturbation import load_wind_field   # <<< unico import aggiuntivo rispetto a PERT

# ── Percorsi ─────────────────────────────────────────────────────
from config import OUTPUT_DIR, INSTANCE_TAG

PKL_DIR      = os.path.join(OUTPUT_DIR, "pkl")
PARALLEL_DIR = os.path.join(PKL_DIR, "parallel_data")
CACHE_PATH   = os.path.join(PKL_DIR, "res_B_cached.pkl")

def _pkl(name):
    return os.path.join(PARALLEL_DIR, f"{name}.pkl")

def _save(name, data):
    os.makedirs(PARALLEL_DIR, exist_ok=True)
    path = _pkl(name)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    os.replace(tmp, path)
    mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Salvato {path} ({mb:.1f} MB)")

def _load(name):
    path = _pkl(name)
    if not os.path.exists(path):
        print(f"ERRORE: {path} non trovato. Lancia prima la fase corrispondente.")
        sys.exit(1)
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  Caricato {path}")
    return data

def _try_load(name):
    path = _pkl(name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"  Checkpoint trovato: {path} -> salto questo step")
    return data


# ── FASE 0: SETUP ────────────────────────────────────────────────
def fase_setup():
    t0 = time.time()
    print("="*60)
    print("SETUP: costruzione I + calibrazione + perturbazioni (CVETT)")
    print("="*60)

    set_seed()
    env = load_env()
    nodes, coords, base_dist, E, root = load_data()

    scenario_ids = SCENARIO_IDS

    if _try_load("setup") is not None:
        print("\nSETUP gia completo, esco.")
        return

    ckpt1 = _try_load("setup_I")
    if ckpt1 is None:
        I, medoid_info = build_I_from_medoid_outgoing_nodes(nodes, E, base_dist, K_MEDOID_NODES)
        b = {(i, j): base_cost_undirected(base_dist, i, j) for (i, j) in I}
        p = {(i, j): PRENOTAZIONE_FRAC * b[i, j] for (i, j) in I}
        C = {(i, j): PENALTY_FRAC * b[i, j] for (i, j) in I}
        _save("setup_I", {"I": I, "medoid_info": medoid_info, "b": b, "p": p, "C": C})
    else:
        I, medoid_info, b, p, C = ckpt1["I"], ckpt1["medoid_info"], ckpt1["b"], ckpt1["p"], ckpt1["C"]

    print(f"\nNODI MEDOIDI: {medoid_info['medoid_nodes']}")
    print(f"Tratte in I: {medoid_info['n_undirected_tratte']}")
    for (i, j) in I:
        print(f"  {{{i},{j}}} | b={b[i,j]:.4f} | p={p[i,j]:.4f} | C={C[i,j]:.4f}")

    # --- CVETT: carica wind field una sola volta per tutto il setup ---
    print(f"\nCaricamento wind field da {WIND_NC_PATH}...")
    wind = load_wind_field(WIND_NC_PATH)

    print("\nCalibrazione archi frequenti...")
    ckpt2 = _try_load("setup_frequent_arcs")
    if ckpt2 is None:
        # NOTA: verifica che find_frequent_arcs in CVETT/scenarios.py accetti wind e u
        frequent_arcs = find_frequent_arcs(
            nodes, E, base_dist, root, env, I,
        )
        _save("setup_frequent_arcs", {"frequent_arcs": frequent_arcs})
    else:
        frequent_arcs = ckpt2["frequent_arcs"]

    print("\nGenerazione perturbazioni scenario...")
    # NOTA: verifica che generate_scenarios in CVETT/scenarios.py accetti wind e u
    results, scenario_probs, total_random_uses = generate_scenarios(
        scenario_ids, nodes, E, base_dist, I, frequent_arcs,
        N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, FINAL_SCENARIO_SEED,
        root=root, env=env, p=p, C=C,
        solve_pi=True,
        wind=wind,
    )

    _save("setup", {
        "nodes": nodes, "coords": coords, "base_dist": base_dist,
        "E": E, "root": root,
        "I": I, "medoid_info": medoid_info,
        "b": b, "p": p, "C": C,
        "frequent_arcs": frequent_arcs,
        "results": results,
        "scenario_probs": scenario_probs,
        "scenario_ids": scenario_ids,
        "total_random_uses": total_random_uses,
    })

    print(f"\nSETUP completato in {time.time()-t0:.1f}s")


# ── FASI 1-4: identiche a PERT ───────────────────────────────────
# (nessuna differenza: usano results/scenario_dist gia calcolati dal setup)

def fase_pi():
    t0 = time.time()
    print("="*60)
    print("PI: TSP esatto per ogni scenario")
    print("="*60)

    set_seed()
    env = load_env()
    s = _load("setup")

    nodes = s["nodes"]; E = s["E"]; root = s["root"]
    results = s["results"]; scenario_ids = s["scenario_ids"]

    pi_results = {}
    for idx, sid in enumerate(scenario_ids):
        print(f"  PI scenario {sid} ({idx+1}/{len(scenario_ids)})...", end=" ")
        t_s = time.time()
        exact_free = solve_exact_tsp(
            nodes, E, results[sid]["scenario_dist"], root, env,
            fixed_arcs=[], fixed_edges_undir=[], output_flag=0,
        )
        dt = time.time() - t_s
        length = exact_free["length"]
        print(f"{'OK' if length else 'FAIL'} | PI = {length or 'N/A'} | {dt:.1f}s")
        pi_results[sid] = exact_free

    _save("pi", {"pi_results": pi_results})
    print(f"\nPI completato in {time.time()-t0:.1f}s")


def fase_eev():
    t0 = time.time()
    print("="*60)
    print("EEV: politica deterministica sullo scenario medio")
    print("="*60)

    set_seed()
    env = load_env()
    s = _load("setup")

    nodes = s["nodes"]; E = s["E"]; root = s["root"]
    base_dist = s["base_dist"]
    I = s["I"]; p = s["p"]; C = s["C"]
    results = s["results"]; scenario_ids = s["scenario_ids"]

    tour_medio, arcs_medio, x_ev, eev_costs, eev_tour_costs, eev_penalty_costs, eev_solutions, EEV, _PI_placeholder = compute_eev_medione(
        nodes, E, root, env, base_dist, I, p, C, results, scenario_ids,
        return_solutions=True,
    )

    _save("eev", {
        "tour_medio": tour_medio, "arcs_medio": arcs_medio,
        "x_ev": x_ev,
        "eev_costs": eev_costs, "eev_tour_costs": eev_tour_costs,
        "eev_penalty_costs": eev_penalty_costs,
        "eev_solutions": eev_solutions, "EEV": EEV,
    })

    print(f"\nEEV = {EEV:.4f}")
    print(f"EEV completato in {time.time()-t0:.1f}s")


def fase_sto():
    t0 = time.time()
    print("="*60)
    print("STO: modello stocastico a due stadi")
    print("="*60)

    set_seed()
    env = load_env()
    s = _load("setup")

    nodes = s["nodes"]; E = s["E"]; root = s["root"]
    base_dist = s["base_dist"]
    I = s["I"]; p = s["p"]; C = s["C"]
    results = s["results"]
    scenario_ids = s["scenario_ids"]
    scenario_probs = s["scenario_probs"]

    scenario_deltas = {sid: results[sid]["pert"] for sid in scenario_ids}

    from config import STO_TIME_LIMIT, STO_MIP_GAP
    print(f"  TimeLimit = {STO_TIME_LIMIT}s ({STO_TIME_LIMIT/3600:.1f}h)")
    print(f"  MIPGap    = {STO_MIP_GAP}")
    print(f"  Inizio: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    res_stoch = solve_stochastic(
        nodes, E, I, base_dist, root, p, C, env,
        scenario_deltas, scenario_probs, force_important=False,
    )

    dt = time.time() - t0
    print(f"  Fine: {time.strftime('%Y-%m-%d %H:%M:%S')} ({dt:.0f}s = {dt/3600:.2f}h)")
    if res_stoch["objective"] is not None:
        print(f"  STO objective = {res_stoch['objective']:.4f}")
        print(f"  STO status    = {res_stoch.get('solver_info', {}).get('status', 'N/A')}")

    _save("sto", {"res_stoch": res_stoch})
    print(f"\nSTO completato in {dt:.1f}s")


def fase_assemble():
    t0 = time.time()
    print("="*60)
    print("ASSEMBLE: unione risultati -> res_B_cached.pkl")
    print("="*60)

    s   = _load("setup")
    pi  = _load("pi")
    eev = _load("eev")
    sto = _load("sto")

    scenario_ids   = s["scenario_ids"]
    results        = s["results"]
    scenario_probs = s["scenario_probs"]
    I = s["I"]; b = s["b"]; p = s["p"]; C = s["C"]

    for sid in scenario_ids:
        results[sid]["exact_free"] = pi["pi_results"][sid]

    pi_lengths = [
        results[sid]["exact_free"]["length"]
        for sid in scenario_ids
        if results[sid]["exact_free"]["length"] is not None
    ]
    PI = sum(pi_lengths) / len(scenario_ids)

    res_stoch         = sto["res_stoch"]
    STO               = res_stoch["objective"]
    stoch_solver_info = res_stoch.get("solver_info", {})
    x_used_set        = set(res_stoch["x_used"])
    stoch_costs       = {sid: res_stoch["scenario_solutions"][sid]["total_cost"]   for sid in scenario_ids}
    stoch_tour_c      = {sid: res_stoch["scenario_solutions"][sid]["tour_cost"]    for sid in scenario_ids}
    stoch_penalty_c   = {sid: res_stoch["scenario_solutions"][sid]["penalty_paid"] for sid in scenario_ids}

    STO_recomputed = sum(scenario_probs[sid] * stoch_costs[sid] for sid in scenario_ids)
    print(f"\nControllo STO:")
    print(f"  da modello     = {STO:.6f}")
    print(f"  ricalcolato    = {STO_recomputed:.6f}")
    print(f"  differenza     = {abs(STO - STO_recomputed):.8f}")

    EEV               = eev["EEV"]
    x_ev              = eev["x_ev"]
    eev_costs         = eev["eev_costs"]
    eev_tour_costs    = eev["eev_tour_costs"]
    eev_penalty_costs = eev["eev_penalty_costs"]
    eev_solutions     = eev["eev_solutions"]
    tour_medio        = eev["tour_medio"]
    arcs_medio        = eev["arcs_medio"]

    random_impact_train = compute_random_edge_usage_stats(
        results, scenario_ids,
        stoch_solutions=res_stoch.get("scenario_solutions", {}),
        eev_solutions=eev_solutions,
    )

    print_and_save_summary(
        "espB", scenario_ids, results,
        eev_costs, eev_tour_costs, eev_penalty_costs,
        stoch_costs, stoch_tour_c, stoch_penalty_c,
        PI, STO, EEV,
        I=I, p=p, C=C, b=b,
        stoch_solver_info=stoch_solver_info,
        frequent_arcs=s["frequent_arcs"],
        total_random_uses=s["total_random_uses"],
        random_impact_stats=random_impact_train,
    )

    res_B = {
        "I": I, "b": b, "p": p, "C": C,
        "results": results,
        "scenario_probs": scenario_probs,
        "frequent_arcs": s["frequent_arcs"],
        "x_ev": x_ev,
        "x_used_sto": x_used_set,
        "eev_costs": eev_costs, "eev_tour_costs": eev_tour_costs,
        "eev_penalty_costs": eev_penalty_costs, "eev_solutions": eev_solutions,
        "stoch_costs": stoch_costs, "stoch_tour_costs": stoch_tour_c,
        "stoch_penalty_costs": stoch_penalty_c,
        "stoch_solutions": res_stoch["scenario_solutions"],
        "res_stoch": res_stoch, "stoch_solver_info": stoch_solver_info,
        "tour_medio": tour_medio, "arcs_medio": arcs_medio,
        "PI": PI, "STO": STO, "EEV": EEV,
        "total_random_uses": s["total_random_uses"],
        "random_impact_stats": random_impact_train,
    }

    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(res_B, f)
    os.replace(tmp, CACHE_PATH)
    mb = os.path.getsize(CACHE_PATH) / 1024 / 1024
    print(f"\n  res_B_cached.pkl salvato: {CACHE_PATH} ({mb:.1f} MB)")
    print(f"\nASSEMBLE completato in {time.time()-t0:.1f}s")


def fase_validate():
    t0 = time.time()
    print("="*60)
    print(f"VALIDATE: {N_VALIDATION_SCENARIOS} scenari out-of-sample")
    print("="*60)

    set_seed()
    env = load_env()
    s   = _load("setup")
    eev = _load("eev")
    sto = _load("sto")

    nodes = s["nodes"]; E = s["E"]; root = s["root"]
    base_dist = s["base_dist"]
    I = s["I"]; p = s["p"]; C = s["C"]

    x_used_set = set(sto["res_stoch"]["x_used"])
    x_ev       = eev["x_ev"]

    # --- CVETT: wind field necessario per generare scenari di validazione ---
    print(f"Caricamento wind field da {WIND_NC_PATH}...")
    wind = load_wind_field(WIND_NC_PATH)

    # NOTA: verifica che validate_policies in CVETT/evaluation.py accetti wind e u
    validate_policies(
        nodes, E, base_dist, root, env, I, p, C,
        x_used_set, x_ev, s["frequent_arcs"], N_VALIDATION_SCENARIOS,
        N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, exp_name="espB",
        wind=wind,
    )

    print(f"\nVALIDATE completato in {time.time()-t0:.1f}s")


# ── CLI ──────────────────────────────────────────────────────────
FASI = {
    "setup":    fase_setup,
    "pi":       fase_pi,
    "eev":      fase_eev,
    "sto":      fase_sto,
    "assemble": fase_assemble,
    "validate": fase_validate,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog="Ordine: setup -> (pi | eev | sto) in parallelo -> assemble -> validate",
    )
    parser.add_argument("fase", choices=FASI.keys())
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"gurobi_parallelo.py CVETT — fase: {args.fase}")
    print(f"  Istanza:      nodi_{INSTANCE_TAG}")
    print(f"  DRONE_U:       {DRONE_U} m/s")
    print(f"  WIND_NC_PATH: {WIND_NC_PATH}")
    print(f"  Cache finale: {CACHE_PATH}")
    print(f"{'='*60}\n")

    FASI[args.fase]()
