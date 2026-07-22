#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
solve_pool_shard.py — risolve PI, STO, EEV e WS per UNA shard del pool di test.

FILOSOFIA: un pool unico di N scenari (default 70.000) risolti UNA volta sola
con tutti e quattro i modelli Gurobi. Ogni risultato e' per scenario_id e
INDIPENDENTE dagli altri scenari:
  - PI  = TSP libero (solve_exact_tsp)
  - STO = policy x_used_sto fissa, recourse per scenario (solve_reservation_tsp)
  - EEV = policy x_ev fissa, recourse per scenario
  - WS  = x libera per scenario (solve_reservation_tsp, fixed_reservations=None)
Nessuno di questi dipende da come raggruppi gli scenari: quindi il pool si
affetta a posteriori in QUALSIASI combinazione IS x dim facendo medie di
sottoinsiemi (vedi slice_pool.py).

COERENZA CON LA RETE: gli scenari sono generati con (base_seed, scenario_id)
esattamente come li genererà il test di UTSP. Stesso seme + stesso id =>
stesso scenario. Quando la rete avrà finito il training, il suo test userà
gli stessi scenario_id e i risultati Gurobi qui calcolati si agganciano senza
ricalcolo. La generazione scenari è CONDIVISA tra i quattro modelli: gli
scenari si generano una volta per shard, non quattro.

EFFICIENZA: sharding a passo n_shards (carico bilanciato), salvataggio
incrementale (riprende dopo timeout), scrittura su file dedicato per shard
(nessuna scrittura concorrente). Ogni modello e' indipendente e cachato:
rilanciare salta ciò che è già risolto.

Uso (un job SLURM per shard):
  TESI_EXPERIMENT=PERT TESI_N_NODES=15 TESI_N_TEST_SCENARIOS_UTSP=70000 \
  python solve_pool_shard.py --shard 0 --n-shards 100

Poi:
  TESI_EXPERIMENT=PERT TESI_N_NODES=15 python merge_pool_shards.py
"""
import argparse
import os
import pickle
import sys
import time

from config import (TEST_SCENARIO_CACHE_DIR, TEST_SCENARIO_IDS_UTSP,
                    TEST_SCENARIO_SEED, PI_TIME_LIMIT,
                    N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC)
from common import load_env, load_data
from scenarios import generate_scenarios
from gurobi_models import solve_exact_tsp, solve_reservation_tsp
from tsp_utils import get_edge_value

# Benchmark => esatti per default (vedi nota in compute_pi_shard).
MIP_GAP = float(os.getenv("TESI_POOL_MIP_GAP", "0.0"))
_gap = MIP_GAP if MIP_GAP > 0 else None

# Quali modelli risolvere (per lanciarne solo alcuni, se serve).
MODELS = os.getenv("TESI_POOL_MODELS", "PI,STO,EEV,WS").split(",")


def shard_dir():
    return os.path.join(TEST_SCENARIO_CACHE_DIR, "pool_shards")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    args = ap.parse_args()
    if not (0 <= args.shard < args.n_shards):
        sys.exit(f"--shard in [0, {args.n_shards})")

    resb_path = os.path.join(TEST_SCENARIO_CACHE_DIR, "res_B_cached.pkl")
    if not os.path.exists(resb_path):
        sys.exit(f"res_B non trovato: {resb_path}. Serve Experiment B.")
    with open(resb_path, "rb") as f:
        res_B = pickle.load(f)
    I, p, C = res_B["I"], res_B["p"], res_B["C"]
    frequent_arcs = res_B["frequent_arcs"]
    x_sto = list(res_B["x_used_sto"])
    x_ev = list(res_B["x_ev"])
    reserv_sto = sum(get_edge_value(p, i, j) for (i, j) in x_sto)
    reserv_ev = sum(get_edge_value(p, i, j) for (i, j) in x_ev)

    ids = TEST_SCENARIO_IDS_UTSP[args.shard::args.n_shards]
    print(f"Shard {args.shard}/{args.n_shards}: {len(ids)} scenari  "
          f"(pool totale {len(TEST_SCENARIO_IDS_UTSP)}, seed {TEST_SCENARIO_SEED})")
    print(f"  modelli: {MODELS}   mip_gap {MIP_GAP} (0=esatto)   time_limit {PI_TIME_LIMIT}s")

    os.makedirs(shard_dir(), exist_ok=True)
    out_path = os.path.join(shard_dir(), f"pool_shard_{args.shard}_of_{args.n_shards}.pkl")
    solved = {}
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            solved = pickle.load(f)
        print(f"  ripresa: {len(solved)} scenari già in questa shard")

    nodes, coords, base_dist, E, root = load_data()

    # Genera SOLO gli scenari non ancora completi (tutti e 4 i modelli presenti).
    def is_done(rec):
        return all(m in rec for m in MODELS)

    todo = [sid for sid in ids if not is_done(solved.get(sid, {}))]
    print(f"  da risolvere: {len(todo)}")
    if not todo:
        print("  Nulla da fare.")
        return

    # Rilascio del token a blocchi: invece di tenere un Env aperto per tutta la
    # shard (bloccando un token per ore), lo apro per un blocco di BLOCK scenari,
    # poi lo chiudo. Tra un blocco e l'altro il token torna libero per altri job;
    # load_env riacquisisce con retry paziente (aspetta se il pool è saturo).
    # BLOCK bilancia due costi: troppo piccolo = riconnessioni continue al token
    # server; troppo grande = token trattenuto a lungo. 50 è un compromesso.
    BLOCK = int(os.getenv("TESI_POOL_ENV_BLOCK", "50"))

    t0 = time.time()
    tl = {m: 0 for m in MODELS}
    done_count = 0

    for block_start in range(0, len(todo), BLOCK):
        block = todo[block_start:block_start + BLOCK]
        env = load_env()          # acquisisce un token (aspetta se serve)
        try:
            for sid in block:
                rec = solved.get(sid, {})
                if "dist" not in rec:
                    gen, _, _ = generate_scenarios(
                        [sid], nodes, E, base_dist, I, frequent_arcs,
                        N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, TEST_SCENARIO_SEED,
                        root=root, env=env, p=p, C=C, solve_pi=False)
                    rec["dist"] = gen[sid]["scenario_dist"]
                d = rec["dist"]

                if "PI" in MODELS and "PI" not in rec:
                    r = solve_exact_tsp(nodes, E, d, root, env, output_flag=0,
                                        time_limit=PI_TIME_LIMIT, mip_gap=_gap)
                    rec["PI"] = {"cost": r.get("length"), "status": r.get("status")}
                    if r.get("status") == "TIME_LIMIT":
                        tl["PI"] += 1

                if "STO" in MODELS and "STO" not in rec:
                    r = solve_reservation_tsp(nodes, E, I, d, root, p, C, env,
                                              fixed_reservations=x_sto, output_flag=0,
                                              model_name=f"sto_{sid}",
                                              time_limit=PI_TIME_LIMIT, mip_gap=_gap)
                    rec["STO"] = {"cost": reserv_sto + (r.get("tour_cost") or 0)
                                  + (r.get("penalty_paid") or 0), "status": r.get("status")}
                    if r.get("status") == "TIME_LIMIT":
                        tl["STO"] += 1

                if "EEV" in MODELS and "EEV" not in rec:
                    r = solve_reservation_tsp(nodes, E, I, d, root, p, C, env,
                                              fixed_reservations=x_ev, output_flag=0,
                                              model_name=f"eev_{sid}",
                                              time_limit=PI_TIME_LIMIT, mip_gap=_gap)
                    rec["EEV"] = {"cost": reserv_ev + (r.get("tour_cost") or 0)
                                  + (r.get("penalty_paid") or 0), "status": r.get("status")}
                    if r.get("status") == "TIME_LIMIT":
                        tl["EEV"] += 1

                if "WS" in MODELS and "WS" not in rec:
                    r = solve_reservation_tsp(nodes, E, I, d, root, p, C, env,
                                              fixed_reservations=None, output_flag=0,
                                              model_name=f"ws_{sid}",
                                              time_limit=PI_TIME_LIMIT, mip_gap=_gap)
                    rec["WS"] = {"cost": r.get("total_cost"),
                                 "x_used": r.get("x_used", []), "status": r.get("status")}
                    if r.get("status") == "TIME_LIMIT":
                        tl["WS"] += 1

                solved[sid] = rec
                done_count += 1
        finally:
            env.dispose()         # RILASCIA il token: libero per altri job

        # salvataggio dopo ogni blocco (il token è già rilasciato)
        tmp = out_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(solved, f)
        os.replace(tmp, out_path)
        el = time.time() - t0
        rate = done_count / el if el else 0
        eta = (len(todo) - done_count) / rate if rate else 0
        print(f"  [{done_count}/{len(todo)}] {rate:.2f} scen/s  {el/60:.1f}m  "
              f"ETA {eta/60:.1f}m  timeout {dict(tl)}  (token rilasciato)", flush=True)

    print(f"\nShard {args.shard} completata: {len(solved)} scenari, "
          f"{(time.time()-t0)/60:.1f} min, timeout {dict(tl)}")
    print(f"  scritta: {out_path}")


if __name__ == "__main__":
    main()