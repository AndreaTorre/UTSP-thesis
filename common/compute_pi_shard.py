#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_pi_shard.py — calcola il PI (TSP libero per scenario) di UNA shard
degli scenari di test, in parallelo e senza toccare la cache condivisa.

Perché sharded: il fill-in dentro generate_test_scenario_blocks riscrive
l'intero test_scenarios_cache.pkl (117 MB). Con N job in parallelo l'ultimo
che scrive cancella il lavoro degli altri. Qui ogni job scrive SOLO il proprio
file pi_shards/pi_shard_<k>.pkl; merge_pi_shards.py li fonde alla fine.

Non serve la GNN, non serve res_B: il PI è un TSP libero sulla scenario_dist
già salvata in cache. Solo Gurobi.

Uso (un job per shard):
  TESI_EXPERIMENT=PERT TESI_N_NODES=15 \
  python compute_pi_shard.py --shard 0 --n-shards 20

Poi, una volta finite tutte:
  python merge_pi_shards.py            # con lo stesso EXPERIMENT/N_NODES
"""
import argparse
import os
import pickle
import sys
import time

from config import TEST_SCENARIO_CACHE_DIR, PI_TIME_LIMIT
from common import load_env, load_data
from gurobi_models import solve_exact_tsp

# NOTA: il PI è il bound inferiore usato per calcolare i gap. PI_MIP_GAP in
# config vale 0.08 ma è pensato per la CALIBRAZIONE ("basta un tour buono").
# Con l'8% di tolleranza il PI potrebbe stare fino all'8% sopra l'ottimo vero,
# rendendo il "gap vs PI" privo di significato (i gap misurati sono 1-7%).
# Qui il default è esatto; alza TESI_PI_MIP_GAP solo se i tempi esplodono.
PI_MIP_GAP_BENCH = float(os.getenv("TESI_PI_MIP_GAP", "0.0"))


def shard_dir():
    return os.path.join(TEST_SCENARIO_CACHE_DIR, "pi_shards")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    args = ap.parse_args()

    if not (0 <= args.shard < args.n_shards):
        sys.exit(f"--shard deve stare in [0, {args.n_shards})")

    cache_path = os.path.join(TEST_SCENARIO_CACHE_DIR, "test_scenarios_cache.pkl")
    if not os.path.exists(cache_path):
        sys.exit(f"Cache scenari non trovata: {cache_path}\n"
                 "Serve almeno un run del test sweep che abbia generato gli scenari.")

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    results = cache["results"]

    # Scenari di QUESTA shard, presi a passo n_shards: così ogni shard prende
    # scenari sparsi e il carico si bilancia anche se la difficoltà varia
    # lungo gli id.
    all_ids = sorted(results.keys())
    my_ids = all_ids[args.shard::args.n_shards]

    todo = [sid for sid in my_ids
            if results[sid].get("exact_free", {}).get("length") is None]

    print(f"Shard {args.shard}/{args.n_shards}: {len(my_ids)} scenari assegnati, "
          f"{len(todo)} senza PI da risolvere")
    print(f"  mip_gap = {PI_MIP_GAP_BENCH} (0 = esatto)   time_limit = {PI_TIME_LIMIT}s")

    if not todo:
        print("  Nulla da fare.")
        return

    os.makedirs(shard_dir(), exist_ok=True)
    out_path = os.path.join(shard_dir(), f"pi_shard_{args.shard}_of_{args.n_shards}.pkl")

    # Riprendi se il job era già partito ed è stato interrotto.
    solved = {}
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            solved = pickle.load(f)
        todo = [sid for sid in todo if sid not in solved]
        print(f"  Shard parziale trovata: {len(solved)} già risolti, restano {len(todo)}")

    nodes, coords, base_dist, E, root = load_data()
    env = load_env()

    t0 = time.time()
    n_timeout = 0
    for k, sid in enumerate(todo, start=1):
        res = solve_exact_tsp(
            nodes, E, results[sid]["scenario_dist"], root, env,
            output_flag=0, time_limit=PI_TIME_LIMIT,
            mip_gap=(PI_MIP_GAP_BENCH if PI_MIP_GAP_BENCH > 0 else None),
        )
        solved[sid] = res
        if res.get("status") == "TIME_LIMIT":
            n_timeout += 1

        if k % 50 == 0 or k == len(todo):
            el = time.time() - t0
            rate = k / el
            eta = (len(todo) - k) / rate if rate > 0 else 0
            print(f"  [{k}/{len(todo)}] {rate:.1f} scen/s  "
                  f"trascorsi {el/60:.1f}m  ETA {eta/60:.1f}m  timeout finora: {n_timeout}",
                  flush=True)
            # Salvataggio incrementale: un timeout SLURM non perde il lavoro.
            tmp = out_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(solved, f)
            os.replace(tmp, out_path)

    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(solved, f)
    os.replace(tmp, out_path)

    print(f"\nShard {args.shard} completata: {len(solved)} PI risolti in "
          f"{(time.time()-t0)/60:.1f} minuti ({n_timeout} in time limit)")
    print(f"  scritta: {out_path}")


if __name__ == "__main__":
    main()
