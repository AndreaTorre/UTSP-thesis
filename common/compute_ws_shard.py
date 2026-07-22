#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compute_ws_shard.py — Wait-and-See (WS): il vero ottimo con informazione perfetta.

PERCHÉ SERVE, e perché il PI non basta:

  PI  = solve_exact_tsp  = TSP LIBERO dello scenario. Ignora p, C e l'insieme I.
        Non è un bound valido per il problema a due stadi: chi usa un arco di I
        DEVE pagare p (prenotandolo) o C (multa), e il PI non paga né l'uno né
        l'altro. Sta sotto il vero ottimo, ma inutilmente lasco.

  PI+pren = tour del PI + costo di prenotazione degli archi di I che quel tour usa.
        Non risolve il problema: prende il tour SBAGLIATO (ottimizzato ignorando p)
        e ci attacca sopra un costo. Con informazione perfetta sceglieresti un tour
        diverso, magari deviando per evitare un arco di I costoso. È un upper bound
        sull'ottimo con informazione perfetta, non l'ottimo.

  WS  = solve_reservation_tsp con x LIBERA, scenario per scenario. Decide insieme
        quali archi prenotare E quale tour fare, conoscendo lo scenario. Questo è
        l'ottimo con informazione perfetta, e il bound inferiore RIGOROSO:

              WS  <=  STO  <=  {UTSP, EEV}

        (WS <= STO perché STO deve usare UNA SOLA x per tutti gli scenari, mentre
        WS può cambiarla per ognuno. La differenza STO - WS è l'EVPI, il valore
        dell'informazione perfetta.)

BONUS: il WS restituisce la x OTTIMA di ogni scenario, quindi

    f_WS(i,j) = frazione di scenari in cui conviene DAVVERO prenotare (i,j)

che è la verifica economica della regola f > p/C — cosa che il PI non può dare,
perché non sa nulla di p e C.

Uso (un job per shard, come per il PI):
  TESI_EXPERIMENT=PERT TESI_N_NODES=15 python compute_ws_shard.py --shard 0 --n-shards 20
Poi:
  TESI_EXPERIMENT=PERT TESI_N_NODES=15 python merge_ws_shards.py
"""
import argparse
import os
import pickle
import sys
import time

from config import TEST_SCENARIO_CACHE_DIR, PI_TIME_LIMIT
from common import load_env, load_data
from gurobi_models import solve_reservation_tsp

# Il WS è un BENCHMARK: risolto esatto per default (vedi nota in compute_pi_shard).
WS_MIP_GAP = float(os.getenv("TESI_WS_MIP_GAP", "0.0"))


def shard_dir():
    return os.path.join(TEST_SCENARIO_CACHE_DIR, "ws_shards")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    args = ap.parse_args()

    if not (0 <= args.shard < args.n_shards):
        sys.exit(f"--shard deve stare in [0, {args.n_shards})")

    cache_path = os.path.join(TEST_SCENARIO_CACHE_DIR, "test_scenarios_cache.pkl")
    resb_path = os.path.join(TEST_SCENARIO_CACHE_DIR, "res_B_cached.pkl")
    for pth in (cache_path, resb_path):
        if not os.path.exists(pth):
            sys.exit(f"File richiesto non trovato: {pth}")

    with open(cache_path, "rb") as f:
        results = pickle.load(f)["results"]
    with open(resb_path, "rb") as f:
        res_B = pickle.load(f)
    I, p, C = res_B["I"], res_B["p"], res_B["C"]

    all_ids = sorted(results.keys())
    my_ids = all_ids[args.shard::args.n_shards]

    os.makedirs(shard_dir(), exist_ok=True)
    out_path = os.path.join(shard_dir(), f"ws_shard_{args.shard}_of_{args.n_shards}.pkl")

    solved = {}
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            solved = pickle.load(f)
    todo = [sid for sid in my_ids if sid not in solved]

    print(f"Shard {args.shard}/{args.n_shards}: {len(my_ids)} scenari assegnati, "
          f"{len(solved)} già risolti, {len(todo)} da fare")
    print(f"  |I| = {len(I)}   mip_gap = {WS_MIP_GAP} (0 = esatto)   time_limit = {PI_TIME_LIMIT}s")
    if not todo:
        print("  Nulla da fare.")
        return

    nodes, coords, base_dist, E, root = load_data()
    env = load_env()

    t0 = time.time()
    n_tl = 0
    for k, sid in enumerate(todo, start=1):
        r = solve_reservation_tsp(
            nodes, E, I, results[sid]["scenario_dist"], root, p, C, env,
            fixed_reservations=None,          # <-- x LIBERA: questo è il WS
            output_flag=0, model_name=f"ws_{sid}",
            time_limit=PI_TIME_LIMIT,
            mip_gap=(WS_MIP_GAP if WS_MIP_GAP > 0 else None),
        )
        solved[sid] = {
            "total_cost": r.get("total_cost"),
            "tour_cost": r.get("tour_cost"),
            "reservation_paid": r.get("reservation_paid"),
            "penalty_paid": r.get("penalty_paid"),
            "x_used": r.get("x_used", []),      # <-- prenotazioni OTTIME dello scenario
            "tour": r.get("tour", []),
            "status": r.get("status"),
        }
        if r.get("status") == "TIME_LIMIT":
            n_tl += 1

        if k % 50 == 0 or k == len(todo):
            el = time.time() - t0
            rate = k / el if el else 0
            eta = (len(todo) - k) / rate if rate else 0
            print(f"  [{k}/{len(todo)}] {rate:.1f} scen/s  trascorsi {el/60:.1f}m  "
                  f"ETA {eta/60:.1f}m  time_limit finora: {n_tl}", flush=True)
            tmp = out_path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(solved, f)
            os.replace(tmp, out_path)

    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(solved, f)
    os.replace(tmp, out_path)
    print(f"\nShard {args.shard} completata: {len(solved)} WS risolti in "
          f"{(time.time()-t0)/60:.1f} minuti ({n_tl} in time limit)")
    print(f"  scritta: {out_path}")


if __name__ == "__main__":
    main()
