#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sto_eev_on_test.py — ricalcola STO / EEV / WS come benchmark OTTIMIZZATI SUL TEST.

Convenzione (concordata): ogni benchmark e' ottimo per il blocco di TEST. Il train
NON entra: serve solo alla rete. Cosi' la catena WS <= STO <= EEV vale per costruzione
sul set su cui misuri tutto.

  WS  : gia' esatto negli shard (test_ws_cache.pkl), media dei total_cost.
  STO : UNA prenotazione fissa, la migliore per il test. Enumero i sottoinsiemi di
        prenotazione ristretti al SUPPORTO del WS (archi con f_WS>0: gli altri non
        conviene mai prenotarli) e prendo quello a costo atteso minimo sul test.
        Ogni scenario valutato con solve_reservation_tsp(fixed_reservations=R) —
        la stessa funzione del WS.
  EEV : prenotazione ottima dello SCENARIO MEDIO del test, valutata sul test.

Riusa le stesse funzioni/dati del WS: nessun MILP a 4228 scenari, nessun train.

USO — prima una PASSATA DI SANITY su un sottocampione (veloce), poi il numero pieno:
  cd CVETT
  TESI_EXPERIMENT=CVETT TESI_N_NODES=15 python sto_eev_on_test.py --max-scen 100 --threads 4
  # se R* = i 4 archi e i numeri sono ~2460, lancia il pieno:
  TESI_EXPERIMENT=CVETT TESI_N_NODES=15 python sto_eev_on_test.py --threads 4

Opzioni:
  --max-scen N : usa solo i primi N scenari (sanity). Default: tutti.
  --threads  T : Gurobi threads per solve. Default 1.
  --all-arcs   : enumera TUTTI i 2^|I| sottoinsiemi (rigoroso ma lento). Default: solo supporto WS.
"""
import argparse
import itertools
import os
import pickle
import sys
import time
from collections import Counter

from config import TEST_SCENARIO_CACHE_DIR, PI_TIME_LIMIT
from common import load_env, load_data
from gurobi_models import solve_reservation_tsp


def canon(e):
    i, j = e
    return (i, j) if i <= j else (j, i)


def load_pkl(name):
    path = os.path.join(TEST_SCENARIO_CACHE_DIR, name)
    if not os.path.exists(path):
        sys.exit(f"ERRORE: manca {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-scen", type=int, default=None)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--all-arcs", action="store_true")
    ap.add_argument("--f-min", type=float, default=0.0,
                    help="enumera solo archi con f_WS >= soglia; i deboli sono valutati a parte")
    args = ap.parse_args()
    os.environ["TESI_GRB_THREADS"] = str(args.threads)

    results = load_pkl("test_scenarios_cache.pkl")["results"]
    resB = load_pkl("res_B_cached.pkl")
    I, p, C = resB["I"], resB["p"], resB["C"]
    wsc = load_pkl("test_ws_cache.pkl")["results"]

    sids = sorted(results.keys())
    if args.max_scen:
        sids = sids[:args.max_scen]
    N = len(sids)
    if N == 0:
        sys.exit("Nessuno scenario nel blocco di test.")

    nodes, coords, base_dist, E, root = load_data()
    env = load_env()

    # ── WS dal cache esatto + frequenza di prenotazione f_WS ──────────────
    ws_vals = [wsc[s]["total_cost"] for s in sids
               if s in wsc and wsc[s].get("total_cost") is not None]
    if len(ws_vals) != N:
        print(f"  ATTENZIONE: WS cache copre {len(ws_vals)}/{N} scenari; "
              f"WS_test su quelli disponibili.")
    WS_test = sum(ws_vals) / len(ws_vals) if ws_vals else float("nan")

    cnt = Counter()
    for s in sids:
        for e in (wsc.get(s, {}).get("x_used") or []):
            cnt[canon(e)] += 1
    fWS = {canon(a): cnt.get(canon(a), 0) / N for a in I}
    support = [a for a in I if fWS[canon(a)] > 0.0]
    strong  = [a for a in I if fWS[canon(a)] >= args.f_min]
    weak    = [a for a in support if fWS[canon(a)] < args.f_min]
    cand = list(I) if args.all_arcs else (strong if args.f_min > 0 else support)

    print("=" * 64)
    print(f"BENCHMARK SUL TEST — N={N} scenari  |I|={len(I)} archi")
    print("=" * 64)
    print(f"  WS_test (shard esatti) = {WS_test:.4f}")
    print(f"  supporto WS (f_WS>0)   = {len(support)} archi"
          + ("" if not args.all_arcs else "  [--all-arcs: enumero TUTTI gli archi]"))
    for a in sorted(I, key=canon):
        print(f"      {canon(a)}  f_WS={fWS[canon(a)]:.3f}")

    # ── valutatore: costo atteso sul test di una prenotazione fissa R ─────
    def eval_R(R):
        Rlist = [tuple(a) for a in R]
        tot = 0.0
        t0 = time.time()
        for k, s in enumerate(sids, 1):
            r = solve_reservation_tsp(
                nodes, E, I, results[s]["scenario_dist"], root, p, C, env,
                fixed_reservations=Rlist, output_flag=0,
                model_name="sto_eval",
                time_limit=PI_TIME_LIMIT, mip_gap=None,   # esatto
            )
            if r["total_cost"] is None:
                sys.exit(f"  ERRORE: nessuna soluzione per sid={s} con R={Rlist}")
            tot += r["total_cost"]
            if k % 1000 == 0:
                print(f"        ...{k}/{N}  ({time.time()-t0:.0f}s)")
        return tot / N

    # ── STO: minimo sui sottoinsiemi candidati ───────────────────────────
    subsets = [frozenset(c) for r in range(len(cand) + 1)
               for c in itertools.combinations([canon(a) for a in cand], r)]
    n_solve = len(subsets) * N
    print(f"\n  STO: {len(subsets)} prenotazioni candidate x {N} scenari = {n_solve} solve")

    # stima tempo dal primo eval
    best = None
    table = []
    for idx, R in enumerate(subsets, 1):
        t0 = time.time()
        cost = eval_R(R)
        dt = time.time() - t0
        table.append((cost, R))
        if best is None or cost < best[0]:
            best = (cost, R)
        flag = "  <== migliore finora" if best[1] == R else ""
        print(f"  [{idx:>2}/{len(subsets)}] |R|={len(R)}  cost={cost:.4f}  ({dt:.0f}s)  R={sorted(R)}{flag}")
        if idx == 1:
            print(f"      (stima totale STO ~ {dt*len(subsets)/60:.1f} min)")
    STO_test, R_star = best

    # Certifica che gli archi deboli (f_WS < soglia) non migliorano l'ottimo:
    # valuta R* + ciascun debole e R* + tutti i deboli.
    if weak:
        print(f"\n  Verifica archi deboli (f_WS<{args.f_min}): {[canon(a) for a in weak]}")
        base = set(R_star)
        checks = [base | {canon(w)} for w in weak] + [base | {canon(w) for w in weak}]
        for Rc in checks:
            cc = eval_R(frozenset(Rc))
            better = "  <-- MIGLIORA (rivedi soglia!)" if cc < STO_test - 1e-6 else "  (non migliora, ok)"
            print(f"    R*+deboli {sorted(Rc)}  cost={cc:.4f}{better}")

    # ── EEV: prenotazione dello scenario MEDIO del test, valutata sul test ─
    proto = results[sids[0]]["scenario_dist"]
    mean_dist = {i: {} for i in proto}
    for i in proto:
        for j in proto[i]:
            mean_dist[i][j] = sum(results[s]["scenario_dist"][i][j] for s in sids) / N
    r_ev = solve_reservation_tsp(
        nodes, E, I, mean_dist, root, p, C, env,
        fixed_reservations=None, output_flag=0,
        model_name="eev_mean", time_limit=PI_TIME_LIMIT, mip_gap=None,
    )
    x_ev = [canon(a) for a in (r_ev["x_used"] or [])]
    print(f"\n  EEV: prenotazione scenario medio = {sorted(x_ev)}")
    EEV_test = eval_R(frozenset(x_ev))

    # ── riferimento: le policy STALE del train congelate sul test ─────────
    x_sto_old = [canon(a) for a in (resB.get("x_used_sto") or [])]
    ref = None
    if x_sto_old:
        ref = eval_R(frozenset(x_sto_old))

    # ── esito ────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("RISULTATO — benchmark ottimizzati sul TEST")
    print("=" * 64)
    print(f"  WS_test  = {WS_test:.4f}")
    print(f"  STO_test = {STO_test:.4f}    R* = {sorted(R_star)}")
    print(f"  EEV_test = {EEV_test:.4f}    x_ev = {sorted(x_ev)}")
    print(f"  VSS  = EEV - STO = {EEV_test - STO_test:+.4f}")
    print(f"  EVPI = STO - WS  = {STO_test - WS_test:+.4f}")
    ok = (WS_test - 1e-6) <= STO_test <= (EEV_test + 1e-6)
    print(f"  catena WS <= STO <= EEV: {'OK' if ok else '!!! NON rispettata, controlla'}")
    if ref is not None:
        print(f"\n  [riferimento] x_STO vecchio del train {sorted(x_sto_old)} valutato sul test = {ref:.4f}")
        print(f"                (e' il 2531 sbagliato: policy fittata sul train, non ottima sul test)")


if __name__ == "__main__":
    main()
