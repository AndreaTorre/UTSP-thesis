#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slice_pool.py — affetta il pool risolto (test_pool_cache.pkl) in combinazioni
IS (numero istanze) x dim (scenari per istanza) e stampa le medie di WS/STO/EEV/PI.

Nessun Gurobi: pura aggregazione. Gli scenari del pool sono ordinati per
scenario_id; l'istanza j di dimensione dim contiene gli scenari
[j*dim+1 .. (j+1)*dim]. Questo affettamento è lo STESSO che usa il test sweep
di UTSP (blocchi consecutivi), quindi le medie qui coincidono con quelle che
otterrà la rete sugli stessi scenari.

  TESI_EXPERIMENT=PERT TESI_N_NODES=15 python slice_pool.py --combos "100x20,100x70,1000x70"
  TESI_EXPERIMENT=PERT TESI_N_NODES=15 python slice_pool.py --is 1000 --dim 20 30 60 70
"""
import argparse
import os
import pickle
import statistics as st
import sys

from config import TEST_SCENARIO_CACHE_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combos", default=None,
                    help='es. "100x20,1000x70" (IShxdim, separati da virgola)')
    ap.add_argument("--is", dest="IS", type=int, default=None)
    ap.add_argument("--dim", type=int, nargs="+", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    path = os.path.join(TEST_SCENARIO_CACHE_DIR, "test_pool_cache.pkl")
    if not os.path.exists(path):
        sys.exit(f"Pool non trovato: {path}. Lancia solve_pool_shard + merge_pool_shards.")
    with open(path, "rb") as f:
        pool = pickle.load(f)["results"]
    ids = sorted(pool.keys())
    npool = len(ids)

    combos = []
    if args.combos:
        for tok in args.combos.split(","):
            a, b = tok.lower().split("x")
            combos.append((int(a), int(b)))
    elif args.IS and args.dim:
        combos = [(args.IS, d) for d in args.dim]
    else:
        sys.exit("Specifica --combos oppure --is e --dim")

    L = []

    def P(*a):
        L.append(" ".join(str(x) for x in a) if a else "")

    P("=" * 78)
    P(f"POOL DI TEST — {npool} scenari risolti  (scenario_id {ids[0]}..{ids[-1]})")
    P("=" * 78)

    def cost(sid, m):
        return pool[sid].get(m, {}).get("cost")

    for (IS, dim) in combos:
        need = IS * dim
        P("")
        P(f"IS={IS} x dim={dim}  ({need} scenari richiesti)")
        if need > npool:
            P(f"  ⚠ il pool ha solo {npool} scenari: combinazione non copribile. Saltata.")
            continue
        # medie per-modello sui primi need scenari, aggregando come il test sweep:
        # media dei costi per scenario (le istanze pesano tutte uguale se IS*dim=need).
        block = ids[:need]
        P(f"    {'modello':<8} {'media':>12} {'std':>10} {'copertura':>10}")
        P("    " + "-" * 44)
        ref = {}
        for m in ("WS", "STO", "EEV", "PI"):
            vals = [cost(sid, m) for sid in block if cost(sid, m) is not None]
            if not vals:
                P(f"    {m:<8} {'—':>12}")
                continue
            ref[m] = st.fmean(vals)
            P(f"    {m:<8} {st.fmean(vals):>12.2f} {st.pstdev(vals):>10.2f} "
              f"{len(vals)}/{need:>6}")
        if "STO" in ref and "WS" in ref:
            P(f"    EVPI (STO-WS) = {ref['STO']-ref['WS']:.2f} "
              f"({100*(ref['STO']-ref['WS'])/ref['WS']:+.2f}%)")
        if "EEV" in ref and "STO" in ref:
            P(f"    EEV vs STO   = {100*(ref['EEV']-ref['STO'])/ref['STO']:+.2f}%")

    out = args.out or os.path.join(os.path.dirname(os.path.dirname(TEST_SCENARIO_CACHE_DIR)),
                                   "sweep_analysis", "pool_slices.txt")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nSalvato in: {out}")


if __name__ == "__main__":
    main()
