#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_ws_shards.py — fonde le shard di compute_ws_shard.py in test_ws_cache.pkl.

Scrive in un file DEDICATO (non dentro test_scenarios_cache.pkl) per non
rischiare di corrompere la cache degli scenari, che è il dato più prezioso:
rigenerarla costerebbe di nuovo tutti i Gurobi.

  TESI_EXPERIMENT=PERT TESI_N_NODES=15 python merge_ws_shards.py
"""
import glob
import os
import pickle
import sys
from collections import Counter

from config import TEST_SCENARIO_CACHE_DIR


def main():
    shards = sorted(glob.glob(os.path.join(TEST_SCENARIO_CACHE_DIR, "ws_shards", "ws_shard_*.pkl")))
    if not shards:
        sys.exit(f"Nessuna shard WS in {TEST_SCENARIO_CACHE_DIR}/ws_shards/")

    merged = {}
    for path in shards:
        with open(path, "rb") as f:
            merged.update(pickle.load(f))
        print(f"  letta {os.path.basename(path)}")

    out = os.path.join(TEST_SCENARIO_CACHE_DIR, "test_ws_cache.pkl")
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"results": merged}, f)
    os.replace(tmp, out)

    statuses = Counter(v.get("status", "?") for v in merged.values())
    n_tl = statuses.get("TIME_LIMIT", 0)

    print(f"\nShard fuse: {len(shards)}")
    print(f"Scenari con WS: {len(merged)}")
    print(f"Stati Gurobi: {dict(statuses)}")
    if n_tl:
        print(f"⚠ {n_tl} scenari chiusi per TIME LIMIT: sono incumbent, non ottimi certificati.")
        print("  Il WS su quegli scenari è una SOVRASTIMA del vero ottimo, quindi il gap")
        print("  UTSP-vs-WS che ne risulta è ottimistico. Riportalo in tesi.")

    # Sanity check: il WS deve essere <= STO su ogni scenario (è un bound inferiore).
    sto_path = os.path.join(TEST_SCENARIO_CACHE_DIR, "test_sto_eev_cache.pkl")
    if os.path.exists(sto_path):
        with open(sto_path, "rb") as f:
            sto = pickle.load(f).get("results", {})
        viol = [sid for sid, v in merged.items()
                if sid in sto and v.get("total_cost") is not None
                and v["total_cost"] > sto[sid]["sto_cost"] + 1e-6]
        if viol:
            print(f"\n⚠⚠ {len(viol)} scenari in cui WS > STO: IMPOSSIBILE se entrambi ottimi.")
            print("   Quasi certamente sono gli scenari chiusi in TIME_LIMIT (WS non ottimo).")
            print(f"   Esempi: {viol[:5]}")
        else:
            print("\n✓ Controllo: WS <= STO su tutti gli scenari (come deve essere).")

    print(f"\nScritta: {out}")


if __name__ == "__main__":
    main()
