#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_pool_shards.py — fonde le shard di solve_pool_shard.py in test_pool_cache.pkl.

  TESI_EXPERIMENT=PERT TESI_N_NODES=15 python merge_pool_shards.py
"""
import glob
import os
import pickle
import sys

from config import TEST_SCENARIO_CACHE_DIR


def main():
    shards = sorted(glob.glob(os.path.join(TEST_SCENARIO_CACHE_DIR, "pool_shards",
                                           "pool_shard_*.pkl")))
    if not shards:
        sys.exit(f"Nessuna shard in {TEST_SCENARIO_CACHE_DIR}/pool_shards/")

    merged = {}
    for path in shards:
        with open(path, "rb") as f:
            merged.update(pickle.load(f))
        print(f"  letta {os.path.basename(path)}")

    # Non salvo le scenario_dist nel file finale: pesano e servono solo durante
    # il calcolo. Tengo costi, x_used (WS), status.
    slim = {}
    for sid, rec in merged.items():
        slim[sid] = {m: rec[m] for m in ("PI", "STO", "EEV", "WS") if m in rec}

    out = os.path.join(TEST_SCENARIO_CACHE_DIR, "test_pool_cache.pkl")
    tmp = out + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"results": slim, "seed": None}, f)
    os.replace(tmp, out)

    n = len(slim)
    print(f"\nShard fuse: {len(shards)}   scenari totali: {n}")
    for m in ("PI", "STO", "EEV", "WS"):
        vals = [r[m]["cost"] for r in slim.values() if m in r and r[m].get("cost") is not None]
        tl = sum(1 for r in slim.values() if m in r and r[m].get("status") == "TIME_LIMIT")
        if vals:
            print(f"  {m:4}: {len(vals)}/{n} risolti, media {sum(vals)/len(vals):.2f}"
                  + (f"  ⚠ {tl} in TIME_LIMIT" if tl else ""))
        else:
            print(f"  {m:4}: assente")

    # Sanity: catena dei bound WS <= STO <= EEV (per scenario, dove tutti presenti).
    def cost(r, m):
        return r.get(m, {}).get("cost")
    viol_ws_sto = viol_sto_eev = 0
    for r in slim.values():
        w, s, e = cost(r, "WS"), cost(r, "STO"), cost(r, "EEV")
        if w is not None and s is not None and w > s + 1e-6:
            viol_ws_sto += 1
        if s is not None and e is not None and s > e + 1e-6:
            viol_sto_eev += 1
    print("")
    if viol_ws_sto:
        print(f"  ⚠ {viol_ws_sto} scenari con WS > STO (atteso 0; probabili TIME_LIMIT sul WS)")
    else:
        print("  ✓ WS <= STO su tutti gli scenari")
    # STO ed EEV sono policy FISSE diverse: la catena STO<=EEV vale solo IN MEDIA,
    # non per singolo scenario. Su scenari dove la prenotazione extra di EEV
    # ripaga, EEV puo' battere STO localmente. NON e' una violazione di bound.
    if viol_sto_eev:
        print(f"  i {viol_sto_eev} scenari con STO > EEV sono NORMALI: STO e' ottimo")
        print("  in media (Experiment B), non per singolo scenario. Solo WS e' ottimo")
        print("  per-scenario, quindi solo WS<=STO e WS<=EEV valgono ovunque.")
    print(f"\nScritta: {out}")


if __name__ == "__main__":
    main()
