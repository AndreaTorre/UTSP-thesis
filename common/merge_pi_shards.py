#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_pi_shards.py — fonde le shard prodotte da compute_pi_shard.py dentro
test_scenarios_cache.pkl, in un unico processo (nessuna scrittura concorrente).

Da lanciare quando tutte le shard sono finite:
  TESI_EXPERIMENT=PERT TESI_N_NODES=15 python merge_pi_shards.py

Dopo il merge, ogni run del test sweep con TESI_TEST_SKIP_PI=0 troverà i PI
già in cache e li userà senza risolvere nulla: i gap vs PI compaiono accanto
a STO/EEV/UTSP sugli STESSI scenari già valutati.
"""
import glob
import os
import pickle
import sys
from collections import Counter

from config import TEST_SCENARIO_CACHE_DIR


def main():
    cache_path = os.path.join(TEST_SCENARIO_CACHE_DIR, "test_scenarios_cache.pkl")
    shard_glob = os.path.join(TEST_SCENARIO_CACHE_DIR, "pi_shards", "pi_shard_*.pkl")

    if not os.path.exists(cache_path):
        sys.exit(f"Cache non trovata: {cache_path}")

    shards = sorted(glob.glob(shard_glob))
    if not shards:
        sys.exit(f"Nessuna shard trovata in {shard_glob}")

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    results = cache["results"]

    merged = {}
    for path in shards:
        with open(path, "rb") as f:
            merged.update(pickle.load(f))
        print(f"  letta {os.path.basename(path)}")

    already = sum(1 for r in results.values()
                  if r.get("exact_free", {}).get("length") is not None)

    applied, skipped_unknown = 0, 0
    statuses = Counter()
    for sid, res in merged.items():
        if sid not in results:
            skipped_unknown += 1
            continue
        results[sid]["exact_free"] = res
        statuses[res.get("status", "?")] += 1
        applied += 1

    tmp = cache_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"key": cache["key"], "results": results}, f)
    os.replace(tmp, cache_path)

    total = len(results)
    with_pi = sum(1 for r in results.values()
                  if r.get("exact_free", {}).get("length") is not None)

    print(f"\nShard fuse: {len(shards)}")
    print(f"PI applicati: {applied}   (già presenti prima: {already})")
    if skipped_unknown:
        print(f"⚠ {skipped_unknown} scenari nelle shard non esistono in cache: ignorati")
    print(f"Stati Gurobi: {dict(statuses)}")
    if statuses.get("TIME_LIMIT"):
        print(f"⚠ {statuses['TIME_LIMIT']} PI chiusi per TIME LIMIT: sono incumbent, "
              f"non ottimi certificati. Il gap vs PI su quegli scenari è ottimistico.")
    print(f"\nCopertura PI: {with_pi}/{total} scenari ({100*with_pi/total:.1f}%)")
    print(f"Cache aggiornata: {cache_path}")


if __name__ == "__main__":
    main()
