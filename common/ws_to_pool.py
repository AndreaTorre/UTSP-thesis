#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Versa i WS certificati (test_ws_cache.pkl, schema 'total_cost') nel pool
test_pool_cache.pkl nello schema che _validate_policies_cached legge:
    {"results": {sid: {"WS": {"cost": <float>}}}}
Preserva un eventuale pool gia' esistente (PI/STO/EEV restano).

    python ws_to_pool.py /percorso/CVETT/RISULTATI_15/pkl
"""
import os
import pickle
import sys


def main(pkl_dir):
    ws_path = os.path.join(pkl_dir, "test_ws_cache.pkl")
    pool_path = os.path.join(pkl_dir, "test_pool_cache.pkl")

    with open(ws_path, "rb") as f:
        ws = pickle.load(f).get("results", {})
    if not ws:
        sys.exit(f"test_ws_cache.pkl vuoto o assente in {pkl_dir}")

    # pool esistente (se c'e'): non lo distruggo, ci aggiungo/aggiorno solo WS
    pool = {}
    if os.path.exists(pool_path):
        try:
            with open(pool_path, "rb") as f:
                pool = pickle.load(f).get("results", {})
        except Exception:
            pool = {}

    n_ok = 0
    for sid, rec in ws.items():
        c = rec.get("total_cost")
        if c is None:
            continue
        pool.setdefault(sid, {})["WS"] = {"cost": float(c)}
        n_ok += 1

    tmp = pool_path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"results": pool}, f)
    os.replace(tmp, pool_path)
    print(f"WS versati nel pool: {n_ok} scenari -> {pool_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("uso: python ws_to_pool.py <cartella_pkl>")
    main(sys.argv[1])
