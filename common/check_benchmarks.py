#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_benchmarks.py — dice cosa è già calcolato (train e test) per un dato
esperimento e numero di nodi, e stampa il comando esatto per ciò che manca.

Solo stdlib: è un controllo di file, non serve torch né Gurobi.

Uso:
  TESI_EXPERIMENT=CVETT TESI_N_NODES=15 python check_benchmarks.py
  python check_benchmarks.py --exp CVETT --nodes 15
"""
import argparse
import glob
import os
import pickle
from pathlib import Path

ROOT_DEFAULT = str(Path(__file__).resolve().parent.parent)


def _load(path):
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        return f"__ERR__ {e}"


def _count_results(obj):
    """Le cache sono {'key':..., 'results': {sid:...}} oppure direttamente {sid:...}."""
    if isinstance(obj, dict):
        r = obj.get("results", obj)
        if isinstance(r, dict):
            return len(r), r
    return 0, {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getenv("TESI_ROOT_DIR", ROOT_DEFAULT))
    ap.add_argument("--exp", default=os.getenv("TESI_EXPERIMENT", "CVETT"))
    ap.add_argument("--nodes", type=int, default=int(os.getenv("TESI_N_NODES", "15")))
    a = ap.parse_args()

    ris = os.path.join(a.root, a.exp, f"RISULTATI_{a.nodes}")
    pkl = os.path.join(ris, "pkl")
    print("=" * 78)
    print(f"CHECK BENCHMARK — exp={a.exp}  nodi={a.nodes}")
    print(f"cartella: {pkl}")
    print("=" * 78)
    if not os.path.isdir(pkl):
        print("✗ La cartella pkl non esiste: nessun benchmark calcolato per questo esperimento.")

    missing = []

    # ── TRAIN: res_B (PI/STO/EEV) ────────────────────────────────────────
    print("\n[TRAIN]  res_B_cached.pkl  (PI, STO, EEV, politiche x_STO/x_EEV)")
    resb_path = os.path.join(pkl, "res_B_cached.pkl")
    if os.path.exists(resb_path):
        d = _load(resb_path)
        if isinstance(d, dict):
            print(f"  ✓ presente | I={len(d.get('I', []))} corridoi | "
                  f"PI={d.get('PI')}  STO={d.get('STO')}  EEV={d.get('EEV')}")
            print(f"    x_STO prenota {len(d.get('x_used_sto', []))} corridoi, "
                  f"x_EEV {len(d.get('x_ev', []))}")
        else:
            print(f"  ⚠ illeggibile: {d}")
    else:
        print("  ✗ ASSENTE")
        missing.append("resB")

    # ── TEST: scenari, STO/EEV, WS ───────────────────────────────────────
    print("\n[TEST]   test_scenarios_cache.pkl  (perturbazioni + scenario_dist)")
    scen_path = os.path.join(pkl, "test_scenarios_cache.pkl")
    n_scen = 0
    if os.path.exists(scen_path):
        n_scen, _ = _count_results(_load(scen_path))
        print(f"  ✓ presente | {n_scen} scenari di test")
    else:
        print("  ✗ ASSENTE  (si crea al primo test della rete)")
        missing.append("test_scen")

    print("\n[TEST]   test_sto_eev_cache.pkl  (STO ed EEV per scenario di test)")
    se_path = os.path.join(pkl, "test_sto_eev_cache.pkl")
    if os.path.exists(se_path):
        n, sample = _count_results(_load(se_path))
        keys = list(next(iter(sample.values())).keys()) if sample else []
        cov = f"{100*n/n_scen:.0f}% degli scenari" if n_scen else f"{n} scenari"
        print(f"  ✓ presente | {n} risolti ({cov}) | campi: {keys}")
    else:
        print("  ✗ ASSENTE  (si popola durante il test della rete, via validate_policies)")
        missing.append("test_sto_eev")

    print("\n[TEST]   test_ws_cache.pkl  (WS = wait-and-see per scenario di test)")
    ws_path = os.path.join(pkl, "test_ws_cache.pkl")
    shards = glob.glob(os.path.join(pkl, "ws_shards", "ws_shard_*.pkl"))
    if os.path.exists(ws_path):
        n, sample = _count_results(_load(ws_path))
        cov = f"{100*n/n_scen:.0f}% degli scenari" if n_scen else f"{n} scenari"
        print(f"  ✓ presente | {n} WS risolti ({cov})")
    else:
        extra = f" | {len(shards)} shard grezze presenti (manca il merge)" if shards else ""
        print(f"  ✗ ASSENTE{extra}")
        missing.append("merge_ws" if shards else "ws")

    # ── Modelli allenati ─────────────────────────────────────────────────
    print("\n[RETE]   modelli UTSP allenati (batch_sweep/BATCH_*/modello/*/utsp_model.pt)")
    models = sorted(glob.glob(os.path.join(ris, "batch_sweep", "BATCH_*", "modello", "*", "utsp_model.pt")))
    if models:
        for m in models:
            rel = os.path.relpath(m, ris)
            print(f"  ✓ {rel}")
    else:
        print("  ✗ Nessun modello allenato: va addestrata la rete.")
        missing.append("train_net")

    # ── Cosa manca e come farlo ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print("COSA MANCA E COME OTTENERLO  (ordine consigliato)")
    print("=" * 78)
    env = f"TESI_EXPERIMENT={a.exp} TESI_N_NODES={a.nodes}"
    steps = []
    if "resB" in missing:
        steps.append(("Benchmark di TRAIN (PI/STO/EEV) → res_B_cached.pkl",
                      f"cd {a.exp} && {env} python main.py --only B"))
    if "train_net" in missing or "test_scen" in missing or "test_sto_eev" in missing:
        steps.append(("Allena la rete CROSS e genera i test (→ test_scenarios + test_sto_eev)",
                      f"cd {a.exp} && {env} TESI_USE_CROSS=1 python main.py --only B_UTSP_LS"))
    if "ws" in missing:
        steps.append(("WS di test (dopo che esiste test_scenarios_cache): shard + merge",
                      f"# come job array (20 shard):\n"
                      f"    for s in $(seq 0 19); do {env} bash run_ws.sh {a.nodes} $s 20; done\n"
                      f"    {env} python merge_ws_shards.py"))
    elif "merge_ws" in missing:
        steps.append(("Fondi le shard WS già calcolate → test_ws_cache.pkl",
                      f"cd common && {env} python merge_ws_shards.py"))

    if not steps:
        print("\n✓ Tutto presente: puoi passare all'analisi (full_report.py, crosspollinate.py).")
    else:
        for i, (what, cmd) in enumerate(steps, 1):
            print(f"\n{i}. {what}\n    {cmd}")

    print()


if __name__ == "__main__":
    main()
