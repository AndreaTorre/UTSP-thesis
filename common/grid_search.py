#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grid search UTSP.

Struttura prodotta (sotto ROOT_DIR/grid_search/):

    grid_search/PERT/NODI_15/BATCH_20/
    ├── combos.json          manifesto indice -> parametri (riproducibilita')
    ├── logs/                stdout/stderr dei job SLURM
    ├── combo_0000/          output completo di run_esperimento_B_UTSP
    │   ├── output/ grafici/ checkpoint/ pkl/
    │   └── result.json      metriche estratte + parametri usati
    ├── combo_0001/ ...
    └── grid_summary.csv     aggregato di tutti i result.json

Modalita':
    --list                    stampa il numero di combinazioni (usato dal submit)
    --run-one IDX             esegue UNA combinazione (un processo dedicato)
    --run-chunk K             esegue le combinazioni del chunk K in sottoprocessi
    --collect                 aggrega i result.json in grid_summary.csv

NOTA: ogni combinazione gira in un processo separato. config.py legge le
variabili d'ambiente al momento dell'import, quindi non e' possibile cambiare
i parametri nello stesso processo senza reload fragili: un processo per combo
e' piu' semplice e isola i crash.
"""

import argparse
import shutil 
import csv
import itertools
import json
import os
import subprocess
import sys
import time

# ============================================================================
# GRIGLIA
# ============================================================================
# Ogni chiave e' il nome ESATTO di una costante di config.py: viene passata al
# processo figlio come TESI_P_<NOME> e config.py la applica dopo il backend.
#
# Chiave speciale "_LAMBDA_B_DIV": UTSP2_LAMBDA_B = batch / divisore.
# La cardinalita' |Omega| coincide con UTSP_BATCH_SIZE, che cambia da cartella
# a cartella: tenere fisso lambda_B renderebbe i BATCH_* non confrontabili.

GRID = {
    "UTSP2_LAMBDA1":      [1.0, 5.0, 20.0],
    "UTSP2_LAMBDA_D":     [1.0, 3.0],
    "UTSP2_TEMP_SCALE":   [0.25, 0.5, 1.0],
    "UTSP2_ALPHA_LOSS":   [0.5, 1.0, 2.0],
    "UTSP2_LAMBDA_B_DIV": [1.0, 4.0, 10.0],
    "UTSP2_LAMBDA_E":     [0.0, 10.0, 100.0],
}

FIXED = {
    "UTSP2_LAMBDA2":  1.0,
    "UTSP2_LS_ALPHA": 0.0,
}

# Metriche estratte da output["local_search"]. Le prime due sono l'obiettivo.
METRICS = [
    "gap_ls_ws", "gap_ls_sto", "gap_ls_eev", "gap_ls_pi",
    "UTSP_LS_test", "WS_test", "STO_test", "EEV_test", "PI_test", "PI_pren_test",
    "UTSP_LS_train", "PI_train",
]
BATCH_SIZES = [20, 30, 40, 50, 55, 60, 65, 70]
NODE_SIZES = [15, 25, 40]
EXPERIMENTS = ["PERT", "CVETT"]


# ============================================================================
# COMBINAZIONI E PERCORSI
# ============================================================================

def build_combos(batch):
    """Prodotto cartesiano deterministico: l'indice IDX identifica la combo."""
    keys = sorted(GRID)
    combos = []
    for values in itertools.product(*(GRID[k] for k in keys)):
        raw = dict(zip(keys, values))
        params = {k: v for k, v in raw.items() if not k.startswith("_")}
        params.update(FIXED)
        div = raw.get("_LAMBDA_B_DIV")
        if div is not None:
            params["UTSP2_LAMBDA_B"] = round(batch / float(div), 6)
        combos.append({"params": params, "meta": {k: raw[k] for k in raw if k.startswith("_")}})
    return combos


def project_root():
    return os.getenv("TESI_ROOT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def leaf_dir(exp, nodes, batch):
    return os.path.join(project_root(), "grid_search", exp, f"NODI_{nodes}", f"BATCH_{batch}")


def write_manifest(leaf, combos):
    os.makedirs(leaf, exist_ok=True)
    path = os.path.join(leaf, "combos.json")
    payload = {
        "grid": GRID,
        "fixed": FIXED,
        "n_combos": len(combos),
        "combos": {f"{i:04d}": c for i, c in enumerate(combos)},
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return path


# ============================================================================
# ESECUZIONE DI UNA COMBINAZIONE
# ============================================================================

def run_one(idx, exp, nodes, batch):
    combos = build_combos(batch)
    if not 0 <= idx < len(combos):
        raise SystemExit(f"IDX {idx} fuori range: 0..{len(combos) - 1}")
    combo = combos[idx]
    leaf = leaf_dir(exp, nodes, batch)
    cdir = os.path.join(leaf, f"combo_{idx:04d}")
    for sub in ("output", "grafici", "checkpoint", "pkl"):
        os.makedirs(os.path.join(cdir, sub), exist_ok=True)

    # --- environment: DEVE essere completo prima di importare config ---
        os.environ["TESI_EXPERIMENT"] = exp
    os.environ["TESI_N_NODES"] = str(nodes)
    # Il backend legge TESI_UTSP_BATCH_SIZE all'import e lì valida la divisibilità
    # dei batch. TESI_BATCH_SWEEP viene applicato DOPO l'import (common/config.py),
    # troppo tardi per quel controllo: senza questa riga ogni combo CVETT muore
    # all'import con "Test non divisibile in batch completi".
    os.environ["TESI_UTSP_BATCH_SIZE"] = str(batch)
    os.environ["TESI_BATCH_SWEEP"] = str(batch)
    os.environ["TESI_OUTPUT_OVERRIDE"] = cdir
    for name, value in combo["params"].items():
        os.environ[f"TESI_P_{name}"] = repr(value) if isinstance(value, str) else str(value)

    t0 = time.time()
    record = {
        "idx": idx, "experiment": exp, "nodes": nodes, "batch": batch,
        "params": combo["params"], "meta": combo["meta"],
        "status": "running", "output_dir": cdir,
    }

    try:
        import config
        from common import load_data, load_env, set_seed
        from utsp import run_esperimento_B_UTSP

        cache = os.path.join(config.TEST_SCENARIO_CACHE_DIR, "res_B_cached.pkl")
        if not os.path.exists(cache):
            raise FileNotFoundError(
                f"res_B_cached.pkl assente: {cache}\n"
                "Esegui prima Esperimento B per questa coppia (esperimento, nodi): "
                "la grid search non ricalcola STO/EEV/PI."
            )
        import pickle
        with open(cache, "rb") as f:
            res_B = pickle.load(f)

        set_seed()
        env = load_env()
        nodes_l, coords, base_dist, E, root = load_data()

        kwargs, exp_name = {}, "espB_UTSP_LS"
        if config.IS_CVETT:
            from wind_perturbation import load_wind_field
            kwargs = {
                "wind_train": load_wind_field(config.ERA5_NC_PATH_TRAIN),
                "wind_test": load_wind_field(config.ERA5_NC_PATH_TEST),
            }
            exp_name = "espB_wind_UTSP_LS"

        out = run_esperimento_B_UTSP(
            nodes_l, coords, base_dist, E, root, env,
            res_B=res_B, mode="local_search",
            scenario_kwargs=kwargs, exp_name=exp_name,
        )
        ls = out.get("local_search", {}) or {}
        record["metrics"] = {k: _scalar(ls.get(k)) for k in METRICS}
        record["aggregato_test"] = _flatten(ls.get("aggregato_test"))
        record["status"] = "ok"

    except Exception as exc:  # noqa: BLE001 — va registrato, non propagato
        import traceback
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    record["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(cdir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)

    # NOTA: la grid legge SOLO result.json (collect + grid_analyze). Il resto
    # (modello/ grafici/ report/ checkpoint/ pkl/) è zavorra: 486 × artefatti
    # saturano la quota inode di $HOME. Lo elimino tenendo solo result.json.
    # try/except: se la pulizia fallisce NON deve far fallire una combo già
    # riuscita — al massimo resta zavorra, come prima della patch.
    if os.getenv("TESI_GRID_KEEP_OUTPUT") != "1":
        try:
            for entry in os.listdir(cdir):
                if entry == "result.json":
                    continue
                path = os.path.join(cdir, entry)
                if os.path.isdir(path):
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    os.remove(path)
        except OSError as e:
            print(f"[combo {idx:04d}] pulizia saltata: {e}")

    print(f"[combo {idx:04d}] {record['status']} in {record['seconds']}s -> {cdir}")
    return 0 if record["status"] == "ok" else 1


def _scalar(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _flatten(d, prefix=""):
    """Appiattisce un dict annidato tenendo solo gli scalari numerici."""
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif _scalar(v) is not None:
            out[key] = _scalar(v)
    return out


# ============================================================================
# CHUNK: piu' combinazioni in un solo job SLURM
# ============================================================================

def run_chunk(chunk, chunk_size, exp, nodes, batch, timeout):
    combos = build_combos(batch)
    leaf = leaf_dir(exp, nodes, batch)
    write_manifest(leaf, combos)

    lo = chunk * chunk_size
    hi = min(lo + chunk_size, len(combos))
    if lo >= len(combos):
        print(f"Chunk {chunk} vuoto (totale {len(combos)} combo).")
        return 0

    print(f"Chunk {chunk}: combo {lo}..{hi - 1} di {len(combos)} — {exp} {nodes} nodi, batch {batch}")
    failed = []
    for idx in range(lo, hi):
        done = os.path.join(leaf, f"combo_{idx:04d}", "result.json")
        if os.path.exists(done):
            with open(done, encoding="utf-8") as f:
                if json.load(f).get("status") == "ok":
                    print(f"[combo {idx:04d}] gia' completata, salto")
                    continue
        cmd = [sys.executable, os.path.abspath(__file__), "--run-one", str(idx),
               "--exp", exp, "--nodes", str(nodes), "--batch", str(batch)]
        try:
            # NOTA: sottoprocesso per combo. Un crash o un OOM non porta giu'
            # le combinazioni successive dello stesso job.
            rc = subprocess.run(cmd, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            print(f"[combo {idx:04d}] TIMEOUT dopo {timeout}s")
            rc = 124
        if rc != 0:
            failed.append(idx)

    print(f"\nChunk {chunk} finito. Fallite: {failed if failed else 'nessuna'}")
    return 0


# ============================================================================
# AGGREGAZIONE
# ============================================================================

def collect(exp, nodes, batch):
    leaf = leaf_dir(exp, nodes, batch)
    rows = []
    for name in sorted(os.listdir(leaf)) if os.path.isdir(leaf) else []:
        rj = os.path.join(leaf, name, "result.json")
        if not name.startswith("combo_") or not os.path.exists(rj):
            continue
        with open(rj, encoding="utf-8") as f:
            r = json.load(f)
        row = {"idx": r["idx"], "status": r["status"], "seconds": r.get("seconds")}
        row.update(r.get("params", {}))
        row.update(r.get("meta", {}))
        row.update(r.get("metrics", {}) or {})
        row.update(_flatten(r.get("aggregato_test")))
        row["error"] = r.get("error", "")
        rows.append(row)

    if not rows:
        print(f"Nessun result.json in {leaf}")
        return 1

    # ordina per obiettivo primario, i falliti in fondo
    rows.sort(key=lambda r: (r["status"] != "ok", r.get("gap_ls_ws") if r.get("gap_ls_ws") is not None else 1e9))
    cols = list(dict.fromkeys(k for r in rows for k in r))
    out = os.path.join(leaf, "grid_summary.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    ok = sum(r["status"] == "ok" for r in rows)
    print(f"{out}\n  {ok}/{len(rows)} combinazioni riuscite")
    if ok:
        best = rows[0]
        print(f"  migliore: combo_{best['idx']:04d}  gap_ls_sto={best.get('gap_ls_sto')}")
    return 0


# ============================================================================
# CLI
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="Grid search UTSP per (esperimento, nodi, batch).")
    ap.add_argument("--exp", choices=EXPERIMENTS, default=os.getenv("TESI_EXPERIMENT", "PERT"))
    ap.add_argument("--nodes", type=int, choices=NODE_SIZES, default=int(os.getenv("TESI_N_NODES", "25")))
    ap.add_argument("--batch", type=int, choices=BATCH_SIZES, default=int(os.getenv("TESI_BATCH_SWEEP", "30")))
    ap.add_argument("--chunk-size", type=int, default=50)
    ap.add_argument("--timeout", type=int, default=7200, help="secondi max per singola combinazione")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true")
    mode.add_argument("--run-one", type=int, metavar="IDX")
    mode.add_argument("--run-chunk", type=int, metavar="K")
    mode.add_argument("--collect", action="store_true")
    args = ap.parse_args()

    if args.list:
        n = len(build_combos(args.batch))
        chunks = (n + args.chunk_size - 1) // args.chunk_size
        # riga parsata da grid_submit.sh: non cambiare il formato
        print(f"{n} {chunks}")
        return 0
    if args.run_one is not None:
        return run_one(args.run_one, args.exp, args.nodes, args.batch)
    if args.run_chunk is not None:
        return run_chunk(args.run_chunk, args.chunk_size, args.exp, args.nodes, args.batch, args.timeout)
    return collect(args.exp, args.nodes, args.batch)


if __name__ == "__main__":
    sys.exit(main())
