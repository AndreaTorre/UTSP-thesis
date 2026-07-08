#!/usr/bin/env python3
import argparse
import csv
import pickle
from pathlib import Path

import numpy as np


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def is_numeric_array(x):
    return isinstance(x, np.ndarray) and np.issubdtype(x.dtype, np.number)


def iter_arrays(obj, prefix="root", max_depth=6):
    if max_depth < 0:
        return

    if is_numeric_array(obj):
        yield prefix, obj
        return

    # NOTA: gestisco anche liste di matrici, senza assumere nomi specifici nei pkl.
    if isinstance(obj, (list, tuple)):
        try:
            arr = np.asarray(obj)
            if is_numeric_array(arr):
                yield prefix, arr
                return
        except Exception:
            pass

        for k, v in enumerate(obj):
            yield from iter_arrays(v, f"{prefix}[{k}]", max_depth - 1)
        return

    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_arrays(v, f"{prefix}.{k}", max_depth - 1)
        return


def is_cost_tensor(arr, nodes=None):
    if arr.ndim != 3:
        return False
    s, n, m = arr.shape
    if s < 2 or n != m:
        return False
    if nodes is not None and n != nodes:
        return False
    return True


def topm_overlap_with_mean(C, top_m):
    S, n, _ = C.shape
    m = min(top_m, n - 1)
    C_mean = C.mean(axis=0)

    overlaps = []

    for s in range(S):
        for i in range(n):
            mean_row = C_mean[i].copy()
            scen_row = C[s, i].copy()

            mean_row[i] = np.inf
            scen_row[i] = np.inf

            mean_top = set(np.argsort(mean_row)[:m])
            scen_top = set(np.argsort(scen_row)[:m])

            inter = len(mean_top & scen_top)
            union = len(mean_top | scen_top)
            overlaps.append(inter / union if union else np.nan)

    return float(np.nanmean(overlaps))


def summarize_tensor(C, top_m):
    C = np.asarray(C, dtype=float)
    S, n, _ = C.shape

    mask = ~np.eye(n, dtype=bool)
    X = C[:, mask]

    finite_cols = np.isfinite(X).all(axis=0)
    X = X[:, finite_cols]

    if X.size == 0:
        raise ValueError("nessun arco valido finito trovato")

    mean_edges = X.mean(axis=0)
    std_edges = X.std(axis=0)
    rel_std = std_edges / (np.abs(mean_edges) + 1e-12)

    abs_delta = np.abs(X - mean_edges)

    scenario_dev = np.linalg.norm(X - mean_edges, axis=1) / (
        np.linalg.norm(mean_edges) + 1e-12
    )

    return {
        "S": S,
        "n": n,
        "valid_edges": int(X.shape[1]),
        "std_mean": float(std_edges.mean()),
        "std_max": float(std_edges.max()),
        "rel_std_mean": float(rel_std.mean()),
        "rel_std_p50": float(np.quantile(rel_std, 0.50)),
        "rel_std_p90": float(np.quantile(rel_std, 0.90)),
        "rel_std_p99": float(np.quantile(rel_std, 0.99)),
        "abs_delta_mean": float(abs_delta.mean()),
        "abs_delta_p90": float(np.quantile(abs_delta, 0.90)),
        "abs_delta_p99": float(np.quantile(abs_delta, 0.99)),
        "scenario_dev_p50": float(np.quantile(scenario_dev, 0.50)),
        "scenario_dev_p90": float(np.quantile(scenario_dev, 0.90)),
        "scenario_dev_max": float(np.quantile(scenario_dev, 1.00)),
        "edges_rel_std_gt_1pct": int(np.sum(rel_std > 0.01)),
        "edges_rel_std_gt_5pct": int(np.sum(rel_std > 0.05)),
        "edges_rel_std_gt_10pct": int(np.sum(rel_std > 0.10)),
        "topm_overlap_with_mean": topm_overlap_with_mean(C, top_m),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Cartella in cui cercare i .pkl")
    parser.add_argument("--nodes", type=int, default=None, help="Filtra tensori con n nodi")
    parser.add_argument("--top-m", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    pkl_files = sorted(root.rglob("*.pkl"))

    if not pkl_files:
        raise SystemExit(f"Nessun .pkl trovato in {root}")

    rows = []

    for path in pkl_files:
        try:
            obj = load_pkl(path)
        except Exception as e:
            print(f"[SKIP] {path}: errore lettura pkl: {e}")
            continue

        found = 0

        for name, arr in iter_arrays(obj):
            if not is_cost_tensor(arr, args.nodes):
                continue

            found += 1

            try:
                stats = summarize_tensor(arr, args.top_m)
            except Exception as e:
                print(f"[SKIP] {path} :: {name}: errore diagnostica: {e}")
                continue

            row = {
                "file": str(path),
                "object": name,
                **stats,
            }
            rows.append(row)

            print("\n=== Candidato C^omega trovato ===")
            print("file:", path)
            print("oggetto:", name)
            print("shape:", tuple(arr.shape))
            print("rel_std media:", row["rel_std_mean"])
            print("rel_std p90/p99:", row["rel_std_p90"], row["rel_std_p99"])
            print("archi rel_std > 1%:", row["edges_rel_std_gt_1pct"])
            print("archi rel_std > 5%:", row["edges_rel_std_gt_5pct"])
            print("overlap top-m con media:", row["topm_overlap_with_mean"])

        if found == 0:
            # NOTA: stampa solo un indizio minimo, utile se i costi hanno nomi/shape inattesi.
            shapes = []
            for name, arr in iter_arrays(obj):
                if is_numeric_array(arr):
                    shapes.append(f"{name}: {arr.shape}")
            if shapes:
                print(f"\n[INFO] Nessun tensore (S,n,n) in {path}. Array trovati:")
                for s in shapes[:20]:
                    print("  ", s)

    if not rows:
        raise SystemExit("Nessun tensore candidato C^omega trovato.")

    out = Path(args.out) if args.out else root / "diagnostica_comega_pkl.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSalvato: {out}")


if __name__ == "__main__":
    main()
