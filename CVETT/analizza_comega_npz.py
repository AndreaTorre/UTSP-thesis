#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import numpy as np


def topm_overlap_with_mean(C, top_m):
    S, n, _ = C.shape
    m = min(top_m, n - 1)
    C_mean = C.mean(axis=0)

    overlaps = []

    for s in range(S):
        for i in range(n):
            row_mean = C_mean[i].copy()
            row_scen = C[s, i].copy()

            row_mean[i] = np.inf
            row_scen[i] = np.inf

            top_mean = set(np.argsort(row_mean)[:m])
            top_scen = set(np.argsort(row_scen)[:m])

            inter = len(top_mean & top_scen)
            union = len(top_mean | top_scen)

            if union > 0:
                overlaps.append(inter / union)

    return float(np.mean(overlaps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz_file")
    parser.add_argument("--top-m", type=int, default=5)
    parser.add_argument("--out-csv", default=None)
    args = parser.parse_args()

    z = np.load(args.npz_file)

    if "C_omega" not in z.files:
        raise SystemExit(f"Chiavi trovate nel file: {z.files}. Manca C_omega.")

    C = np.asarray(z["C_omega"], dtype=float)

    if C.ndim != 3 or C.shape[1] != C.shape[2]:
        raise SystemExit(f"C_omega deve avere forma (S,n,n). Trovata: {C.shape}")

    S, n, _ = C.shape

    mask = ~np.eye(n, dtype=bool)
    X = C[:, mask]

    finite_cols = np.isfinite(X).all(axis=0)
    X = X[:, finite_cols]

    mean_edges = X.mean(axis=0)
    std_edges = X.std(axis=0)
    rel_std = std_edges / (np.abs(mean_edges) + 1e-12)
    abs_delta = np.abs(X - mean_edges)

    scenario_dev = np.linalg.norm(X - mean_edges, axis=1) / (
        np.linalg.norm(mean_edges) + 1e-12
    )

    print("\n=== Diagnostica C^omega ===")
    print("file:", args.npz_file)
    print("shape C_omega:", C.shape)
    print("numero scenari:", S)
    print("numero nodi:", n)
    print("archi validi:", X.shape[1])

    print("\n--- Scala dei costi ---")
    print(f"costo medio archi: {mean_edges.mean():.8f}")
    print(f"costo minimo: {np.nanmin(X):.8f}")
    print(f"costo massimo: {np.nanmax(X):.8f}")

    print("\n--- Variabilità assoluta ---")
    print(f"std media archi: {std_edges.mean():.8f}")
    print(f"std massima archi: {std_edges.max():.8f}")
    print(f"abs_delta medio: {abs_delta.mean():.8f}")
    print(f"abs_delta p90: {np.quantile(abs_delta, 0.90):.8f}")
    print(f"abs_delta p99: {np.quantile(abs_delta, 0.99):.8f}")

    print("\n--- Variabilità relativa ---")
    print(f"rel_std media: {rel_std.mean():.8f}")
    print(f"rel_std p50: {np.quantile(rel_std, 0.50):.8f}")
    print(f"rel_std p90: {np.quantile(rel_std, 0.90):.8f}")
    print(f"rel_std p99: {np.quantile(rel_std, 0.99):.8f}")

    print("\n--- Archi con variazione relativa rilevante ---")
    print(f"archi con rel_std > 1% : {np.sum(rel_std > 0.01)}")
    print(f"archi con rel_std > 5% : {np.sum(rel_std > 0.05)}")
    print(f"archi con rel_std > 10%: {np.sum(rel_std > 0.10)}")

    print("\n--- Deviazione degli scenari dalla matrice media ---")
    print(f"scenario_dev p50: {np.quantile(scenario_dev, 0.50):.8f}")
    print(f"scenario_dev p90: {np.quantile(scenario_dev, 0.90):.8f}")
    print(f"scenario_dev max: {scenario_dev.max():.8f}")

    overlap = topm_overlap_with_mean(C, args.top_m)

    print("\n--- Stabilità degli archi più economici ---")
    print(f"overlap top-{args.top_m} con matrice media: {overlap:.8f}")

    if "scenario_ids" in z.files:
        print("\nPrimi scenario_ids:", z["scenario_ids"][:10])

    if "nodes" in z.files:
        print("Nodi:", z["nodes"])

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)

        rows = []

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                vals = C[:, i, j]

                if not np.isfinite(vals).all():
                    continue

                mean = float(vals.mean())
                std = float(vals.std())

                rows.append({
                    "i": i,
                    "j": j,
                    "mean": mean,
                    "std": std,
                    "rel_std": std / (abs(mean) + 1e-12),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "range": float(vals.max() - vals.min()),
                })

        rows.sort(key=lambda r: r["rel_std"], reverse=True)

        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nCSV archi salvato in: {out}")


if __name__ == "__main__":
    main()
