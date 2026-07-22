#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_test_sweep.py — raccoglie e analizza i risultati del test sweep.

Legge i file scritti da _run_utsp_test_only_branch in ogni combinazione
  <ROOT>/<EXP>/RISULTATI_<N>/batch_sweep/BATCH_<B>/test/IS_<i>_DIM_<d>/grafici/
e produce:
  1. test_sweep_instances.csv  — una riga per istanza di test (dati appaiati)
  2. test_sweep_combos.csv     — una riga per combinazione (medie e std)
  3. test_sweep_report.txt     — matrici BATCH×DIM, vincitori appaiati,
                                 analisi diagonale (batch == DIM), ranking.

Solo stdlib: gira ovunque senza dipendenze.

Uso:
  python collect_test_sweep.py                    # tutto ciò che trova
  python collect_test_sweep.py --exp PERT --nodes 15
  python collect_test_sweep.py --metric gap_eev   # default: gap_sto
"""
import argparse
import csv
import glob
import os
import pickle
import re
import statistics as st
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.dirname(HERE)  # .../UTSP se lo script sta in common/

# Righe "COSTI TEST SCENARIO PER SCENARIO":  sid | pre_total | post_total | ...
SCEN_ROW_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*([-\d.]+|N/A)\s*\|\s*([-\d.]+|N/A)\s*\|", re.M)


def load_pi_from_cache(root, exp, n_nodes):
    """
    PI per scenario_id dalla cache. Serve per calcolare gap_pi A POSTERIORI:
    i file di stats scritti con TESI_TEST_SKIP_PI=1 hanno PI=N/A, ma i costi
    UTSP sono lì scenario per scenario e i PI sono in cache — si uniscono per
    scenario_id senza rieseguire la local search.
    """
    path = os.path.join(root, exp, f"RISULTATI_{n_nodes}", "pkl",
                        "test_scenarios_cache.pkl")
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        results = pickle.load(f)["results"]
    pi = {}
    for sid, rec in results.items():
        length = rec.get("exact_free", {}).get("length")
        if length is not None:
            pi[sid] = float(length)
    return pi


def parse_scenario_costs(text):
    """Costo UTSP post-booking per scenario, dal file di stats."""
    out = {}
    for m in SCEN_ROW_RE.finditer(text):
        sid, post = m.group(1), m.group(3)
        if post != "N/A":
            out[int(sid)] = float(post)
    return out


# Campi del file per-istanza -> nome colonna
FIELD_RE = {
    "PI":      re.compile(r"PI test\s*=\s*([-\d.naN/A]+)"),
    "PI_pren": re.compile(r"PI\+pren test\s*=\s*([-\d.naN/A]+)"),
    "UTSP":    re.compile(r"UTSP test\s*=\s*([-\d.naN/A]+)"),
    "STO":     re.compile(r"STO test\s*=\s*([-\d.naN/A]+)"),
    "EEV":     re.compile(r"EEV test\s*=\s*([-\d.naN/A]+)"),
    "gap_sto": re.compile(r"Gap UTSP vs STO\s*=\s*([-\d.naN/A]+)%"),
    "gap_eev": re.compile(r"Gap UTSP vs EEV\s*=\s*([-\d.naN/A]+)%"),
    "gap_pi":  re.compile(r"Gap UTSP vs PI\s*=\s*([-\d.naN/A]+)%"),
}
COMBO_RE = re.compile(
    r"/(?P<exp>PERT|CVETT)/RISULTATI_(?P<n>\d+)/batch_sweep/BATCH_(?P<batch>\d+)"
    r"/test/IS_(?P<is_>\d+)_DIM_(?P<dim>\d+)/"
)
INSTANCE_FILE_RE = re.compile(r"_test_only_i(?P<idx>\d+)_test_only_stats\.txt$")
SINGLE_FILE_RE = re.compile(r"_test_only_test_only_stats\.txt$")  # caso 1 istanza


def _num(s):
    s = s.strip()
    if s in ("N/A", "nan", "NaN", ""):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def discover(root, exp_filter=None, nodes_filter=None, with_pi=True):
    """Trova tutti i file per-istanza del test sweep. Ritorna righe-istanza."""
    pattern = os.path.join(root, "*", "RISULTATI_*", "batch_sweep",
                           "BATCH_*", "test", "IS_*_DIM_*", "grafici", "*_test_only_*stats.txt")
    pi_cache = {}   # (exp, n) -> {sid: pi}
    rows = []
    for path in sorted(glob.glob(pattern)):
        cm = COMBO_RE.search(path.replace(os.sep, "/"))
        if not cm:
            continue
        exp, n = cm.group("exp"), int(cm.group("n"))
        if exp_filter and exp != exp_filter:
            continue
        if nodes_filter and n != nodes_filter:
            continue
        im = INSTANCE_FILE_RE.search(path)
        if im:
            idx = int(im.group("idx"))
        elif SINGLE_FILE_RE.search(path):
            idx = 0
        else:
            continue  # es. cost_distributions_*_stats.txt: non è un file istanza

        text = open(path, encoding="utf-8", errors="replace").read()
        row = {}
        for name, rx in FIELD_RE.items():
            m = rx.search(text)
            row[name] = _num(m.group(1)) if m else None

        # PI a posteriori: se manca nel file ma la cache ce l'ha, lo ricostruisco
        # dai costi per-scenario (stessi scenari, join per scenario_id).
        if with_pi and row.get("PI") is None:
            if (exp, n) not in pi_cache:
                pi_cache[(exp, n)] = load_pi_from_cache(root, exp, n)
            pi_by_sid = pi_cache[(exp, n)]
            if pi_by_sid:
                costs = parse_scenario_costs(text)
                pis = [pi_by_sid[s] for s in costs if s in pi_by_sid]
                if pis and len(pis) == len(costs):   # PI completo per questa istanza
                    row["PI"] = st.fmean(pis)
                    if row.get("UTSP") is not None and row["PI"]:
                        row["gap_pi"] = (row["UTSP"] - row["PI"]) / abs(row["PI"]) * 100
                    row["_pi_postumo"] = 1

        row.update(exp=exp, n_nodes=n, batch=int(cm.group("batch")),
                   IS=int(cm.group("is_")), DIM=int(cm.group("dim")), istanza=idx)
        row.setdefault("_pi_postumo", 0)
        rows.append(row)
    return rows


def _mean_std(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return st.fmean(vals), (st.pstdev(vals) if len(vals) > 1 else 0.0)


def build_combos(instances):
    """Aggrega le righe-istanza in righe-combinazione."""
    groups = defaultdict(list)
    for r in instances:
        groups[(r["exp"], r["n_nodes"], r["batch"], r["IS"], r["DIM"])].append(r)
    combos = []
    for (exp, n, b, is_, dim), rows in sorted(groups.items()):
        c = {"exp": exp, "n_nodes": n, "batch": b, "IS": is_, "DIM": dim,
             "n_istanze_trovate": len(rows)}
        for f in FIELD_RE:
            m, s = _mean_std([r[f] for r in rows])
            c[f + "_mean"], c[f + "_std"] = m, s
        combos.append(c)
    return combos


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def fmt(x, nd=3):
    return "   —  " if x is None else f"{x:+.{nd}f}" if isinstance(x, float) else str(x)


def matrix_report(out, combos, instances, exp, n, metric):
    """Matrici BATCH×DIM, vincitori appaiati, diagonale, ranking per un grafo."""
    sub = [c for c in combos if c["exp"] == exp and c["n_nodes"] == n]
    if not sub:
        return
    batches = sorted({c["batch"] for c in sub})
    dims = sorted({c["DIM"] for c in sub})
    get = {(c["batch"], c["DIM"]): c for c in sub}
    mkey, skey = metric + "_mean", metric + "_std"

    P = out.append
    P("")
    P("=" * 78)
    P(f"{exp} — {n} nodi   (metrica: {metric}, % rispetto al benchmark; più basso = meglio)")
    P("=" * 78)

    # copertura: combinazioni assenti o incomplete
    missing = [(b, d) for b in batches for d in dims if (b, d) not in get]
    partial = [(c["batch"], c["DIM"], c["n_istanze_trovate"])
               for c in sub if c["n_istanze_trovate"] < c["IS"]]
    if missing:
        P(f"  ⚠ combinazioni assenti (non ancora girate?): {missing}")
    if partial:
        P(f"  ⚠ combinazioni incomplete (istanze trovate < IS): {partial}")

    # matrice media ± std
    hdr = "  batch\\DIM |" + "".join(f" {d:>13}" for d in dims)
    P("")
    P(hdr)
    P("  " + "-" * (len(hdr) - 2))
    for b in batches:
        cells = []
        for d in dims:
            c = get.get((b, d))
            cells.append(f" {fmt(c[mkey]) + '±' + f'{c[skey]:.2f}' if c and c[mkey] is not None else '      —      ':>13}")
        P(f"  {b:>9} |" + "".join(cells))

    # miglior batch per ogni DIM (sulla media)
    P("")
    P("  Miglior batch per DIM (media):")
    best_per_dim = {}
    for d in dims:
        cands = [(get[(b, d)][mkey], b) for b in batches
                 if (b, d) in get and get[(b, d)][mkey] is not None]
        if cands:
            v, b = min(cands)
            best_per_dim[d] = b
            P(f"    DIM {d:>4}: BATCH_{b}  ({metric}={v:+.3f})")

    # vincitori APPAIATI: a parità di (DIM, istanza) gli scenari sono identici
    # per tutti i batch -> conto su quante istanze ogni batch è il migliore.
    P("")
    P("  Vittorie appaiate per istanza (stessi scenari per tutti i batch):")
    inst = [r for r in instances if r["exp"] == exp and r["n_nodes"] == n]
    for d in dims:
        by_inst = defaultdict(dict)
        for r in inst:
            if r["DIM"] == d and r[metric.replace("_mean", "")] is not None:
                by_inst[r["istanza"]][r["batch"]] = r[metric]
        wins = defaultdict(int)
        n_conf = 0
        for _i, per_batch in by_inst.items():
            if len(per_batch) < 2:
                continue
            n_conf += 1
            wins[min(per_batch.items(), key=lambda kv: kv[1])[0]] += 1
        if n_conf:
            ranking = sorted(wins.items(), key=lambda kv: -kv[1])
            P(f"    DIM {d:>4} ({n_conf} istanze confrontabili): "
              + ", ".join(f"BATCH_{b}:{w}" for b, w in ranking))

    # diagonale: la rete allenata con batch B rende meglio sul test DIM==B?
    diag_dims = [d for d in dims if d in batches]
    if diag_dims:
        P("")
        P("  Analisi diagonale (batch di training == DIM di test):")
        for d in diag_dims:
            c_diag = get.get((d, d))
            if not (c_diag and c_diag[mkey] is not None):
                continue
            b_best = best_per_dim.get(d)
            v_best = get[(b_best, d)][mkey] if b_best is not None else None
            verdict = ("è il migliore" if b_best == d else
                       f"NON è il migliore (meglio BATCH_{b_best}: {v_best:+.3f})")
            P(f"    DIM {d:>3}: BATCH_{d} = {c_diag[mkey]:+.3f}  → {verdict}")

    # ranking complessivo: rank medio dei batch attraverso i DIM
    P("")
    P("  Ranking complessivo dei batch (rank medio attraverso i DIM; 1 = migliore):")
    ranks = defaultdict(list)
    for d in dims:
        cands = sorted((get[(b, d)][mkey], b) for b in batches
                       if (b, d) in get and get[(b, d)][mkey] is not None)
        for pos, (_v, b) in enumerate(cands, start=1):
            ranks[b].append(pos)
    for b, rs in sorted(ranks.items(), key=lambda kv: st.fmean(kv[1])):
        vals = [get[(b, d)][mkey] for d in dims
                if (b, d) in get and get[(b, d)][mkey] is not None]
        P(f"    BATCH_{b:<3} rank medio {st.fmean(rs):.2f}  "
          f"({metric} medio sui DIM: {st.fmean(vals):+.3f}, "
          f"stabilità tra DIM: std {st.pstdev(vals):.3f})")

    P("")
    P("  Nota: tra DIM diversi gli scenari differiscono, quindi confronta i")
    P("  batch DENTRO ogni colonna; tra colonne usa i gap (scale-free), non i")
    P("  costi assoluti UTSP.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--exp", choices=["PERT", "CVETT"], default=None)
    ap.add_argument("--nodes", type=int, default=None)
    ap.add_argument("--metric", default="gap_sto",
                    choices=["gap_sto", "gap_eev", "gap_pi", "UTSP"])
    ap.add_argument("--no-pi", action="store_true",
                    help="non ricostruire il PI dalla cache")
    ap.add_argument("--out-dir", default=None,
                    help="default: <root>/sweep_analysis")
    args = ap.parse_args()

    instances = discover(args.root, args.exp, args.nodes, with_pi=not args.no_pi)
    if not instances:
        sys.exit(f"Nessun file di test sweep trovato sotto {args.root} "
                 f"(filtri: exp={args.exp}, nodes={args.nodes}). "
                 "I job hanno già scritto in .../test/IS_*_DIM_*/grafici/ ?")

    combos = build_combos(instances)

    out_dir = args.out_dir or os.path.join(args.root, "sweep_analysis")
    os.makedirs(out_dir, exist_ok=True)
    write_csv(os.path.join(out_dir, "test_sweep_instances.csv"), instances)
    write_csv(os.path.join(out_dir, "test_sweep_combos.csv"), combos)

    n_postumi = sum(r.get("_pi_postumo", 0) for r in instances)
    report = [f"TEST SWEEP — {len(instances)} istanze in {len(combos)} combinazioni",
              f"root: {args.root}"]
    if n_postumi:
        report.append(f"PI ricostruito dalla cache per {n_postumi} istanze "
                      f"(i file di stats erano stati scritti con SKIP_PI=1)")
    pairs = sorted({(c["exp"], c["n_nodes"]) for c in combos})
    for exp, n in pairs:
        matrix_report(report, combos, instances, exp, n, args.metric)

    text = "\n".join(report)
    print(text)
    with open(os.path.join(out_dir, "test_sweep_report.txt"), "w",
              encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"\nScritti: {out_dir}/test_sweep_instances.csv, "
          f"test_sweep_combos.csv, test_sweep_report.txt")


if __name__ == "__main__":
    main()
