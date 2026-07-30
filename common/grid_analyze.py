#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
grid_analyze.py — legge i grid_summary.csv e dice quali parametri contano.

Perche' non basta ordinare per gap e prendere la prima riga: con 486
configurazioni la migliore singola puo' essere rumore di una particolare
estrazione di scenari. Su un fattoriale completo l'effetto marginale di ogni
parametro si legge direttamente, mediando sugli altri.

Uso:
  python grid_analyze.py --exp PERT --nodes 15 --batch 20
  python grid_analyze.py --all                      # tutte le foglie disponibili
  python grid_analyze.py --all --metric gap_ls_sto --top 10

NOTA: solo stdlib. f_pvalue e size_label arrivano da anova_train_test.py,
che le implementa gia': non le riscrivo.
"""

import argparse
import csv
import glob
import os
import statistics as st
from collections import defaultdict

from anova_train_test import f_pvalue, size_label

DEFAULT_METRIC = "gap_ls_ws"


# ---------------------------------------------------------------- lettura

def project_root():
    return os.getenv("TESI_ROOT_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_leaves(exp=None, nodes=None, batch=None):
    pat = os.path.join(project_root(), "grid_search",
                       exp or "*", f"NODI_{nodes}" if nodes else "NODI_*",
                       f"BATCH_{batch}" if batch else "BATCH_*", "grid_summary.csv")
    return sorted(glob.glob(pat))


def read_leaf(path, metric):
    """Righe riuscite di una foglia, con i valori numerici convertiti."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status") != "ok":
                continue
            out = {}
            for k, v in r.items():
                if v in (None, "", "nan"):
                    continue
                try:
                    out[k] = float(v)
                except ValueError:
                    out[k] = v
            if isinstance(out.get(metric), float):
                rows.append(out)
    return rows


def grid_params(rows):
    """Colonne che variano davvero: sono i parametri della griglia.
    NOTA: dedotte dai dati, non da GRID, cosi' l'analisi resta valida anche
    su CSV prodotti da una griglia diversa da quella attuale."""
    if not rows:
        return []
    cand = [k for k in rows[0] if k.startswith(("UTSP", "_"))]
    return [k for k in cand if len({r.get(k) for r in rows}) > 1]


# ---------------------------------------------------------------- analisi

def one_way(rows, factor, metric):
    """ANOVA a una via sul fattore, mediando su tutti gli altri.
    Su un fattoriale completo questo E' l'effetto marginale."""
    by = defaultdict(list)
    for r in rows:
        if factor in r and isinstance(r.get(metric), float):
            by[r[factor]].append(r[metric])
    if len(by) < 2:
        return None
    vals = [v for vs in by.values() for v in vs]
    N, k = len(vals), len(by)
    if N - k <= 0:
        return None
    grand = st.fmean(vals)
    ss_tot = sum((v - grand) ** 2 for v in vals)
    ss_b = sum(len(vs) * (st.fmean(vs) - grand) ** 2 for vs in by.values())
    ss_w = ss_tot - ss_b
    df_b, df_w = k - 1, N - k
    ms_w = ss_w / df_w
    F = (ss_b / df_b) / ms_w if ms_w > 0 else float("inf")
    return {
        "factor": factor, "F": F, "p": f_pvalue(F, df_b, df_w),
        "eta2": ss_b / ss_tot if ss_tot > 0 else 0.0,
        "levels": {lv: (len(vs), st.fmean(vs),
                        st.pstdev(vs) if len(vs) > 1 else 0.0)
                   for lv, vs in sorted(by.items())},
    }


def report_leaf(path, metric, top):
    rows = read_leaf(path, metric)
    label = os.path.relpath(os.path.dirname(path), os.path.join(project_root(), "grid_search"))
    print("\n" + "=" * 78)
    print(f"{label}   ({len(rows)} configurazioni riuscite, metrica: {metric})")
    print("=" * 78)
    if len(rows) < 4:
        print("  Troppo poche righe per un'analisi.")
        return None

    params = grid_params(rows)
    effects = [e for e in (one_way(rows, p, metric) for p in params) if e]
    effects.sort(key=lambda e: -e["eta2"])

    print("\n  QUALI PARAMETRI CONTANO  (eta2 = quota di varianza spiegata)\n")
    print(f"  {'parametro':<22} {'eta2':>7} {'p':>9}   giudizio")
    print(f"  {'-'*22} {'-'*7} {'-'*9}   {'-'*34}")
    for e in effects:
        print(f"  {e['factor']:<22} {e['eta2']:>7.4f} {e['p']:>9.2e}   {size_label(e['eta2'])}")

    print("\n  EFFETTO MARGINALE PER LIVELLO  (media su tutte le altre combinazioni)\n")
    for e in effects:
        best = min(e["levels"].items(), key=lambda kv: kv[1][1])[0]
        print(f"  {e['factor']}")
        for lv, (n, mu, sd) in e["levels"].items():
            star = "  <-- migliore" if lv == best else ""
            print(f"      {lv:>10} : {mu:>9.4f}  (sd {sd:>7.4f}, n={n}){star}")
        if e["eta2"] < 0.01:
            print("      NOTA: effetto trascurabile, il valore si puo' fissare.")
        elif best == min(e["levels"]) or best == max(e["levels"]):
            print("      NOTA: ottimo sul BORDO del range, conviene estenderlo.")
        print()

    rows.sort(key=lambda r: r[metric])
    std_col = f"{metric}_std" if f"{metric}_std" in rows[0] else None
    print(f"  MIGLIORI {top} CONFIGURAZIONI\n")
    head = f"  {'idx':>5} {metric:>11}"
    if std_col:
        head += f" {'sd_istanze':>11}"
    head += "   " + "  ".join(f"{p.replace('UTSP2_',''):>10}" for p in params)
    print(head)
    for r in rows[:top]:
        line = f"  {int(r['idx']):>5} {r[metric]:>11.4f}"
        if std_col:
            line += f" {r.get(std_col, float('nan')):>11.4f}"
        line += "   " + "  ".join(f"{r.get(p, ''):>10}" for p in params)
        print(line)

    # La differenza fra le prime e' distinguibile dal rumore fra istanze?
    if std_col and len(rows) > 1:
        n_ist = int(rows[0].get("n_istanze", 0)) or None
        sd = rows[0].get(std_col)
        if sd and n_ist:
            se = sd / (n_ist ** 0.5)
            entro = sum(1 for r in rows if r[metric] - rows[0][metric] < 2 * se)
            print(f"\n  Errore standard della migliore: {se:.4f} (sd {sd:.4f} su {n_ist} istanze)")
            print(f"  Configurazioni entro 2 SE dalla migliore: {entro}")
            if entro > 1:
                print("  => la prima NON e' distinguibile dalle altre: scegli in base")
                print("     agli effetti marginali qui sopra, non alla singola riga.")

    return {"label": label, "rows": rows, "params": params, "effects": effects}


def cross_leaf(reports, metric, top):
    """I parametri migliori sono gli stessi su tutte le foglie?"""
    print("\n" + "=" * 78)
    print(f"CONSISTENZA FRA LE {len(reports)} FOGLIE")
    print("=" * 78)

    ranks = defaultdict(list)
    for rep in reports:
        for pos, e in enumerate(rep["effects"], 1):
            ranks[e["factor"]].append((pos, e["eta2"]))
    print("\n  Importanza media dei parametri\n")
    print(f"  {'parametro':<22} {'rango medio':>12} {'eta2 medio':>12}")
    for p, vs in sorted(ranks.items(), key=lambda kv: st.fmean(x[0] for x in kv[1])):
        print(f"  {p:<22} {st.fmean(x[0] for x in vs):>12.2f} {st.fmean(x[1] for x in vs):>12.4f}")

    best_lv = defaultdict(lambda: defaultdict(int))
    for rep in reports:
        for e in rep["effects"]:
            best_lv[e["factor"]][min(e["levels"].items(), key=lambda kv: kv[1][1])[0]] += 1
    print("\n  Livello migliore per foglia (quante foglie lo preferiscono)\n")
    for p, c in best_lv.items():
        votes = ", ".join(f"{lv}={n}" for lv, n in sorted(c.items(), key=lambda kv: -kv[1]))
        unan = "  [unanime]" if len(c) == 1 else ""
        print(f"  {p:<22} {votes}{unan}")

    common = None
    for rep in reports:
        s = {int(r["idx"]) for r in rep["rows"][:top]}
        common = s if common is None else (common & s)
    print(f"\n  Configurazioni nelle prime {top} di TUTTE le foglie: "
          f"{sorted(common) if common else 'nessuna'}")
    if not common:
        print("  => nessun vincitore unico: usa i livelli marginali qui sopra.")


def main():
    ap = argparse.ArgumentParser(description="Analizza i grid_summary.csv della grid search.")
    ap.add_argument("--exp", choices=["PERT", "CVETT"])
    ap.add_argument("--nodes", type=int, choices=[15, 25, 40])
    ap.add_argument("--batch", type=int)
    ap.add_argument("--all", action="store_true", help="tutte le foglie disponibili")
    ap.add_argument("--metric", default=DEFAULT_METRIC)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()

    leaves = find_leaves(a.exp, a.nodes, a.batch) if (a.all or a.exp or a.nodes or a.batch) \
        else find_leaves()
    if not leaves:
        raise SystemExit("Nessun grid_summary.csv trovato. Lancia prima: "
                         "python grid_search.py --collect --exp ... --nodes ... --batch ...")

    reports = [r for r in (report_leaf(p, a.metric, a.top) for p in leaves) if r]
    if len(reports) > 1:
        cross_leaf(reports, a.metric, a.top)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
