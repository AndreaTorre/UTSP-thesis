#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnose_booking.py — la regola f > p/C è valida?

Scompone l'errore della decisione di prenotazione in due parti:

  (A) errore di STIMA   : f^LS (local search) vs f^PI (Gurobi esatto)
  (B) errore di REGOLA  : decisione da f^PI vs x_STO (ottimo two-stage)
  (C) errore TOTALE     : x_test (quella usata davvero) vs x_STO

Perché f^PI è la quantità giusta: la regola prenota se p < f·C, dove f è la
frequenza con cui useresti l'arco SE FOSSE GRATUITO (cioè dopo la prenotazione).
Il tour PI è il TSP libero ottimo dello scenario — ignora p e C — quindi la
frequenza degli archi di I nei tour PI è ESATTAMENTE quella f, calcolata da
Gurobi invece che stimata dalla local search.

LIMITE NOTO DELLA REGOLA (da discutere in tesi): assume che, non prenotando,
pagheresti comunque la multa f·C. In realtà la LS può DEVIARE ed evitare
l'arco, quindi il costo vero del non-prenotare è T_pen - T_free <= f·C.
La regola sovrastima il beneficio della prenotazione => tende a SOVRA-prenotare.
Il confronto (B) contro x_STO misura proprio questo bias.

Uso:
  TESI_EXPERIMENT=PERT TESI_N_NODES=15 python diagnose_booking.py
  TESI_EXPERIMENT=PERT TESI_N_NODES=15 python diagnose_booking.py --batch 20 --dim 60
"""
import argparse
import ast
import glob
import os
import pickle
import re
import sys
from collections import Counter, defaultdict

from config import TEST_SCENARIO_CACHE_DIR, EXPERIMENT, N_NODES
from tsp_utils import canon_edge, get_edge_value

RESULTS_ROOT = os.path.dirname(os.path.dirname(TEST_SCENARIO_CACHE_DIR))  # .../RISULTATI_N/..
X_TEST_RE = re.compile(r"x_test\s*=\s*(\[.*?\])", re.S)


def load_res_B():
    path = os.path.join(TEST_SCENARIO_CACHE_DIR, "res_B_cached.pkl")
    if not os.path.exists(path):
        sys.exit(f"res_B non trovato: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_scenarios():
    path = os.path.join(TEST_SCENARIO_CACHE_DIR, "test_scenarios_cache.pkl")
    if not os.path.exists(path):
        sys.exit(f"Cache scenari non trovata: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)["results"]


def f_from_pi(results, I_set):
    """Frequenza esatta d'uso degli archi di I nei tour PI (TSP libero)."""
    used = Counter()
    n = 0
    n_no_pi = 0
    for sid, rec in results.items():
        tour = rec.get("exact_free", {}).get("tour")
        if not tour:
            n_no_pi += 1
            continue
        n += 1
        arcs = [(tour[k], tour[(k + 1) % len(tour)]) for k in range(len(tour))]
        for (a, b) in arcs:
            ce = canon_edge(a, b)
            if ce in I_set:
                used[ce] += 1
    return ({e: used[e] / n for e in I_set} if n else {}), n, n_no_pi


def x_test_per_instance(batch=None, dim=None):
    """Legge x_test da ogni file di stats delle combinazioni del test sweep."""
    root = os.path.dirname(os.path.dirname(TEST_SCENARIO_CACHE_DIR))
    pattern = os.path.join(
        root, f"RISULTATI_{N_NODES}", "batch_sweep",
        f"BATCH_{batch}" if batch else "BATCH_*",
        "test", f"IS_*_DIM_{dim}" if dim else "IS_*_DIM_*",
        "grafici", "*_test_only_i*_test_only_stats.txt")
    out = defaultdict(list)
    for path in glob.glob(pattern):
        m = re.search(r"BATCH_(\d+)/test/IS_(\d+)_DIM_(\d+)/", path.replace(os.sep, "/"))
        if not m:
            continue
        text = open(path, encoding="utf-8", errors="replace").read()
        xm = X_TEST_RE.search(text)
        if not xm:
            continue
        try:
            arcs = ast.literal_eval(xm.group(1))
        except Exception:
            continue
        key = (int(m.group(1)), int(m.group(3)))  # (batch, DIM)
        out[key].append({canon_edge(i, j) for (i, j) in arcs})
    return out


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--out", default=None, help="percorso del file di report")
    args = ap.parse_args()

    res_B = load_res_B()
    I = res_B["I"]
    p, C = res_B["p"], res_B["C"]
    I_set = {canon_edge(i, j) for (i, j) in I}
    x_sto = {canon_edge(i, j) for (i, j) in res_B["x_used_sto"]}
    x_ev = {canon_edge(i, j) for (i, j) in res_B["x_ev"]}

    results = load_scenarios()
    f_pi, n_pi, n_missing = f_from_pi(results, I_set)

    LINES = []

    def P(*a):
        line = " ".join(str(x) for x in a) if a else ""
        LINES.append(line)
        print(line)

    def save():
        root = os.path.dirname(os.path.dirname(TEST_SCENARIO_CACHE_DIR))
        out_path = args.out or os.path.join(
            os.path.dirname(root), "sweep_analysis",
            f"prenotazioni_{EXPERIMENT}_{N_NODES}nodi.txt")
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(LINES) + "\n")
        print(f"\nReport salvato in: {out_path}")

    P("=" * 92)
    P(f"DIAGNOSI REGOLA DI PRENOTAZIONE  —  {EXPERIMENT} {N_NODES} nodi")
    P("=" * 92)
    if not f_pi:
        sys.exit("\nNessun tour PI in cache. Lancia compute_pi_shard.py + merge_pi_shards.py.")
    P(f"Scenari con PI risolto: {n_pi}" + (f"   (senza PI: {n_missing})" if n_missing else ""))
    P(f"Archi prenotabili |I| = {len(I_set)}\n")

    # ---- (B) decisione della regola con f esatta, vs STO ----
    P("Per ogni arco di I:  f^PI = frequenza d'uso nel tour PI (TSP libero, esatta)")
    P("                     soglia = p/C ;  la regola prenota se f^PI > soglia\n")
    hdr = f"  {'arco':>12} | {'f^PI':>6} | {'soglia p/C':>10} | {'REGOLA':>8} | {'STO':>5} | {'EEV':>5} | esito"
    P(hdr)
    P("  " + "-" * (len(hdr) - 2))

    x_rule_pi = set()
    agree = disagree_over = disagree_under = 0
    for e in sorted(I_set):
        i, j = e
        f = f_pi[e]
        pv, Cv = get_edge_value(p, i, j), get_edge_value(C, i, j)
        soglia = pv / Cv if Cv > 0 else float("inf")
        book = f > soglia
        if book:
            x_rule_pi.add(e)
        in_sto, in_ev = e in x_sto, e in x_ev

        if book == in_sto:
            esito, _ = "concorde", agree
            agree += 1
        elif book and not in_sto:
            esito = "SOVRA-prenota (regola sì, STO no)"
            disagree_over += 1
        else:
            esito = "SOTTO-prenota (regola no, STO sì)"
            disagree_under += 1

        P(f"  {str(e):>12} | {f:>6.3f} | {soglia:>10.3f} | "
              f"{'PRENOTA' if book else '   no  ':>8} | {'sì' if in_sto else 'no':>5} | "
              f"{'sì' if in_ev else 'no':>5} | {esito}")

    P("\n  (B) REGOLA con input esatto vs STO (ottimo two-stage):")
    P(f"      concordi: {agree}/{len(I_set)}   "
          f"sovra-prenota: {disagree_over}   sotto-prenota: {disagree_under}")
    P(f"      x_regola(f^PI) = {sorted(x_rule_pi)}")
    P(f"      x_STO          = {sorted(x_sto)}")
    P(f"      Jaccard        = {jaccard(x_rule_pi, x_sto):.3f}")
    if disagree_over > disagree_under:
        P("      → bias di SOVRA-prenotazione, come atteso: la regola assume che senza")
        P("        prenotazione pagheresti f·C, ignorando che la LS può deviare.")

    # ---- (C) x_test davvero usata, vs STO / vs regola-esatta ----
    per_combo = x_test_per_instance(args.batch, args.dim)
    if not per_combo:
        P("\n(Nessun x_test trovato nei file di stats: salto i confronti (A) e (C).)")
        save()
        return

    P("\n" + "=" * 92)
    P("  (C) x_test EFFETTIVA (dalla local search, per istanza) vs STO e vs regola-esatta")
    P("=" * 92)
    P(f"  {'batch':>5} {'DIM':>5} {'ist.':>5} | {'J vs STO':>9} | {'J vs regola(f^PI)':>18} | archi instabili")
    P("  " + "-" * 88)

    for (batch, dim) in sorted(per_combo):
        xs = per_combo[(batch, dim)]
        j_sto = sum(jaccard(x, x_sto) for x in xs) / len(xs)
        j_rule = sum(jaccard(x, x_rule_pi) for x in xs) / len(xs)
        # archi che cambiano decisione tra istanze: instabilità della regola
        freq = Counter()
        for x in xs:
            freq.update(x)
        unstable = {e: freq[e] / len(xs) for e in I_set if 0 < freq[e] < len(xs)}
        us = ", ".join(f"{e}:{v:.0%}" for e, v in sorted(unstable.items())) or "nessuno"
        P(f"  {batch:>5} {dim:>5} {len(xs):>5} | {j_sto:>9.3f} | {j_rule:>18.3f} | {us}")

    P("\n  J = Jaccard (1.0 = decisioni identiche). 'archi instabili' = archi prenotati")
    P("  in alcune istanze e non in altre: se ce ne sono, la regola è al confine e")
    P("  piccole fluttuazioni di f fanno flippare la decisione (spiega la varianza")
    P("  extra tra istanze osservata sui 15 nodi).")

    save()


if __name__ == "__main__":
    main()
