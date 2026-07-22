#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
regen_graphs.py — grafo pesato POST-BOOKING, uno per (variante, batch, dimensione test).

Approccio (riscritto da zero):
  - itera ESPLICITAMENTE su ogni cartella test/IS_*_DIM_*/ (nessun find_stats_file
    ambiguo): ogni cella ha il suo file e produce il suo grafo;
  - legge i tour post-booking (unica colonna 'tour post' del formato test_only);
  - peso arco = frazione di tour che usano l'arco (frequenza d'uso reale);
  - archi in I (prenotabili) in rosso, gli altri blu graduato.

NOTA: i file di test contengono SOLO il tour post-booking. Il grafo pre-booking
non è generabile da qui (il percorso pre non è salvato, solo il suo costo).

Uso:
  python regen_graphs.py                         # tutte le celle, tutte le dim
  python regen_graphs.py --nodes 15 --variante pen1
  python regen_graphs.py --dim 60                # solo DIM_60
  python regen_graphs.py --threshold 0.02
"""
import argparse
import ast
import glob
import os
import re
import sys

PROJECT_ROOT = os.environ.get("TESI_ROOT_DIR",
                              os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COMMON_DIR = os.path.join(PROJECT_ROOT, "common")
sys.path.insert(0, COMMON_DIR)

import numpy as np
from common import load_data
import plots

# riga scenario: "  <sid> | ... | [tour]" — il tour è l'ULTIMA lista sulla riga.
TOUR_RE = re.compile(r"\[[\d,\s]+\]\s*$")


def load_nodes_coords():
    data = load_data()
    if len(data) == 5:
        nodes, coords, _b, _E, _r = data
    elif len(data) == 4:
        nodes, coords, _E, _r = data
    else:
        raise ValueError(f"load_data(): {len(data)} valori inattesi.")
    return list(nodes), coords


def parse_post_tours(path):
    """Estrae i tour post-booking. Ogni riga-scenario ha il tour come ultima
    lista [...] della riga. Ignora header, righe '# ISTANZA', testo."""
    tours = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            # solo righe che iniziano con uno scenario numerico e contengono '|'
            if "|" not in line:
                continue
            first = line.split("|", 1)[0].strip()
            if not first.isdigit():
                continue
            m = TOUR_RE.search(line.rstrip())
            if not m:
                continue
            try:
                tour = ast.literal_eval(m.group(0))
            except Exception:
                continue
            tour = [int(v) for v in tour]
            if len(tour) >= 2 and tour[0] == tour[-1]:
                tour = tour[:-1]          # togli chiusura; build la riaggiunge
            if len(tour) >= 2:
                tours.append(tour)
    return tours


def edge_frequency(tours, nodes):
    """freq[i,j] = frazione di tour che usano l'arco orientato i->j (incl. ritorno).
    Ritorna (freq, n_validi)."""
    idx = {v: k for k, v in enumerate(nodes)}
    counts = np.zeros((len(nodes), len(nodes)))
    n = 0
    scartati = 0
    for tour in tours:
        if any(v not in idx for v in tour):
            scartati += 1
            continue
        for a, b in zip(tour, tour[1:] + tour[:1]):
            counts[idx[a], idx[b]] += 1.0
        n += 1
    if scartati:
        print(f"    (scartati {scartati} tour con nodi fuori mappa)")
    return (counts / n if n else counts), n


def parse_x_test(path):
    """Legge x_test dal primo blocco: gli archi prenotati (per colorarli in rosso)."""
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip().startswith("x_test"):
                m = re.search(r"\[.*\]", line)
                if m:
                    try:
                        return [tuple(e) for e in ast.literal_eval(m.group(0))]
                    except Exception:
                        return None
    return None


def find_test_cells(root, exp, nodes_n, variante=None, dim=None):
    """Trova ogni cartella test/IS_*_DIM_*/ e il suo file test_all_instances.
    Ritorna lista di (variante, batch, dim, filepath)."""
    base = os.path.join(root, exp, f"RISULTATI_{nodes_n}", "variants")
    vpat = variante or "*"
    dpat = f"DIM_{dim}" if dim else "DIM_*"
    pattern = os.path.join(base, vpat, "batch_sweep", "BATCH_*",
                           "test", f"IS_*_{dpat}", "grafici",
                           "*_test_all_instances.txt")
    cells = []
    for f in sorted(glob.glob(pattern)):
        m = re.search(r"variants/([^/]+)/batch_sweep/(BATCH_\d+)/test/IS_\d+_(DIM_\d+)/", f)
        if m:
            cells.append((m.group(1), m.group(2), m.group(3), f))
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=15)
    ap.add_argument("--exp", default="PERT")
    ap.add_argument("--variante", default=None, help="pen0 | pen1 (default: tutte)")
    ap.add_argument("--dim", type=int, default=None, help="20|30|60|70 (default: tutte)")
    ap.add_argument("--threshold", type=float, default=0.01)
    args = ap.parse_args()

    nodes, coords = load_nodes_coords()

    # I archi prenotabili, da res_B (per colorarli in rosso)
    I = None
    resb = os.path.join(PROJECT_ROOT, args.exp, f"RISULTATI_{args.nodes}",
                        "pkl", "res_B_cached.pkl")
    if os.path.exists(resb):
        import pickle
        try:
            I = pickle.load(open(resb, "rb")).get("I")
        except Exception:
            I = None

    cells = find_test_cells(PROJECT_ROOT, args.exp, args.nodes, args.variante, args.dim)
    print(f"Celle di test trovate: {len(cells)}")
    if not cells:
        print("Nessuna cella. Controlla --nodes/--variante/--dim e che i test siano finiti.")
        return

    fatti = 0
    for variante, batch, dimtag, path in cells:
        tours = parse_post_tours(path)
        if not tours:
            print(f"  {variante}/{batch}/{dimtag}: 0 tour, salto.")
            continue
        freq, n = edge_frequency(tours, nodes)
        if n == 0:
            print(f"  {variante}/{batch}/{dimtag}: 0 tour validi, salto.")
            continue
        x_test = parse_x_test(path)
        I_plot = I if I is not None else x_test  # se manca res_B, usa x_test per il rosso
        out_dir = os.path.dirname(path)
        out = os.path.join(out_dir, f"{variante}_{batch}_{dimtag}_post_freq.png")
        title = f"Post-booking — {variante} {batch} {dimtag} (n={n} tour)"
        plots.grafo_pesato_frequenza(nodes, coords, freq, out,
                                     I=I_plot, threshold=args.threshold, title=title)
        fatti += 1

    print(f"\nGrafi generati: {fatti}/{len(cells)}")


if __name__ == "__main__":
    main()