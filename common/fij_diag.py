#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostica f_ij: frequenza d'uso di ogni corridoio nel TOUR LIBERO per scenario.
Legge SOLO res_B_cached.pkl (contiene 'I' e 'results' con exact_free['arcs']).
Nessun solver, nessun config da caricare.

    python fij_diag.py /percorso/RISULTATI_N/pkl/res_B_cached.pkl

Lettura:
- se ogni f e' 0.00 o 1.00  -> tour rigido: VSS=0 strutturale, i parametri non bastano.
- se qualche f e' ~0.3..0.7 -> il tour flippa tra scenari: c'e' margine per VSS>0.
"""
import sys
import pickle
from collections import Counter


def canon(i, j):
    return (i, j) if i < j else (j, i)


def main(path):
    with open(path, "rb") as f:
        data = pickle.load(f)

    I = data["I"]
    results = data["results"]
    x_ev = data.get("x_ev", set())
    x_sto = data.get("x_used_sto", data.get("x_sto", set()))

    I_set = {canon(*e) for e in I}
    x_ev = {canon(*e) for e in x_ev}
    x_sto = {canon(*e) for e in x_sto}

    cnt = Counter()
    n_ok = 0
    n_missing = 0
    for sid, r in results.items():
        ef = r.get("exact_free") or {}
        arcs = ef.get("arcs")
        if not arcs:
            n_missing += 1
            continue
        n_ok += 1
        used = {canon(*a) for a in arcs}
        for e in I_set & used:
            cnt[e] += 1

    if n_ok == 0:
        print("NESSUN exact_free['arcs'] popolato: la cache non ha i tour liberi "
              "(generata con solve_pi=False?). Serve rigenerare Experiment B con i PI.")
        return

    print(f"Scenari con tour libero: {n_ok}"
          + (f"  (saltati senza arcs: {n_missing})" if n_missing else ""))
    print(f"x_sto == x_ev ? {'SI' if x_sto == x_ev else 'NO'}"
          f"   |x_sto|={len(x_sto)}  |x_ev|={len(x_ev)}")
    print("-" * 52)
    print(f"{'corridoio':>14}  {'f':>5}   book_sto  book_ev")
    intermediate = []
    for e in sorted(I_set):
        f = cnt[e] / n_ok
        tag_s = "*" if e in x_sto else " "
        tag_e = "*" if e in x_ev else " "
        print(f"{str(e):>14}  {f:5.2f}     {tag_s}         {tag_e}")
        if 0.05 < f < 0.95:
            intermediate.append((e, f))
    print("-" * 52)
    if intermediate:
        print(f"CORRIDOI CHE FLIPPANO (0.05<f<0.95): {len(intermediate)}")
        for e, f in intermediate:
            print(f"   {e}  f={f:.2f}")
        print(">> C'e' struttura latente: ha senso una probe Gurobi mirata.")
    else:
        print("TUTTI gli f sono ~0 o ~1: tour rigido su ogni scenario.")
        print(">> VSS=0 strutturale a questo setup; nessun parametro lo sposta.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("uso: python fij_diag.py <res_B_cached.pkl>")
    main(sys.argv[1])
