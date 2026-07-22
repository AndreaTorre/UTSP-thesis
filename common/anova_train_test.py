#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
anova_train_test.py — ANOVA a due vie sulla performance di UTSP.

Risposta = gap per ISTANZA di test.  Fattore A = IS_train (batch), B = dim_test (DIM).

NOTA: il disegno è APPAIATO (per una data (DIM, indice istanza) il blocco di
scenari è identico per tutti i batch), ma questa ANOVA è between-subjects: la
varianza tra istanze finisce nel residuo, quindi la F su IS_train è
CONSERVATIVA. Il test dei segni appaiato (full_report sez. 6) è il riferimento
primario; questa ANOVA è la conferma secondaria.

NOTA: celle mancanti renderebbero SS_interazione e df non validi, quindi i dati
vengono ridotti al rettangolo completo (Tipo I = Tipo III su griglia piena).

Solo stdlib: p-value dalla F via beta incompleta.

Uso:
  python anova_train_test.py
  python anova_train_test.py --csv percorso.csv --metric gap_sto --variant pen0
"""
import argparse
import csv
import math
import os
import statistics as st
from collections import defaultdict


# ---------- distribuzione F (p-value) in stdlib ----------
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def _betai(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def f_pvalue(F, df1, df2):
    """P(F_{df1,df2} > F)."""
    if F <= 0 or df1 <= 0 or df2 <= 0:
        return 1.0
    x = df2 / (df2 + df1 * F)
    return _betai(df2 / 2.0, df1 / 2.0, x)


# ---------- riduzione al rettangolo completo ----------
def complete_rectangle(rows, fa, fb, y):
    """Tiene solo i livelli presenti in TUTTE le combinazioni.
    NOTA: due passate bastano per griglie piccole come questa."""
    cells = {(r[fa], r[fb]) for r in rows if r.get(y) is not None}
    if not cells:
        return [], set(), set()
    A = {a for a, _ in cells}
    B = {b for _, b in cells}
    A = {a for a in A if all((a, b) in cells for b in B)}
    B = {b for b in B if all((a, b) in cells for a in A)}
    keep = [r for r in rows if r[fa] in A and r[fb] in B and r.get(y) is not None]
    return keep, A, B


# ---------- ANOVA a due vie, Tipo I ----------
def two_way_anova(rows, fa, fb, y):
    data = [(r[fa], r[fb], r[y]) for r in rows if r.get(y) is not None]
    N = len(data)
    if N < 4:
        return None
    grand = st.fmean(v for _, _, v in data)
    ss_tot = sum((v - grand) ** 2 for _, _, v in data)

    by_a, by_b, by_ab = defaultdict(list), defaultdict(list), defaultdict(list)
    for a, b, v in data:
        by_a[a].append(v)
        by_b[b].append(v)
        by_ab[(a, b)].append(v)

    la, lb, n_cells = len(by_a), len(by_b), len(by_ab)

    ss_a = sum(len(vs) * (st.fmean(vs) - grand) ** 2 for vs in by_a.values())
    ss_b = sum(len(vs) * (st.fmean(vs) - grand) ** 2 for vs in by_b.values())
    ss_cells = sum(len(vs) * (st.fmean(vs) - grand) ** 2 for vs in by_ab.values())
    ss_ab = ss_cells - ss_a - ss_b
    ss_res = ss_tot - ss_cells

    # df generalizzati: validi anche con un solo livello di B (degenera a una via)
    df_a, df_b = la - 1, lb - 1
    df_ab = n_cells - 1 - df_a - df_b
    df_res = N - n_cells
    if df_res <= 0 or df_a <= 0:
        return None

    ms_res = ss_res / df_res

    def row(name, ss, df):
        if df <= 0:
            return None
        ms = ss / df
        F = ms / ms_res if ms_res > 0 else float("inf")
        return dict(name=name, ss=ss, df=df, ms=ms, F=F,
                    p=f_pvalue(F, df, df_res),
                    eta2=ss / ss_tot if ss_tot > 0 else 0.0)

    return {
        "N": N, "grand": grand, "levels": (la, lb),
        "A": row(fa, ss_a, df_a),
        "B": row(fb, ss_b, df_b),
        "AB": row(f"{fa} x {fb}", ss_ab, df_ab),
        "res": dict(ss=ss_res, df=df_res, ms=ms_res),
        "tot": dict(ss=ss_tot, df=N - 1),
        "cells": {k: (len(v), st.fmean(v)) for k, v in by_ab.items()},
    }


def size_label(eta2):
    if eta2 < 0.01:
        return "effetto trascurabile (<1% varianza)"
    if eta2 < 0.06:
        return "effetto piccolo"
    if eta2 < 0.14:
        return "effetto medio"
    return "effetto grande"


def main():
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--csv", default=os.path.join(here, "sweep_analysis",
                                                  "test_sweep_instances.csv"))
    ap.add_argument("--metric", default="gap_sto",
                    choices=["gap_sto", "gap_eev", "gap_pi"])
    ap.add_argument("--exp", default=None)
    ap.add_argument("--nodes", type=int, default=None)
    ap.add_argument("--variant", default=os.getenv("TESI_VARIANT"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV non trovato: {args.csv}\n"
                         "Genera prima con collect_test_sweep.py")

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if args.exp and r.get("exp") != args.exp:
                continue
            if args.nodes and int(r.get("n_nodes", -1)) != args.nodes:
                continue
            # NOTA: filtro variante solo se la colonna esiste davvero nel CSV
            if args.variant and "variant" in r and r["variant"] != args.variant:
                continue
            v = r.get(args.metric, "")
            try:
                r[args.metric] = float(v) if v not in ("", "None") else None
            except ValueError:
                r[args.metric] = None
            rows.append(r)

    n_raw = sum(1 for r in rows if r.get(args.metric) is not None)
    rows, A_keep, B_keep = complete_rectangle(rows, "batch", "DIM", args.metric)
    n_drop = n_raw - len(rows)

    res = two_way_anova(rows, "batch", "DIM", args.metric)
    if res is None:
        raise SystemExit("Dati insufficienti per l'ANOVA (servono più celle/istanze).")

    L = []

    def P(*a):
        L.append(" ".join(str(x) for x in a) if a else "")

    la, lb = res["levels"]
    P("=" * 84)
    P(f"ANOVA A DUE VIE — gap di UTSP ({args.metric}) vs IS_train e dim_test")
    filt = [x for x in (f"exp={args.exp}" if args.exp else None,
                        f"nodi={args.nodes}" if args.nodes else None,
                        f"variante={args.variant}" if args.variant else None) if x]
    if filt:
        P("(filtro: " + ", ".join(filt) + ")")
    P("=" * 84)
    P(f"Osservazioni (istanze): {res['N']}")
    P(f"Livelli IS_train: {la} {sorted(A_keep)}")
    P(f"Livelli dim_test: {lb} {sorted(B_keep)}")
    P(f"Gap medio complessivo: {res['grand']:+.4f}%")
    if n_drop:
        P(f"Righe scartate per rendere il disegno completo: {n_drop}")
        P("  (celle mancanti renderebbero SS_interazione e df non validi)")
    if lb < 2:
        P("ATTENZIONE: un solo livello di dim_test -> l'analisi degenera a UNA via")
        P("  su IS_train; le righe dim_test e interazione non sono calcolabili.")
    P("")
    P("Somme dei quadrati di Tipo I (sequenziale IS_train -> dim_test -> interazione).")
    P("Su griglia completa coincidono con Tipo II/III.")
    P("")
    P(f"  {'sorgente':<22} {'SS':>12} {'df':>5} {'MS':>12} {'F':>9} {'p-value':>10} {'eta^2':>8}")
    P("  " + "-" * 80)
    for key in ("A", "B", "AB"):
        r = res[key]
        if r is None:
            continue
        P(f"  {r['name']:<22} {r['ss']:>12.3f} {r['df']:>5} {r['ms']:>12.4f} "
          f"{r['F']:>9.3f} {r['p']:>10.4g} {r['eta2']:>8.4f}")
    rr = res["res"]
    P(f"  {'Residuo':<22} {rr['ss']:>12.3f} {rr['df']:>5} {rr['ms']:>12.4f}")
    tt = res["tot"]
    P(f"  {'Totale':<22} {tt['ss']:>12.3f} {tt['df']:>5}")
    P("")

    P("NUMEROSITÀ PER CELLA")
    P("-" * 84)
    for (a, b), (n, m) in sorted(res["cells"].items()):
        P(f"  IS{a:<4} DIM{b:<4}  n={n:<6}  gap medio = {m:+.3f}%")
    P("")

    P("INTERPRETAZIONE")
    P("-" * 84)
    for key, label in (("A", "IS_train (dimensione batch di training)"),
                       ("B", "dim_test (dimensione istanza di test)"),
                       ("AB", "interazione IS_train x dim_test")):
        r = res[key]
        if r is None:
            P(f"  {label}: non calcolabile con i livelli disponibili.")
            continue
        sig = "significativo" if r["p"] < 0.05 else "NON significativo"
        P(f"  {label}:")
        P(f"    p = {r['p']:.4g} ({sig});  eta^2 = {r['eta2']:.4f} ({size_label(r['eta2'])})")
    P("")
    P("  Con molte istanze per cella anche effetti minuscoli risultano significativi")
    P("  (p piccolo). eta^2 dice se l'effetto è anche GRANDE. Un fattore con p<0.05")
    P("  ma eta^2<0.01 è rilevabile ma praticamente irrilevante.")
    P("")

    P("CONCLUSIONE")
    P("-" * 84)
    for key, label in (("A", "IS_train"), ("B", "dim_test"), ("AB", "interazione")):
        r = res[key]
        if r is None:
            continue
        P(f"  {label}: spiega il {100*r['eta2']:.2f}% della varianza del gap "
          f"(p={r['p']:.3g}).")
    P("  L'assenza di effetto non si dimostra: si mostra che è al più di questa")
    P("  ampiezza. Inoltre il disegno non è bloccato sull'istanza, quindi la stima")
    P("  su IS_train è CONSERVATIVA: incrociare col test dei segni appaiato")
    P("  (full_report, sezione 6) prima di concludere.")

    out = args.out or os.path.join(os.path.dirname(args.csv), "anova_train_test.txt")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nSalvato in: {out}")


if __name__ == "__main__":
    main()