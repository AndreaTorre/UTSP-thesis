#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_mtsp.py — confronto APPAIATO tra il modello single-TSP e il modello
multi-grafo (mtsp), valutati sullo STESSO grafo di test.

Perché è pulito: STO/EEV/PI sono policy FISSE (decise in Experiment B), quindi
valgono lo stesso identico numero per entrambi i modelli. L'unica cosa che
cambia è UTSP. Il confronto è quindi appaiato scenario-per-scenario: differenza
di costo UTSP (mtsp - single), a parità di tutto il resto.

Interpretazione: il modello single-TSP si è allenato SUL grafo di test
(memorizzazione); il mtsp si è allenato su ALTRI grafi (generalizzazione). Se
mtsp ~ single sul grafo di test, la generalizzazione regge quanto la memorizzazione.

Uso:
  TESI_EXPERIMENT=PERT python compare_mtsp.py --nodes 15 25
  (opz.) --root <repo>  --exp PERT  --mtsp-tag mtsp  --out confronto_mtsp.txt
Legge i file report/*_test_all_instances.txt di entrambe le run (con o senza
batch_sweep). Non ricalcola nulla: usa i risultati già prodotti dall'eval.
"""
import argparse
import glob
import os
import statistics as st

# riuso del parser autorevole: etichette/colonne sono congelate ("NON cambiare")
from full_report import FIELD_RE, SCEN_RE, BLOCK_RE, _num


def _find_reports(root, exp, n, variant):
    vseg = os.path.join("variants", variant) if variant else ""
    pats = [
        os.path.join(root, exp, f"RISULTATI_{n}", vseg, "report", "*_test_all_instances.txt"),
        os.path.join(root, exp, f"RISULTATI_{n}", vseg, "batch_sweep", "BATCH_*",
                     "report", "*_test_all_instances.txt"),
    ]
    return sorted({f for p in pats for f in glob.glob(p)})


def _parse(files):
    """Lista di istanze: {UTSP,STO,EEV,PI, istanza, scen:{sid:post_cost}, src}."""
    out = []
    for path in files:
        text = open(path, encoding="utf-8", errors="replace").read()
        blocks = [(m.start(), int(m.group("i"))) for m in BLOCK_RE.finditer(text)]
        for k, (pos, idx) in enumerate(blocks):
            end = blocks[k + 1][0] if k + 1 < len(blocks) else len(text)
            blk = text[pos:end]
            r = {kk: (_num(rx.search(blk).group(1)) if rx.search(blk) else None)
                 for kk, rx in FIELD_RE.items()}
            scen = {}
            for m in SCEN_RE.finditer(blk):
                post = _num(m.group(3))
                if post is not None:
                    scen[int(m.group(1))] = post
            r.update(istanza=idx, scen=scen, src=path)
            out.append(r)
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else float("nan")


def _agg(rows):
    def gap(a, b):
        g = [(r[a] - r[b]) / r[b] * 100 for r in rows
             if r.get(a) is not None and r.get(b) not in (None, 0)]
        return st.mean(g) if g else float("nan")
    return {
        "istanze": len(rows),
        "scenari": sum(len(r["scen"]) for r in rows),
        "UTSP": _mean([r["UTSP"] for r in rows]),
        "STO": _mean([r["STO"] for r in rows]),
        "EEV": _mean([r["EEV"] for r in rows]),
        "PI": _mean([r["PI"] for r in rows]),
        "gap_STO": gap("UTSP", "STO"),
        "gap_EEV": gap("UTSP", "EEV"),
        "gap_PI": gap("UTSP", "PI"),
    }


def _paired(single_rows, mtsp_rows):
    """Confronto appaiato del costo UTSP per (istanza, scenario). >0 = mtsp peggiore."""
    def index(rows):
        d = {}
        for r in rows:
            for sid, c in r["scen"].items():
                d[(r["istanza"], sid)] = c
        return d
    a, b = index(single_rows), index(mtsp_rows)
    common = sorted(set(a) & set(b))
    diffs = [b[k] - a[k] for k in common]
    if not diffs:
        return None
    mtsp_better = sum(1 for d in diffs if d < -1e-9)
    ties = sum(1 for d in diffs if abs(d) <= 1e-9)
    return {
        "n": len(diffs),
        "mean_diff": st.mean(diffs),
        "median_diff": st.median(diffs),
        "mtsp_better": mtsp_better,
        "ties": ties,
        "single_better": len(diffs) - mtsp_better - ties,
        "pct_mtsp_better": 100 * mtsp_better / len(diffs),
    }


def _fmt(x, p=4):
    return "N/A" if x is None or (isinstance(x, float) and x != x) else f"{x:.{p}f}"


def report_for_n(root, exp, n, mtsp_tag, single_tag):
    single = _parse(_find_reports(root, exp, n, single_tag))
    mtsp = _parse(_find_reports(root, exp, n, mtsp_tag))
    lines = [f"{'='*70}", f"N = {n}", f"{'='*70}"]
    if not single:
        lines.append(f"  [!] nessun report single-TSP trovato (variant={single_tag or 'base'}).")
    if not mtsp:
        lines.append(f"  [!] nessun report mtsp trovato (variant={mtsp_tag}).")
    if not single or not mtsp:
        lines.append("  Confronto impossibile: manca una delle due run.")
        return "\n".join(lines), None

    sa, ma = _agg(single), _agg(mtsp)
    lines += [
        f"  Istanze: single={sa['istanze']}  mtsp={ma['istanze']}   "
        f"(baseline STO/EEV/PI identiche: {_fmt(sa['STO'])}/{_fmt(sa['EEV'])}/{_fmt(sa['PI'])})",
        "",
        f"  {'metrica':<22}{'single-TSP':>14}{'mtsp':>14}{'Δ (mtsp-single)':>18}",
        f"  {'-'*66}",
        f"  {'UTSP costo medio':<22}{_fmt(sa['UTSP']):>14}{_fmt(ma['UTSP']):>14}"
        f"{_fmt((ma['UTSP']-sa['UTSP']) if sa['UTSP']==sa['UTSP'] else None):>18}",
        f"  {'gap UTSP vs STO %':<22}{_fmt(sa['gap_STO'],2):>14}{_fmt(ma['gap_STO'],2):>14}"
        f"{_fmt(ma['gap_STO']-sa['gap_STO'],2):>18}",
        f"  {'gap UTSP vs EEV %':<22}{_fmt(sa['gap_EEV'],2):>14}{_fmt(ma['gap_EEV'],2):>14}"
        f"{_fmt(ma['gap_EEV']-sa['gap_EEV'],2):>18}",
        f"  {'gap UTSP vs PI %':<22}{_fmt(sa['gap_PI'],2):>14}{_fmt(ma['gap_PI'],2):>14}"
        f"{_fmt(ma['gap_PI']-sa['gap_PI'],2):>18}",
    ]

    pr = _paired(single, mtsp)
    if pr:
        verdetto = ("mtsp ≈ single" if abs(pr["mean_diff"]) < 1e-6 else
                    "mtsp meglio (costo minore)" if pr["mean_diff"] < 0 else
                    "single meglio (costo minore)")
        lines += [
            "",
            f"  Confronto appaiato UTSP ({pr['n']} scenari comuni):",
            f"    differenza media di costo (mtsp-single) = {_fmt(pr['mean_diff'])}  "
            f"(mediana {_fmt(pr['median_diff'])})",
            f"    mtsp migliore in {pr['mtsp_better']} scenari, "
            f"single in {pr['single_better']}, pari {pr['ties']}  "
            f"({_fmt(pr['pct_mtsp_better'],1)}% a favore di mtsp)",
            f"    → {verdetto}",
        ]
    else:
        lines.append("  [!] nessuno scenario comune appaiabile (test config diverse?).")
    return "\n".join(lines), (sa, ma, pr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--exp", default=os.getenv("TESI_EXPERIMENT", "PERT"))
    ap.add_argument("--nodes", type=int, nargs="+", default=[15, 25])
    ap.add_argument("--mtsp-tag", default="mtsp")
    ap.add_argument("--single-tag", default="", help="variant del single-TSP (vuoto = base)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    blocks = [f"CONFRONTO single-TSP  vs  multigraph (mtsp) — exp={a.exp}",
              "STO/EEV/PI sono baseline identiche; il confronto è sul solo UTSP.\n"]
    for n in a.nodes:
        txt, _ = report_for_n(a.root, a.exp, n, a.mtsp_tag, a.single_tag)
        blocks.append(txt)
    out_txt = "\n\n".join(blocks) + "\n"
    print(out_txt)
    out = a.out or os.path.join(a.root, a.exp, f"confronto_mtsp_{'_'.join(map(str, a.nodes))}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(out_txt)
    print(f"[salvato] {out}")


if __name__ == "__main__":
    main()
