#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_cross.py — confronto appaiato di DUE modelli UTSP sugli STESSI dati di
test: uno con canali cross-scenario, uno senza (file col token "nocross").

Riusa il parsing e le cache di full_report.py e l'ANOVA di anova_train_test.py
(nessuna logica riscritta). Aggiunge ciò che serve per l'ablation cross-scenario:

  - K-sweep: ogni metrica vs dimensione istanza di test (DIM), due modelli;
  - confronto APPAIATO per DIM (stessa istanza = stesso blocco di scenari):
    Wilcoxon signed-rank + test dei segni sulle differenze (nocross - cross);
  - coerenza delle PRENOTAZIONI: F1 di x_UTSP vs x_STO / x_EEV per istanza, e
    concordanza per-corridoio con WS/STO/EEV (la domanda: il cross-scenario
    prenota meglio, cioè più coerente coi benchmark?);
  - controllo a K=1 (nessuna informazione cross: le curve devono coincidere);
  - diagnostica meccanicistica: massa di attention sui canali cross (dai
    metadata/history del modello), per spiegare il PERCHÉ;
  - ANOVA a due vie modello × DIM sul gap primario;
  - grafici del K-sweep e della differenza appaiata.

Uso:
  TESI_EXPERIMENT=PERT TESI_N_NODES=15 python compare_cross.py
  python compare_cross.py --exp PERT --nodes 15 --token nocross --primary gap_ws
"""
import argparse
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Riuso diretto: parser dei blocchi-istanza, cache e helper già scritti.
from full_report import (
    canon, tour_arcs, load_ws, load_sto_per_scenario, load_resB,
    sign_test, BLOCK_RE, FIELD_RE, X_RE, SCEN_RE, PATH_RE, _num,
)
from anova_train_test import two_way_anova, size_label

try:
    from scipy.stats import wilcoxon
except Exception:                      # scipy c'è già (evaluation.py), ma non fermarsi se manca
    wilcoxon = None

ROOT_DEFAULT = str(Path(__file__).resolve().parent.parent)


# ── Discovery + parsing dei file per-istanza, ETICHETTATI per modello ────────

def _parse_blocks(text):
    """Un file *_test_all_instances.txt contiene N blocchi '# ISTANZA N'.
    Stessa logica di full_report.load, ma qui isolata per poterla etichettare."""
    out = []
    blocks = [(m.start(), int(m.group("i"))) for m in BLOCK_RE.finditer(text)]
    for k, (pos, idx_ist) in enumerate(blocks):
        end = blocks[k + 1][0] if k + 1 < len(blocks) else len(text)
        blk = text[pos:end]
        rec = {kk: (_num(rx.search(blk).group(1)) if rx.search(blk) else None)
               for kk, rx in FIELD_RE.items()}
        xm = X_RE.search(blk)
        try:
            import ast
            rec["x"] = {canon(i, j) for (i, j) in ast.literal_eval(xm.group(1))} if xm else set()
        except (ValueError, SyntaxError):
            rec["x"] = set()
        sids, scen_by_sid = [], {}
        for m in SCEN_RE.finditer(blk):
            sid = int(m.group(1)); post = _num(m.group(3))
            if post is None:
                continue
            sids.append(sid); scen_by_sid[sid] = post
        rec.update(istanza=idx_ist, DIM=len(sids), sids=sids, scen_by_sid=scen_by_sid)
        out.append(rec)
    return out


def discover(root, exp, nodes, token):
    """Trova tutti i *_test_all_instances.txt sotto RISULTATI_<nodes> e li
    etichetta 'nocross' se il token compare nel percorso, altrimenti 'cross'."""
    pat = os.path.join(root, exp, f"RISULTATI_{nodes}", "**", "*_test_all_instances.txt")
    rows = []
    files = sorted(glob.glob(pat, recursive=True))
    for path in files:
        up = path.replace(os.sep, "/")
        model = "nocross" if token.lower() in up.lower() else "cross"
        pm = PATH_RE.search(up)
        batch = int(pm.group("b")) if pm else None
        text = open(path, encoding="utf-8", errors="replace").read()
        for rec in _parse_blocks(text):
            rec.update(model=model, batch=batch, path=path)
            rows.append(rec)
    return rows, files


# ── Arricchimento: WS per istanza + gap + F1 prenotazioni ────────────────────

def enrich(rows, ws_by_sid, x_sto, x_ev):
    """Aggiunge WS di istanza (media sui suoi scenari), i tre gap e le F1 di
    prenotazione vs STO/EEV. WS può mancare: in quel caso gap_ws resta None."""
    def f1(pred, ref):
        P, R = set(pred), set(ref)
        if not P and not R:
            return 1.0
        tp = len(P & R)
        prec = tp / len(P) if P else 0.0
        rec = tp / len(R) if R else 0.0
        return 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    for r in rows:
        ws_vals = [ws_by_sid[s] for s in r["sids"] if s in ws_by_sid]
        r["WS"] = float(st.fmean(ws_vals)) if len(ws_vals) == len(r["sids"]) and ws_vals else None
        u = r["UTSP"]
        r["gap_ws"]  = 100 * (u - r["WS"]) / abs(r["WS"])   if r["WS"]  and u is not None else None
        r["gap_sto"] = 100 * (u - r["STO"]) / abs(r["STO"]) if r["STO"] and u is not None else None
        r["gap_eev"] = 100 * (u - r["EEV"]) / abs(r["EEV"]) if r["EEV"] and u is not None else None
        r["f1_sto"] = f1(r["x"], x_sto)
        r["f1_ev"]  = f1(r["x"], x_ev)


# ── Aggregazione K-sweep e confronto appaiato ────────────────────────────────

def _ms(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    if not vals:
        return (float("nan"), float("nan"), 0)
    return (float(np.mean(vals)), float(np.std(vals)), len(vals))


def sweep_table(rows, metric):
    """media ± std di `metric` per (modello, DIM)."""
    by = defaultdict(list)
    for r in rows:
        by[(r["model"], r["DIM"])].append(r.get(metric))
    dims = sorted({d for (_, d) in by})
    return dims, {(m, d): _ms(by[(m, d)]) for (m, d) in by}


def paired(rows, metric):
    """Per ogni DIM appaia le istanze presenti in ENTRAMBI i modelli (stessa
    istanza = stesso blocco di scenari) e restituisce le differenze nocross-cross."""
    idx = defaultdict(dict)     # (batch, DIM, istanza) -> {model: value}
    for r in rows:
        v = r.get(metric)
        if v is not None and np.isfinite(v):
            idx[(r["batch"], r["DIM"], r["istanza"])][r["model"]] = v
    per_dim = defaultdict(list)
    for (batch, dim, ist), mv in idx.items():
        if "cross" in mv and "nocross" in mv:
            per_dim[dim].append((mv["cross"], mv["nocross"]))
    return per_dim


def paired_stats(pairs):
    """pairs = lista di (cross, nocross). Convenzione: delta = nocross - cross,
    quindi delta<0 = cross MEGLIO (gap più basso). Ritorna test appaiati."""
    c = np.array([a for a, _ in pairs], float)
    nc = np.array([b for _, b in pairs], float)
    d = nc - c
    n = len(d)
    wins_cross = int(np.sum(d > 0))                       # nocross peggiore => cross vince
    med = float(np.median(d)) if n else float("nan")
    mean = float(np.mean(d)) if n else float("nan")
    sem = float(np.std(d, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    ci = (mean - 1.96 * sem, mean + 1.96 * sem) if n > 1 and np.isfinite(sem) else (float("nan"),) * 2
    p_sign = sign_test(wins_cross, n)
    p_w = float("nan")
    if wilcoxon is not None and n >= 6 and np.any(d != 0):
        try:
            p_w = float(wilcoxon(nc, c).pvalue)
        except Exception:
            p_w = float("nan")
    return dict(n=n, mean=mean, median=med, ci=ci, wins_cross=wins_cross,
                p_sign=p_sign, p_wilcoxon=p_w)


# ── Coerenza prenotazioni per-corridoio (vs WS/STO/EEV) ──────────────────────

def booking_freq_by_model(rows):
    """Per (modello, corridoio): frazione di istanze in cui il modello prenota."""
    tot = defaultdict(int); cnt = defaultdict(int)
    for r in rows:
        tot[r["model"]] += 1
        for e in r["x"]:
            cnt[(r["model"], e)] += 1
    return tot, cnt


def f_ws_booking(ws):
    """f_WS(e) = frazione di scenari in cui il WS (info perfetta) prenota davvero e."""
    n = 0; c = defaultdict(int)
    for v in ws.values():
        xu = v.get("x_used")
        if xu is None:
            continue
        n += 1
        for e in {canon(i, j) for (i, j) in xu}:
            c[e] += 1
    return n, c


# ── Diagnostica meccanicistica: attention sui canali cross ───────────────────

def cross_attention(root, exp, nodes, token):
    """Legge, per ciascun modello, la frazione media di attention sui canali
    cross-scenario per layer, dai metadata/history salvati dal training.
    Ritorna {model: [frac_layer0, ...] | None}."""
    out = {}
    base = os.path.join(root, exp, f"RISULTATI_{nodes}")
    for meta in glob.glob(os.path.join(base, "**", "modello", "**", "utsp_history.json"), recursive=True):
        model = "nocross" if token.lower() in meta.lower() else "cross"
        try:
            h = json.load(open(meta))
            cs = h.get("cross_share")            # lista per epoca di liste per layer
            out.setdefault(model, cs[-1] if cs else None)
        except Exception:
            out.setdefault(model, None)
    return out


# ── Report ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getenv("TESI_ROOT_DIR", ROOT_DEFAULT))
    ap.add_argument("--exp", default=os.getenv("TESI_EXPERIMENT", "PERT"))
    ap.add_argument("--nodes", type=int, default=int(os.getenv("TESI_N_NODES", "15")))
    ap.add_argument("--token", default="nocross", help="token nel path del modello SENZA cross")
    ap.add_argument("--primary", default="gap_ws", choices=["gap_ws", "gap_sto", "gap_eev"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root, exp, nodes = a.root, a.exp, a.nodes
    lines = []
    def P(*x):
        s = " ".join(str(t) for t in x)
        print(s); lines.append(s)

    rows, files = discover(root, exp, nodes, a.token)
    models = sorted({r["model"] for r in rows})
    P("=" * 92)
    P(f"CONFRONTO CROSS vs NO-CROSS   exp={exp}  nodi={nodes}  token='{a.token}'")
    P("=" * 92)
    P(f"File trovati: {len(files)} | blocchi-istanza: {len(rows)} | modelli: {models}")
    if set(models) != {"cross", "nocross"}:
        P("\n⚠ Attesi entrambi i modelli 'cross' e 'nocross'. Controlla il --token e i path.")
        for f in files:
            P("   ", ("nocross" if a.token.lower() in f.lower() else "cross"), f)
        if not rows:
            _save(a, lines); return

    ws = load_ws(root, exp, nodes)
    ws_by_sid = {sid: v.get("total_cost") for sid, v in ws.items() if v.get("total_cost") is not None}
    if not ws_by_sid:
        sto_eev_ws = load_sto_per_scenario(root, exp, nodes)   # solo per messaggio
        P("\n⚠ WS non disponibile (test_ws_cache.pkl mancante o vuoto).")
        P("  gap_ws resta None: lancia compute_ws_shard.py + merge_ws_shards.py, oppure usa --primary gap_sto.")
    resB = load_resB(root, exp, nodes)
    if resB is None:
        P("\n⚠ res_B_cached.pkl mancante: niente x_STO/x_EEV, salto le F1 di prenotazione.")
        x_sto = x_ev = set(); I_set = []
        p_, C_ = {}, {}
    else:
        x_sto = {canon(i, j) for (i, j) in resB["x_used_sto"]}
        x_ev = {canon(i, j) for (i, j) in resB["x_ev"]}
        I_set = sorted({canon(i, j) for (i, j) in resB["I"]})
        p_, C_ = resB["p"], resB["C"]

    enrich(rows, ws_by_sid, x_sto, x_ev)

    # ── 1. K-SWEEP ────────────────────────────────────────────────────────
    P("\n" + "=" * 92)
    P("1.  K-SWEEP: media ± std per dimensione istanza di test (DIM)")
    P("=" * 92)
    for metric, name in [("UTSP", "costo UTSP"), ("gap_ws", "gap vs WS %"),
                         ("gap_sto", "gap vs STO %"), ("gap_eev", "gap vs EEV %"),
                         ("f1_sto", "F1 prenot. vs STO"), ("f1_ev", "F1 prenot. vs EEV")]:
        dims, tab = sweep_table(rows, metric)
        if not dims:
            continue
        P(f"\n  {name}")
        P(f"    {'DIM':>5} | {'cross (m±s, n)':>26} | {'nocross (m±s, n)':>26}")
        for d in dims:
            cm, cs, cn = tab.get(("cross", d), (float('nan'),) * 3)
            nm, ns, nn = tab.get(("nocross", d), (float('nan'),) * 3)
            P(f"    {d:>5} | {cm:>10.4f} ± {cs:>7.4f} (n={cn:>3}) | "
              f"{nm:>10.4f} ± {ns:>7.4f} (n={nn:>3})")

    # ── 2. CONFRONTO APPAIATO ─────────────────────────────────────────────
    P("\n" + "=" * 92)
    P(f"2.  CONFRONTO APPAIATO per DIM  —  delta = nocross - cross  (delta<0 ⇒ CROSS meglio)")
    P("    stessa istanza = stesso blocco di scenari ⇒ test appaiato (varianza fra istanze cancellata)")
    P("=" * 92)
    for metric in ["gap_ws", "gap_sto", "gap_eev", "f1_sto"]:
        per_dim = paired(rows, metric)
        if not per_dim:
            continue
        better = "delta>0" if metric.startswith("f1") else "delta<0"
        P(f"\n  {metric}   (per le F1 conviene delta>0: cross prenota più simile a STO)")
        P(f"    {'DIM':>5} | {'n':>4} | {'media Δ':>10} | {'mediana Δ':>10} | "
          f"{'IC95%':>20} | {'p(segni)':>9} | {'p(Wilcoxon)':>11} | vince cross")
        for d in sorted(per_dim):
            s = paired_stats(per_dim[d])
            frac = 1 - s["wins_cross"] / s["n"] if s["n"] else float("nan")
            ci = f"[{s['ci'][0]:+.3f},{s['ci'][1]:+.3f}]" if np.isfinite(s['ci'][0]) else "n.d."
            note = ""
            if d == 1:
                note = "  ← controllo K=1: atteso ≈0"
            P(f"    {d:>5} | {s['n']:>4} | {s['mean']:>+10.4f} | {s['median']:>+10.4f} | "
              f"{ci:>20} | {s['p_sign']:>9.4f} | {s['p_wilcoxon']:>11.4f} | "
              f"{100*frac:>5.0f}%{note}")

    # ── 3. COERENZA PRENOTAZIONI per-corridoio (vs WS/STO/EEV) ────────────
    if I_set:
        P("\n" + "=" * 92)
        P("3.  COERENZA PRENOTAZIONI per-corridoio: quanto ogni modello prenota vs STO/EEV/WS")
        P("=" * 92)
        tot, cnt = booking_freq_by_model(rows)
        nws, cws = f_ws_booking(ws)
        def gv(dd, e):
            return dd.get(e, dd.get((e[0], e[1]), 0.0))
        P(f"    {'corr.':>10} | {'p/C':>6} | {'STO':>3} | {'EEV':>3} | "
          f"{'f_WS':>6} | {'cross':>6} | {'nocross':>7} | esito(cross|nocross vs WS)")
        for e in I_set:
            soglia = gv(p_, e) / gv(C_, e) if gv(C_, e) else float("inf")
            fws = cws[e] / nws if nws else None
            fc = cnt[("cross", e)] / tot.get("cross", 1) if tot.get("cross") else 0.0
            fn = cnt[("nocross", e)] / tot.get("nocross", 1) if tot.get("nocross") else 0.0
            def esito(fm):
                if fws is None:
                    return "WS?"
                if abs(fm - fws) < 0.35:
                    return "concorde"
                return "sovra" if fm > fws else "sotto"
            fwss = f"{fws:.3f}" if fws is not None else "  —  "
            P(f"    {str(e):>10} | {soglia:>6.3f} | {'SI' if e in x_sto else 'no':>3} | "
              f"{'SI' if e in x_ev else 'no':>3} | {fwss:>6} | {100*fc:>5.0f}% | "
              f"{100*fn:>6.0f}% | {esito(fc)}|{esito(fn)}")
        P("\n  Lettura: se il cross-scenario aiuta le prenotazioni, la colonna 'cross' deve")
        P("  essere più vicina a f_WS (e all'accordo con STO) della colonna 'nocross'.")

    # ── 4. ANOVA a due vie modello × DIM sul gap primario ─────────────────
    P("\n" + "=" * 92)
    P(f"4.  ANOVA a due vie  modello × DIM  su  {a.primary}")
    P("=" * 92)
    arows = [{"model": r["model"], "DIM": r["DIM"], a.primary: r[a.primary]}
             for r in rows if r.get(a.primary) is not None and np.isfinite(r[a.primary])]
    if len({r["model"] for r in arows}) == 2 and len({r["DIM"] for r in arows}) >= 2:
        res = two_way_anova(arows, "model", "DIM", a.primary)
        if not res:
            P("    ANOVA non calcolabile (troppo pochi dati o celle incomplete).")
        else:
            P(f"    {'sorgente':<22} {'SS':>12} {'df':>5} {'F':>9} {'p-value':>10} {'eta^2':>8}")
            for key in ("A", "B", "AB"):
                r = res.get(key)
                if not r:
                    continue
                P(f"    {r['name']:<22} {r['ss']:>12.4f} {r['df']:>5} "
                  f"{r['F']:>9.3f} {r['p']:>10.4g} {r['eta2']:>8.4f}  {size_label(r['eta2'])}")
            P("\n  'model' con p<0.05 ed eta^2 non trascurabile ⇒ il cross-scenario conta.")
            P("  L'interazione model×DIM significativa ⇒ l'effetto del cross CAMBIA con K")
            P("  (tipicamente cresce con K): è l'evidenza cercata.")
    else:
        P("    Servono 2 modelli e ≥2 valori di DIM per l'ANOVA a due vie.")

    # ── 5. DIAGNOSTICA MECCANICISTICA: attention sui canali cross ─────────
    P("\n" + "=" * 92)
    P("5.  DIAGNOSTICA: massa di attention sui canali cross-scenario (dai metadata del modello)")
    P("=" * 92)
    ca = cross_attention(root, exp, nodes, a.token)
    if ca:
        for m, shares in ca.items():
            if shares is None:
                P(f"    {m:>8}: cross_share non registrato (history senza 'cross_share').")
            else:
                P(f"    {m:>8}: attention media sui canali cross per layer = "
                  f"{[round(float(s),3) for s in shares]}")
        P("\n  Se per il modello 'cross' questa massa è ~0, la rete ha imparato a ignorare")
        P("  il cross-scenario: i due modelli devono allora coincidere (coerente con K=1).")
    else:
        P("    Nessun utsp_history.json trovato sotto modello/.")

    # ── 6. GRAFICI ────────────────────────────────────────────────────────
    outdir = a.out or os.path.join(root, exp, f"RISULTATI_{nodes}", "report")
    os.makedirs(outdir, exist_ok=True)
    _plots(rows, a.primary, outdir, exp, nodes, P)

    _save_to(os.path.join(outdir, "compare_cross_report.txt"), lines)
    print(f"\n  → Report salvato in: {os.path.join(outdir, 'compare_cross_report.txt')}")


def _plot_curve(ax, rows, metric, title, ylabel):
    dims, tab = sweep_table(rows, metric)
    for model, mark in [("cross", "o-"), ("nocross", "s--")]:
        xs = [d for d in dims if not math.isnan(tab.get((model, d), (float('nan'),))[0])]
        ys = [tab[(model, d)][0] for d in xs]
        es = [tab[(model, d)][1] for d in xs]
        if xs:
            ax.errorbar(xs, ys, yerr=es, fmt=mark, capsize=3, label=model)
    ax.set_title(title); ax.set_xlabel("DIM (scenari per istanza di test)")
    ax.set_ylabel(ylabel); ax.grid(True, ls="--", alpha=.4); ax.legend()


def _plots(rows, primary, outdir, exp, nodes, P):
    # Fig A: gap primario e F1-vs-STO vs K
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _plot_curve(axes[0], rows, primary, f"{primary} vs K", primary)
    _plot_curve(axes[1], rows, "f1_sto", "F1 prenotazioni vs STO", "F1 (x_UTSP vs x_STO)")
    fig.suptitle(f"Cross vs no-cross — {exp} {nodes} nodi", fontweight="bold")
    fig.tight_layout()
    fa = os.path.join(outdir, "compare_cross_ksweep.png")
    fig.savefig(fa, dpi=150, bbox_inches="tight"); plt.close(fig)

    # Fig B: differenza appaiata del gap primario vs K, con IC95%
    per_dim = paired(rows, primary)
    if per_dim:
        dims = sorted(per_dim)
        stats = [paired_stats(per_dim[d]) for d in dims]
        means = [s["mean"] for s in stats]
        errs = [(s["ci"][1] - s["mean"]) if np.isfinite(s["ci"][0]) else 0.0 for s in stats]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axhline(0, color="k", lw=1)
        ax.errorbar(dims, means, yerr=errs, fmt="D-", capsize=4, color="#0055CC")
        ax.set_title(f"Δ appaiato (nocross - cross) di {primary}  —  <0 ⇒ cross meglio")
        ax.set_xlabel("DIM"); ax.set_ylabel(f"Δ {primary} (nocross - cross)")
        ax.grid(True, ls="--", alpha=.4)
        fb = os.path.join(outdir, "compare_cross_paired_delta.png")
        fig.savefig(fb, dpi=150, bbox_inches="tight"); plt.close(fig)
        P(f"\n  Grafici: {fa}\n           {fb}")
    else:
        P(f"\n  Grafico: {fa}  (nessuna coppia appaiata per il delta)")


def _save_to(path, lines):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _save(a, lines):
    outdir = a.out or os.path.join(a.root, a.exp, f"RISULTATI_{a.nodes}", "report")
    os.makedirs(outdir, exist_ok=True)
    _save_to(os.path.join(outdir, "compare_cross_report.txt"), lines)


if __name__ == "__main__":
    main()
