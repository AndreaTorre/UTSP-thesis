#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
full_report.py — report diagnostico completo del test sweep.

FATTO STRUTTURALE che rende tutto più potente: STO ed EEV sono POLICY FISSE
(x decisa una volta in Experiment B). Per una data coppia (DIM, istanza) il
blocco di scenari è lo stesso per tutti i batch, quindi STO ed EEV valgono
ESATTAMENTE lo stesso numero qualunque sia il checkpoint. L'unica cosa che
varia tra batch è UTSP. Conseguenza: ogni confronto tra batch è appaiato, e
si può separare la varianza dovuta al batch da quella dovuta all'istanza.

Sezioni:
  1. Costi assoluti medi per batch (UTSP, STO, EEV) + composizione
  2. Scomposizione della varianza: BETWEEN batch vs WITHIN batch (ANOVA, eta^2)
  3. Archi prenotati: STO vs EEV vs UTSP, batch per batch
  4. Distribuzione dei risultati per ogni combinazione (batch x DIM)
  5. Effetto di DIM sulla forma della distribuzione (a batch fisso)
  6. Confronti appaiati batch-contro-batch (test dei segni)

Uso:
  TESI_EXPERIMENT=PERT TESI_N_NODES=15 python full_report.py
  TESI_EXPERIMENT=PERT TESI_N_NODES=15 python full_report.py --dim 60
"""
import argparse
import ast
import glob
import math
import os
import pickle
import re
import statistics as st
import sys
from collections import Counter, defaultdict

FIELD_RE = {
    "UTSP": re.compile(r"UTSP test\s*=\s*([-\d.]+|N/A|nan)"),
    "STO":  re.compile(r"STO test\s*=\s*([-\d.]+|N/A|nan)"),
    "EEV":  re.compile(r"EEV test\s*=\s*([-\d.]+|N/A|nan)"),
    "PI":   re.compile(r"PI test\s*=\s*([-\d.]+|N/A|nan)"),
}
X_RE = re.compile(r"x_test\s*=\s*(\[.*?\])", re.S)
SCEN_RE = re.compile(
    r"^\s*(\d+)\s*\|\s*([-\d.]+|N/A)\s*\|\s*([-\d.]+|N/A)\s*\|"
    r"\s*([-\d.]+|N/A)\s*\|\s*([-\d.]+|N/A)\s*\|", re.M)
PATH_RE = re.compile(
    r"/(?P<exp>PERT|CVETT)/RISULTATI_(?P<n>\d+)/(?:variants/(?P<variant>[^/]+)/)?batch_sweep/BATCH_(?P<b>\d+)"
    r"/report/")
IDX_RE = re.compile(r"_test_only_i(?P<i>\d+)_test_only_stats\.txt$")
BLOCK_RE = re.compile(r"^#{5,}\s*\n#\s*ISTANZA\s+(?P<i>\d+)\s*\n#{5,}\s*\n", re.M)


def _num(s):
    if s in ("N/A", "nan", "NaN", "", None):
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def canon(i, j):
    return (i, j) if i < j else (j, i)


def skew(v):
    if len(v) < 3:
        return 0.0
    m, s = st.fmean(v), st.pstdev(v)
    if s == 0:
        return 0.0
    return sum(((x - m) / s) ** 3 for x in v) / len(v)


def load(root, exp, nodes, dim_filter=None):
    _variant = os.getenv("TESI_VARIANT")
    _vseg = os.path.join("variants", _variant) if _variant else ""
    pat = os.path.join(root, exp, f"RISULTATI_{nodes}", _vseg, "batch_sweep", "BATCH_*",
                       "report", "*_test_all_instances.txt")
    rows = []
    for path in sorted(glob.glob(pat)):
        pm = PATH_RE.search(path.replace(os.sep, "/"))
        if not pm:
            continue
        text = open(path, encoding="utf-8", errors="replace").read()
        # il file unico contiene N istanze in blocchi "# ISTANZA N": itero sui blocchi
        blocks = [(m.start(), int(m.group("i"))) for m in BLOCK_RE.finditer(text)]
        if not blocks:
            continue
        for k, (pos, idx_ist) in enumerate(blocks):
            end = blocks[k + 1][0] if k + 1 < len(blocks) else len(text)
            blk = text[pos:end]
            r = {kk: _num(rx.search(blk).group(1)) if rx.search(blk) else None
                 for kk, rx in FIELD_RE.items()}
            xm = X_RE.search(blk)
            try:
                r["x"] = {canon(i, j) for (i, j) in ast.literal_eval(xm.group(1))} if xm else set()
            except (ValueError, SyntaxError):
                r["x"] = set()
            tour = multa = pren_post = 0.0
            ns = nm = dim_blk = 0
            scen_costs = []
            scen_by_sid = {}
            for m in SCEN_RE.finditer(blk):
                dim_blk += 1              # DIM = numero di righe-scenario nel blocco
                sid = int(m.group(1))
                post, tc, pc = _num(m.group(3)), _num(m.group(4)), _num(m.group(5))
                if post is None:
                    continue
                ns += 1
                pren_post += post
                tour += tc or 0.0
                multa += pc or 0.0
                scen_costs.append(post)
                scen_by_sid[sid] = post
                if (pc or 0) > 1e-9:
                    nm += 1
            r.update(batch=int(pm.group("b")), DIM=dim_blk,
                     istanza=idx_ist, n_scen=ns, n_scen_multa=nm,
                     tot_post=pren_post, tot_tour=tour, tot_multa=multa,
                     scen_costs=scen_costs, scen_by_sid=scen_by_sid)
            rows.append(r)
    if dim_filter is not None:
        rows = [r for r in rows if r["DIM"] == dim_filter]
    return rows


def load_pi_per_scenario(root, exp, nodes):
    """
    PI (TSP libero ottimo) per scenario_id, dalla cache scenari.
    I file di stats scritti con TESI_TEST_SKIP_PI=1 hanno 'PI test = N/A':
    il merge delle shard riempie la CACHE, non riscrive quei file. Qui il PI
    viene ricostruito unendo cache e file di stats per scenario_id — stessi
    scenari, nessuna local search da rifare.
    """
    p = os.path.join(root, exp, f"RISULTATI_{nodes}", "pkl", "test_scenarios_cache.pkl")
    if not os.path.exists(p):
        return {}
    with open(p, "rb") as f:
        res = pickle.load(f)["results"]
    out = {}
    for sid, rec in res.items():
        length = rec.get("exact_free", {}).get("length")
        if length is not None:
            out[sid] = float(length)
    return out


def load_pi_tours(root, exp, nodes):
    """Tour PI (TSP libero) per scenario_id: serve per f^PI e PI+pren."""
    pth = os.path.join(root, exp, f"RISULTATI_{nodes}", "pkl", "test_scenarios_cache.pkl")
    if not os.path.exists(pth):
        return {}
    with open(pth, "rb") as f:
        res = pickle.load(f)["results"]
    return {sid: rec["exact_free"]["tour"]
            for sid, rec in res.items()
            if rec.get("exact_free", {}).get("tour")}


def load_ws(root, exp, nodes):
    """Wait-and-See per scenario_id: costo ottimo e x ottima con info perfetta."""
    pth = os.path.join(root, exp, f"RISULTATI_{nodes}", "pkl", "test_ws_cache.pkl")
    if not os.path.exists(pth):
        return {}
    with open(pth, "rb") as f:
        return pickle.load(f).get("results", {})


def tour_arcs(tour):
    return [canon(tour[k], tour[(k + 1) % len(tour)]) for k in range(len(tour))]


def load_sto_per_scenario(root, exp, nodes):
    """
    Costo della policy STO per singolo scenario_id (dalla cache STO/EEV).
    Serve per calcolare il GAP PER-SCENARIO: senza questo si può solo
    confrontare le medie di istanza, e il rumore comune UTSP/STO non si
    cancella.
    """
    p = os.path.join(root, exp, f"RISULTATI_{nodes}", "pkl", "test_sto_eev_cache.pkl")
    if not os.path.exists(p):
        return {}
    with open(p, "rb") as f:
        d = pickle.load(f)
    return {sid: v["sto_cost"] for sid, v in d.get("results", {}).items()}


def load_resB(root, exp, nodes):
    p = os.path.join(root, exp, f"RISULTATI_{nodes}", "pkl", "res_B_cached.pkl")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def sign_test(k, n):
    """p-value bilaterale, H0: p=0.5."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--exp", default=os.getenv("TESI_EXPERIMENT", "PERT"))
    ap.add_argument("--nodes", type=int, default=int(os.getenv("TESI_N_NODES", "15")))
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = load(args.root, args.exp, args.nodes, args.dim)
    if not rows:
        sys.exit(f"Nessun risultato per {args.exp} {args.nodes} nodi sotto {args.root}")
    res_B = load_resB(args.root, args.exp, args.nodes)

    # PI ricostruito dalla cache per le istanze che ce l'hanno a N/A
    pi_sid = load_pi_per_scenario(args.root, args.exp, args.nodes)
    n_pi_ricostruiti = 0
    if pi_sid:
        for r in rows:
            if r.get("PI") is None and r.get("scen_by_sid"):
                vals = [pi_sid[sid] for sid in r["scen_by_sid"] if sid in pi_sid]
                # solo se il PI c'è per TUTTI gli scenari dell'istanza:
                # una media parziale sarebbe silenziosamente sbagliata.
                if vals and len(vals) == len(r["scen_by_sid"]):
                    r["PI"] = st.fmean(vals)
                    n_pi_ricostruiti += 1
    # WS per istanza: media del wait-and-see sugli scenari dell'istanza
    ws_sid = load_ws(args.root, args.exp, args.nodes)
    for r in rows:
        sids = r.get("scen_by_sid") or {}
        wv = [ws_sid[s]["total_cost"] for s in sids
              if s in ws_sid and ws_sid[s].get("total_cost") is not None]
        # NOTA: solo se il WS c'è per TUTTI gli scenari, come per il PI
        r["WS"] = st.fmean(wv) if wv and len(wv) == len(sids) else None

    L = []

    def P(*a):
        s = " ".join(str(x) for x in a) if a else ""
        L.append(s)
        print(s)

    batches = sorted({r["batch"] for r in rows})
    dims = sorted({r["DIM"] for r in rows})
    by = defaultdict(list)
    for r in rows:
        by[(r["batch"], r["DIM"])].append(r)

    P("=" * 100)
    P(f"REPORT COMPLETO  —  {args.exp}  {args.nodes} nodi")
    P("=" * 100)
     
     
    # ---------------- 1. COSTI ASSOLUTI ----------------
    P("")
    P("=" * 100)
    P("1.  COSTI ASSOLUTI MEDI  (unità del problema, non percentuali)")
    P("=" * 100)
    for dim in dims:
        sub = [r for r in rows if r["DIM"] == dim]
        if not sub:
            continue
        sto = [r["STO"] for r in sub if r["STO"] is not None]
        eev = [r["EEV"] for r in sub if r["EEV"] is not None]
        wss = [r["WS"] for r in sub if r["WS"] is not None]
        P("")
        P(f"  ISTANZA TEST = {dim} scenari")
        P(f"    {'ist.train':>9} | {'UTSP medio':>12} | {'std':>7} | {'vs STO':>8} | {'vs EEV':>8} | {'vs WS':>8} | {'multe %':>8}")
        P("    " + "-" * 68)
        for b in batches:
            rr = by.get((b, dim), [])
            u = [r["UTSP"] for r in rr if r["UTSP"] is not None]
            if not u:
                continue
            s_ = [r["STO"] for r in rr if r["STO"] is not None]
            e_ = [r["EEV"] for r in rr if r["EEV"] is not None]
            gs = st.fmean([(r["UTSP"] - r["STO"]) / abs(r["STO"]) * 100
                           for r in rr if r["UTSP"] and r["STO"]]) if s_ else float("nan")
            ge = st.fmean([(r["UTSP"] - r["EEV"]) / abs(r["EEV"]) * 100
                           for r in rr if r["UTSP"] and r["EEV"]]) if e_ else float("nan")
            gw_l = [(r["UTSP"] - r["WS"]) / abs(r["WS"]) * 100
                    for r in rr if r["UTSP"] and r["WS"]]
            gw = st.fmean(gw_l) if gw_l else float("nan")
            tp = sum(r["tot_post"] for r in rr)
            tm = sum(r["tot_multa"] for r in rr)
            pm = 100 * tm / tp if tp else 0.0
            gw_s = f"{gw:>+7.3f}%" if gw_l else f"{'—':>8}"
            P(f"    {b:>9} | {st.fmean(u):>12.2f} | {st.pstdev(u):>7.2f} | "
              f"{gs:>+7.3f}% | {ge:>+7.3f}% | {gw_s} | {pm:>7.2f}%")
        if sto:
            P(f"    {'STO':>9} | {st.fmean(sto):>12.2f} | {st.pstdev(sto):>7.2f} |"
              f" {'—':>8} | {'—':>8} | {'—':>8} | {'—':>8}    ")
        if eev:
            P(f"    {'EEV':>9} | {st.fmean(eev):>12.2f} | {st.pstdev(eev):>7.2f} |"
              f" {'—':>8} | {'—':>8} | {'—':>8} | {'—':>8}    ")
        if wss:
            P(f"    {'WS':>9} | {st.fmean(wss):>12.2f} | {st.pstdev(wss):>7.2f} |"
              f" {'—':>8} | {'—':>8} | {'—':>8} | {'—':>8}   ")
        else:
            P("    WS: non disponibile. Controlla che test_ws_cache.pkl copra tutti")
            P("        gli scenari delle istanze (altrimenti WS resta None per riga).")

     
         

    # ---------------- 3. ARCHI PRENOTATI ----------------
    # ---------------- 3. ARCHI PRENOTATI ----------------
    P("")
    P("=" * 100)
    P("3.  ARCHI PRENOTATI:  STO  vs  EEV  vs  UTSP (per dimensione istanza di train)")
    P("=" * 100)
    if res_B:
        I_set = {canon(i, j) for (i, j) in res_B["I"]}
        x_sto = {canon(i, j) for (i, j) in res_B["x_used_sto"]}
        x_ev = {canon(i, j) for (i, j) in res_B["x_ev"]}
        p_, C_ = res_B["p"], res_B["C"]

        def gv(d, i, j):
            return d.get(canon(i, j), d.get((i, j), 0.0))

        P(f"  |I| = {len(I_set)} archi prenotabili")
        P(f"  x_STO = {sorted(x_sto)}")
        P(f"  x_EEV = {sorted(x_ev)}")
        P("")
        for dim in dims:
            P(f"  ISTANZA TEST = {dim} scenari:  frequenza con cui ogni istanza-train prenota l'arco")
            P(f"  (su {len(by.get((batches[0], dim), []))} istanze di test)")
            hdr = f"    {'arco':>10} | {'p/C':>6} | {'STO':>4} | {'EEV':>4} |" + \
                  "".join(f" IS{b:<4}" for b in batches)
            P(hdr)
            P("    " + "-" * (len(hdr) - 4))
            for e in sorted(I_set):
                soglia = gv(p_, *e) / gv(C_, *e) if gv(C_, *e) else float("inf")
                cells = ""
                for b in batches:
                    rr = by.get((b, dim), [])
                    if not rr:
                        cells += f"{'-':>7}"
                        continue
                    f = sum(1 for r in rr if e in r["x"]) / len(rr)
                    cells += f" {100*f:>5.0f}%"
                P(f"    {str(e):>10} | {soglia:>6.3f} | {'SI' if e in x_sto else 'no':>4} |"
                  f" {'SI' if e in x_ev else 'no':>4} |{cells}")
            P("")
         

    # ---------------- 4. DISTRIBUZIONI PER COMBINAZIONE ----------------
    P("")
    P("=" * 100)
    P("4.  DISTRIBUZIONE DEL GAP vs STO, per ogni combinazione (istanza train x istanza test)")
    P("=" * 100)
    P(f"  {'ist.tr':>6} {'ist.te':>6} {'n':>4} | {'media':>8} {'std':>6} | {'p5':>7} {'p25':>7} "
      f"{'p50':>7} {'p75':>7} {'p95':>7} | {'asimm.':>7} | {'vince':>6}")
    P("  " + "-" * 94)
    for b in batches:
        for dim in dims:
            rr = by.get((b, dim), [])
            g = [(r["UTSP"] - r["STO"]) / abs(r["STO"]) * 100
                 for r in rr if r["UTSP"] and r["STO"]]
            if len(g) < 2:
                continue
            v = sorted(g)

            def q(pc):
                pos = pc / 100 * (len(v) - 1)
                lo = int(pos)
                hi = min(lo + 1, len(v) - 1)
                return v[lo] + (v[hi] - v[lo]) * (pos - lo)
            wins = sum(1 for x in g if x < 0)
            P(f"  {b:>6} {dim:>6} {len(g):>4} | {st.fmean(g):>+8.3f} {st.pstdev(g):>6.3f} | "
              f"{q(5):>+7.3f} {q(25):>+7.3f} {q(50):>+7.3f} {q(75):>+7.3f} {q(95):>+7.3f} | "
              f"{skew(g):>+7.2f} | {100*wins/len(g):>5.0f}%")
    P("")
    P("  'vince' = % di istanze in cui UTSP costa MENO di STO.  asimm. > 0 = coda a destra")
    P("  (poche istanze molto peggiori della mediana); < 0 = coda a sinistra.")

    # ---------------- 5. EFFETTO DI DIM ----------------
    P("")
    P("=" * 100)
    P("5.  EFFETTO DELLA DIMENSIONE DELL'ISTANZA DI TEST (a istanza-train fissa)")
    P("=" * 100)
    P("  Cambiare la dimensione dell'istanza di test raggruppa gli STESSI scenari in blocchi")
    P("  non dovrebbe cambiare, la DISPERSIONE sì (come 1/sqrt(DIM)). Un eccesso di")
    P("  dispersione rispetto all'atteso segnala una componente sistematica per istanza.")
    for b in batches:
        ds = [d for d in dims if len(by.get((b, d), [])) > 1]
        if len(ds) < 2:
            continue
        P("")
        P(f"  ISTANZA TRAIN = {b} scenari")
        P(f"    {'ist.test':>8} | {'media gap':>10} | {'std oss.':>9} | {'std attesa':>10} | rapporto")
        P("    " + "-" * 58)
        base = ds[0]
        g0 = [(r["UTSP"] - r["STO"]) / abs(r["STO"]) * 100
              for r in by[(b, base)] if r["UTSP"] and r["STO"]]
        s0 = st.pstdev(g0)
        for d in ds:
            g = [(r["UTSP"] - r["STO"]) / abs(r["STO"]) * 100
                 for r in by[(b, d)] if r["UTSP"] and r["STO"]]
            s = st.pstdev(g)
            att = s0 * math.sqrt(base / d)
            ratio = s / att if att else float("nan")
            flag = "" if ratio < 1.35 else "  <-- ECCESSO"
            P(f"    {d:>8} | {st.fmean(g):>+10.3f} | {s:>9.3f} | {att:>10.3f} | {ratio:>5.2f}x{flag}")

 

    # ---------------- 7. BENCHMARK CON INFORMAZIONE PERFETTA ----------------
    P("")
    P("=" * 100)
    P("7.  BENCHMARK CON INFORMAZIONE PERFETTA:  PI, PI+pren, WS  —  e la regola f > p/C")
    P("=" * 100)
    P("  PI      = TSP LIBERO dello scenario. Ignora p, C e I: NON è un bound valido per")
    P("            il problema a due stadi (chi usa un arco di I deve pagare p oppure C).")
    P("            Sta sotto il vero ottimo, ma è inutilmente lasco.")
    P("  PI+pren = tour del PI + costo di prenotazione degli archi di I che quel tour usa.")
    P("            Prende il tour SBAGLIATO (ottimizzato ignorando p) e ci attacca un costo:")
    P("            è un UPPER bound sull'ottimo con info perfetta, non l'ottimo.")
    P("  WS      = Wait-and-See: x e y decisi INSIEME conoscendo lo scenario.")
    P("            Questo è il vero ottimo con informazione perfetta e il bound RIGOROSO:")
    P("                    WS  <=  STO  <=  {UTSP, EEV}")
    P("            STO - WS = EVPI, il valore dell'informazione perfetta.")
    P("")

    pi_tours = load_pi_tours(args.root, args.exp, args.nodes)
    ws = load_ws(args.root, args.exp, args.nodes)

    if res_B:
        I_set2 = {canon(i, j) for (i, j) in res_B["I"]}
        p2, C2 = res_B["p"], res_B["C"]

        def gv2(d, e):
            return d.get(e, d.get((e[1], e[0]), 0.0))

        # ---- catena dei bound ----
        if ws:
            wsv = [v["total_cost"] for v in ws.values() if v.get("total_cost") is not None]
            stov = load_sto_per_scenario(args.root, args.exp, args.nodes)
            if wsv and stov:
                common = [sid for sid in ws if sid in stov
                          and ws[sid].get("total_cost") is not None]
                ws_m = st.fmean([ws[sid]["total_cost"] for sid in common])
                sto_m = st.fmean([stov[sid] for sid in common])
                evpi = sto_m - ws_m
                P(f"  CATENA DEI BOUND (media su {len(common)} scenari):")
                P(f"    WS  (info perfetta)   = {ws_m:>10.2f}")
                P(f"    STO (una x per tutti) = {sto_m:>10.2f}")
                P(f"    EVPI = STO - WS       = {evpi:>10.2f}  ({100*evpi/ws_m:+.2f}% di WS)")
                P("      = quanto varrebbe conoscere lo scenario in anticipo.")
                n_tl = sum(1 for v in ws.values() if v.get("status") == "TIME_LIMIT")
                if n_tl:
                    P(f"    ⚠ {n_tl} scenari WS chiusi in TIME_LIMIT: incumbent, non ottimi.")
                P("")

                # gap di UTSP verso WS, per cella
                P("  GAP di UTSP verso WS (bound corretto):")
                P(f"    {'ist.train':>9} {'ist.test':>9} | {'vs WS':>9} | {'vs STO':>9}")
                P("    " + "-" * 44)
                for b_ in batches:
                    for dim in dims:
                        rr = [r for r in by.get((b_, dim), []) if r.get("scen_by_sid")]
                        if not rr:
                            continue
                        gws, gst = [], []
                        for r in rr:
                            wv = [ws[sid]["total_cost"] for sid in r["scen_by_sid"]
                                  if sid in ws and ws[sid].get("total_cost") is not None]
                            sv = [stov[sid] for sid in r["scen_by_sid"] if sid in stov]
                            if len(wv) == len(r["scen_by_sid"]) and r["UTSP"]:
                                m = st.fmean(wv)
                                gws.append((r["UTSP"] - m) / abs(m) * 100)
                            if len(sv) == len(r["scen_by_sid"]) and r["UTSP"]:
                                m = st.fmean(sv)
                                gst.append((r["UTSP"] - m) / abs(m) * 100)
                        if gws:
                            gs_ = f"{st.fmean(gst):>+8.3f}%" if gst else f"{'—':>9}"
                            P(f"    {b_:>9} {dim:>9} | {st.fmean(gws):>+8.3f}% | {gs_}")
                P("")
        else:
            P("  WS non disponibile: lancia compute_ws_shard.py + merge_ws_shards.py.")
            P("")

        # ---- la regola f > p/C, con i tre riferimenti ----
        if pi_tours or ws:
            P("  VERIFICA DELLA REGOLA  f > p/C,  arco per arco:")
            P("")
            P("    f_PI = frazione di scenari in cui il tour PI (libero) USA l'arco.")
            P("           È la 'f' teorica della regola: frequenza d'uso a costo zero.")
            P("           Ma il PI non sa nulla di p e C: dice se l'arco è UTILE,")
            P("           non se conviene PRENOTARLO.")
            P("    f_WS = frazione di scenari in cui l'ottimo con info perfetta PRENOTA")
            P("           davvero l'arco. Questa è la verifica ECONOMICA della regola.")
            P("")
            hdr = (f"    {'arco':>12} | {'p/C':>6} | {'f_PI':>6} | {'f_WS':>6} | "
                   f"{'regola':>7} | {'STO':>4} | {'UTSP(freq)':>10} | esito")
            P(hdr)
            P("    " + "-" * (len(hdr) - 4))
            x_sto2 = {canon(i, j) for (i, j) in res_B["x_used_sto"]}

            n_pi = len(pi_tours)
            use_pi = Counter()
            for t in pi_tours.values():
                for e in set(tour_arcs(t)):
                    if e in I_set2:
                        use_pi[e] += 1
            n_ws = 0
            book_ws = Counter()
            for v in ws.values():
                if v.get("x_used") is None:
                    continue
                n_ws += 1
                for e in {canon(i, j) for (i, j) in v["x_used"]}:
                    book_ws[e] += 1

            # frequenza con cui UTSP prenota (media su tutte le celle)
            utsp_book = Counter()
            n_utsp = 0
            for r in rows:
                n_utsp += 1
                for e in r["x"]:
                    utsp_book[e] += 1

            for e in sorted(I_set2):
                soglia = gv2(p2, e) / gv2(C2, e) if gv2(C2, e) else float("inf")
                fpi = use_pi[e] / n_pi if n_pi else None
                fws = book_ws[e] / n_ws if n_ws else None
                fut = utsp_book[e] / n_utsp if n_utsp else 0.0
                regola = (fpi is not None and fpi > soglia)
                in_sto = e in x_sto2

                if fws is None:
                    esito = "WS mancante"
                elif regola and fws < 0.25:
                    esito = "SOVRA-prenota (WS quasi mai)"
                elif (not regola) and fws > 0.75:
                    esito = "SOTTO-prenota (WS quasi sempre)"
                elif abs((1 if regola else 0) - fws) < 0.35:
                    esito = "concorde con WS"
                else:
                    esito = "parzialmente discorde"

                fpis = f"{fpi:>6.3f}" if fpi is not None else f"{'—':>6}"
                fwss = f"{fws:>6.3f}" if fws is not None else f"{'—':>6}"
                P(f"    {str(e):>12} | {soglia:>6.3f} | {fpis} | {fwss} | "
                  f"{'SI' if regola else 'no':>7} | {'SI' if in_sto else 'no':>4} | "
                  f"{100*fut:>9.0f}% | {esito}")

                 

                fpis = f"{fpi:>6.3f}" if fpi is not None else f"{'—':>6}"
                fwss = f"{fws:>6.3f}" if fws is not None else f"{'—':>6}"
                 
             

    # ---------------- salvataggio ----------------
    out = args.out or os.path.join(args.root, "sweep_analysis",
                                   f"report_completo_{args.exp}_{args.nodes}nodi"
                                   + (f"_DIM{args.dim}" if args.dim else "") + ".txt")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\nReport salvato in: {out}")


if __name__ == "__main__":
    main()