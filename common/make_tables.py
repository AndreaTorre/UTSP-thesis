#!/usr/bin/env python3
"""
Tabelle PERT per la tesi: assoluti (WS/STO/EEV/UTSP+LS) + gap% (vs STO, WS, EEV) + VSS/EVPI.
STO = riferimento; WS lower bound, EEV upper bound.
Due assi per foglia: TRAIN (DIM fisso, righe=batch) e TEST (batch fisso, righe=DIM).
Output: tesi_test_full.csv  +  tesi_tables.tex (booktabs).
"""
import re, csv, argparse
from pathlib import Path

def parse_txt(txt):
    KEYS=["UTSP_LS_test","WS_test","STO_test","EEV_test","PI_test"]
    d={}
    for k in KEYS:
        m=re.search(rf"{re.escape(k)}_mean=([-\d.eE+]+)",txt)
        d[k]=float(m.group(1)) if m else None
    return d

def collect(root, only_exp):
    rows=[]
    for exp in only_exp:
        for n in [15,25,40]:
            base=root/exp/f"RISULTATI_{n}"/"batch_sweep"
            if not base.exists(): continue
            for f in base.glob("BATCH_*/test/IS_*_DIM_*/report/*_aggregate.txt"):
                m=re.search(r"BATCH_(\d+)/test/IS_(\d+)_DIM_(\d+)",str(f))
                if not m: continue
                b,is_,dim=map(int,m.groups())
                rec={"exp":exp,"nodi":n,"batch_train":b,"IS":is_,"DIM":dim}
                rec.update(parse_txt(f.read_text()))
                rows.append(rec)
    return rows

def derive(r):
    ls,ws,sto,eev=r["UTSP_LS_test"],r["WS_test"],r["STO_test"],r["EEV_test"]
    pct=lambda a,base: None if (a is None or base in (None,0)) else 100*(a-base)/base
    r["gapSTO"]=pct(ls,sto); r["gapWS"]=pct(ls,ws); r["gapEEV"]=pct(ls,eev)
    r["VSS"]=None if (eev is None or sto is None) else eev-sto
    r["EVPI"]=None if (sto is None or ws is None) else sto-ws
    r["chain_ok"]=(ws is not None and sto is not None and eev is not None and ws-1e-6<=sto<=eev+1e-6)
    return r

def fnum(x,d=2): return "--" if x is None else f"{x:.{d}f}"

LABELS={"WS_test":"WS","STO_test":"STO","EEV_test":"EEV","UTSP_LS_test":r"UTSP+LS",
        "gapWS":r"$\Delta_{\text{WS}}$\%","gapSTO":r"$\Delta_{\text{STO}}$\%",
        "gapEEV":r"$\Delta_{\text{EEV}}$\%","VSS":"VSS","EVPI":"EVPI"}

def table(rows, axis, fixed, cols, caption, label):
    sub=sorted([r for r in rows if all(r[k]==v for k,v in fixed.items())], key=lambda r:r[axis])
    if not sub: return ""
    head={"batch_train":"batch","DIM":"DIM"}[axis]
    L=[r"\begin{table}[htbp]\centering",
       rf"\caption{{{caption}}}\label{{{label}}}",
       r"\begin{tabular}{r"+"r"*len(cols)+"}", r"\toprule",
       f"{head} & "+" & ".join(LABELS[c] for c in cols)+r" \\", r"\midrule"]
    for r in sub:
        L.append(f"{r[axis]} & "+" & ".join(fnum(r[c]) for c in cols)+r" \\")
    L+=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    return "\n".join(L)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default="/home/atorre/UTSP/unione/git/UTSP")
    ap.add_argument("--outdir",default=None)
    ap.add_argument("--exclude-dim",type=int,nargs="*",default=[300])
    ap.add_argument("--exp",nargs="*",default=["PERT"])   # PERT only di default
    a=ap.parse_args()
    root=Path(a.root); outdir=Path(a.outdir) if a.outdir else root
    rows=[derive(r) for r in collect(root,a.exp) if r["DIM"] not in a.exclude_dim]
    if not rows: print("Nessun risultato (job in corso?)"); return
    rows.sort(key=lambda r:(r["exp"],r["nodi"],r["DIM"],r["batch_train"]))

    allcols=["exp","nodi","batch_train","IS","DIM","WS_test","STO_test","EEV_test",
             "UTSP_LS_test","gapWS","gapSTO","gapEEV","VSS","EVPI","chain_ok"]
    with open(outdir/"tesi_test_full.csv","w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=allcols,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    bad=[r for r in rows if not r["chain_ok"]]
    if bad:
        print(f"ATTENZIONE: {len(bad)} righe con WS<=STO<=EEV violata (controlla prima di usarle):")
        for r in bad[:15]:
            print(f"  {r['exp']} {r['nodi']} DIM={r['DIM']} b={r['batch_train']}: "
                  f"WS={fnum(r['WS_test'])} STO={fnum(r['STO_test'])} EEV={fnum(r['EEV_test'])}")

    ABS=["WS_test","STO_test","EEV_test","UTSP_LS_test"]
    GAP=["gapWS","gapSTO","gapEEV"]
    tex=[]
    for exp in a.exp:
        for n in [15,25,40]:
            leaf=[r for r in rows if r["exp"]==exp and r["nodi"]==n]
            if not leaf: continue
            dims=sorted({r["DIM"] for r in leaf}); batches=sorted({r["batch_train"] for r in leaf})
            IS=leaf[0]["IS"]; dimF=max(dims); batchF=min(batches)
            tex.append(table(leaf,"batch_train",{"DIM":dimF},
                ABS+GAP, f"{exp} {n} nodi — variazione col batch di train (test IS={IS}, DIM={dimF}).",
                f"tab:{exp.lower()}{n}_train"))
            tex.append(table(leaf,"DIM",{"batch_train":batchF},
                ABS+GAP+["VSS","EVPI"], f"{exp} {n} nodi — variazione con la dim.\\ di test (batch={batchF}, IS={IS}).",
                f"tab:{exp.lower()}{n}_test"))
    (outdir/"tesi_tables.tex").write_text("\n\n".join(t for t in tex if t))
    print(f"\nOK: {len(rows)} righe -> {outdir}/tesi_test_full.csv")
    print(f"    LaTeX -> {outdir}/tesi_tables.tex")

if __name__=="__main__": main()
