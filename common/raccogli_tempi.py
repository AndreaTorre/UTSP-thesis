#!/usr/bin/env python3
"""
Raccoglie i tempi per fase dai phase_times.json e costruisce la tabella tempi per la tesi.
Rete = training + ls_test ; Baseline Gurobi = bench_gurobi.
Distingue train-sweep (train+test interno) e test-sweep (solo test, per DIM).
"""
import json, re, csv
from pathlib import Path

ROOT = Path("/home/atorre/UTSP/unione/git/UTSP")
PHASES = ["gen_scenari_train","training","gen_scenari_test","ls_train","ls_test","bench_gurobi"]

def load(fp):
    try: return json.loads(Path(fp).read_text())
    except Exception: return None

rows=[]
for exp in ["PERT","CVETT"]:
    for n in [15,25,40]:
        base = ROOT/exp/f"RISULTATI_{n}"/"batch_sweep"
        if not base.exists(): continue
        for pt in base.glob("BATCH_*/**/phase_times.json"):
            d = load(pt)
            if not d: continue
            s=str(pt)
            mb=re.search(r"BATCH_(\d+)", s); batch=int(mb.group(1)) if mb else None
            mt=re.search(r"test/IS_(\d+)_DIM_(\d+)", s)
            kind = "test_sweep" if mt else "train_sweep"
            IS  = int(mt.group(1)) if mt else None
            DIM = int(mt.group(2)) if mt else None
            rec={"exp":exp,"nodi":n,"batch":batch,"kind":kind,"IS":IS,"DIM":DIM}
            for ph in PHASES: rec[ph]=d.get(ph)
            rec["rete_train_test"] = (d.get("training") or 0)+(d.get("ls_test") or 0)
            rec["baseline_gurobi"] = d.get("bench_gurobi")
            rows.append(rec)

rows.sort(key=lambda r:(r["exp"],r["nodi"],r["kind"],r["DIM"] or 0,r["batch"] or 0))
cols=["exp","nodi","kind","batch","IS","DIM"]+PHASES+["rete_train_test","baseline_gurobi"]
out=ROOT/"tesi_tempi.csv"
with open(out,"w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=cols,extrasaction="ignore"); w.writeheader(); w.writerows(rows)
print(f"scritte {len(rows)} righe -> {out}\n")
# anteprima leggibile
hdr=["exp","nodi","kind","batch","DIM","training","ls_test","bench_gurobi"]
print(" | ".join(f"{h:>11}" for h in hdr))
for r in rows:
    print(" | ".join(f"{(round(r[h],1) if isinstance(r.get(h),(int,float)) else r.get(h))!s:>11}" for h in hdr))
