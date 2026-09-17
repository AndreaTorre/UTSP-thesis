#!/usr/bin/env python3
"""
diag_sto_cvett.py — READ ONLY. Conferma perche' x_STO sotto-prenota:
  1) lo STO in parallel_data/sto.pkl era troncato? (status / gap / tempo)
  2) quali archi prenota (atteso 2; l'ottimo esatto ne prenoterebbe 4)
  3) gli scenari di setup (su cui gira STO/EEV) combaciano col blocco di test?
Non modifica nulla.

Uso (dalla root del checkout):
    python common/diag_sto_cvett.py CVETT/RISULTATI_15
"""
import sys, os, pickle, hashlib

OUT = sys.argv[1] if len(sys.argv) > 1 else "CVETT/RISULTATI_15"
PAR = os.path.join(OUT, "pkl", "parallel_data")
PKL = os.path.join(OUT, "pkl")

def load(path):
    if not os.path.exists(path):
        print(f"  [manca] {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def canon(e):
    i, j = e
    return (i, j) if i <= j else (j, i)

def arcs(x):
    try:
        return sorted(set(canon(tuple(e)) for e in x))
    except Exception:
        return list(x)

def results_of(o):
    if isinstance(o, dict) and "results" in o:
        return o["results"]
    return o if isinstance(o, dict) else {}

def scen_hash(res):
    if not res:
        return None, None
    rec = res[next(iter(res))]
    key = "scenario_dist" if isinstance(rec, dict) and "scenario_dist" in rec else \
          ("pert" if isinstance(rec, dict) and "pert" in rec else None)
    if key is None:
        return None, (list(rec.keys()) if isinstance(rec, dict) else None)
    blob = []
    for sid in sorted(res.keys(), key=lambda s: str(s)):
        d = res[sid].get(key)
        try:
            blob.append((str(sid), tuple(sorted((str(i), str(j), round(float(v), 6))
                        for i, row in d.items() for j, v in row.items()))
                        if isinstance(d, dict) else round(float(d), 6)))
        except Exception:
            blob.append((str(sid), "??"))
    return hashlib.md5(repr(blob).encode()).hexdigest()[:10], key

# 1) STO cache
print(f"\n--- parallel_data/sto.pkl ---")
sto = load(os.path.join(PAR, "sto.pkl"))
if isinstance(sto, dict):
    rs = sto.get("res_stoch") or {}
    xs = rs.get("x_used")
    print(f"  objective   = {rs.get('objective')}")
    print(f"  x_used      = {len(arcs(xs)) if xs is not None else '??'} archi: {arcs(xs) if xs is not None else None}")
    info = rs.get("solver_info") or {}
    print(f"  solver_info = {info}")
    # segnali di troncamento tipici
    st = str(info.get('status', '')).upper()
    gap = info.get('mip_gap', info.get('gap'))
    if 'TIME' in st or (gap is not None and gap and float(gap) > 1e-3):
        print("  => STO TRONCATO (status TIME_LIMIT o gap non chiuso): causa della sotto-prenotazione.")
    else:
        print("  => nessun segnale ovvio di troncamento nei metadati; guarda status/gap sopra.")
else:
    print("  n/d")

# 2) allineamento scenari setup vs test
print("\n--- scenari: setup (fit STO/EEV) vs blocco di test ---")
setup = load(os.path.join(PAR, "setup.pkl"))
res_s = None
if isinstance(setup, dict):
    for k in ("results", "scenari", "scenarios"):
        if k in setup:
            res_s = setup[k]; break
    if res_s is None and all(isinstance(v, dict) for v in list(setup.values())[:1]):
        res_s = setup
hs, ks = scen_hash(res_s) if res_s else (None, None)
test = results_of(load(os.path.join(PKL, "test_scenarios_cache.pkl")))
ht, kt = scen_hash(test) if test else (None, None)
print(f"  setup : {len(res_s) if res_s else 0} sid  hash={hs} (campo={ks})")
print(f"  test  : {len(test) if test else 0} sid  hash={ht} (campo={kt})")
if hs and ht:
    print("  => scenari COINCIDONO: re-risolvi STO senza toccare setup/scenari."
          if (ks == kt and hs == ht) else
          "  => scenari DIVERSI: fermati, non basta re-risolvere STO (vedi nota agente).")

print("\n--- res_B_cached.pkl corrente ---")
rb = load(os.path.join(PKL, "res_B_cached.pkl"))
if isinstance(rb, dict):
    print(f"  x_used_sto = {arcs(rb.get('x_used_sto')) if rb.get('x_used_sto') is not None else None}")
    print(f"  x_ev       = {arcs(rb.get('x_ev')) if rb.get('x_ev') is not None else None}")
    print(f"  STO={rb.get('STO')}  EEV={rb.get('EEV')}  WS={rb.get('WS')}")
