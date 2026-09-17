#!/usr/bin/env python3
"""
ws_provenance_check.py — perche' WS_test_mean (2474.30) finisce SOPRA UTSP_LS (2460.05).
Zero Gurobi. Confronta, per scenario_id, le tre sorgenti WS che convivono nella pkl dir
e l'hash della scenario_dist pool-vs-blocco. Dice quale sorgente riproduce la media
dell'aggregate e se e' sui giusti scenari.

Uso (dalla root del checkout):
    python common/ws_provenance_check.py CVETT/RISULTATI_15/pkl
"""
import sys, os, pickle, hashlib

PKL_DIR = sys.argv[1] if len(sys.argv) > 1 else "CVETT/RISULTATI_15/pkl"

def load(name):
    p = os.path.join(PKL_DIR, name)
    if not os.path.exists(p):
        print(f"  [manca] {name}")
        return None
    with open(p, "rb") as f:
        return pickle.load(f)

def results_of(obj):
    # tutte queste cache sono {"key":..., "results": {...}} oppure gia' il dict per-sid
    if isinstance(obj, dict) and "results" in obj:
        return obj["results"]
    return obj if isinstance(obj, dict) else {}

def dist_hash(d):
    # scenario_dist e' dict-of-dict dist[i][j]; hash canonico e stabile
    if d is None:
        return None
    try:
        items = sorted((str(i), str(j), round(float(v), 6))
                       for i, row in d.items() for j, v in row.items())
    except Exception:
        return "??"
    return hashlib.md5(repr(items).encode()).hexdigest()[:8]

def ws_cost_from_record(rec):
    # pool: rec["WS"]["cost"] ; sto_eev: rec["ws_cost"] ; ws_cache: rec["total_cost"] o rec["cost"]
    if not isinstance(rec, dict):
        return None
    if "ws_cost" in rec:                         return rec.get("ws_cost")
    if "WS" in rec and isinstance(rec["WS"], dict): return rec["WS"].get("cost")
    if "total_cost" in rec:                      return rec.get("total_cost")
    if "cost" in rec:                            return rec.get("cost")
    return None

def dist_from_record(rec):
    if not isinstance(rec, dict):
        return None
    for k in ("dist", "scenario_dist"):
        if k in rec:
            return rec[k]
    if "WS" in rec and isinstance(rec["WS"], dict):
        return rec["WS"].get("dist")
    return None

# --- carica ---
scen = results_of(load("test_scenarios_cache.pkl"))   # verita' sugli scenari del blocco
pool = results_of(load("test_pool_cache.pkl"))
wsc  = results_of(load("test_ws_cache.pkl"))
stev = results_of(load("test_sto_eev_cache.pkl"))      # cio' che ha alimentato l'aggregate

print(f"\nPKL_DIR = {PKL_DIR}")
for nm, obj in [("scenarios", scen), ("pool", pool), ("ws_cache", wsc), ("sto_eev", stev)]:
    print(f"  {nm:10s}: {len(obj) if obj else 0} sid")

# struttura di un record campione, per non indovinare le chiavi
def peek(nm, obj):
    if not obj: return
    sid0 = next(iter(obj))
    rec = obj[sid0]
    ks = list(rec.keys()) if isinstance(rec, dict) else type(rec).__name__
    print(f"    [{nm}] sid={sid0!r} keys={ks}")
print("\nStruttura record:")
for nm, obj in [("scenarios", scen), ("pool", pool), ("ws_cache", wsc), ("sto_eev", stev)]:
    peek(nm, obj)

# --- il blocco: gli sid della scenario-cache sono la verita' ---
block = list(scen.keys()) if scen else []
if not block:
    print("\n[STOP] scenario cache vuota: non so quali sid formano il blocco.")
    sys.exit(0)

# hash della scenario_dist del blocco (verita')
block_hash = {sid: dist_hash(scen[sid].get("scenario_dist")) for sid in block}

# --- confronto per-sid ---
rows = []
n_pool_cov = n_pool_distmis = n_ws_disagree = 0
for sid in block:
    h_blk  = block_hash[sid]
    p_rec  = pool.get(sid) if pool else None
    s_rec  = stev.get(sid) if stev else None
    w_rec  = wsc.get(sid) if wsc else None
    ws_pool = ws_cost_from_record(p_rec) if p_rec else None
    ws_stev = ws_cost_from_record(s_rec) if s_rec else None
    ws_wsc  = ws_cost_from_record(w_rec) if w_rec else None
    h_pool  = dist_hash(dist_from_record(p_rec)) if p_rec else None
    if ws_pool is not None: n_pool_cov += 1
    if h_pool is not None and h_pool not in ("??",) and h_pool != h_blk:
        n_pool_distmis += 1
    # disaccordo WS tra la sorgente-aggregate (sto_eev) e la ws_cache esatta
    if ws_stev is not None and ws_wsc is not None and abs(ws_stev - ws_wsc) > 1e-4:
        n_ws_disagree += 1
    rows.append((sid, ws_stev, ws_pool, ws_wsc, h_blk, h_pool))

def mean(vals):
    v = [x for x in vals if x is not None]
    return sum(v) / len(v) if v else float("nan")

m_stev = mean([r[1] for r in rows])
m_pool = mean([r[2] for r in rows])
m_wsc  = mean([r[3] for r in rows])

print(f"\n--- medie WS su {len(block)} sid del blocco ---")
print(f"  sto_eev.ws_cost  (usato dall'aggregate) = {m_stev:.4f}")
print(f"  pool.WS.cost                            = {m_pool:.4f}")
print(f"  ws_cache (shard, esatto)                = {m_wsc:.4f}")
print(f"  [atteso aggregate WS_test_mean          = 2474.30]")

print(f"\n--- provenienza pool ---")
print(f"  sid coperti dal pool          : {n_pool_cov}/{len(block)}")
print(f"  dist pool != dist blocco      : {n_pool_distmis}"
      f"  {'(pool NON porta dist: check impossibile senza re-solve)' if all(r[5] is None for r in rows) else ''}")
print(f"  sto_eev.ws != ws_cache(esatto): {n_ws_disagree}  (se >0: il WS dell'aggregate non e' l'esatto)")

# prime righe discordanti
print("\n--- prime 8 righe (sid | ws_aggregate | ws_pool | ws_esatto | hblk | hpool) ---")
for r in rows[:8]:
    print("  {:>8} | {} | {} | {} | {} | {}".format(
        r[0],
        f"{r[1]:.2f}" if r[1] is not None else "  --  ",
        f"{r[2]:.2f}" if r[2] is not None else "  --  ",
        f"{r[3]:.2f}" if r[3] is not None else "  --  ",
        r[4], r[5]))
