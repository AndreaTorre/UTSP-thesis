#!/usr/bin/env bash
# DUE patch a CVETT/gurobi_parallelo.py, entrambe necessarie per CVETT-40:
#  (1) fase_pi: cap WS (mip_gap/time_limit/skip_pi) -> WS a n=40 fattibile con gap dichiarato.
#  (2) fase_assemble: PI OPZIONALE -> con skip_pi, pi_results e' vuoto e l'assemble
#      crasherebbe (KeyError su exact_free). Il PI serve solo alla regola f^PI.
# NON tocca PERT (file diverso). Backup automatico.
set -euo pipefail
F=/home/atorre/UTSP/unione/git/UTSP/CVETT/gurobi_parallelo.py
cp "$F" "$F.bak_cvett40"

python3 - "$F" << 'PYEOF'
import sys, py_compile
F=sys.argv[1]
src=open(F).read()

# ---- PATCH 1: fase_pi cap WS ----
old1='''    pi_results = {}   # TSP libero: SOLO per archi frequenti / regola f^PI (non un bound)
    ws_results = {}   # WS (wait-and-see): bound di informazione perfetta
    for idx, sid in enumerate(scenario_ids):
        print(f"  scenario {sid} ({idx+1}/{len(scenario_ids)})...", end=" ")
        t_s = time.time()
        dist = results[sid]["scenario_dist"]
        exact_free = solve_exact_tsp(
            nodes, E, dist, root, env,
            fixed_arcs=[], fixed_edges_undir=[], output_flag=0,
        )
        pi_results[sid] = exact_free
        ws = solve_reservation_tsp(
            nodes, E, I, dist, root, p, C, env,
            fixed_reservations=None, output_flag=0, model_name=f"ws_{sid}",
        )
        ws_results[sid] = ws
        dt = time.time() - t_s
        print(f"WS = {ws.get('total_cost')} | {dt:.1f}s")'''
new1='''    pi_results = {}   # TSP libero: SOLO per archi frequenti / regola f^PI (non un bound)
    ws_results = {}   # WS (wait-and-see): bound di informazione perfetta
    ws_gap  = float(os.getenv("TESI_WS_MIP_GAP", "0.0"))
    _tl     = os.getenv("TESI_WS_TIME_LIMIT")
    ws_tl   = float(_tl) if _tl else None
    skip_pi = os.getenv("TESI_SKIP_PI", "0") == "1"
    print(f"  WS: mip_gap={ws_gap}  time_limit={ws_tl}  skip_PI={skip_pi}")
    for idx, sid in enumerate(scenario_ids):
        print(f"  scenario {sid} ({idx+1}/{len(scenario_ids)})...", end=" ")
        t_s = time.time()
        dist = results[sid]["scenario_dist"]
        if not skip_pi:
            exact_free = solve_exact_tsp(
                nodes, E, dist, root, env,
                fixed_arcs=[], fixed_edges_undir=[], output_flag=0,
            )
            pi_results[sid] = exact_free
        ws = solve_reservation_tsp(
            nodes, E, I, dist, root, p, C, env,
            fixed_reservations=None, output_flag=0, model_name=f"ws_{sid}",
            time_limit=ws_tl, mip_gap=(ws_gap if ws_gap > 0 else None),
        )
        ws_results[sid] = ws
        dt = time.time() - t_s
        info = ws.get("solver_info", {})
        print(f"WS = {ws.get('total_cost')}  gap={info.get('mip_gap')}  status={info.get('status')} | {dt:.1f}s")'''

# ---- PATCH 2: fase_assemble PI opzionale ----
old2='''    for sid in scenario_ids:
        results[sid]["exact_free"] = pi["pi_results"][sid]
        results[sid]["ws"] = pi["ws_results"][sid]

    # PI (TSP libero): resta solo per gli archi frequenti / regola f^PI.
    pi_lengths = [
        results[sid]["exact_free"]["length"]
        for sid in scenario_ids
        if results[sid]["exact_free"]["length"] is not None
    ]
    PI = sum(pi_lengths) / len(scenario_ids)'''
new2='''    _pi_res = pi.get("pi_results", {}) or {}
    _have_pi = len(_pi_res) > 0
    for sid in scenario_ids:
        if _have_pi and sid in _pi_res:
            results[sid]["exact_free"] = _pi_res[sid]
        results[sid]["ws"] = pi["ws_results"][sid]

    if _have_pi:
        pi_lengths = [
            results[sid]["exact_free"]["length"]
            for sid in scenario_ids
            if results[sid].get("exact_free") and results[sid]["exact_free"]["length"] is not None
        ]
        PI = sum(pi_lengths) / len(scenario_ids) if pi_lengths else None
    else:
        PI = None
        print("  [assemble] PI assente (WS con skip_pi): salto regola f^PI, WS/STO/EEV OK.")'''

n1, n2 = src.count(old1), src.count(old2)
if n1!=1: sys.exit(f"ABORT patch1 (fase_pi): trovato {n1} volte (atteso 1).")
if n2!=1: sys.exit(f"ABORT patch2 (assemble): trovato {n2} volte (atteso 1).")
src=src.replace(old1,new1).replace(old2,new2)
open(F,"w").write(src)
py_compile.compile(F, doraise=True)
print("PATCH CVETT (fase_pi WS + assemble PI-opzionale) applicate, sintassi OK. Backup:", F+".bak_cvett40")
PYEOF
