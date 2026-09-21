#!/usr/bin/env bash
# Rende il PI OPZIONALE in fase_assemble di PERT: se il WS ha girato con TESI_SKIP_PI=1,
# pi_results e' vuoto e l'assemble crasherebbe (KeyError su exact_free). Il PI serve solo
# per la regola diagnostica f^PI, NON per WS/STO/EEV. Con questa patch l'assemble salta il
# PI mancante e produce comunque res_B con i benchmark che servono.
set -euo pipefail
F=/home/atorre/UTSP/unione/git/UTSP/PERT/gurobi_parallelo.py
cp "$F" "$F.bak_pi"

python3 - "$F" << 'PYEOF'
import sys
F=sys.argv[1]
src=open(F).read()

old='''    # Inserisci i risultati PI e WS dentro results
    for sid in scenario_ids:
        results[sid]["exact_free"] = pi["pi_results"][sid]
        results[sid]["ws"] = pi["ws_results"][sid]

    # PI (TSP libero): resta solo per gli archi frequenti / regola f^PI.
    pi_lengths = [
        results[sid]["exact_free"]["length"]
        for sid in scenario_ids
        if results[sid]["exact_free"]["length"] is not None
    ]
    PI = sum(pi_lengths) / len(scenario_ids)'''

new='''    # Inserisci i risultati PI e WS dentro results.
    # PI (exact_free) e' OPZIONALE: se il WS ha girato con TESI_SKIP_PI=1, pi_results
    # e' vuoto. Il PI serve solo alla regola diagnostica f^PI, non a WS/STO/EEV.
    _pi_res = pi.get("pi_results", {}) or {}
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
        print("  [assemble] PI assente (WS con skip_pi): salto la regola f^PI, benchmark WS/STO/EEV OK.")'''

if src.count(old)!=1:
    sys.exit(f"ABORT: blocco atteso trovato {src.count(old)} volte (atteso 1). Nessuna modifica.")
open(F,"w").write(src.replace(old,new))
import py_compile; py_compile.compile(F, doraise=True)
print("PATCH assemble PI-opzionale applicata, sintassi OK. Backup:", F+".bak_pi")
PYEOF
