#!/usr/bin/env python3
"""check_seeds.py — verifica l'allineamento dei semi PERT/CVETT.

Importa la config per PERT e CVETT su 15/25/40 nodi (in sottoprocessi, perché
la config legge l'ambiente all'import) e controlla che:
  1. i semi NON cambino tra i nodi;
  2. ogni seme sia identico PERT vs CVETT;
  3. i tre split (train/test/val) siano distinti tra loro.

Uso (dopo `source env.sh`, con i .nc CVETT presenti):
    python check_seeds.py
Esce 0 se tutto ok, 2 altrimenti.
"""
import json
import os
import subprocess
import sys

SEEDS = [
    "GLOBAL_SEED", "CALIBRATION_SCENARIO_SEED", "FINAL_SCENARIO_SEED",
    "VALIDATION_SEED", "TRAIN_SCENARIO_SEED", "TEST_SCENARIO_SEED",
    "VAL_SCENARIO_SEED", "UTSP_LS_RANDOM_SEED", "UTSP_TRAINING_SEED",
]
SPLIT = {"TRAIN_SCENARIO_SEED", "TEST_SCENARIO_SEED", "VAL_SCENARIO_SEED"}
NODES = [15, 25, 40]

_SNIPPET = (
    "import config as c, json;"
    "print('SEEDS' + json.dumps({k: getattr(c, k) for k in %r}))" % SEEDS
)


def seeds_for(exp, n):
    env = dict(os.environ, TESI_EXPERIMENT=exp, TESI_N_NODES=str(n))
    if exp == "CVETT":
        # necessario perché il backend CVETT valida la divisibilità all'import
        env.update(
            TESI_UTSP_BATCH_SIZE="20", TESI_BATCH_SWEEP="20",
            TESI_N_ISTANZE_TEST="100", TESI_DIM_ISTANZA_TEST="20",
            TESI_N_TEST_SCENARIOS="2000",
        )
    out = subprocess.run([sys.executable, "-c", _SNIPPET], env=env,
                         capture_output=True, text=True)
    line = [l for l in out.stdout.splitlines() if l.startswith("SEEDS")]
    if not line:
        print(f"IMPORT FALLITO {exp} {n}:\n{out.stderr[-1000:]}")
        sys.exit(1)
    return json.loads(line[0][5:])


def main():
    data = {(e, n): seeds_for(e, n) for e in ("PERT", "CVETT") for n in NODES}
    ok = True

    # 1) invarianza per nodo
    for e in ("PERT", "CVETT"):
        base = data[(e, NODES[0])]
        for n in NODES[1:]:
            if data[(e, n)] != base:
                ok = False
                print(f"[X] {e}: i semi cambiano tra {NODES[0]} e {n} nodi")

    # 2) ogni seme identico PERT vs CVETT
    for n in NODES:
        for k in SEEDS:
            pv, cv = data[("PERT", n)][k], data[("CVETT", n)][k]
            if pv != cv:
                ok = False
                print(f"[X] {k} @ {n} nodi: PERT={pv} != CVETT={cv}")

    # 3) i tre split distinti tra loro
    s = {k: data[("PERT", NODES[0])][k] for k in SPLIT}
    if len(set(s.values())) != 3:
        ok = False
        print(f"[X] i tre split non sono distinti: {s}")

    print()
    if ok:
        ref = data[("PERT", NODES[0])]
        for k in SEEDS:
            tag = " (split)" if k in SPLIT else ""
            print(f"  {k:28s} = {ref[k]}{tag}")
        print("\nTUTTO OK: semi allineati PERT/CVETT, distinti solo per split.")
    else:
        print("CI SONO DISALLINEAMENTI (vedi sopra).")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
