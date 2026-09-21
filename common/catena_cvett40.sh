#!/usr/bin/env bash
# ============================================================================
# CATENA CVETT-40 completa, agganciata al setup in corso.
# Da lanciare dopo aver applicato: bash patch_cvett_pi_ws.sh
#
# ORDINE: setup(in corso) --> [eev, sto, pi/ws] --> assemble --> (rete a mano)
#   Correzioni vs run_tutto.sh: STO 192G (no OOM), gap realistici a n=40,
#   assemble afterany (tollera STO/WS troncati), WS con gap dichiarato.
#
# PREREQUISITO: patch_cvett_pi_ws.sh applicata (senza, il WS va esatto = proibitivo).
# ============================================================================
set -euo pipefail
ROOT=/home/atorre/UTSP/unione/git/UTSP
ACCT=def-tms
SETUP=3342137          # <-- setup CVETT-40 in corso. Verifica: squeue -u atorre
cd "$ROOT"

# safety: patch applicata?
if ! grep -q "skip_PI" CVETT/gurobi_parallelo.py; then
  echo "ERRORE: patch CVETT fase_pi WS non applicata. Esegui: bash patch_cvett_pi_ws.sh"
  exit 1
fi

ENV='module load gurobi/13.0.0
  source '"$ROOT"'/venv/bin/activate
  unset GRB_WLSACCESSID GRB_WLSSECRET GRB_LICENSEID
  unset TESI_BENCH_MIP_GAP TESI_BENCH_TIME_LIMIT
  export PYTHONPATH='"$ROOT"'/common:$PYTHONPATH
  export TESI_EXPERIMENT=CVETT TESI_N_NODES=40 TESI_DROP_LAST_TEST_BATCH=0'

# EEV - dopo setup (veloce)
JOB_EEV=$(sbatch --parsable --job-name=cvett40_eev --account=$ACCT \
  --dependency=afterok:$SETUP --time=06:00:00 --cpus-per-task=4 --mem=32G \
  --output=CVETT/RISULTATI_40/output/eev_%j.txt --error=CVETT/RISULTATI_40/output/eev_%j.txt \
  --wrap="$ENV
    cd $ROOT/CVETT && python gurobi_parallelo.py eev")
echo "eev = $JOB_EEV"

# STO - dopo setup, 192G, gap 5% (realistico a n=40), 20h time limit
JOB_STO=$(sbatch --parsable --job-name=cvett40_sto --account=$ACCT \
  --dependency=afterok:$SETUP --time=24:00:00 --cpus-per-task=1 --mem=192G \
  --output=CVETT/RISULTATI_40/output/sto_%j.txt --error=CVETT/RISULTATI_40/output/sto_%j.txt \
  --wrap="$ENV
    export TESI_STO_MIP_GAP=0.05 TESI_STO_TIME_LIMIT=72000
    cd $ROOT/CVETT && python gurobi_parallelo.py sto")
echo "sto = $JOB_STO"

# PI/WS - dopo setup. skip_pi + gap 1% -> solo WS con gap dichiarato (n=40).
# time_limit 2h/scenario. Serializzato dopo lo STO per non contendere la licenza.
JOB_WS=$(sbatch --parsable --job-name=cvett40_ws --account=$ACCT \
  --dependency=afterany:$JOB_STO --time=48:00:00 --cpus-per-task=4 --mem=64G \
  --output=CVETT/RISULTATI_40/output/ws_%j.txt --error=CVETT/RISULTATI_40/output/ws_%j.txt \
  --wrap="$ENV
    export TESI_SKIP_PI=1 TESI_WS_MIP_GAP=0.01 TESI_WS_TIME_LIMIT=7200 TESI_GRB_THREADS=4
    cd $ROOT/CVETT && python gurobi_parallelo.py pi")
echo "ws  = $JOB_WS   (dopo STO, serializzato per licenza)"

# ASSEMBLE - dopo eev+sto+ws (afterany: tollera troncati)
JOB_ASM=$(sbatch --parsable --job-name=cvett40_assemble --account=$ACCT \
  --dependency=afterany:$JOB_EEV:$JOB_STO:$JOB_WS --time=00:30:00 --mem=16G \
  --output=CVETT/RISULTATI_40/output/assemble_%j.txt --error=CVETT/RISULTATI_40/output/assemble_%j.txt \
  --wrap="$ENV
    cd $ROOT/CVETT && python gurobi_parallelo.py assemble")
echo "assemble = $JOB_ASM"

echo
echo "CATENA CVETT-40: setup($SETUP) -> [eev,sto,ws] -> assemble."
echo "La RETE va lanciata A MANO dopo l'assemble (main.py --only B_UTSP_LS),"
echo "  poi i benchmark sul test con sto_eev_on_test.py (CVETT ha le cache)."
echo "Nota: PI saltato (skip_pi) -> serve patch assemble PI-opzionale ANCHE per CVETT"
echo "  se l'assemble CVETT legge pi_results. VERIFICARE prima."
