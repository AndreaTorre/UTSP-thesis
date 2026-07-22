#!/bin/bash
# run_variant_sweep.sh — allena il batch sweep per le 4 combinazioni della loss
# (penalty on/off) x (aggregation sum/mean), ognuna nel suo albero isolato.
#
# Ogni variante finisce in:
#   <EXP>/RISULTATI_<N>/variants/<nome>/batch_sweep/BATCH_<X>/train/...
# dove <nome> in {pen1_sum, pen1_mean, pen0_sum, pen0_mean}.
# I test si lanceranno poi con lo STESSO TESI_VARIANT, così pescano il
# checkpoint e la cache giusti.
#
# PREREQUISITI nel repo (da allineare al codice locale prima di lanciare):
#   - PERT/config_backend.py definisce UTSP2_AGGREGATION da TESI_UTSP_AGGREGATION
#   - two_stage_utsp_loss.py usa UTSP2_AGGREGATION per sum/mean nei termini
#   - config_backend.py definisce UTSP2_INCLUDE_PENALTY da TESI_UTSP_INCLUDE_PENALTY
#
# Uso:
#   bash run_variant_sweep.sh PERT 15                       # tutte e 4, batch di default
#   bash run_variant_sweep.sh PERT 15 "20 30 40 50 60 70"   # batch custom

set -euo pipefail

module load python
module load gurobi/13.0.0
unset GRB_WLSACCESSID GRB_WLSSECRET GRB_LICENSEID

ROOT=/home/atorre/UTSP/unione/git/UTSP
source "$ROOT/venv/bin/activate"
export PYTHONPATH="$ROOT/common:$PYTHONPATH"
cd "$ROOT/common"

EXP=${1:?Specifica EXP (PERT o CVETT)}
N=${2:?Specifica N (15, 25, 40)}
SIZES=${3:-"20 30 40 50 55 60 65 70"}

# (nome variante, INCLUDE_PENALTY, AGGREGATION)
run_variant () {
  local name=$1 pen=$2 agg=$3
  echo ""
  echo "########################################################################"
  echo "# VARIANTE ${name}  (penalty=${pen}, aggregation=${agg})  —  ${EXP} ${N} nodi"
  echo "########################################################################"
  for B in $SIZES; do
    echo ""
    echo "=== ${name} | batch ${B} ==="
    TESI_EXPERIMENT="$EXP" \
    TESI_N_NODES="$N" \
    TESI_VARIANT="$name" \
    TESI_UTSP_INCLUDE_PENALTY="$pen" \
    TESI_UTSP_AGGREGATION="$agg" \
    TESI_BATCH_SWEEP="$B" \
    python main.py --only B_UTSP_LS
  done
}

run_variant pen1_sum  1 sum
run_variant pen1_mean 1 mean
run_variant pen0_sum  0 sum
run_variant pen0_mean 0 mean

echo ""
echo "Fatto. Checkpoint in: ${EXP}/RISULTATI_${N}/variants/<variante>/batch_sweep/BATCH_*/train/"
