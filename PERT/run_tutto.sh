#!/bin/bash
UTSP_ROOT="${TESI_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# PERT/run_tutto.sh — Pipeline parallela esperimento B per PERT
# Uso:
#   bash run_tutto.sh 15
#   bash run_tutto.sh 25
#   bash run_tutto.sh 40

set -euo pipefail

N=${1:-${TESI_N_NODES:-40}}

if [[ "$N" != "15" && "$N" != "25" && "$N" != "40" ]]; then
  echo "Errore: N deve essere 15, 25 oppure 40. Valore ricevuto: $N"
  exit 1
fi

ROOT=${UTSP_ROOT}
PERT=$ROOT/PERT
VENV=$ROOT/venv
PYPATH=$ROOT/common

OUTPUT=$PERT/RISULTATI_$N/output
mkdir -p "$OUTPUT"
mkdir -p "$PERT/RISULTATI_$N/grafici"
mkdir -p "$PERT/RISULTATI_$N/checkpoint"
mkdir -p "$PERT/RISULTATI_$N/pkl"

COMMON_ARGS="
  --output=$OUTPUT/output_%x_%j.txt
  --error=$OUTPUT/error_%x_%j.txt
  --nodes=1
  --ntasks=1
"

wrap_cmd() {
  # $1 = fase
  echo "
    cd $ROOT
    module load gurobi/13.0.0
    source $VENV/bin/activate

    unset GRB_WLSACCESSID
    unset GRB_WLSSECRET
    unset GRB_LICENSEID

    export PYTHONPATH=$PYPATH:\$PYTHONPATH
    export TESI_EXPERIMENT=PERT
    export TESI_N_NODES=$N
    cd $PERT

    python gurobi_parallelo.py $1
  "
}

echo "======================================================"
echo "PERT pipeline B — $N nodi"
echo "Output: $OUTPUT"
echo "======================================================"

JOB_SETUP=$(sbatch --parsable $COMMON_ARGS \
  --job-name="pert_setup_$N" \
  --time=12:00:00 \
  --cpus-per-task=8 \
  --mem=16G \
  --wrap="$(wrap_cmd setup)")

echo "  setup    $JOB_SETUP"

JOB_PI=$(sbatch --parsable $COMMON_ARGS \
  --dependency=afterok:$JOB_SETUP \
  --job-name="pert_pi_$N" \
  --time=24:00:00 \
  --cpus-per-task=4 \
  --mem=8G \
  --wrap="$(wrap_cmd pi)")

echo "  pi       $JOB_PI"

JOB_EEV=$(sbatch --parsable $COMMON_ARGS \
  --dependency=afterok:$JOB_SETUP \
  --job-name="pert_eev_$N" \
  --time=24:00:00 \
  --cpus-per-task=4 \
  --mem=8G \
  --wrap="$(wrap_cmd eev)")

echo "  eev      $JOB_EEV"

JOB_STO=$(sbatch --parsable $COMMON_ARGS \
  --dependency=afterok:$JOB_SETUP \
  --job-name="pert_sto_$N" \
  --time=24:00:00 \
  --cpus-per-task=8 \
  --mem=32G \
  --wrap="$(wrap_cmd sto)")

echo "  sto      $JOB_STO"

JOB_ASS=$(sbatch --parsable $COMMON_ARGS \
  --dependency=afterok:$JOB_PI:$JOB_EEV:$JOB_STO \
  --job-name="pert_assemble_$N" \
  --time=02:00:00 \
  --cpus-per-task=4 \
  --mem=8G \
  --wrap="$(wrap_cmd assemble)")

echo "  assemble $JOB_ASS"

echo ""
echo "Dipendenze: setup -> (pi | eev | sto in parallelo) -> assemble"
echo "Monitora: squeue -u atorre"
echo ""
echo "Log setup:"
echo "  tail -f $OUTPUT/output_pert_setup_${JOB_SETUP}.txt"
