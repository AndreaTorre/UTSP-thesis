#!/bin/bash
# Lancia solo STO e poi ASSEMBLE per CVETT.
# Uso:
#   bash submit_sto_assemble.sh 25

set -euo pipefail

N=${1:-25}

ROOT=/home/atorre/UTSP/unione/git/UTSP
CVETT=$ROOT/CVETT
VENV=$ROOT/venv
PYPATH=$ROOT/common

OUTPUT=$CVETT/RISULTATI_$N/output
mkdir -p "$OUTPUT"

COMMON_ARGS="
  --account=def-tms
  --output=$OUTPUT/output_%x_%j.txt
  --error=$OUTPUT/error_%x_%j.txt
  --nodes=1
  --ntasks=1
"

wrap_cmd() {
  FASE=$1
  echo "
    cd $ROOT

    module load gurobi/13.0.0
    source $VENV/bin/activate

    unset GRB_WLSACCESSID
    unset GRB_WLSSECRET
    unset GRB_LICENSEID

    export PYTHONUNBUFFERED=1
    export PYTHONPATH=$PYPATH:\$PYTHONPATH
    export TESI_EXPERIMENT=CVETT
    export TESI_N_NODES=$N

    cd $CVETT

    echo '===== FASE: $FASE ====='
    echo 'N_NODES='$N
    echo 'PWD='$(pwd)

    python gurobi_parallelo.py $FASE
  "
}

echo "=== CVETT solo STO + ASSEMBLE ==="
echo "N_NODES=$N"
echo "Output: $OUTPUT"

JOB_STO=$(sbatch --parsable $COMMON_ARGS \
  --job-name="cvett_sto_$N" \
  --time=24:00:00 \
  --cpus-per-task=8 \
  --mem=32G \
  --wrap="$(wrap_cmd sto)")

echo "STO job: $JOB_STO"

JOB_ASS=$(sbatch --parsable $COMMON_ARGS \
  --dependency=afterok:$JOB_STO \
  --job-name="cvett_assemble_$N" \
  --time=00:30:00 \
  --cpus-per-task=2 \
  --mem=8G \
  --wrap="$(wrap_cmd assemble)")

echo "ASSEMBLE job: $JOB_ASS"

echo ""
echo "Dipendenza:"
echo "  STO -> ASSEMBLE"
echo ""
echo "Monitora:"
echo "  squeue -u atorre"
echo ""
echo "Log STO:"
echo "  tail -f $OUTPUT/output_cvett_sto_${JOB_STO}.txt"
echo ""
echo "Log ASSEMBLE:"
echo "  tail -f $OUTPUT/output_cvett_assemble_${JOB_ASS}.txt"
