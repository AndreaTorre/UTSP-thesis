#!/bin/bash
# CVETT/run_tutto.sh — Pipeline parallela esperimento B per CVETT
# Uso: bash run_tutto.sh 15|25|40
#
# Dipendenze SLURM: setup -> (pi | eev | sto) -> assemble
# La fase sto per 40 nodi puo richiedere molte ore: regola STO_TIME_LIMIT in config.

set -euo pipefail

N=${1:-${TESI_N_NODES:-40}}

ROOT=/home/atorre/UTSP/unione/git/UTSP
CVETT=$ROOT/CVETT
VENV=$ROOT/venv
PYPATH=$ROOT/common

OUTPUT=$CVETT/RISULTATI_$N/output
mkdir -p "$OUTPUT"

# NOTA: adatta --account, --partition e i time limit al tuo cluster
COMMON_ARGS="
  --account=def-tms
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
    export TESI_EXPERIMENT=CVETT
    export TESI_N_NODES=$N

    cd $CVETT

    echo '===== DEBUG GUROBI ====='
    module list
    which gurobi_cl || true
    gurobi_cl --version || true
    env | grep -i gurobi || true
    env | grep -i grb || true
    python -c 'import gurobipy as gp; print(\"gurobipy\", gp.gurobi.version()); print(gp.__file__)'

    python gurobi_parallelo.py $1
  "
}

echo "======================================================"
echo "CVETT pipeline B — $N nodi"
echo "Output: $OUTPUT"
echo "======================================================"

# ── SETUP (calibrazione + perturbazioni) ───────────────────────
JOB_SETUP=$(sbatch --parsable $COMMON_ARGS \
  --job-name="cvett_setup_$N" \
  --time=12:00:00 \
  --cpus-per-task=8 \
  --mem=16G \
  --wrap="$(wrap_cmd setup)")
echo "  setup    $JOB_SETUP"

# ── PI (TSP esatto per scenario — indipendente da eev e sto) ───
JOB_PI=$(sbatch --parsable $COMMON_ARGS \
  --dependency=afterok:$JOB_SETUP \
  --job-name="cvett_pi_$N" \
  --time=10:00:00 \
  --cpus-per-task=4 \
  --mem=8G \
  --wrap="$(wrap_cmd pi)")
echo "  pi       $JOB_PI"

# ── EEV (scenario medio — veloce) ──────────────────────────────
JOB_EEV=$(sbatch --parsable $COMMON_ARGS \
  --dependency=afterok:$JOB_SETUP \
  --job-name="cvett_eev_$N" \
  --time=12:00:00 \
  --cpus-per-task=4 \
  --mem=8G \
  --wrap="$(wrap_cmd eev)")
echo "  eev      $JOB_EEV"

# ── STO (2-stage MILP — il piu pesante) ────────────────────────
# NOTA: 40 nodi con scenari ERA5 puo essere molto piu lento di 15/25.
# Se va in timeout aumenta STO_TIME_LIMIT in config e il --time qui sotto.
JOB_STO=$(sbatch --parsable $COMMON_ARGS \
  --dependency=afterok:$JOB_SETUP \
  --job-name="cvett_sto_$N" \
  --time=24:00:00 \
  --cpus-per-task=8 \
  --mem=32G \
  --wrap="$(wrap_cmd sto)")
echo "  sto      $JOB_STO"

# ── ASSEMBLE (unisce tutto) ─────────────────────────────────────
JOB_ASS=$(sbatch --parsable $COMMON_ARGS \
  --dependency=afterok:$JOB_PI:$JOB_EEV:$JOB_STO \
  --job-name="cvett_assemble_$N" \
  --time=00:30:00 \
  --cpus-per-task=2 \
  --mem=8G \
  --wrap="$(wrap_cmd assemble)")
echo "  assemble $JOB_ASS"

echo ""
echo "Dipendenze: setup -> (pi | eev | sto in parallelo) -> assemble"
echo "Monitora: squeue -u atorre"
echo ""
echo "Per il log in tempo reale:"
echo "  tail -f $OUTPUT/output_cvett_setup_${JOB_SETUP}.txt"
