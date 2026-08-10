#!/bin/bash
#SBATCH --account=def-tms_cpu
#SBATCH --job-name=cross_sweep
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-7
#SBATCH --output=cross_sweep_%A_%a.out
#
# sweep_two_models.sh — testa DUE modelli (cross vs no-cross) sugli STESSI
# scenari, a più DIM, come JOB ARRAY. Ogni task = una combinazione (modello,DIM),
# indipendente e cache-backed: un OOM/timeout su una non tocca le altre.
#
# Perché array e non un unico job: 8 combinazioni serie sarebbero ~9h in un
# colpo solo (fragile). L'array le fa in parallelo, ~2h l'una.
#
# ── Sottomissione ────────────────────────────────────────────────────────────
#   cd /home/atorre/UTSP/unione/git/UTSP/common
#   sbatch sweep_two_models.sh                 # DIM = 70 60 30 20, IS = 100  → 8 task
#   sbatch sweep_two_models.sh "60 30" 100     # DIM custom → ATTENZIONE: aggiorna --array!
#
#   L'array DEVE valere 0..(2*numero_DIM - 1). Con 4 DIM → 0-7 (default).
#   Con 2 DIM → cambia l'header in "#SBATCH --array=0-3".
#
#   Interattivo (login node sconsigliato; usa salloc): bash sweep_two_models.sh
#   → senza SLURM esegue TUTTE le combinazioni in serie.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

UTSP_ROOT="${TESI_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXP="${EXP:-PERT}"; NODES="${NODES:-15}"; BATCH="${BATCH:-20}"
read -ra DIM_VALUES <<< "${1:-70 60 30 20}"
N_ISTANZE_TEST="${2:-100}"
SKIP_PI="${TESI_TEST_SKIP_PI:-1}"

source "$UTSP_ROOT/common/env.sh"          # module load + venv + PYTHONPATH
cd "$UTSP_ROOT/common"

# (train_name in modello/) : (use_cross) : (prefisso cartella col token)
MODELS=( "espB_UTSP_LS:1:cross" "espB_UTSP_LS_nocross:0:nocross" )
COMBOS=()
for spec in "${MODELS[@]}"; do
  for dim in "${DIM_VALUES[@]}"; do COMBOS+=("$spec:$dim"); done
done

run_one() {
  IFS=":" read -r NAME USE_CROSS PREFIX DIM <<< "$1"
  local model_pt="$UTSP_ROOT/$EXP/RISULTATI_${NODES}/batch_sweep/BATCH_${BATCH}/modello/${NAME}/utsp_model.pt"
  if [ ! -e "$model_pt" ]; then
    echo "⚠ Modello mancante: $model_pt — salto '${NAME}' (symlink fatto?)"; return 0
  fi
  local out_tag="${PREFIX}_IS_${N_ISTANZE_TEST}_DIM_${DIM}"
  echo "=== ${EXP} ${NODES}n BATCH_${BATCH} | ${NAME} (cross=${USE_CROSS}) | DIM=${DIM} IS=${N_ISTANZE_TEST} → test/${out_tag} ==="
  TESI_EXPERIMENT="$EXP" TESI_N_NODES="$NODES" TESI_BATCH_SWEEP="$BATCH" \
  TESI_USE_CROSS="$USE_CROSS" TESI_UTSP_TEST_ONLY=1 TESI_UTSP_TRAIN_NAME="$NAME" \
  TESI_DIM_ISTANZA_TEST="$DIM" TESI_N_ISTANZE_TEST="$N_ISTANZE_TEST" \
  TESI_TEST_SKIP_PI="$SKIP_PI" TESI_TEST_OUTPUT_SUBDIR="test/${out_tag}" \
  python main.py --only B_UTSP_LS
}

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  idx="$SLURM_ARRAY_TASK_ID"
  if [ "$idx" -ge "${#COMBOS[@]}" ]; then
    echo "task $idx oltre le ${#COMBOS[@]} combinazioni: niente da fare (riduci --array)."; exit 0
  fi
  run_one "${COMBOS[$idx]}"
else
  echo "Nessun SLURM_ARRAY_TASK_ID: eseguo tutte le ${#COMBOS[@]} combinazioni in serie."
  for c in "${COMBOS[@]}"; do run_one "$c"; done
fi