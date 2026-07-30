#!/usr/bin/env bash
# ============================================================================
# Lancia la grid search per una o piu' triple (esperimento, nodi, batch).
#
# Uso:
#   bash grid_submit.sh PERT 25 40          # una sola tripla
#   bash grid_submit.sh PERT 25 all         # tutti i batch di PERT/NODI_25
#   bash grid_submit.sh PERT all all        # tutto PERT
#   bash grid_submit.sh all all all         # tutte le 48 triple
#   DRY=1 bash grid_submit.sh PERT 25 all   # stampa e basta, non lancia
#
# Ogni tripla diventa UN job array SLURM: un task per gruppo di CHUNK_SIZE
# combinazioni. Cosi' nessun task rischia il time limit e i task girano in
# parallelo. Al termine parte un job di raccolta che scrive grid_summary.csv.
#
# Variabili regolabili:
#   CHUNK_SIZE=50     combinazioni per task
#   MAX_PARALLEL=8    task simultanei per array
#   TIME=12:00:00     wall time per task
#   MEM=16G  CPUS=8
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

EXP_IN=${1:?"esperimento: PERT | CVETT | all"}
NOD_IN=${2:?"nodi: 15 | 25 | 40 | all"}
BAT_IN=${3:?"batch: 20|30|40|50|55|60|65|70 | all"}

CHUNK_SIZE=${CHUNK_SIZE:-50}
TIMEOUT=${TIMEOUT:-7200}
MAX_PARALLEL=${MAX_PARALLEL:-8}
TIME=${TIME:-12:00:00}
MEM=${MEM:-16G}
CPUS=${CPUS:-8}
DRY=${DRY:-0}

[[ "$EXP_IN" == "all" ]] && EXPS="PERT CVETT" || EXPS="$EXP_IN"
[[ "$NOD_IN" == "all" ]] && NODS="15 25 40"   || NODS="$NOD_IN"
[[ "$BAT_IN" == "all" ]] && BATS="20 30 40 50 55 60 65 70" || BATS="$BAT_IN"

for EXP in $EXPS; do
for N   in $NODS; do

  CACHE="$UTSP_ROOT/$EXP/RISULTATI_$N/pkl/res_B_cached.pkl"
  if [[ ! -f "$CACHE" ]]; then
    echo "SALTO $EXP/$N: manca $CACHE (lancia prima Esperimento B)" >&2
    continue
  fi

for B in $BATS; do

  LEAF="$UTSP_ROOT/grid_search/$EXP/NODI_$N/BATCH_$B"
  mkdir -p "$LEAF/logs"

  read -r N_COMBOS N_CHUNKS < <(python grid_search.py --list --batch "$B" --chunk-size "$CHUNK_SIZE")
  LAST=$((N_CHUNKS - 1))

  echo "=== $EXP | $N nodi | batch $B : $N_COMBOS combo in $N_CHUNKS task (max $MAX_PARALLEL paralleli)"
  echo "    -> $LEAF"
  [[ "$DRY" == "1" ]] && continue

JOBID=$(sbatch --parsable \
    --job-name="grid_${EXP}_${N}_${B}" \
    --array="0-${LAST}%${MAX_PARALLEL}" \
    --time="$TIME" --cpus-per-task="$CPUS" --mem="$MEM" \
    --output="$LEAF/logs/chunk_%a_%A.out" \
    --error="$LEAF/logs/chunk_%a_%A.err" \
    --export=ALL,TESI_ROOT_DIR="$UTSP_ROOT",GRID_EXP="$EXP",GRID_N="$N",GRID_B="$B",GRID_CHUNK="$CHUNK_SIZE",GRID_TIMEOUT="$TIMEOUT" <<'EOF_JOB'
#!/usr/bin/env bash
source "$TESI_ROOT_DIR/common/env.sh"
python grid_search.py --run-chunk "$SLURM_ARRAY_TASK_ID" --exp "$GRID_EXP" --nodes "$GRID_N" --batch "$GRID_B" --chunk-size "$GRID_CHUNK" --timeout "$GRID_TIMEOUT"
EOF_JOB
)

  sbatch --parsable \
    --job-name="collect_${EXP}_${N}_${B}" \
    --dependency="afterany:$JOBID" \
    --time=00:10:00 --cpus-per-task=1 --mem=2G \
    --output="$LEAF/logs/collect_%j.out" \
    --error="$LEAF/logs/collect_%j.err" \
    --export=ALL,TESI_ROOT_DIR="$UTSP_ROOT",GRID_EXP="$EXP",GRID_N="$N",GRID_B="$B" <<'EOF_JOB' > /dev/null
#!/usr/bin/env bash
source "$TESI_ROOT_DIR/common/env.sh"
python grid_search.py --collect --exp "$GRID_EXP" --nodes "$GRID_N" --batch "$GRID_B"
EOF_JOB

  echo "    array=$JOBID (+ collect in dipendenza)"

done; done; done
