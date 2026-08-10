#!/usr/bin/env bash
# common/run.sh — entry point UNICO per i job UTSP (PERT/CVETT).
#
# Possiede in un posto solo l'ambiente per-esperimento (batch al backend CVETT,
# pool di test divisibile, semi, split) e instrada al worker già unificato.
#
# Uso:
#   bash run.sh <kind> <exp> <nodi> [batch | "lista batch"]
#     kind = b | train | test | grid
#     exp  = pert | cvett
#     nodi = 15 | 25 | 40
#
# Assi (override via env, con default):
#   NINST=100   istanze di training   (PERT: richiede N_ISTANZE_TARGET=env, patch 1-riga)
#   IS=100      istanze di test
#   DIM=20      dimensione istanza di test (test: può essere "300 100 60")
#   BATCH=20    batch size di default se non passi il 4º argomento
#   EVAL_SPLIT=test   test | val   (val = pool di validation per la grid)
#   SKIP_PI=1   salta il PI a test
#   DRY=1       stampa i comandi e NON lancia
#
# Regola CVETT: TESI_N_TEST_SCENARIOS = IS*DIM RIDOTTO al multiplo del batch
# (così la divisibilità del backend torna sempre, senza aggiustare a mano).
#
# Esempi:
#   DRY=1 bash run.sh b     cvett 25
#   bash run.sh train pert  25 "20 30 40"
#   bash run.sh grid  cvett 25 "20 30"
#   DIM="300 100 60" bash run.sh test cvett 25 20
#   EVAL_SPLIT=val bash run.sh grid pert 25 20      # grid su validation

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${TESI_ROOT_DIR:-$(cd "$HERE/.." && pwd)}"

KIND=$(echo "${1:?kind: b|train|test|grid}" | tr '[:upper:]' '[:lower:]')
EXP=$(echo  "${2:?exp: pert|cvett}"        | tr '[:lower:]' '[:upper:]')
N=${3:?nodi: 15|25|40}
LIST=${4:-${BATCH:-20}}

[[ "$KIND" =~ ^(b|train|test|grid)$ ]] || { echo "kind non valido: $KIND" >&2; exit 1; }
[[ "$EXP" == PERT || "$EXP" == CVETT ]] || { echo "exp non valido: $EXP" >&2; exit 1; }
[[ "$N" =~ ^(15|25|40)$ ]]             || { echo "nodi non validi: $N" >&2; exit 1; }

NINST=${NINST:-100}; IS=${IS:-100}; DIM=${DIM:-20}
EVAL_SPLIT=${EVAL_SPLIT:-test}; SKIP_PI=${SKIP_PI:-1}; DRY=${DRY:-0}

# --- assi comuni a tutti i kind ---
export TESI_ROOT_DIR="$ROOT" TESI_EXPERIMENT="$EXP" TESI_N_NODES="$N"
export TESI_N_UTSP_TRAIN_INSTANCES="$NINST"
export TESI_N_ISTANZE_TEST="$IS" TESI_DIM_ISTANZA_TEST="$DIM"
export TESI_EVAL_SPLIT="$EVAL_SPLIT" TESI_TEST_SKIP_PI="$SKIP_PI"

run() { echo "+ $*"; [[ "$DRY" == "1" ]] || "$@"; }

# imposta le variabili batch-dipendenti; per CVETT riduce il pool test al
# multiplo del batch. $2 = ampiezza del pool da coprire (default IS*DIM).
set_batch_env() {
  local b=$1 span=${2:-$((IS*DIM))}
  export TESI_BATCH_SWEEP="$b"
  [[ "$EXP" == CVETT ]] || return 0
  export TESI_UTSP_BATCH_SIZE="$b"
  local ntest=$(( (span / b) * b ))
  (( ntest > 0 )) || { echo "STOP: pool test ($span) < batch $b" >&2; exit 2; }
  export TESI_N_TEST_SCENARIOS="$ntest"
  (( ntest == span )) || echo "  [CVETT] N_TEST ridotto a $ntest (multiplo di $b, da $span)"
}

echo "=== run.sh: kind=$KIND exp=$EXP nodi=$N | NINST=$NINST IS=$IS DIM=$DIM | split=$EVAL_SPLIT | DRY=$DRY"

case "$KIND" in
  b)
    # B via pipeline parallela (wind-aware per CVETT). Batch singolo: serve solo
    # a far tornare la validazione di divisibilità della config CVETT.
    set_batch_env "$LIST"
    run bash "$ROOT/$EXP/run_tutto.sh" "$N"
    ;;

  train)
    # batch sweep di training: un submit_exp per batch, con l'env CVETT corretto.
    for b in $LIST; do
      set_batch_env "$b"
      run bash "$HERE/submit_exp.sh" "$EXP" "$N"
    done
    ;;

  grid)
    # grid search: un array grid_submit per batch, con l'env CVETT corretto.
    for b in $LIST; do
      set_batch_env "$b"
      run bash "$HERE/grid_submit.sh" "$EXP" "$N" "$b"
    done
    ;;

  test)
    # batch sweep di test: run_test_sweep gira INLINE, quindi lo avvolgo in UN job.
    # Il pool CVETT deve coprire la combinazione più grande IS*max(DIM).
    maxdim=$(printf '%s\n' $DIM | sort -n | tail -1)
    set_batch_env "$LIST" $((IS * maxdim))
    run sbatch --job-name="${EXP}_${N}_TEST" \
        --time=12:00:00 --cpus-per-task=8 --mem=16G \
        --output="$ROOT/$EXP/RISULTATI_$N/output/test_%j.txt" \
        --export=ALL \
        --wrap="bash '$HERE/run_test_sweep.sh' '$EXP' '$N' '$DIM' '$LIST'"
    ;;
esac
