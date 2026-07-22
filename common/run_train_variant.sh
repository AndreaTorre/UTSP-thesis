#!/usr/bin/env bash
# Training di una variante. Sostituisce anche run_one_variant.sh.
# Uso: bash run_train_variant.sh <VARIANTE> <PENALTY 0|1> <N_NODI> <BATCH>
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export TESI_EXPERIMENT="${TESI_EXPERIMENT:-PERT}"
export TESI_VARIANT="${1:?variante mancante}"
export TESI_UTSP_INCLUDE_PENALTY="${2:?penalty mancante}"
export TESI_N_NODES="${3:?N nodi mancante}"
export TESI_BATCH_SWEEP="${4:?batch mancante}"
python main.py --only B_UTSP_LS
