#!/usr/bin/env bash
# Ambiente comune. Sourcelo come prima riga di ogni script:
#   source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
# ROOT si autodetermina dalla posizione del file: niente path assoluti.
UTSP_ROOT="${TESI_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export TESI_ROOT_DIR="$UTSP_ROOT"
module load python 2>/dev/null || true
module load gurobi/13.0.0 2>/dev/null || true
unset GRB_WLSACCESSID GRB_WLSSECRET GRB_LICENSEID
source "$UTSP_ROOT/venv/bin/activate"
export PYTHONPATH="$UTSP_ROOT/common:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
cd "$UTSP_ROOT/common"
