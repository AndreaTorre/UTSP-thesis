#!/bin/bash
set -euo pipefail
module load python
module load gurobi/13.0.0
unset GRB_WLSACCESSID GRB_WLSSECRET GRB_LICENSEID
ROOT=/home/atorre/UTSP/unione/git/UTSP
source "$ROOT/venv/bin/activate"
export PYTHONPATH="$ROOT/common:$PYTHONPATH"
cd "$ROOT/common"
# $1=variante $2=pen $3=N $4=batch
export TESI_EXPERIMENT=PERT
export TESI_N_NODES=$3
export TESI_VARIANT=$1
export TESI_UTSP_INCLUDE_PENALTY=$2
export TESI_BATCH_SWEEP=$4
export PYTHONUNBUFFERED=1
python main.py --only B_UTSP_LS
