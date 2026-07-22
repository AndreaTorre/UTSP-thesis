#!/bin/bash
module load python; module load gurobi/13.0.0
unset GRB_WLSACCESSID GRB_WLSSECRET GRB_LICENSEID
source /home/atorre/UTSP/unione/git/UTSP/venv/bin/activate
export PYTHONPATH=/home/atorre/UTSP/unione/git/UTSP/common:$PYTHONPATH
cd /home/atorre/UTSP/unione/git/UTSP/common
# $1=variante $2=pen $3=N $4=batch
export TESI_EXPERIMENT=PERT TESI_N_NODES=$3
export TESI_VARIANT=$1 TESI_UTSP_INCLUDE_PENALTY=$2
export TESI_BATCH_SWEEP=$4
python main.py --only B_UTSP_LS
