#!/bin/bash
module load python
module load gurobi/13.0.0
unset GRB_WLSACCESSID GRB_WLSSECRET GRB_LICENSEID
source /home/atorre/UTSP/unione/git/UTSP/venv/bin/activate
export PYTHONPATH=/home/atorre/UTSP/unione/git/UTSP/common:$PYTHONPATH
cd /home/atorre/UTSP/unione/git/UTSP/common
export TESI_EXPERIMENT=PERT
export TESI_N_NODES=$1
python compute_pi_shard.py --shard $2 --n-shards 20
