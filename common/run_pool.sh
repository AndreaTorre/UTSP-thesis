#!/bin/bash
module load python; module load gurobi/13.0.0
unset GRB_WLSACCESSID GRB_WLSSECRET GRB_LICENSEID
source /home/atorre/UTSP/unione/git/UTSP/venv/bin/activate
export PYTHONPATH=/home/atorre/UTSP/unione/git/UTSP/common:$PYTHONPATH
cd /home/atorre/UTSP/unione/git/UTSP/common
export TESI_EXPERIMENT=PERT TESI_N_NODES=15
export TESI_N_TEST_SCENARIOS_UTSP=70000
export TESI_POOL_ENV_BLOCK=200
export TESI_POOL_MIP_GAP=0.01
export TESI_POOL_MODELS=STO,EEV,WS
python solve_pool_shard.py --shard $SLURM_ARRAY_TASK_ID --n-shards 32
