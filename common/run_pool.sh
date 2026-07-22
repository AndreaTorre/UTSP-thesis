#!/usr/bin/env bash
# Uso: sbatch --array=0-31 run_pool.sh <N_NODI>     (sostituisce run_pool.sh + run_pool25.sh)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export TESI_EXPERIMENT="${TESI_EXPERIMENT:-PERT}"
export TESI_N_NODES="${1:?N nodi mancante}"
export TESI_N_TEST_SCENARIOS_UTSP="${TESI_N_TEST_SCENARIOS_UTSP:-70000}"
export TESI_POOL_ENV_BLOCK="${TESI_POOL_ENV_BLOCK:-200}"
export TESI_POOL_MIP_GAP="${TESI_POOL_MIP_GAP:-0.01}"
export TESI_POOL_MODELS="${TESI_POOL_MODELS:-STO,EEV,WS}"
python solve_pool_shard.py --shard "$SLURM_ARRAY_TASK_ID" --n-shards "${2:-32}"
