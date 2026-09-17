#!/usr/bin/env bash
#SBATCH --job-name=ws_CVETT
#SBATCH --account=def-tms_cpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch/atorre/ws_CVETT_%j.out
#SBATCH --error=/scratch/atorre/ws_CVETT_%j.err
# Uso:  sbatch ws_cvett2.sh 15        (oppure 25)
# Autoconsistente: l'env sta dentro, non dipende dalla shell/nodo.
set -uo pipefail
cd /home/atorre/UTSP/unione/git/UTSP/common
source env.sh

N_NODES="${1:?passa il numero di nodi, es: sbatch ws_cvett2.sh 15}"

export TESI_EXPERIMENT=CVETT
export TESI_N_NODES="$N_NODES"
export TESI_DROP_LAST_TEST_BATCH=0
export TESI_DIM_ISTANZA_TEST=20
export TESI_N_ISTANZE_TEST=100

echo "CACHE_DIR = $(python -c 'from config import TEST_SCENARIO_CACHE_DIR as d; print(d)')"
python compute_ws_shard.py --shard 0 --n-shards 1
python merge_ws_shards.py
