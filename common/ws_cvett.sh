#!/usr/bin/env bash
#SBATCH --time=1:00:00
#SBATCH --output=ws_cvett_%j.log
# Uso:  sbatch ws_cvett.sh 15        (oppure 25)
#   o:  bash    ws_cvett.sh 15        (foreground, WS su 15 nodi e' veloce)
# Autoconsistente: NON dipende dagli export della shell (che si perdono cambiando nodo).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source env.sh

N_NODES="${1:?passa il numero di nodi, es: sbatch ws_cvett.sh 15}"

# env che conta per il path della cache e per l'assert CVETT all'import:
export TESI_EXPERIMENT=CVETT
export TESI_N_NODES="$N_NODES"
export TESI_DROP_LAST_TEST_BATCH=0
export TESI_DIM_ISTANZA_TEST=20
export TESI_N_ISTANZE_TEST=100

echo "CACHE_DIR = $(python -c 'from config import TEST_SCENARIO_CACHE_DIR as d; print(d)')"
python compute_ws_shard.py --shard 0 --n-shards 1
python merge_ws_shards.py
