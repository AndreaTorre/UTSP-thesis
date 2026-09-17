#!/usr/bin/env bash
#SBATCH --job-name=test_CVETT
#SBATCH --account=def-tms_cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch/atorre/test_CVETT_%j.out
#SBATCH --error=/scratch/atorre/test_CVETT_%j.err
# Uso:  sbatch test_only_cvett.sh 15        (oppure 25)
# Carica il checkpoint UTSP gia' addestrato e rigenera SOLO il test,
# rileggendo il pool WS appena creato (test_ws_cache.pkl). Nessun training.
set -uo pipefail
cd /home/atorre/UTSP/unione/git/UTSP/common
source env.sh

N_NODES="${1:?passa il numero di nodi, es: sbatch test_only_cvett.sh 15}"

export TESI_EXPERIMENT=CVETT
export TESI_N_NODES="$N_NODES"
export TESI_DROP_LAST_TEST_BATCH=0
export TESI_DIM_ISTANZA_TEST=20
export TESI_N_ISTANZE_TEST=100
export TESI_UTSP_TEST_ONLY=1

echo "CACHE_DIR = $(python -c 'from config import TEST_SCENARIO_CACHE_DIR as d; print(d)')"
python main.py --only B_UTSP_LS
