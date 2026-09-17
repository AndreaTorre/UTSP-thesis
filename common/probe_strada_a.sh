#!/usr/bin/env bash
#SBATCH --job-name=probeA_CVETT
#SBATCH --account=def-tms_cpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch/atorre/probeA_CVETT_%j.out
#SBATCH --error=/scratch/atorre/probeA_CVETT_%j.err
set -uo pipefail
cd /home/atorre/UTSP/unione/git/UTSP/common
source env.sh
N_NODES="${1:?N nodi, es: sbatch probe_strada_a.sh 15 12}"
DRONE_U="${2:?DRONE_U, es: sbatch probe_strada_a.sh 15 12}"
export TESI_EXPERIMENT=CVETT
export TESI_N_NODES="$N_NODES"
export TESI_DRONE_U="$DRONE_U"
export TESI_DROP_LAST_TEST_BATCH=0
export TESI_DIM_ISTANZA_TEST="${TESI_DIM_ISTANZA_TEST:-10}"
export TESI_N_ISTANZE_TEST="${TESI_N_ISTANZE_TEST:-5}"
export TESI_TEST_SKIP_PI=1
echo "CACHE_DIR = $(python -c 'from config import TEST_SCENARIO_CACHE_DIR as d; print(d)')"
echo "DIM=$TESI_DIM_ISTANZA_TEST N_ISTANZE=$TESI_N_ISTANZE_TEST U=$TESI_DRONE_U (gap default ~esatto)"
python probe_strada_a.py
