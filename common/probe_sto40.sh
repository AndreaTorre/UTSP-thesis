#!/usr/bin/env bash
#SBATCH --job-name=probeSTO_P40
#SBATCH --account=def-tms_cpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/atorre/probeSTO_P40_%j.out
#SBATCH --error=/scratch/atorre/probeSTO_P40_%j.err
set -uo pipefail
cd /home/atorre/UTSP/unione/git/UTSP/common && source env.sh
export TESI_EXPERIMENT=PERT TESI_N_NODES=40
export TESI_STO_TIME_LIMIT=9000 TESI_STO_MIP_GAP=0.0001   # ~2.5h, gap stretto: lascia correre
export TESI_TEST_SKIP_PI=1
export TESI_OUTPUT_OVERRIDE=/scratch/atorre/probeSTO_P40
export TESI_CACHE_DIR_OVERRIDE=/scratch/atorre/probeSTO_P40/pkl
unset TESI_P_UTSP2_LAMBDA1 TESI_P_UTSP2_LAMBDA_D TESI_P_UTSP2_TEMP_SCALE \
      TESI_P_UTSP2_LAMBDA_B_DIV TESI_P_UTSP2_ALPHA_LOSS TESI_P_UTSP2_LAMBDA_E 2>/dev/null || true
python main.py --only B
