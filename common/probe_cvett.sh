#!/usr/bin/env bash
#SBATCH --job-name=probe_CVETT25
#SBATCH --account=def-tms_cpu
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/scratch/atorre/probe_CVETT25_%j.out
#SBATCH --error=/scratch/atorre/probe_CVETT25_%j.err
set -uo pipefail
cd /home/atorre/UTSP/unione/git/UTSP/common
source env.sh

export TESI_EXPERIMENT=CVETT TESI_N_NODES=25
export TESI_N_VALIDATION_SCENARIOS=8
export TESI_TEST_SKIP_PI=1 TESI_DROP_LAST_TEST_BATCH=0
export TESI_BENCH_TIME_LIMIT=120 TESI_BENCH_MIP_GAP=0.02
unset TESI_P_UTSP2_LAMBDA1 TESI_P_UTSP2_LAMBDA_D TESI_P_UTSP2_TEMP_SCALE \
      TESI_P_UTSP2_LAMBDA_B_DIV TESI_P_UTSP2_ALPHA_LOSS TESI_P_UTSP2_LAMBDA_E 2>/dev/null || true

for U in 12 8 6; do
  echo "########## DRONE_U = $U ##########"
  export TESI_DRONE_U=$U
  export TESI_OUTPUT_OVERRIDE=/scratch/atorre/probe_CVETT25_U$U
  export TESI_CACHE_DIR_OVERRIDE=/scratch/atorre/probe_CVETT25_U$U/pkl   # <-- cache isolata!
  python main.py --only B || echo "  !! U=$U FALLITO, continuo"
  echo "########## FINE U = $U ##########"
done
