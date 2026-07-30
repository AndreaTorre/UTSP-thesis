#!/bin/bash
#SBATCH --job-name=pert15_b20_sanity
#SBATCH --account=def-tms
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=sanity_pert15_b20_%j.out
#SBATCH --error=sanity_pert15_b20_%j.err

# --- ambiente (stesso di una sessione interattiva) ---
cd /home/atorre/UTSP/unione/git/UTSP/common
source env.sh

# --- configurazione della prova ---
export TESI_EXPERIMENT=PERT
export TESI_N_NODES=15
export TESI_BATCH_SWEEP=20
export TESI_N_ISTANZE_TEST=100
export TESI_DIM_ISTANZA_TEST=20

python main.py --only B_UTSP_LS
