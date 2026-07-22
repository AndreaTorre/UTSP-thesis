#!/bin/bash
UTSP_ROOT="${TESI_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# launch_variant_tests.sh — lancia i test delle 4 varianti della loss in PARALLELO.
#
# Il parallelismo naturale del test NON è spezzare la local search (un'istanza
# di test è un blocco indivisibile: il pre-booking decide x_test sull'intero
# blocco). Il parallelismo è tra i 32 CHECKPOINT indipendenti:
#   4 varianti (pen1_sum, pen1_mean, pen0_sum, pen0_mean) x 8 batch (20..70).
# Un job SLURM per checkpoint => 32 job che girano insieme.
#
# Ogni job usa TESI_VARIANT per trovare il checkpoint sotto
#   variants/<nome>/batch_sweep/BATCH_<B>/train/espB_UTSP_LS/utsp_model.pt
# e scrive i risultati sotto la stessa variante. Il guard di ripresa di
# run_test_sweep.sh salta le combinazioni già complete.
#
# Uso:
#   bash launch_variant_tests.sh 15 "60"           # test dim=60, tutte le varianti
#   bash launch_variant_tests.sh 15 "20 30 60 70"  # più dimensioni di test

N=${1:-15}
DIMS=${2:-"60"}
ROOT=${UTSP_ROOT}

cat > "$ROOT/common/run_test_variant.sh" << 'EOF'
#!/bin/bash
module load python; module load gurobi/13.0.0
unset GRB_WLSACCESSID GRB_WLSSECRET GRB_LICENSEID
source ${UTSP_ROOT}/venv/bin/activate
export PYTHONPATH=${UTSP_ROOT}/common:$PYTHONPATH
cd ${UTSP_ROOT}/common
# $1=variante $2=pen $3=agg $4=N $5=batch $6=dimlist
export TESI_EXPERIMENT=PERT TESI_N_NODES=$4
export TESI_VARIANT=$1 TESI_UTSP_INCLUDE_PENALTY=$2 TESI_UTSP_AGGREGATION=$3
bash run_test_sweep.sh PERT $4 "$6" $5
EOF
chmod +x "$ROOT/common/run_test_variant.sh"

for combo in "pen1_sum 1 sum" "pen1_mean 1 mean" "pen0_sum 0 sum" "pen0_mean 0 mean"; do
  set -- $combo; name=$1; pen=$2; agg=$3
  for B in 20 30 40 50 55 60 65 70; do
    sbatch --job-name=t_${name}_b${B} --time=04:00:00 --cpus-per-task=4 --mem=8G \
      --output=${ROOT}/PERT/RISULTATI_${N}/tvar_${name}_b${B}_%j.out \
      --error=${ROOT}/PERT/RISULTATI_${N}/tvar_${name}_b${B}_%j.err \
      "$ROOT/common/run_test_variant.sh" ${name} ${pen} ${agg} ${N} ${B} "$DIMS"
  done
done

echo "Lanciati 32 job (4 varianti x 8 batch) su dim=[$DIMS] per $N nodi."
echo "Monitor: squeue -u atorre | grep t_pen"
