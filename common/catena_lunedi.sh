#!/usr/bin/env bash
# ============================================================================
# CATENA PERT-40 COMPLETA, concatenata dal WS gia' in corso (job 3260647).
# Da lanciare STANOTTE. Gira da sola per 3 giorni. Lunedi' -> risultato PERT-40.
#
# ORDINE:  WS(in corso) --> STO --> assemble --> rete(train+test)
#   Tutto SERIALIZZATO (contesa licenza Gurobi): un MILP alla volta.
#
# PREREQUISITO OBBLIGATORIO: applicare PRIMA la patch assemble PI-opzionale:
#   bash patch_assemble_pi_opzionale.sh
# (senza, l'assemble CRASHA perche' il WS gira con skip_pi -> pi_results vuoto.)
#
# NOTE:
#  - STO afterany:WS -> parte quando il WS finisce, comunque sia finito.
#  - assemble afterany:STO -> assembla anche se lo STO e' troncato (basta l'incumbent).
#  - rete afterok:assemble -> senza res_B non ha senso.
#  - I benchmark WS/STO/EEV sono IN-SAMPLE sul train (8 scenari SAA): la rete li
#    riporta nel suo aggregate. Il confronto rete-vs-benchmark e' quello che conta.
#  - CVETT-40 NON incluso (da zero, mai testato): fallo presidiato al ritorno.
# ============================================================================
set -euo pipefail
ROOT=/home/atorre/UTSP/unione/git/UTSP
ACCT=def-tms
WS_JOB=3260647              # <-- il WS PERT-40 in corso. Verifica con: squeue -u atorre
cd "$ROOT"

# safety: la patch e' stata applicata?
if ! grep -q "PI assente (WS con skip_pi)" PERT/gurobi_parallelo.py; then
  echo "ERRORE: patch assemble PI-opzionale NON applicata. Esegui prima:"
  echo "   bash patch_assemble_pi_opzionale.sh"
  exit 1
fi

ENVSET='module load gurobi/13.0.0
  source '"$ROOT"'/venv/bin/activate
  unset GRB_WLSACCESSID GRB_WLSSECRET GRB_LICENSEID
  unset TESI_BENCH_MIP_GAP TESI_BENCH_TIME_LIMIT
  export PYTHONPATH='"$ROOT"'/common:$PYTHONPATH'

# backup dell'incumbent STO troncato esistente (5.71%), non si sa mai
cp PERT/RISULTATI_40/pkl/parallel_data/sto.pkl \
   PERT/RISULTATI_40/pkl/parallel_data/sto.oom57.bak 2>/dev/null || true

# 1) STO: aspetta il WS, poi parte. 192G contro OOM, gap-target 3% (Opzione 1).
JOB_STO=$(sbatch --parsable --job-name=pert40_sto --account=$ACCT \
  --dependency=afterany:$WS_JOB \
  --time=12:00:00 --cpus-per-task=1 --mem=192G \
  --output=PERT/RISULTATI_40/output/sto3_%j.txt \
  --error=PERT/RISULTATI_40/output/sto3_%j.txt \
  --wrap="$ENVSET
    export TESI_EXPERIMENT=PERT TESI_N_NODES=40
    export TESI_STO_MIP_GAP=0.03 TESI_STO_TIME_LIMIT=36000
    cd $ROOT/PERT && python gurobi_parallelo.py sto")
echo "STO      = $JOB_STO   (dopo WS $WS_JOB)"

# 2) assemble: dopo lo STO (comunque finito). Niente Gurobi, veloce.
JOB_ASM=$(sbatch --parsable --job-name=pert40_assemble --account=$ACCT \
  --dependency=afterany:$JOB_STO \
  --time=00:30:00 --mem=16G \
  --output=PERT/RISULTATI_40/output/assemble_%j.txt \
  --error=PERT/RISULTATI_40/output/assemble_%j.txt \
  --wrap="$ENVSET
    export TESI_EXPERIMENT=PERT TESI_N_NODES=40
    cd $ROOT/PERT && python gurobi_parallelo.py assemble")
echo "assemble = $JOB_ASM   (dopo STO)"

# 3) rete train+test: afterok (serve res_B). Tempo largo: test-LS a n=40 e' lento.
JOB_NET=$(sbatch --parsable --job-name=pert40_net --account=$ACCT \
  --dependency=afterok:$JOB_ASM \
  --time=23:00:00 --cpus-per-task=4 --mem=32G \
  --output=PERT/RISULTATI_40/output/net_%j.txt \
  --error=PERT/RISULTATI_40/output/net_%j.txt \
  --wrap="$ENVSET
    export TESI_EXPERIMENT=PERT TESI_N_NODES=40 TESI_GRB_THREADS=4
    cd $ROOT/PERT && python main.py --only B_UTSP_LS")
echo "rete     = $JOB_NET   (dopo assemble)"

echo
echo "CATENA LANCIATA. Ordine: WS($WS_JOB) -> STO($JOB_STO) -> assemble($JOB_ASM) -> rete($JOB_NET)"
echo "Lunedi': PERT/RISULTATI_40/output/net_*.txt  e  PERT/RISULTATI_40/report/"
echo "Controlla con: squeue -u atorre   (o i log se sacct e' giu)"
