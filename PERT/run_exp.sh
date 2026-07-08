#!/bin/bash
# Wrapper PERT: lancia rete/eval UTSP tramite pipeline comune.
#
# Uso:
#   bash run_exp.sh 15
#   bash run_exp.sh 25
#   bash run_exp.sh 40

set -euo pipefail

N=${1:-${TESI_N_NODES:-40}}

if [[ "$N" != "15" && "$N" != "25" && "$N" != "40" ]]; then
  echo "Errore: N deve essere 15, 25 oppure 40. Valore ricevuto: $N"
  exit 1
fi

cd /home/atorre/UTSP/unione/git/UTSP/common
bash submit_exp.sh PERT "$N"
