#!/usr/bin/env bash
# Uso: bash run_pi.sh <N_NODI> <SHARD> [N_SHARDS]
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export TESI_EXPERIMENT="${TESI_EXPERIMENT:-PERT}"
export TESI_N_NODES="${1:?N nodi mancante}"
python compute_pi_shard.py --shard "${2:?shard mancante}" --n-shards "${3:-20}"
