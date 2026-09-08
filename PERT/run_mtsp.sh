#!/usr/bin/env bash
# PERT/run_mtsp.sh — SOTTOMETTE job SLURM per il training multi-grafo e/o la
# valutazione, con output isolati in RISULTATI_<N>/variants/mtsp/. Allineato a
# submit_exp.sh: stesso ambiente (common/env.sh), CPU, --export=ALL.
#
# Uso:
#   bash run_mtsp.sh 15 train    # sottomette il training multigraph
#   bash run_mtsp.sh 15 eval     # sottomette la valutazione su nodi_15 (riusa i benchmark)
#   bash run_mtsp.sh 15 all      # train -> eval  (eval parte in afterok sul train)
#
# Parametri via env (default tra parentesi):
#   TESI_MG_MODE=subsample|uniform (subsample)   TESI_MG_BIG (data/pool_tsplib)
#   TESI_MG_INSTANCES (300)   TESI_MG_K (20)
#   TESI_SLURM_ACCOUNT (nessuno -> usa il tuo account di default, come submit_exp.sh)
#   TESI_MTSP_LOCAL=1  -> gira in foreground senza SLURM (per test su workstation)

set -euo pipefail

ROOT="${TESI_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
EXP=PERT
N=${1:-15}
PHASE=${2:-all}
[[ "$N" =~ ^(15|25|40)$ ]] || { echo "N non valido: $N (usa 15|25|40)"; exit 1; }

MODE="${TESI_MG_MODE:-subsample}"
BIG="${TESI_MG_BIG:-$ROOT/data/pool_tsplib}"
INST="${TESI_MG_INSTANCES:-300}"
K="${TESI_MG_K:-20}"

OUT="$ROOT/$EXP/RISULTATI_$N/variants/mtsp"
BASE_PKL="$ROOT/$EXP/RISULTATI_$N/pkl"
mkdir -p "$OUT/output" "$OUT/pkl"

ACCT=""; [[ -n "${TESI_SLURM_ACCOUNT:-}" ]] && ACCT="--account=$TESI_SLURM_ACCOUNT"
COMMON_EXPORT="TESI_ROOT_DIR=$ROOT,TESI_EXPERIMENT=$EXP,TESI_N_NODES=$N,TESI_VARIANT=mtsp"

reuse_benchmarks() {
  # res_B e cache di test dipendono solo dal grafo di test (identico al single-TSP):
  # symlink dalla pkl/ base invece di ricalcolare ore di STO.
  for f in res_B_cached.pkl test_scenarios_cache.pkl test_sto_eev_cache.pkl test_ws_cache.pkl ws_shards; do
    if [[ -e "$BASE_PKL/$f" && ! -e "$OUT/pkl/$f" ]]; then
      ln -sfn "$(cd "$BASE_PKL" && pwd)/$f" "$OUT/pkl/$f"
    fi
  done
  [[ -e "$BASE_PKL/res_B_cached.pkl" ]] || \
    echo "ATTENZIONE: manca $BASE_PKL/res_B_cached.pkl — genera prima i benchmark:  bash PERT/run_tutto.sh $N"
}

# ── modalità LOCALE (senza SLURM), per una prova rapida ─────────────
if [[ "${TESI_MTSP_LOCAL:-0}" == "1" ]]; then
  source "$ROOT/common/env.sh"
  export TESI_EXPERIMENT=$EXP TESI_N_NODES=$N TESI_VARIANT=mtsp
  _train(){ export TESI_MG_MODE=$MODE TESI_MG_BIG=$BIG TESI_MG_INSTANCES=$INST TESI_MG_K=$K; python multigraph.py; }
  _eval(){ reuse_benchmarks; export TESI_REUSE_UTSP_TRAIN=1; python main.py --only B_UTSP_LS; }
  case "$PHASE" in train) _train;; eval) _eval;; all) _train; _eval;; *) echo "fase: train|eval|all"; exit 1;; esac
  echo "fatto (locale). Output in: $OUT"; exit 0
fi

# ── sottomissione SLURM (default) ───────────────────────────────────
submit_train() {
  sbatch --parsable $ACCT \
    --job-name="MTSP_${N}_train" \
    --time=08:00:00 --cpus-per-task=8 --mem=16G \
    --output="$OUT/output/train_%j.out" --error="$OUT/output/train_%j.err" \
    --export=ALL,$COMMON_EXPORT,TESI_MG_MODE=$MODE,TESI_MG_BIG=$BIG,TESI_MG_INSTANCES=$INST,TESI_MG_K=$K \
    <<'EOF_JOB'
#!/usr/bin/env bash
source "$TESI_ROOT_DIR/common/env.sh"
python multigraph.py
EOF_JOB
}

submit_eval() {
  local dep="$1" depflag=""
  [[ -n "$dep" ]] && depflag="--dependency=afterok:$dep"
  reuse_benchmarks
  sbatch --parsable $ACCT $depflag \
    --job-name="MTSP_${N}_eval" \
    --time=04:00:00 --cpus-per-task=8 --mem=16G \
    --output="$OUT/output/eval_%j.out" --error="$OUT/output/eval_%j.err" \
    --export=ALL,$COMMON_EXPORT,TESI_REUSE_UTSP_TRAIN=1 \
    <<'EOF_JOB'
#!/usr/bin/env bash
source "$TESI_ROOT_DIR/common/env.sh"
python main.py --only B_UTSP_LS
EOF_JOB
}

case "$PHASE" in
  train) J=$(submit_train); echo "  train  -> job $J" ;;
  eval)  J=$(submit_eval ""); echo "  eval   -> job $J" ;;
  all)
    T=$(submit_train); echo "  train  -> job $T"
    E=$(submit_eval "$T"); echo "  eval   -> job $E (parte dopo il train, afterok:$T)"
    ;;
  *) echo "fase ignota: $PHASE (train|eval|all)"; exit 1 ;;
esac

echo "  output/log in: $OUT/output/   |   monitora:  squeue -u \$USER"
