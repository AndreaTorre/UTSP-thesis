#!/bin/bash
UTSP_ROOT="${TESI_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# run_test_sweep.sh — testa ogni modello UTSP già allenato (batch_sweep) su
# più combinazioni (n_istanze_test, dim_istanza_test), senza riallenare.
#
# Sia il PI che il costo delle policy STO/EEV sono cacheati per scenario_id
# (vedi TEST_SCENARIO_CACHE_DIR in config.py, generate_test_scenario_blocks
# e _validate_policies_cached in local_search.py): la prima combinazione
# DIM=300 IS=100 risolve con Gurobi tutti i 30.000 scenari (PI + reservation-
# TSP per x_sto + reservation-TSP per x_ev) una volta sola. Ogni combinazione
# successiva — DIM diverso, IS diverso, persino BATCH_X diverso o un run
# lanciato giorni dopo — pesca dalla cache senza richiamare Gurobi, perché
# x_sto/x_ev sono policy già fisse: il costo di ogni scenario non dipende da
# quali altri scenari sono nello stesso blocco di test.
#
# Uso:
#   bash run_test_sweep.sh                          # tutti i checkpoint trovati
#   bash run_test_sweep.sh PERT                      # solo PERT, tutti gli N_NODES
#   bash run_test_sweep.sh PERT 25                   # solo PERT, 25 nodi
#   bash run_test_sweep.sh PERT 25 "300 100 70 60 30 20"   # DIM custom
#   bash run_test_sweep.sh PERT 25 "60" 40           # solo BATCH_40, solo DIM=60
#
# Output di ogni combinazione:
#   <EXP>/RISULTATI_<N>/batch_sweep/BATCH_<X>/test/IS_100_DIM_<dim>/

set -euo pipefail

module load python
module load gurobi/13.0.0
unset GRB_WLSACCESSID
unset GRB_WLSSECRET
unset GRB_LICENSEID

ROOT=${UTSP_ROOT}
N_ISTANZE_TEST=${TESI_N_ISTANZE_TEST:-100}

# PI saltato di default: per confrontare STO/EEV/UTSP non serve, e toglie
# un solve_exact_tsp per scenario. Gli scenari restano in cache completi di
# scenario_dist: un run futuro con TESI_TEST_SKIP_PI=0 riempie i PI mancanti
# sugli STESSI scenari (fill-in), agganciando il quarto modello ai tre già
# calcolati senza rigenerare nulla.
SKIP_PI=${TESI_TEST_SKIP_PI:-1}

FILTER_EXP=${1:-}
FILTER_N=${2:-}
read -ra DIM_VALUES <<< "${3:-300 100 70 60 30 20}"
FILTER_BATCH=${4:-}

source "$ROOT/venv/bin/activate"
export PYTHONPATH="$ROOT/common:$PYTHONPATH"
cd "$ROOT/common"

# NOTA: il primo DIM della lista (il più grande) scalda la cache PI per
# tutti gli altri di quello stesso checkpoint: ordine decrescente non è
# solo estetico, riduce il tempo totale della coda.
find "$ROOT" -type f -path "*/batch_sweep/BATCH_*/train/*/utsp_model.pt" | sort | while read -r ckpt; do

  if [[ ! "$ckpt" =~ /(PERT|CVETT)/RISULTATI_([0-9]+)/(variants/[^/]+/)?batch_sweep/BATCH_([0-9]+)/train/([^/]+)/utsp_model\.pt$ ]]; then
    echo "  Salto (path inatteso, non combacia con lo schema noto): $ckpt"
    continue
  fi
  exp="${BASH_REMATCH[1]}"
  n_nodes="${BASH_REMATCH[2]}"
  variant_seg="${BASH_REMATCH[3]}"   # "variants/<nome>/" oppure vuoto
  batch_x="${BASH_REMATCH[4]}"
  train_name="${BASH_REMATCH[5]}"

  # FILTRO_VARIANTE: se TESI_VARIANT è settata, testa SOLO i checkpoint di
  # quella variante. Senza, il find trova i checkpoint di tutte le varianti
  # e il job li testerebbe con l'ambiente sbagliato, mescolando i risultati.
  if [ -n "${TESI_VARIANT:-}" ]; then
    if [ "$variant_seg" != "variants/${TESI_VARIANT}/" ]; then
      continue
    fi
  else
    if [ -n "$variant_seg" ]; then
      continue   # senza TESI_VARIANT, ignora i checkpoint dentro variants/
    fi
  fi

  if [[ -n "$FILTER_EXP" && "$exp" != "$FILTER_EXP" ]]; then continue; fi
  if [[ -n "$FILTER_N" && "$n_nodes" != "$FILTER_N" ]]; then continue; fi
  if [[ -n "$FILTER_BATCH" && "$batch_x" != "$FILTER_BATCH" ]]; then continue; fi

  for dim in "${DIM_VALUES[@]}"; do
    out_tag="IS_${N_ISTANZE_TEST}_DIM_${dim}"
    batch_dir="$ROOT/$exp/RISULTATI_${n_nodes}/${variant_seg}batch_sweep/BATCH_${batch_x}"
    log_dir="$batch_dir/test"
    mkdir -p "$log_dir"

    # Ripresa dopo timeout: se la combinazione ha già tutte le IS istanze
    # (un file di stats per istanza), saltala. Le cache Gurobi sopravvivono
    # al timeout, ma la local search no: senza questo guard un rilancio
    # rifarebbe da zero anche le combinazioni già complete.
    # NOTA: il test -d è necessario. Con `set -euo pipefail`, un find su una
    # cartella inesistente esce 1, pipefail propaga il fallimento e lo script
    # muore in silenzio — cioè a ogni combinazione nuova.
    done_dir="$batch_dir/test/${out_tag}/grafici"
    n_done=0
    if [ -d "$done_dir" ]; then
      n_done=$(find "$done_dir" -maxdepth 1 -name "*_test_only_i*_test_only_stats.txt" | wc -l)
    fi
    if [ "$n_done" -ge "$N_ISTANZE_TEST" ]; then
      echo "=== ${exp} | ${n_nodes} nodi | BATCH_${batch_x} | ${out_tag} — GIÀ COMPLETA ($n_done istanze), salto ==="
      continue
    fi
    if [ "$n_done" -gt 0 ]; then
      echo "    (combinazione parziale: $n_done/${N_ISTANZE_TEST} istanze — viene rifatta da capo)"
    fi

    echo ""
    echo "=== ${exp} | ${n_nodes} nodi | BATCH_${batch_x} | train=${train_name} | ${out_tag} ==="

    TESI_EXPERIMENT="$exp" \
    TESI_N_NODES="$n_nodes" \
    TESI_BATCH_SWEEP="$batch_x" \
    TESI_UTSP_TEST_ONLY=1 \
    TESI_UTSP_TRAIN_NAME="$train_name" \
    TESI_DIM_ISTANZA_TEST="$dim" \
    TESI_N_ISTANZE_TEST="$N_ISTANZE_TEST" \
    TESI_TEST_SKIP_PI="$SKIP_PI" \
    TESI_TEST_OUTPUT_SUBDIR="test/${out_tag}" \
    python main.py --only B_UTSP_LS \
      2>&1 | tee "${log_dir}/${out_tag}.log"
  done
done

echo ""
echo "Fatto. Combinazioni lanciate: vedi i log in <EXP>/RISULTATI_<N>/batch_sweep/BATCH_<X>/test/*.log"
