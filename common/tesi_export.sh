#!/usr/bin/env bash
# tesi_export.sh — inventario e archiviazione dei risultati per la tesi.
#
#   bash tesi_export.sh            # INVENTARIO: cosa c'è per PERT/CVETT x {15,25,40} x {single,mtsp}
#   bash tesi_export.sh export     # come sopra + copia gli artefatti presentabili in
#                                    tesi_export/ e crea tesi_export.tar.gz
#
# Non ricalcola niente: legge solo ciò che le run hanno già prodotto.
# Gestisce sia il layout base sia quello sotto batch_sweep/BATCH_*/ e variants/mtsp/.

set -uo pipefail
ROOT="${TESI_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODE="${1:-inventory}"
EXPS=(PERT CVETT)
NS=(15 25 40)
VARIANTS=("" mtsp)      # "" = single-TSP (base),  mtsp = multigraph

_first(){ ls $@ 2>/dev/null | head -1; }                    # primo match o vuoto
_n(){ ls $@ 2>/dev/null | wc -l | tr -d ' '; }              # conteggio match
_mark(){ [ -n "$1" ] && echo "sì " || echo " — "; }

base_dir(){ local exp=$1 n=$2 var=$3; local b="$ROOT/$exp/RISULTATI_$n"; [ -n "$var" ] && b="$b/variants/$var"; echo "$b"; }

echo "======================================================================"
echo " INVENTARIO ESPERIMENTI   (root: $ROOT)"
echo "======================================================================"
printf " %-5s %-4s %-7s | %-9s | %-9s | %-9s | %s\n" EXP N tipo benchmark modello report grafici
printf " %s\n" "----------------------------------------------------------------------"

for exp in "${EXPS[@]}"; do
  for n in "${NS[@]}"; do
    [ -d "$ROOT/$exp/RISULTATI_$n" ] || continue
    for var in "${VARIANTS[@]}"; do
      b=$(base_dir "$exp" "$n" "$var")
      [ -d "$b" ] || { [ -z "$var" ] || continue; }   # salta mtsp se la cartella non esiste
      bench=$(_first "$ROOT/$exp/RISULTATI_$n/pkl/res_B_cached.pkl")     # benchmark: sempre nella base
      mdl=$(_first "$b"/modello/*/utsp_model.pt "$b"/batch_sweep/*/modello/*/utsp_model.pt)
      rep=$(_first "$b"/report/*_test_all_instances.txt "$b"/batch_sweep/*/report/*_test_all_instances.txt)
      nplt=$(_n "$b"/grafici/*.png "$b"/batch_sweep/*/grafici/*.png)
      printf " %-5s %-4s %-7s |   %s     |   %s     |   %s     |  %s\n" \
        "$exp" "$n" "${var:-single}" "$(_mark "$bench")" "$(_mark "$mdl")" "$(_mark "$rep")" "$nplt"
    done
  done
done
echo ""

if [ "$MODE" != "export" ]; then
  echo "Legenda: benchmark=res_B_cached.pkl | modello=utsp_model.pt | report=*_test_all_instances.txt"
  echo "Per archiviare per la tesi:  bash tesi_export.sh export"
  exit 0
fi

# ── EXPORT ──────────────────────────────────────────────────────────
DEST="$ROOT/tesi_export"
rm -rf "$DEST"
echo ">> copio gli artefatti presentabili in $DEST/ ..."
for exp in "${EXPS[@]}"; do
  for n in "${NS[@]}"; do
    for var in "${VARIANTS[@]}"; do
      b=$(base_dir "$exp" "$n" "$var")
      [ -d "$b" ] || continue
      # c'è qualcosa da esportare?
      [ -n "$(_first "$b"/report/*.txt "$b"/batch_sweep/*/report/*.txt "$b"/grafici/*.png)" ] || continue
      out="$DEST/$exp/N$n/${var:-single}"
      mkdir -p "$out/report" "$out/grafici" "$out/modello"
      # report testuali (anche sotto batch_sweep), grafici, metadati modello (NON i pesi .pt: pesanti)
      cp -f "$b"/report/*.txt          "$out/report/"   2>/dev/null || true
      cp -f "$b"/batch_sweep/*/report/*.txt "$out/report/" 2>/dev/null || true
      cp -f "$b"/grafici/*.png         "$out/grafici/"  2>/dev/null || true
      cp -f "$b"/batch_sweep/*/grafici/*.png "$out/grafici/" 2>/dev/null || true
      cp -f "$b"/modello/*/utsp_metadata.json "$out/modello/" 2>/dev/null || true
      cp -f "$b"/modello/*/utsp_history.json  "$out/modello/" 2>/dev/null || true
      cp -f "$b"/batch_sweep/*/modello/*/utsp_metadata.json "$out/modello/" 2>/dev/null || true
      cp -f "$b"/batch_sweep/*/modello/*/utsp_history.json  "$out/modello/" 2>/dev/null || true
      # pulizia cartelle vuote
      rmdir "$out"/* 2>/dev/null || true
    done
    # confronti a livello di esperimento (output di compare_mtsp.py)
    cp -f "$ROOT/$exp"/confronto_mtsp*.txt "$DEST/$exp/" 2>/dev/null || true
  done
done

tar czf "$ROOT/tesi_export.tar.gz" -C "$ROOT" tesi_export 2>/dev/null
echo ">> fatto."
echo "   cartella:  $DEST"
echo "   archivio:  $ROOT/tesi_export.tar.gz"
echo "   (i pesi .pt restano dove sono; ho copiato metadati/history, report e grafici)"
