#!/usr/bin/env bash
# Lancia Esperimento B.   Uso: bash submit_b.sh PERT 40
set -euo pipefail
ROOT="${TESI_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

EXP=$(echo "${1:-${TESI_EXPERIMENT:-PERT}}" | tr '[:lower:]' '[:upper:]')
N=${2:-${TESI_N_NODES:-40}}
[[ "$EXP" == "PERT" || "$EXP" == "CVETT" ]] || { echo "EXP non valido: $EXP"; exit 1; }
[[ "$N" =~ ^(15|25|40)$ ]] || { echo "N non valido: $N"; exit 1; }

OUTPUT_DIR="$ROOT/$EXP/RISULTATI_$N"
mkdir -p "$OUTPUT_DIR"/{output,grafici,checkpoint,pkl}
echo "=== Submit B === EXP=$EXP N=$N OUT=$OUTPUT_DIR"

sbatch \
  --job-name="${EXP}_${N}_B" \
  --time=24:00:00 --cpus-per-task=8 --mem=32G \
  --output="$OUTPUT_DIR/output/output_B_%j.txt" \
  --error="$OUTPUT_DIR/output/error_B_%j.txt" \
  --export=ALL,TESI_ROOT_DIR="$ROOT",TESI_EXPERIMENT="$EXP",TESI_N_NODES="$N" <<'EOF_JOB'
#!/usr/bin/env bash
source "$TESI_ROOT_DIR/common/env.sh"
python main.py --only B
EOF_JOB
