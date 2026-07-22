#!/usr/bin/env bash
UTSP_ROOT="${TESI_ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# collect_cvett_results.sh
#
# Raccoglie, per ogni esperimento CVETT (ogni cartella
# RISULTATI_<N>/batch_sweep/BATCH_<M>/), le metriche di train e test
# per i modelli UTSP, STO, PI, PI+pren, EEV, piu' varianza (within/between,
# pre e post booking) e skewness/curtosi di UTSP (calcolate sui valori
# grezzi post_total, sia train che test).
#
# Fonti dati per ciascun esperimento:
#   grafici/espB_UTSP_LS_cost_distributions_train_stats.txt
#     -> PI, PI+pren, UTSP (train/test), varianza per batch (pre/post booking),
#        valori scenario-per-scenario (sezioni "COSTI TRAIN/TEST SCENARIO PER SCENARIO")
#   output/output_UTSP_*.txt  (log SLURM; se piu' di uno, si usa il piu' recente)
#     -> STO, EEV (train/test), che non sono nel file delle statistiche
#
# Uso:
#   ./collect_cvett_results.sh [CVETT_ROOT] [OUT_CSV]
#
# Default:
#   CVETT_ROOT = ${UTSP_ROOT}/CVETT
#   OUT_CSV   = ./cvett_batch_sweep_summary.csv
#
# NOTA: skewness/curtosi sono calcolate qui (non nel codice Python del progetto)
# leggendo i 3000/300 valori "post_total" dalle sezioni "COSTI ... SCENARIO PER
# SCENARIO" del file delle statistiche, tramite un piccolo script Python embedded.

set -uo pipefail   # niente -e: i grep "a vuoto" sono normali (campi mancanti -> nan/vuoto), non devono killare lo script

CVETT_ROOT="${1:-${UTSP_ROOT}/CVETT}"
OUT_CSV="${2:-./cvett_batch_sweep_summary.csv}"

STATS_FILENAME="espB_wind_UTSP_LS_cost_distributions_train_stats.txt"

if [[ ! -d "$CVETT_ROOT" ]]; then
    echo "Errore: directory non trovata: $CVETT_ROOT" >&2
    exit 1
fi

# Trova python (preferisco quello del venv UTSP se siamo dentro quell'albero, altrimenti python3 di sistema)
PY="python3"
command -v "$PY" >/dev/null 2>&1 || PY="python"

# --- helper: estrae un numero (anche nan) dopo "label = " o "label =" ---
extract_num() {
    local file="$1" label="$2"
    grep -m1 -oE "${label}[[:space:]]*=[[:space:]]*(nan|-?[0-9.]+)" "$file" 2>/dev/null \
        | grep -oE '(nan|-?[0-9.]+)$'
}

# --- helper: skewness e curtosi sui valori "post_total" di una sezione ---
# argv: file sezione_header colonna_index(0-based dopo lo split su '|')
compute_skew_kurt() {
    local file="$1" section="$2"
    "$PY" - "$file" "$section" <<'PYEOF'
import sys

path, section = sys.argv[1], sys.argv[2]
vals = []
in_section = False
col_idx = None  # indice di post_total, determinato dall'header della sezione

with open(path, "r", errors="replace") as f:
    for line in f:
        s = line.strip()
        if s == section:
            in_section = True
            col_idx = None
            continue
        if not in_section:
            continue
        if not s:
            continue
        if s.lower().startswith("scenario"):
            # riga di header: trova l'indice della colonna "post_total"
            cols = [c.strip() for c in s.split("|")]
            try:
                col_idx = cols.index("post_total")
            except ValueError:
                col_idx = None
            continue
        if "|" not in s:
            break  # fine sezione
        if col_idx is None:
            continue
        parts = [p.strip() for p in s.split("|")]
        if len(parts) <= col_idx:
            continue
        try:
            post_total = float(parts[col_idx])
        except ValueError:
            continue
        vals.append(post_total)

n = len(vals)
if n < 3:
    print("nan,nan,nan")
    sys.exit(0)

mean = sum(vals) / n
m2 = sum((v - mean) ** 2 for v in vals) / n
m3 = sum((v - mean) ** 3 for v in vals) / n
m4 = sum((v - mean) ** 4 for v in vals) / n
std = m2 ** 0.5

skew = m3 / (std ** 3) if std > 0 else float("nan")
kurt = (m4 / (std ** 4) - 3) if std > 0 else float("nan")  # excess kurtosis

print(f"{n},{skew:.6f},{kurt:.6f}")
PYEOF
}

# Header CSV
header="batch_size"
header="${header},UTSP_train,STO_train,EEV_train,PI_train,PIpren_train"
header="${header},UTSP_test,STO_test,EEV_test,PI_test,PIpren_test"
header="${header},train_var_total_pre,train_var_within_pre,train_var_between_pre,train_pct_within_pre,train_pct_between_pre"
header="${header},train_var_total_post,train_var_within_post,train_var_between_post,train_pct_within_post,train_pct_between_post"
header="${header},UTSP_train_n,UTSP_train_skew,UTSP_train_kurt"
header="${header},UTSP_test_n,UTSP_test_skew,UTSP_test_kurt"
echo "$header" > "$OUT_CSV"

shopt -s nullglob nocaseglob
batch_dirs=("$CVETT_ROOT"/RISULTATI_*/batch_sweep/batch_*/ "$CVETT_ROOT"/RISULTATI_*/batch_sweep/BATCH_*/)
shopt -u nullglob nocaseglob
batch_dirs=($(printf '%s\n' "${batch_dirs[@]}" | sort -u))

if [[ ${#batch_dirs[@]} -eq 0 ]]; then
    echo "Errore: nessuna cartella RISULTATI_*/batch_sweep/BATCH_*/ trovata in $CVETT_ROOT" >&2
    exit 1
fi

for bdir in "${batch_dirs[@]}"; do
    bname="$(basename "$bdir")"
    bsize="${bname#BATCH_}"
    bsize="${bsize#batch_}"
    risN="$(basename "$(dirname "$(dirname "$bdir")")")"   # RISULTATI_N
    label="${risN}_${bname}"

    stats_file="${bdir}grafici/${STATS_FILENAME}"
    if [[ ! -f "$stats_file" ]]; then
        echo "Attenzione: manca $stats_file, salto $label" >&2
        continue
    fi

    # --- PI / PI+pren / UTSP (train + test) dal file delle statistiche ---
    pi_train=$(extract_num "$stats_file" "PI train")
    pipren_train=$(extract_num "$stats_file" "PI\+pren train")
    utsp_train=$(extract_num "$stats_file" "UTSP train post")
    pi_test=$(extract_num "$stats_file" "PI test")
    pipren_test=$(extract_num "$stats_file" "PI\+pren test")
    utsp_test=$(extract_num "$stats_file" "UTSP test post")

    # --- STO / EEV dal log SLURM (output_UTSP_*.txt); se piu' di uno, prendo il piu' recente ---
    out_dir="${bdir}output"
    sto_train=""; eev_train=""; sto_test=""; eev_test=""
    if [[ -d "$out_dir" ]]; then
        mapfile -t out_files < <(ls -t "$out_dir"/output_UTSP_*.txt 2>/dev/null)
        if [[ ${#out_files[@]} -gt 1 ]]; then
            echo "Attenzione: ${#out_files[@]} file output_UTSP_*.txt in $out_dir, uso il piu' recente: ${out_files[0]}" >&2
        fi
        if [[ ${#out_files[@]} -ge 1 ]]; then
            log_file="${out_files[0]}"
            sto_train=$(extract_num "$log_file" "STO train B")
            eev_train=$(extract_num "$log_file" "EEV train B")
            sto_test=$(extract_num "$log_file" "STO test")
            eev_test=$(extract_num "$log_file" "EEV test")
        else
            echo "Attenzione: nessun output_UTSP_*.txt in $out_dir, STO/EEV mancanti per $label" >&2
        fi
    else
        echo "Attenzione: manca $out_dir, STO/EEV mancanti per $label" >&2
    fi

    # --- varianza train pre/post booking (totale/within/between/quote) ---
    read_var_block() {
        local section="$1"
        local block
        block=$(awk -v sect="$section" '
            $0==sect {flag=1; next}
            flag && /^$/ {if (started) exit}
            flag {started=1; print}
        ' "$stats_file")
        echo "$block"
    }

    pre_block=$(read_var_block "VARIANZA TRAIN PRE-BOOKING")
    post_block=$(read_var_block "VARIANZA TRAIN POST-BOOKING")

    get_field() {
        local block="$1" label="$2"
        echo "$block" | grep -m1 -oE "${label}[[:space:]]*=[[:space:]]*-?[0-9.]+" \
            | grep -oE '\-?[0-9.]+$'
    }
    get_pct() {
        local block="$1" label="$2"
        echo "$block" | grep -m1 -oE "${label}[[:space:]]*=[[:space:]]*-?[0-9.]+%" \
            | grep -oE '\-?[0-9.]+'
    }

    var_tot_pre=$(get_field "$pre_block" "varianza totale")
    var_win_pre=$(get_field "$pre_block" "varianza within")
    var_btw_pre=$(get_field "$pre_block" "varianza between")
    pct_win_pre=$(get_pct "$pre_block" "quota within")
    pct_btw_pre=$(get_pct "$pre_block" "quota between")

    var_tot_post=$(get_field "$post_block" "varianza totale")
    var_win_post=$(get_field "$post_block" "varianza within")
    var_btw_post=$(get_field "$post_block" "varianza between")
    pct_win_post=$(get_pct "$post_block" "quota within")
    pct_btw_post=$(get_pct "$post_block" "quota between")

    # --- skewness / curtosi su UTSP, train e test, da post_total grezzi ---
    train_skk=$(compute_skew_kurt "$stats_file" "COSTI TRAIN SCENARIO PER SCENARIO")
    test_skk=$(compute_skew_kurt "$stats_file" "COSTI TEST SCENARIO PER SCENARIO")
    IFS=',' read -r train_n train_skew train_kurt <<< "$train_skk"
    IFS=',' read -r test_n test_skew test_kurt <<< "$test_skk"

    row="${bsize}"
    row="${row},${utsp_train},${sto_train},${eev_train},${pi_train},${pipren_train}"
    row="${row},${utsp_test},${sto_test},${eev_test},${pi_test},${pipren_test}"
    row="${row},${var_tot_pre},${var_win_pre},${var_btw_pre},${pct_win_pre},${pct_btw_pre}"
    row="${row},${var_tot_post},${var_win_post},${var_btw_post},${pct_win_post},${pct_btw_post}"
    row="${row},${train_n},${train_skew},${train_kurt}"
    row="${row},${test_n},${test_skew},${test_kurt}"
    echo "$row" >> "$OUT_CSV"
done

# Rimuovo righe vuote residue e ordino per batch_size
sed -i '/^[[:space:]]*$/d' "$OUT_CSV"
{
    head -n1 "$OUT_CSV"
    tail -n +2 "$OUT_CSV" | sort -t',' -k1,1n
} > "${OUT_CSV}.sorted" && mv "${OUT_CSV}.sorted" "$OUT_CSV"

echo "Fatto. Riepilogo salvato in: $OUT_CSV"
echo ""
echo "--- Anteprima ---"
if command -v column >/dev/null 2>&1; then
    column -s',' -t "$OUT_CSV" | cut -c1-220
else
    cat "$OUT_CSV"
fi
