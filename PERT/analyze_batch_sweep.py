#!/usr/bin/env python3
"""
analyze_batch_sweep.py

Legge il CSV prodotto da collect_pert_results.sh (anche se nel frattempo e'
stato aperto/risalvato con Excel e i numeri sono stati spaccati con punti
spuri tipo "1.862.830.171"), lo corregge automaticamente se serve, e per
ogni esperimento (gruppo di righe consecutive, tipicamente una per N_NODES)
stampa la quota di varianza between (pre e post booking) per batch_size
crescente, con il delta rispetto al batch precedente, per individuare il
plateau ("ginocchio della curva"): il punto dove aumentare ulteriormente
il batch_size porta un guadagno sempre piu' piccolo.

NOTA: gli esperimenti vengono separati automaticamente individuando i punti
in cui il batch_size NON cresce rispetto alla riga precedente (es. torna a
20 dopo essere arrivato a 50) o le righe vuote nel CSV originale. Funziona
con qualunque insieme di batch_size, anche non multipli di 10 (es. 55, 65).

Uso:
    python3 analyze_batch_sweep.py pert_batch_sweep_summary.csv
"""

import csv
import sys


# Decimali noti per ciascuna colonna del CSV originale (vedi collect_pert_results.sh).
# None = valore intero o non numerico (es. "nan").
DECIMALS = {
    "batch_size": 0,
    "UTSP_train": 6, "STO_train": 4, "EEV_train": 4, "PI_train": None, "PIpren_train": None,
    "UTSP_test": 6, "STO_test": 4, "EEV_test": 4, "PI_test": None, "PIpren_test": None,
    "train_var_total_pre": 6, "train_var_within_pre": 6, "train_var_between_pre": 6,
    "train_pct_within_pre": 4, "train_pct_between_pre": 4,
    "train_var_total_post": 6, "train_var_within_post": 6, "train_var_between_post": 6,
    "train_pct_within_post": 4, "train_pct_between_post": 4,
    "UTSP_train_n": 0, "UTSP_train_skew": 6, "UTSP_train_kurt": 6,
    "UTSP_test_n": 0, "UTSP_test_skew": 6, "UTSP_test_kurt": 6,
}


def fix_excel_number(raw, dec):
    """Ricostruisce un numero che Excel ha spaccato con punti-migliaia spuri
    su tutta la stringa (es. "1.862.830.171" -> "1862.830171" se dec=6)."""
    raw = raw.strip()
    if raw == "" or raw.lower() == "nan":
        return raw
    digits = raw.replace(".", "").replace(",", "")
    neg = digits.startswith("-")
    if neg:
        digits = digits[1:]
    if dec is None or dec == 0:
        return ("-" if neg else "") + digits
    intpart = digits[:-dec] if len(digits) > dec else "0"
    decpart = digits[-dec:].rjust(dec, "0")
    return ("-" if neg else "") + f"{intpart}.{decpart}"


def load_rows(path):
    """Rileva se il CSV e' nel formato originale (virgola, punto decimale)
    o rotto da Excel (punto-e-virgola, punti-migliaia spuri), e lo carica
    correttamente in entrambi i casi."""
    with open(path, encoding="utf-8-sig") as f:
        sample = f.readline()

    delimiter = ";" if sample.count(";") > sample.count(",") else ","

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        raw_rows = [r for r in reader if any(v.strip() for v in r.values() if v is not None)]

    if delimiter == ";":
        print("[info] Rilevato CSV nel formato rotto da Excel (';' + punti-migliaia): correggo i numeri.", file=sys.stderr)
        fixed_rows = []
        for r in raw_rows:
            fixed_rows.append({k: fix_excel_number(v, DECIMALS.get(k)) for k, v in r.items()})
        return fixed_rows

    return raw_rows


def to_float(s):
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def split_into_experiments(rows):
    """Separa le righe in esperimenti distinti: un nuovo esperimento comincia
    quando il batch_size non e' strettamente crescente rispetto alla riga
    precedente (tipicamente torna al valore piu' piccolo, es. 50 -> 20)."""
    experiments = []
    current = []
    prev_b = None
    for r in rows:
        b = to_float(r["batch_size"])
        if b is None:
            continue
        if prev_b is not None and b <= prev_b:
            if current:
                experiments.append(current)
            current = []
        current.append(r)
        prev_b = b
    if current:
        experiments.append(current)
    return experiments


def analyze_experiment(rows, idx):
    rows = sorted(rows, key=lambda r: to_float(r["batch_size"]))
    print(f"\n=== Esperimento {idx} ({len(rows)} batch: {', '.join(r['batch_size'] for r in rows)}) ===")

    for label, col in [("PRE-booking", "train_pct_between_pre"), ("POST-booking", "train_pct_between_post")]:
        print(f"\n  Quota varianza between, {label}:")
        prev_pct = None
        prev_b = None
        deltas = []
        for r in rows:
            b = r["batch_size"]
            pct = to_float(r[col])
            if pct is None:
                print(f"    BATCH_{b:>3}: dato mancante")
                continue
            if prev_pct is None:
                print(f"    BATCH_{b:>3}: {pct:6.3f}%")
            else:
                delta = pct - prev_pct
                deltas.append(delta)
                print(f"    BATCH_{b:>3}: {pct:6.3f}%   (delta da BATCH_{prev_b}: {delta:+.3f} pp)")
            prev_pct, prev_b = pct, b

        # individua il primo punto dove il miglioramento si "appiattisce" davvero:
        # il delta resta negativo (la varianza between continua a scendere) ma la
        # sua magnitudine scende sotto il 25% del primo delta osservato.
        # Un delta positivo (la between RISALE) non e' un plateau: e' rumore o un
        # vero peggioramento, e viene segnalato a parte senza essere scambiato per "stabilizzazione".
        if len(deltas) >= 2:
            first_delta = deltas[0]  # tipicamente negativo (la between scende all'inizio)
            plateau_found = False
            if first_delta < 0:
                for i, d in enumerate(deltas[1:], start=1):
                    if d > 0:
                        b_noisy = rows[i + 1]["batch_size"]
                        b_prev = rows[i]["batch_size"]
                        print(f"    -> Attenzione: da BATCH_{b_prev} a BATCH_{b_noisy} la quota between RISALE "
                              f"({d:+.3f} pp). Possibile rumore statistico (pochi mini-batch / campionamento "
                              f"scenari), non un plateau: non interrotto il confronto, continuo a guardare i batch successivi.")
                        continue
                    if abs(d) < 0.25 * abs(first_delta):
                        plateau_batch = rows[i + 1]["batch_size"]
                        prev_batch = rows[i]["batch_size"]
                        print(f"    -> Possibile plateau: il guadagno si appiattisce passando da "
                              f"BATCH_{prev_batch} a BATCH_{plateau_batch} (delta sceso sotto il 25% del primo delta, "
                              f"ma ancora in miglioramento).")
                        print(f"       Secondo il criterio 'batch piu' piccolo quando il miglioramento si appiattisce', "
                              f"BATCH_{prev_batch} potrebbe essere la scelta preferibile a batch piu' grandi.")
                        plateau_found = True
                        break
            if not plateau_found:
                print("    -> Nessun plateau chiaro nei batch testati: il guadagno e' ancora sostanziale "
                      "(o il segnale e' troppo rumoroso) fino all'ultimo batch disponibile. "
                      "Considerare di testare batch ancora piu' grandi.")

    # riepilogo costo test, per contesto
    print(f"\n  Costo UTSP_test per batch (per contesto, non solo varianza):")
    for r in rows:
        cost = to_float(r["UTSP_test"])
        cost_str = f"{cost:.2f}" if cost is not None else "n/a"
        print(f"    BATCH_{r['batch_size']:>3}: {cost_str}")


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 analyze_batch_sweep.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    rows = load_rows(path)
    if not rows:
        print("Errore: nessuna riga valida trovata nel CSV.", file=sys.stderr)
        sys.exit(1)

    experiments = split_into_experiments(rows)
    for i, exp_rows in enumerate(experiments, start=1):
        analyze_experiment(exp_rows, i)


if __name__ == "__main__":
    main()
