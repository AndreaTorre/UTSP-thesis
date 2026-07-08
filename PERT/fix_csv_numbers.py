#!/usr/bin/env python3
"""
fix_csv_numbers.py

Corregge IN-PLACE un CSV che e' stato rotto da Excel: quando Excel riapre/risalva
un CSV con virgola decimale usando un locale italiano, trasforma il separatore
di campo in ';' e spacca ogni numero decimale mettendo un punto ogni 3 cifre su
TUTTA la stringa (es. "1863.190590" diventa "1.863.190.590"), perdendo la vera
posizione del punto decimale.

Le cifre non vengono mai perse: basta sapere quanti decimali aveva originariamente
ogni colonna per rimettere il punto al posto giusto. Quei numeri di decimali sono
noti dal formato con cui lo script di raccolta (collect_pert_results.sh) scrive
il CSV originale.

Se il file e' gia' nel formato corretto (separatore ',', punto decimale), lo
script non lo tocca.

Uso:
    python3 fix_csv_numbers.py pert_batch_sweep_summary.csv
"""

import csv
import sys
import shutil

# Decimali noti per ciascuna colonna del CSV originale.
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


def main():
    if len(sys.argv) < 2:
        print("Uso: python3 fix_csv_numbers.py <csv_path>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]

    with open(path, encoding="utf-8-sig") as f:
        first_line = f.readline()
    delimiter = ";" if first_line.count(";") > first_line.count(",") else ","

    if delimiter == ",":
        print("[info] Il file e' gia' nel formato corretto (separatore ','): nessuna modifica.", file=sys.stderr)
        sys.exit(0)

    print("[info] Rilevato formato rotto da Excel (';' + punti-migliaia): correggo in-place.", file=sys.stderr)

    backup_path = path + ".bak"
    shutil.copyfile(path, backup_path)
    print(f"[info] Backup salvato in: {backup_path}", file=sys.stderr)

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        fieldnames = reader.fieldnames
        rows = [r for r in reader if any(v.strip() for v in r.values() if v is not None)]

    fixed_rows = []
    for r in rows:
        fixed_rows.append({k: fix_excel_number(v, DECIMALS.get(k)) for k, v in r.items()})

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=",")
        writer.writeheader()
        writer.writerows(fixed_rows)

    print(f"[info] Fatto. {len(fixed_rows)} righe corrette e riscritte in: {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
