#!/usr/bin/env python3
"""
Audit degli import "interni" di una cartella esperimento (PERT o CVETT)
rispetto a common/: per ogni file .py in cima alla cartella, mostra da
dove verrebbe risolto ogni import — copia locale, common/, oppure
ambiguo (esiste in entrambi i posti).

Uso:
    python audit_imports.py <cartella_esperimento> <cartella_common>

Esempio:
    python audit_imports.py /home/atorre/UTSP/unione/git/UTSP/PERT \
                             /home/atorre/UTSP/unione/git/UTSP/common

    python audit_imports.py /home/atorre/UTSP/unione/git/UTSP/CVETT \
                             /home/atorre/UTSP/unione/git/UTSP/common

Non serve alcuna libreria esterna: solo ast/pathlib della stdlib.
"""
import ast
import sys
from pathlib import Path


def top_level_py_files(folder: Path):
    return sorted(folder.glob("*.py"))


def get_imports(path: Path):
    """Ritorna [(nome_modulo, riga), ...] e le righe con sys.path.append/insert."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    modules, path_hacks = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append((alias.name.split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            if node.module:  # ignora `from . import x` (relativo, module è None)
                modules.append((node.module.split(".")[0], node.lineno))
        elif isinstance(node, ast.Attribute) and node.attr in ("append", "insert"):
            if isinstance(node.value, ast.Attribute) and node.value.attr == "path":
                path_hacks.append(node.lineno)
    return modules, path_hacks


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)

    exp_dir = Path(sys.argv[1]).resolve()
    common_dir = Path(sys.argv[2]).resolve()

    local_names = {p.stem for p in top_level_py_files(exp_dir)}
    common_names = {p.stem for p in top_level_py_files(common_dir)}
    other_exp_names = {"PERT", "CVETT"} - {exp_dir.name}

    print(f"# Audit import — {exp_dir.name}  (common = {common_dir})\n")

    ambiguous_summary = {}

    for f in top_level_py_files(exp_dir):
        modules, path_hacks = get_imports(f)
        rows = []
        for name, lineno in modules:
            if name == "config_backend":
                tag = ("BYPASS: import diretto di config_backend, salta l'iniezione "
                       "di common/config.py (N_NODES, WIND_NC_PATH_*). Deve importare `config`.")
            elif name in other_exp_names:
                tag = f"RIFERIMENTO DIRETTO ALLA CARTELLA {name} (quasi sempre un errore)"
            elif name in local_names and name in common_names:
                tag = "AMBIGUO: esiste sia qui sia in common/ (vince questa cartella se lanciato da qui)"
                ambiguous_summary.setdefault(name, []).append(f.name)
            elif name in local_names:
                tag = "locale (copia propria)"
            elif name in common_names:
                tag = "common/ (nessuna copia locale)"
            else:
                continue  # libreria esterna o stdlib, non ci interessa
            rows.append((lineno, name, tag))

        if rows or path_hacks:
            print(f"## {f.name}")
            for lineno, name, tag in sorted(rows):
                print(f"  L{lineno:<4} import {name:<22} -> {tag}")
            for lineno in path_hacks:
                print(f"  L{lineno:<4} sys.path.append/insert -- verificare manualmente")
            print()

    if ambiguous_summary:
        print("## Riepilogo moduli ambigui (esistono sia locali che in common/)")
        for name, files in ambiguous_summary.items():
            print(f"  - {name}: usato in {', '.join(files)}")
        print(
            "\n  Se il file e' nato come wrapper (config.py, experiment_B.py, "
            "scenarios.py) va bene cosi'. Se invece e' una copia dimenticata "
            "(evaluation.py, utsp.py, two_stage_utsp_loss.py...) va deciso quale "
            "delle due copie e' quella vera e va tolta l'altra."
        )


if __name__ == "__main__":
    main()
