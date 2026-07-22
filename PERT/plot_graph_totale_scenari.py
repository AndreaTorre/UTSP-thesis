"""
Crea UN SOLO grafo pesato per ogni cartella BATCH_*.

Per ogni batch:
- legge tutti i file
  espB_UTSP_LS_test_i*_cost_distributions_train_stats.txt
  dentro .../BATCH_*/grafici/
- estrae tutti i "tour post"
- unisce tutti i tour di tutti i file del batch
- conta quante volte compare ogni arco orientato i -> j
- normalizza sul numero totale di tour/scenari letti nel batch
- salva una sola immagine per batch

Quindi:
UNA immagine per BATCH
NON una immagine per singolo file / singola istanza
"""

import ast
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

PROJECT_ROOT = "/home/atorre/UTSP/unione/git/UTSP"
COMMON_DIR = os.path.join(PROJECT_ROOT, "common")
PERT_DIR = os.path.join(PROJECT_ROOT, "PERT")

sys.path.insert(0, COMMON_DIR)

from tsp_utils import canon_edge
from evaluation import _draw_nodes
import config  # noqa: F401
from common import load_data


def load_project_data():
    """
    Gestisce sia load_data() che restituisce 4 valori
    sia load_data() che ne restituisce 5.
    """
    data = load_data()

    if len(data) == 5:
        nodes, coords, base_dist, E, root_node = data
    elif len(data) == 4:
        nodes, coords, E, root_node = data
        base_dist = None
    else:
        raise ValueError(
            f"load_data() restituisce {len(data)} valori: firma non attesa."
        )

    return nodes, coords, base_dist, E, root_node


def parse_tour_stats_file(path):
    """
    Estrae tutti i 'tour post' dal file.
    Ogni riga dati ha 7 campi separati da '|'
    e il campo 7 (indice 6) contiene il tour post.
    """
    tours = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split("|")

            if len(parts) != 7:
                continue

            try:
                int(parts[0].strip())
            except ValueError:
                continue

            try:
                tour = ast.literal_eval(parts[6].strip())
            except Exception:
                continue

            if not isinstance(tour, (list, tuple)):
                continue

            tour = list(tour)

            # Se il tour è già chiuso tipo [a, ..., a], tolgo l'ultimo nodo
            # per evitare di chiudere due volte il ciclo.
            if len(tour) >= 2 and tour[0] == tour[-1]:
                tour = tour[:-1]

            if len(tour) >= 2:
                tours.append(tour)

    return tours


def build_edge_counts(tours, nodes):
    """
    Conta quante volte ogni arco orientato compare
    nell'insieme totale dei tour.
    """
    idx = {node: k for k, node in enumerate(nodes)}
    counts = np.zeros((len(nodes), len(nodes)), dtype=float)

    valid_tours = 0

    for tour in tours:
        # ignora eventuali tour con nodi non presenti
        if any(node not in idx for node in tour):
            continue

        closed_tour = tour[1:] + tour[:1]

        for a, b in zip(tour, closed_tour):
            counts[idx[a], idx[b]] += 1.0

        valid_tours += 1

    return counts, valid_tours


def plot_utsp_graph_total_scenari(
    nodes,
    coords,
    edge_counts,
    n_scenari,
    save_path,
    threshold=0.01,
    title_suffix="",
    I=None,
):
    """
    Disegna il grafo pesato:
    peso(i,j) = frazione di scenari/tour in cui compare l'arco i->j
    """
    if n_scenari <= 0:
        print(f"Nessuno scenario valido per {save_path}, salto.")
        return

    idx = {v: k for k, v in enumerate(nodes)}
    I_set = {canon_edge(i, j) for i, j in (I or [])}
    freq = edge_counts / float(n_scenari)

    fig, ax = plt.subplots(figsize=(10, 9))
    _draw_nodes(ax, nodes, coords)

    lw_min, lw_max = 0.4, 6.0
    alpha_min, alpha_max = 0.12, 0.88
    ms_min, ms_max = 6, 18

    drawn_edges = 0

    for i in nodes:
        for j in nodes:
            if i == j:
                continue

            w = float(freq[idx[i], idx[j]])

            if w < threshold:
                continue

            xi, yi = coords[i]
            xj, yj = coords[j]

            lw = lw_min + (lw_max - lw_min) * w
            alpha = alpha_min + (alpha_max - alpha_min) * w
            ms = ms_min + (ms_max - ms_min) * w

            if canon_edge(i, j) in I_set:
                color = "crimson"
            else:
                color = (0.05, 0.25 + 0.45 * (1 - w), 0.85, alpha)

            ax.annotate(
                "",
                xy=(xj, yj),
                xytext=(xi, yi),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=lw,
                    mutation_scale=ms,
                    alpha=alpha,
                    shrinkA=8,
                    shrinkB=8,
                ),
                zorder=2,
            )
            drawn_edges += 1

    sm = ScalarMappable(cmap="Blues", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label("frazione di scenari che usano l'arco i→j")

    title = f"Grafo pesato su totale scenari del batch (n={n_scenari})"
    if title_suffix:
        title += f"\n{title_suffix}"

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"  Salvato: {save_path}  | archi disegnati: {drawn_edges}")


def extract_pen_and_batch_from_path(batch_dir):
    """
    Esempio input:
    /.../variants/pen0_mean/batch_sweep/BATCH_55
    """
    m = re.search(r"variants[/\\]([^/\\]+)[/\\]batch_sweep[/\\](BATCH_\d+)", batch_dir)
    if m:
        return m.group(1), m.group(2)
    return "unknown_pen", os.path.basename(batch_dir)


def process_single_batch(batch_dir, nodes, coords, I=None, threshold=0.01):
    """
    Legge TUTTI i file txt nel batch e costruisce UN solo grafo aggregato.
    """
    grafici_dir = os.path.join(batch_dir, "grafici")

    pattern = os.path.join(
        grafici_dir,
        "espB_UTSP_LS_test_i*_cost_distributions_train_stats.txt"
    )

    files = sorted(glob.glob(pattern))

    pen_name, batch_name = extract_pen_and_batch_from_path(batch_dir)

    print("\n" + "=" * 80)
    print(f"Batch: {batch_dir}")
    print(f"Pen:   {pen_name}")
    print(f"Nome:  {batch_name}")
    print(f"File trovati: {len(files)}")

    if not files:
        print("  Nessun file txt trovato, salto.")
        return

    all_tours = []
    n_files_with_data = 0

    for path in files:
        tours = parse_tour_stats_file(path)

        if tours:
            all_tours.extend(tours)
            n_files_with_data += 1

    print(f"File con almeno un tour valido: {n_files_with_data}")
    print(f"Numero totale di tour/scenari aggregati nel batch: {len(all_tours)}")

    if not all_tours:
        print("  Nessun tour valido nel batch, salto.")
        return

    edge_counts, n_valid_tours = build_edge_counts(all_tours, nodes)

    print(f"Tour validi effettivamente usati nel conteggio: {n_valid_tours}")

    if n_valid_tours == 0:
        print("  Tutti i tour sono risultati invalidi, salto.")
        return

    save_path = os.path.join(
        grafici_dir,
        f"{pen_name}_{batch_name}_graph_total_scenari.png"
    )

    title_suffix = f"{pen_name} — {batch_name} — file aggregati: {len(files)}"

    plot_utsp_graph_total_scenari(
        nodes=nodes,
        coords=coords,
        edge_counts=edge_counts,
        n_scenari=n_valid_tours,
        save_path=save_path,
        threshold=threshold,
        title_suffix=title_suffix,
        I=I,
    )


def generate_one_graph_per_batch(results_root, nodes, coords, I=None, threshold=0.01):
    """
    Scansiona tutte le cartelle:
    RISULTATI_15/variants/pen*_/batch_sweep/BATCH_*

    e per ciascuna crea UNA sola immagine aggregata.
    """
    batch_pattern = os.path.join(
        results_root,
        "variants",
        "pen*_*",
        "batch_sweep",
        "BATCH_*",
    )

    batch_dirs = sorted(glob.glob(batch_pattern))

    print(f"Root risultati: {results_root}")
    print(f"Batch trovati: {len(batch_dirs)}")

    if not batch_dirs:
        print("Nessuna cartella BATCH_* trovata.")
        return

    for batch_dir in batch_dirs:
        process_single_batch(
            batch_dir=batch_dir,
            nodes=nodes,
            coords=coords,
            I=I,
            threshold=threshold,
        )


if __name__ == "__main__":
    nodes, coords, base_dist, E, root_node = load_project_data()

    results_root = os.path.join(PERT_DIR, "RISULTATI_15")

    generate_one_graph_per_batch(
        results_root=results_root,
        nodes=nodes,
        coords=coords,
        I=None,
        threshold=0.01,
    )