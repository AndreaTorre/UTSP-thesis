#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plots.py — funzioni di DISEGNO pure per l'intero progetto UTSP.

Regola di questo modulo: SOLO matplotlib/numpy (+ tsp_utils, leggero). Nessun
import di torch, gurobi, local_search o della pipeline. Ogni funzione riceve
dati GIÀ PRONTI (matrici, coordinate, liste) e produce una figura. I dati si
preparano altrove (vedi regen_graphs.py), così un grafico si rigenera sul login
node in un secondo, senza far partire l'ambiente pesante.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

try:
    from tsp_utils import canon_edge
except Exception:
    def canon_edge(i, j):
        return (i, j) if i <= j else (j, i)


def _draw_nodes(ax, nodes, coords, node_size=90):
    for v in nodes:
        x, y = coords[v]
        ax.scatter([x], [y], s=node_size, c="black", zorder=3)
        ax.annotate(str(v), (x, y), fontsize=8, xytext=(3, 3),
                    textcoords="offset points", zorder=4)


def grafo_pesato_frequenza(nodes, coords, edge_freq, save_path,
                           I=None, threshold=0.01, title=None):
    """
    Grafo pesato dove lo spessore/opacità di i->j = FREQUENZA d'uso reale
    (frazione di scenari in cui il tour usa quell'arco). edge_freq è già
    normalizzata in [0,1]: peso = n_volte_arco / n_scenari.

    Archi in I (prenotabili) in crimson, gli altri in blu graduato.
    """
    idx = {v: k for k, v in enumerate(nodes)}
    I_set = {canon_edge(i, j) for i, j in (I or [])}

    fig, ax = plt.subplots(figsize=(10, 9))
    _draw_nodes(ax, nodes, coords)

    lw_min, lw_max = 0.4, 6.0
    a_min, a_max = 0.12, 0.88
    ms_min, ms_max = 6, 18
    drawn = 0

    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            w = float(edge_freq[idx[i], idx[j]])
            if w < threshold:
                continue
            xi, yi = coords[i]
            xj, yj = coords[j]
            lw = lw_min + (lw_max - lw_min) * w
            alpha = a_min + (a_max - a_min) * w
            ms = ms_min + (ms_max - ms_min) * w
            color = "crimson" if canon_edge(i, j) in I_set else (0.05, 0.25 + 0.45 * (1 - w), 0.85, alpha)
            ax.annotate("", xy=(xj, yj), xytext=(xi, yi),
                        arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                        mutation_scale=ms, alpha=alpha,
                                        shrinkA=8, shrinkB=8),
                        zorder=2)
            drawn += 1

    sm = ScalarMappable(cmap="Blues", norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7)
    cbar.set_label("frazione di scenari che usano l'arco i\u2192j")

    ax.set_title(title or "Grafo pesato UTSP (frequenza d'uso reale)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {save_path}  (archi disegnati: {drawn})")
    return drawn
