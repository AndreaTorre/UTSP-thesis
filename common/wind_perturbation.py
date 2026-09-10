# -*- coding: utf-8 -*-
"""
Perturbazioni basate su campo vettoriale di vento ERA5.

Formula additiva (integrale di linea discretizzato, midpoint rule):

    Delta_ij = -alpha * sum_{k=1}^{m} (L_ij / m) * [w(s_k) · d_hat_ij]

    c_ij     = max(L_ij + Delta_ij, eps)

Dipendenze: numpy, h5py  (niente netCDF4, niente scipy)
"""

import math
import numpy as np
import h5py

# NOTA: le costanti drone esistono solo in CVETT/config_backend.py. Sul ramo
# PERT questo modulo viene importato da scenarios.py ma non usato: import
# protetto per non far esplodere la pipeline PERT.
try:
    from config import DRONE_U, DRONE_A0, DRONE_A2, DRONE_A3
except ImportError:
    DRONE_U = DRONE_A0 = DRONE_A2 = DRONE_A3 = None

# ---------------------------------------------------------------------------
# CARICAMENTO
# ---------------------------------------------------------------------------

def load_wind_field(nc_path: str) -> dict:
    """
    Legge il file ERA5 (NetCDF 4 / HDF5) e restituisce un dizionario con
    tutti i dati in memoria.  Chiamare una volta sola all'avvio, poi passare
    l'oggetto alle funzioni.
    """
    with h5py.File(nc_path, "r") as ds:
        u100 = ds["u100"][:].astype(float)   # (T, lat, lon)
        v100 = ds["v100"][:].astype(float)
        lats = ds["latitude"][:].astype(float)
        lons = ds["longitude"][:].astype(float)
        n_times = u100.shape[0]

    return {
        "u100":    u100,
        "v100":    v100,
        "lats":    lats,
        "lons":    lons,
        "n_times": n_times,
    }


# ---------------------------------------------------------------------------
# MAPPATURA COORDINATE PIXEL → GRIGLIA ERA5
# ---------------------------------------------------------------------------

def _make_geo_coords(nodes, coords, n_lat, n_lon):
    """
    Mappa coordinate pixel (x, y) in indici frazionari sulla griglia ERA5.
    Restituisce dict: node_id -> (lat_frac, lon_frac).

    Convenzione:
      x cresce verso destra  → longitudine cresce verso est   (stessa direzione)
      y cresce verso il basso → latitudine cresce verso nord  (direzione invertita)
    """
    xs = [coords[i][0] for i in nodes]
    ys = [coords[i][1] for i in nodes]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    geo = {}
    for i in nodes:
        x, y = coords[i]
        lon_frac = (x - x_min) / (x_max - x_min + 1e-12) * (n_lon - 1)
        lat_frac = (1.0 - (y - y_min) / (y_max - y_min + 1e-12)) * (n_lat - 1)
        geo[i] = (lat_frac, lon_frac)
    return geo


def _bilinear(field_2d, lat_frac, lon_frac):
    """Interpolazione bilineare su griglia 2D (lat, lon)."""
    n_lat, n_lon = field_2d.shape
    r0 = max(0, min(int(math.floor(lat_frac)), n_lat - 2))
    c0 = max(0, min(int(math.floor(lon_frac)), n_lon - 2))
    dr = lat_frac - r0
    dc = lon_frac - c0
    return (field_2d[r0,     c0    ] * (1 - dr) * (1 - dc) +
            field_2d[r0 + 1, c0    ] *      dr  * (1 - dc) +
            field_2d[r0,     c0 + 1] * (1 - dr) *      dc  +
            field_2d[r0 + 1, c0 + 1] *      dr  *      dc)


# ---------------------------------------------------------------------------
# PERTURBAZIONE COMPLETA PER UNO SCENARIO  (formula additiva)
# ---------------------------------------------------------------------------



def _power(v):
    """P(v) = a0 + a2*v^2 + a3*v^3, modello di potenza propulsiva."""
    return DRONE_A0 + DRONE_A2 * v**2 + DRONE_A3 * v**3


def build_wind_perturbation(
    scenario_id,
    nodes,
    base_dist,
    coords,
    wind,
    n_samples=None,
    eps=1e-6,
    samples_per_cell=2,
    n_samples_min=5,
):
    """
    Costo wind-adjusted basato sul modello energetico (paper droni_vento.pdf):

        c_ij = d_ij * (1/K) * sum_k [ P(v_a,k) / P(u) ]

    dove v_a,k è la velocità relativa all'aria nel k-esimo punto campionato
    lungo l'arco i->j, ottenuta scomponendo il vento in componente parallela
    e perpendicolare alla direzione dell'arco.

    Sostituisce integralmente la vecchia regola a soglia (nessun costo V,
    nessuna turbolenza locale): il modello è continuo e moltiplicativo.

    Restituisce
    -----------
    dict (i, j) -> delta, drop-in compatibile con
        c_ij = max(base_dist[i][j] + pert[(i, j)], eps)
    """
    # Gli scenario_id sono 1-based e locali al file vento passato.
    # Quindi:
    # scenario_id = 1 -> primo istante del file
    # scenario_id = 2 -> secondo istante del file
    # ecc.
    t_idx = int(scenario_id) - 1
    
    if t_idx < 0 or t_idx >= wind["n_times"]:
        raise ValueError(
            f"scenario_id={scenario_id} fuori range per il file vento: "
            f"n_times={wind['n_times']}."
        )
    n_lat = len(wind["lats"])
    n_lon = len(wind["lons"])
    u_field = wind["u100"][t_idx]
    v_field = wind["v100"][t_idx]
    geo = _make_geo_coords(nodes, coords, n_lat, n_lon)

    P_u = _power(DRONE_U)
    _fixed_fracs = None
    if n_samples is not None:
        _fixed_fracs = [(k + 0.5) / n_samples for k in range(n_samples)]

    pert = {}
    for i in nodes:
        xi, yi = coords[i]
        lat_i, lon_i = geo[i]
        for j in nodes:
            if i == j:
                continue
            xj, yj = coords[j]
            L_ij = float(base_dist[i][j])
            if L_ij < eps:
                pert[(i, j)] = 0.0
                continue

            dx, dy = xj - xi, yj - yi
            length = math.hypot(dx, dy)
            tau_x, tau_y = dx / length, -dy / length  # direzione arco: est, nord

            lat_j, lon_j = geo[j]
            if _fixed_fracs is not None:
                fracs = _fixed_fracs
            else:
                # celle di griglia ERA5 attraversate dall'arco
                n_cells = math.hypot(lat_j - lat_i, lon_j - lon_i)
                m = max(n_samples_min, int(math.ceil(samples_per_cell * n_cells)))
                fracs = [(k + 0.5) / m for k in range(m)]
            P_sum = 0.0
            for frac in fracs:
                lat_s = lat_i + frac * (lat_j - lat_i)
                lon_s = lon_i + frac * (lon_j - lon_i)
                u_w = _bilinear(u_field, lat_s, lon_s)
                v_w = _bilinear(v_field, lat_s, lon_s)

                w_par = u_w * tau_x + v_w * tau_y
                w_perp_sq = u_w**2 + v_w**2 - w_par**2
                # NOTA: max(...,0.0) per errori di arrotondamento su w_perp_sq
                v_a = math.sqrt((DRONE_U - w_par) ** 2 + max(w_perp_sq, 0.0))
                P_sum += _power(v_a)

            m_ij = (P_sum / len(fracs)) / P_u
            pert[(i, j)] = L_ij * (m_ij - 1.0)  # delta = c_ij - d_ij

    return pert

# ---------------------------------------------------------------------------
# DIAGNOSTICHE
# ---------------------------------------------------------------------------

def perturbation_diagnostics(nodes, base_dist, all_pert, print_output=True):
    """Statistiche sulle perturbazioni relative Delta_ij / L_ij.

    Parametri
    ---------
    all_pert : dict[omega -> dict[(i,j) -> delta]]

    Restituisce dict con 'per_scenario' e 'global'.
    """
    all_rel = []
    per_scenario = []

    for omega, pert in all_pert.items():
        rel_deltas = []
        n_negative = 0
        asym_vals = []

        for i in nodes:
            for j in nodes:
                if i == j:
                    continue
                L = float(base_dist[i][j])
                if L < 1e-12:
                    continue
                d = pert[(i, j)]
                rel_deltas.append(d / L)
                all_rel.append(d / L)

                if L + d <= 0:
                    n_negative += 1

                d_rev = pert[(j, i)]
                asym_vals.append(abs(d + d_rev) / L)

        rd = np.array(rel_deltas)
        sc = {
            "omega": omega,
            "mean": rd.mean(), "std": rd.std(),
            "min": rd.min(), "max": rd.max(),
            "p1": np.percentile(rd, 1), "p5": np.percentile(rd, 5),
            "p50": np.percentile(rd, 50),
            "p95": np.percentile(rd, 95), "p99": np.percentile(rd, 99),
            "n_negative_dist": n_negative,
            "mean_asym": np.mean(asym_vals),
        }
        per_scenario.append(sc)

        if print_output:
            print(f"--- Scenario {omega} ---")
            print(f"  rel delta:  mean={sc['mean']:+.4f}  std={sc['std']:.4f}")
            print(f"  range:      [{sc['min']:+.4f}, {sc['max']:+.4f}]")
            print(f"  percentili: 1%={sc['p1']:+.4f}  5%={sc['p5']:+.4f}"
                  f"  50%={sc['p50']:+.4f}  95%={sc['p95']:+.4f}"
                  f"  99%={sc['p99']:+.4f}")
            print(f"  archi c<=0: {sc['n_negative_dist']}")
            print(f"  asimmetria: {sc['mean_asym']:.6f}")

    ar = np.array(all_rel)
    glob = {
        "mean": ar.mean(), "std": ar.std(),
        "min": ar.min(), "max": ar.max(),
        "p1": np.percentile(ar, 1), "p99": np.percentile(ar, 99),
    }
    if print_output:
        print("=== GLOBALE ===")
        print(f"  mean={glob['mean']:+.4f}  std={glob['std']:.4f}"
              f"  range=[{glob['min']:+.4f}, {glob['max']:+.4f}]"
              f"  1%={glob['p1']:+.4f}  99%={glob['p99']:+.4f}")

    return {"per_scenario": per_scenario, "global": glob}