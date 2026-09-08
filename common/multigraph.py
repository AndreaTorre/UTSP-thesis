# -*- coding: utf-8 -*-
"""
multigraph.py — training UTSP su MOLTI grafi (stesso N), integrato con la pipeline.

Perché
------
Oggi rete allenata su UN grafo perturbato: vede una sola geometria e la overfitta.
Qui ogni *istanza* è un grafo diverso (stesso N) perturbato K volte, così la rete
generalizza sulle geometrie. La valutazione NON è un secondo binario: riusa il ramo
`TESI_REUSE_UTSP_TRAIN=1` di `run_esperimento_B_UTSP` su un grafo held-out.

Teoria (le scelte non sono cosmetiche)
--------------------------------------
1. Un forward = un grafo. La media cross-scenario (`_scenario_diffusion_media`, dim=0)
   è il consenso d'istanza: mischiare scenari di grafi diversi la rovinerebbe.
2. Normalizzazione PER GRAFO. τ è legata alla mediana delle distanze DELL'ISTANZA e
   `dist_scale` alla media delle distanze positive: si ricalcolano per grafo, al
   training e — fix di correttezza — anche al test (vedi patch a utsp.py). Usare la
   τ/scale del training su un grafo nuovo mis-scala il kernel.
3. I/p/C sono funzioni deterministiche della geometria (I dal k-medoids del grafo,
   p=PRENOTAZIONE_FRAC·b, C=PENALTY_FRAC·b). La rete non riceve I in input, ma poiché
   I dipende solo dalla geometria (che la rete vede), può imparare una mappatura
   geometria→prenotazioni che TRANSFERISCE. Vale finché i rapporti p/C restano fissi
   tra i grafi: è la frontiera di validità.
4. Disgiunzione train/test a livello di GRAFO: il pool di training NON deve contenere
   la source instance del grafo di test, altrimenti la generalizzazione non è testata.
5. Distribuzione: se il test è un sottocampione TSPLIB, allena su sottocampioni dello
   stesso tipo (mode='subsample' con un pool di istanze). 'uniform' è solo ablazione.
6. Compromesso n_istanze ↔ K: più grafi = più diversità geometrica; K più grande =
   consenso per-grafo meno rumoroso. A budget fisso è un trade-off da esplorare.

Integrazione (niente pipeline parallela)
----------------------------------------
- Il loop `_optimize_units` riproduce ESATTAMENTE il corpo di `_train_utsp_2stage`
  (stesse loss/step/clip/best-state), solo iterando su unità = grafi invece che su
  slice di un grafo. NOTA: l'ho tenuto qui, fuori da `_train_utsp_2stage`, per non
  toccare codice già validato; se vuoi azzerare la quasi-duplicazione, il loop
  single-graph può chiamare questa stessa funzione (edit opzionale nel messaggio).
- Il checkpoint è nel formato/percorso dei tuoi artefatti (`_utsp_train_paths`), con
  `metadata["multigraph"]=True`. La patch a utsp.py fa accettare questo checkpoint al
  loader e usare la normalizzazione del grafo di test.

Uso
---
    # 1) allena su molti grafi (pool = istanze TSPLIB TRANNE quella di test)
    TESI_EXPERIMENT=PERT TESI_N_NODES=40 python common/multigraph.py \
        TESI_MG_INSTANCES=300 TESI_MG_K=20 TESI_MG_MODE=subsample \
        TESI_MG_BIG=/path/pool_dir   TESI_MG_EXP=espB_UTSP_LS
    # 2) valuta sul grafo held-out con la pipeline standard:
    TESI_REUSE_UTSP_TRAIN=1  (stesso exp_name)  → run_esperimento_B_UTSP
oppure da codice:  run_multigraph(n_instances=300, K=20, mode="subsample",
                                  big_paths="/path/pool_dir", exp_name="espB_UTSP_LS")
"""
import os
import glob
import time
import json

import numpy as np
import torch
import torch.optim as optim

from config import (
    N_NODES, UTSP_TRAINING_SEED, K_MEDOID_NODES,
    PRENOTAZIONE_FRAC, PENALTY_FRAC,
    N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, OUTPUT_DIR,
    UTSP2_HIDDEN, UTSP2_NLAYERS, UTSP2_EPOCHS, UTSP2_LR, UTSP2_STEP_LR,
    UTSP2_LOG_FREQ, UTSP2_LAMBDA1, UTSP2_LAMBDA2, UTSP2_LAMBDA_D, UTSP2_LAMBDA_E,
    UTSP2_ALPHA_LOSS, UTSP2_LAMBDA_B_DIV, UTSP2_INCLUDE_PENALTY, UTSP2_INCLUDE_ENTROPY,
)
from tsp_utils import base_cost_undirected
from gurobi_models import build_I_from_medoid_outgoing_nodes
from scenarios import generate_scenario_batches
from two_stage_utsp_loss import build_I_tensors
# Riuso: rete, tensori d'ingresso, serializzazione, path artefatti. Niente duplicato.
from utsp import UTSP_GNN, _build_input_tensors, _json_safe, _utsp_train_paths, _optimize_units


# ─────────────────────────────────────────────────────────────────────
# k-medoids (PAM greedy, deterministico). Definisce I su ogni grafo nuovo.
# ─────────────────────────────────────────────────────────────────────
def k_medoids(dist, k, seed, max_iter=100):
    """Indici (0..M-1) dei k medoidi sulla matrice `dist` (M×M simmetrica).
    NOTA: sostituisce `K_MEDOID_NODES` (hardcoded sul grafo unico). M=N ⇒ costo nullo."""
    dist = np.asarray(dist, dtype=float)
    M = dist.shape[0]
    if not (0 < k <= M):
        raise ValueError(f"k-medoids: k={k} non valido per M={M}.")
    if k == M:
        return list(range(M))
    rng = np.random.default_rng(seed)
    medoids = np.sort(rng.choice(M, size=k, replace=False))
    tc = lambda meds: float(dist[:, meds].min(axis=1).sum())
    best = tc(medoids)
    for _ in range(max_iter):
        improved = False
        med_set = set(int(m) for m in medoids)
        non_medoids = [o for o in range(M) if o not in med_set]
        for mi in range(k):
            for o in non_medoids:
                cand = medoids.copy(); cand[mi] = o
                c = tc(cand)
                if c < best - 1e-12:
                    medoids, best, improved = np.sort(cand), c, True
                    break
            if improved:
                break
        if not improved:
            break
    return sorted(int(m) for m in medoids)


# ─────────────────────────────────────────────────────────────────────
# Geometria dei grafi (pool TSPLIB oppure uniforme)
# ─────────────────────────────────────────────────────────────────────
def _load_graph_json(path):
    """JSON schema `nodi_*.json` → (coords (M,2), dist (M,M))."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    ids = [int(v) for v in d["node_ids"]]
    coords = np.array([d["coordinates"][str(v)] for v in ids], dtype=float)
    dm = d["distance_matrix"]
    dist = np.array([[float(dm[str(i)][str(j)]) for j in ids] for i in ids], dtype=float)
    if coords.shape[0] != dist.shape[0] or dist.shape[0] != dist.shape[1]:
        raise ValueError(f"{path}: coordinate/distanze incoerenti.")
    return coords, dist


def _resolve_pool(big_paths):
    """big_paths: dir | lista di path | stringa comma-separated → lista (coords,dist)."""
    if isinstance(big_paths, str):
        if os.path.isdir(big_paths):
            paths = sorted(glob.glob(os.path.join(big_paths, "*.json")))
        else:
            paths = [s for s in big_paths.split(",") if s.strip()]
    else:
        paths = list(big_paths or [])
    if not paths:
        raise ValueError("mode='subsample' richiede un pool non vuoto (TESI_MG_BIG).")
    return [_load_graph_json(p) for p in paths]


def _make_graph(mode, N, rng, big):
    """(nodes, coords, base_dist, E) per un grafo a N nodi, id 0..N-1."""
    if mode == "uniform":
        xy = rng.random((N, 2))
        diff = xy[:, None, :] - xy[None, :, :]
        D = np.sqrt((diff ** 2).sum(axis=-1))
    elif mode == "subsample":
        big_coords, big_dist = big
        M = big_coords.shape[0]
        if M < N:
            raise ValueError(f"subsample: istanza con {M} nodi < N={N}.")
        pick = np.sort(rng.choice(M, size=N, replace=False))
        xy = big_coords[pick]
        # NOTA: distanze dalla sottomatrice, non ricalcolate da xy: preserva la metrica
        # TSPLIB originale (che può non essere euclidea pura). xy resta il prior geometrico.
        D = big_dist[np.ix_(pick, pick)]
    else:
        raise ValueError(f"mode ignoto: {mode!r} (usa 'uniform' o 'subsample').")

    nodes = list(range(N))
    coords = {i: (float(xy[i, 0]), float(xy[i, 1])) for i in nodes}
    base_dist = {i: {j: float(D[i, j]) for j in nodes} for i in nodes}
    E = [(i, j) for i in nodes for j in nodes if i != j]
    return nodes, coords, base_dist, E


# ─────────────────────────────────────────────────────────────────────
# Un'istanza = un grafo + K scenari + I,p,C (riusando le tue primitive)
# ─────────────────────────────────────────────────────────────────────
def _build_graph_instance(nodes, coords, base_dist, E, K, seed):
    dist_np = np.array([[base_dist[i][j] for j in nodes] for i in nodes], dtype=float)
    medoids = [nodes[t] for t in k_medoids(dist_np, len(K_MEDOID_NODES), seed=seed)]

    I, _ = build_I_from_medoid_outgoing_nodes(nodes, E, base_dist, medoid_nodes=medoids)
    if not I:
        raise ValueError("I vuota sul grafo generato.")
    b = {e: base_cost_undirected(base_dist, e[0], e[1]) for e in I}
    p = {e: PRENOTAZIONE_FRAC * b[e] for e in I}
    C = {e: PENALTY_FRAC * b[e] for e in I}

    # frequent_arcs=[] ⇒ nessuna calibrazione MIP; il batcher forza solve_pi=False ⇒ env=None.
    batches = generate_scenario_batches(
        nodes, E, base_dist, I, [],
        N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, seed,
        nodes[0], None, p, C,
        n_scenarios=K, batch_size=K, drop_last=False,
    )
    bt = batches[0]
    return {"nodes": nodes, "coords": coords, "I": I, "p": p, "C": C,
            "scenario_ids": bt["scenario_ids"], "results": bt["results"],
            "scenario_probs": bt["scenario_probs"]}


def generate_graph_instances(n_instances, K, mode, seed=UTSP_TRAINING_SEED,
                             big_paths=None, N=None):
    """Lista di istanze. Ogni grafo è determinato dal solo `seed+1+g`:
    riproducibile e indipendente dall'ordine e da n_instances."""
    N = int(N or N_NODES)
    if n_instances < 1:
        raise ValueError("n_instances ≥ 1.")
    if K < 2:
        raise ValueError("K ≥ 2 (più scenari per istanza).")
    if len(K_MEDOID_NODES) > N:
        raise ValueError(f"len(K_MEDOID_NODES)={len(K_MEDOID_NODES)} > N={N}.")

    pool = _resolve_pool(big_paths) if mode == "subsample" else None
    if pool is not None and any(c.shape[0] < N for c, _ in pool):
        raise ValueError(f"il pool contiene istanze con < {N} nodi.")

    instances = []
    for g in range(n_instances):
        g_seed = int(seed) + 1 + g
        g_rng = np.random.default_rng(g_seed)
        big = pool[int(g_rng.integers(len(pool)))] if pool is not None else None
        nodes, coords, base_dist, E = _make_graph(mode, N, g_rng, big)
        inst = _build_graph_instance(nodes, coords, base_dist, E, K, seed=g_seed)
        if len(inst["nodes"]) != N:
            raise ValueError(f"istanza {g}: N atteso {N}, ottenuto {len(inst['nodes'])}.")
        instances.append(inst)
    src = f"{len(pool)} istanze pool" if pool is not None else "uniforme"
    print(f"[multigraph] {n_instances} istanze | N={N} K={K} mode={mode} ({src})")
    return instances


# ─────────────────────────────────────────────────────────────────────
# Tensori per grafo + loop di ottimizzazione CONDIVISO col single-graph
# ─────────────────────────────────────────────────────────────────────
def _prepare_unit(inst, device):
    """Un grafo → unità pronta per il loop, con la SUA normalizzazione."""
    xy, _dr, dist_model, dist_scale, temperature = _build_input_tensors(
        inst["scenario_ids"], inst["results"], inst["nodes"], inst["coords"], device
    )
    adj = torch.exp(-dist_model / max(float(temperature), 1e-9))
    I_mask, p_mat, C_mat, _ = build_I_tensors(
        inst["I"], inst["nodes"], inst["p"], inst["C"], device
    )
    p_mat = p_mat / dist_scale                      # identico a _train_utsp_2stage
    C_mat = C_mat / dist_scale
    K = len(inst["scenario_ids"])
    probs = torch.full((K,), 1.0 / K, dtype=torch.float32, device=device)
    return {"xy": xy, "adj": adj, "dist": dist_model, "I_mask": I_mask,
            "p_mat": p_mat, "C_mat": C_mat, "probs": probs, "K": K}


# NOTA: il loop di ottimizzazione e' _optimize_units, importato da utsp: unica
# sorgente condivisa con il training single-graph (nessuna duplicazione).


def train_utsp_multigraph(instances, device, seed=UTSP_TRAINING_SEED,
                          exp_name="espB_UTSP_LS"):
    N = len(instances[0]["nodes"])
    torch.manual_seed(int(seed))                    # init GNN, come in _train_utsp_2stage
    model = UTSP_GNN(N, UTSP2_HIDDEN, UTSP2_NLAYERS).to(device)
    n_par = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
    optimizer = optim.Adam(model.parameters(), lr=UTSP2_LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=UTSP2_STEP_LR, gamma=0.8)

    units = [_prepare_unit(inst, device) for inst in instances]
    print(f"\n  Training UTSP MULTIGRAPH | device={device} | n_params={n_par:,}")
    print(f"  GNN({N}→{UTSP2_HIDDEN}×{UTSP2_NLAYERS}) | istanze={len(units)} "
          f"K={units[0]['K']} | epoche={UTSP2_EPOCHS} lr={UTSP2_LR}")

    t0 = time.time()
    history, best_state, best_loss = _optimize_units(
        model, units, optimizer, scheduler, device, UTSP2_EPOCHS, shuffle_seed=seed
    )
    print(f"\n  MULTIGRAPH completato | best={best_loss:.5f} "
          f"(ep {int(np.argmin(history['loss'])) + 1}) | {time.time() - t0:.1f}s")

    model.load_state_dict(best_state)
    model.eval()
    return model, history


# ─────────────────────────────────────────────────────────────────────
# Salvataggio nel formato esistente (+ flag multigraph)
# ─────────────────────────────────────────────────────────────────────
def _save_multigraph_artifact(exp_name, model, history, meta):
    paths = _utsp_train_paths(exp_name)
    metadata = {
        "multigraph": True,            # <- riconosciuto dalla patch nel loader
        "exp_name": str(exp_name),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(OUTPUT_DIR),
        "n_nodes": int(N_NODES),
        "utsp2_hidden": int(UTSP2_HIDDEN),
        "utsp2_nlayers": int(UTSP2_NLAYERS),
        "utsp2_epochs": int(UTSP2_EPOCHS),
        "utsp2_lr": float(UTSP2_LR),
        "utsp_training_seed": int(UTSP_TRAINING_SEED),
        **meta,
    }
    torch.save({
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "history": _json_safe(history),
        "metadata": metadata,
    }, paths["model"])
    with open(paths["history"], "w", encoding="utf-8") as f:
        json.dump(_json_safe(history), f, indent=2)
    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(_json_safe(metadata), f, indent=2)
    print(f"  Artefatto MULTIGRAPH salvato: {paths['model']}")


# ─────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────
def run_multigraph(n_instances, K, mode, big_paths=None,
                   seed=UTSP_TRAINING_SEED, exp_name="espB_UTSP_LS"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    instances = generate_graph_instances(n_instances, K, mode, seed=seed, big_paths=big_paths)
    model, history = train_utsp_multigraph(instances, device, seed=seed, exp_name=exp_name)
    _save_multigraph_artifact(exp_name, model, history, {
        "n_instances": int(n_instances), "K": int(K), "mode": mode,
        "seed": int(seed), "big_paths": big_paths if isinstance(big_paths, str) else None,
    })
    return model, history


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


if __name__ == "__main__":
    run_multigraph(
        n_instances=_env_int("TESI_MG_INSTANCES", 300),
        K=_env_int("TESI_MG_K", 20),
        mode=os.environ.get("TESI_MG_MODE", "uniform").strip().lower(),
        big_paths=os.environ.get("TESI_MG_BIG", "").strip() or None,
        seed=_env_int("TESI_MG_SEED", UTSP_TRAINING_SEED),
        exp_name=os.environ.get("TESI_MG_EXP", "espB_UTSP_LS").strip(),
    )
