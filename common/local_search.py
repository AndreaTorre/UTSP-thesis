# -*- coding: utf-8 -*-
"""
local_search.py

Modulo isolato dalla parte "rete" di utsp.py: contiene tutta la local search
di secondo stadio (mosse, decodifica heatmap->tour, esecuzione su scenari),
le funzioni di generazione/gestione delle istanze di test, i confronti con
PI/STO/EEV e il salvataggio dei riepiloghi.

Estratto meccanicamente da utsp.py (nessuna riga di logica riscritta), per
poter lavorare sulla struttura della local search senza toccare la GNN.

Rimasto in utsp.py: UTSP_GNN, diffusione GCN/scattering, costruzione tensori
di input, training della rete, salvataggio/caricamento checkpoint,
run_esperimento_B_UTSP (che ora importa _run_local_search_branch e
_run_utsp_test_only_branch da qui).
"""
import os
import time
import math
import random
import pickle
import hashlib
import numpy as np
import torch

from config import (
    OUTPUT_DIR, N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, UTSP_BATCH_SIZE,
    TEST_SCENARIO_IDS_UTSP, TEST_SCENARIO_SEED,
    TEST_SCENARIO_CACHE_DIR, TEST_SKIP_PI, PI_TIME_LIMIT, PI_MIP_GAP,
    UTSP2_EPOCHS, UTSP2_LS_ALPHA,
    UTSP_LS_MAX_ACTIONS, UTSP_LS_ACTIONS_PER_ROUND, UTSP_LS_MAX_RESTARTS,
    UTSP_LS_M, UTSP_LS_K, UTSP_LS_BETA, UTSP_LS_RANDOM_SEED,
    UTSP_LS_APPLY_INITIAL_2OPT, DIM_ISTANZA_TEST, N_ISTANZE_TEST,
)
from tsp_utils import get_edge_value, canon_edge
from gurobi_models import solve_exact_tsp, solve_reservation_tsp
from scenarios import generate_scenarios
from evaluation import (
    validate_policies, genera_grafici_utsp, plot_cost_distributions,
)
from two_stage_utsp_loss import compute_heatmap

# DIPENDENZE E LOCAL SEARCH 
 
# Costo di un tour su una matrice/dizionario di distanze orientate
def tour_cost(tour, dist): 
    n = len(tour)
    return sum(dist[tour[k]][tour[(k + 1) % n]] for k in range(n))

# decodifica H e tour con gurobi e non local search
def _decode_tour_gurobi(H, nodes, E, root, env):
    node_idx = {v: k for k, v in enumerate(nodes)}
    dist_neg = {
        i: {j: -float(H[node_idx[i], node_idx[j]])
            for j in nodes if j != i}
        for i in nodes
    }
    return solve_exact_tsp(nodes, E, dist_neg, root, env)


 
# LOCAL SEARCH STILE UTSP PAPER, ADATTATA AL CASO ORIENTATO 
# Ruota un tour senza ripetizione finale in modo che inizi da start.
def _rotate_tour_to_start(tour, start):
    
    if not tour or start not in tour:
        return list(tour)
    k = tour.index(start)
    return list(tour[k:] + tour[:k])

# Mantiene una rappresentazione canonica del tour con root in prima posizione.
def _rotate_tour_to_root(tour, root):
    return _rotate_tour_to_start(list(tour), root)

# Archi orientati del tour, incluso l'arco di ritorno all'inizio
def _tour_edges(tour):
    n = len(tour)
    return [(tour[i], tour[(i + 1) % n]) for i in range(n)]

# Costo di un tour su una matrice/dizionario di distanze orientate
def _tour_cost_on_dist(tour, dist):
    return tour_cost(tour, dist)


# 2-opt con ricalcolo completo del costo, quindi valido anche con costi orientati.
# Nel TSP asimmetrico l'inversione cambia i versi degli archi: non uso formule
# incrementali simmetriche, ma rivaluto tutto il tour.

def _two_opt_descent_directed(tour, dist, root, max_passes=50):
    
    if not tour or len(tour) <= 3:
        return list(tour), float("inf")

    best = _rotate_tour_to_root(tour, root)
    best_cost = _tour_cost_on_dist(best, dist)
    n = len(best)

    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 2, n + 1):
                if i == 1 and j == n:
                    continue
                cand = best[:i] + list(reversed(best[i:j])) + best[j:]
                cand = _rotate_tour_to_root(cand, root)
                cand_cost = _tour_cost_on_dist(cand, dist)
                if cand_cost + 1e-9 < best_cost:
                    best, best_cost = cand, cand_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best, best_cost


# Or-opt (single-node relocation) per ATSP.
# Sposta un nodo alla volta nella posizione che riduce il costo.
# Non inverte segmenti: valida per grafi orientati.
def _or_opt_descent_directed(tour, dist, root, max_passes=50):
    
    if not tour or len(tour) <= 3:
        return list(tour), float("inf")

    best = _rotate_tour_to_root(list(tour), root)
    best_cost = _tour_cost_on_dist(best, dist)
    n = len(best)

    for _ in range(max_passes):
        improved = False
        for i in range(n):
            node = best[i]
            remaining = best[:i] + best[i + 1:]
            for j in range(len(remaining)):
                cand = remaining[:j + 1] + [node] + remaining[j + 1:]
                cand = _rotate_tour_to_root(cand, root)
                cand_cost = _tour_cost_on_dist(cand, dist)
                if cand_cost + 1e-9 < best_cost:
                    best, best_cost = cand, cand_cost
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break

    return best, best_cost

#Per ogni nodo i costruisce i candidati j da provare nella local search.
# Priorità: top-M valori di H[i,j]. Se la riga è tutta nulla, fallback sui nodi
#più vicini secondo la distanza media reale.
def _build_heatmap_candidates(H, nodes, M, avg_dist=None):   
    idx = {v: k for k, v in enumerate(nodes)}
    candidates = {}

    for i in nodes:
        ii = idx[i]
        vals = []
        for j in nodes:
            if i == j:
                continue
            jj = idx[j]
            vals.append((float(H[ii, jj]), j))

        vals_sorted = sorted(vals, key=lambda x: x[0], reverse=True)
        top = [j for val, j in vals_sorted[:max(1, min(M, len(vals_sorted)))] if val > 0]

        if not top and avg_dist is not None:
            near = sorted(
                [(avg_dist[i][j], j) for j in nodes if j != i],
                key=lambda x: x[0]
            )
            top = [j for _, j in near[:max(1, min(M, len(near)))]]

        if not top:
            top = [j for _, j in vals_sorted[:max(1, min(M, len(vals_sorted)))]]

        candidates[i] = top

    return candidates

#Selezione stocastica guidata dalla heatmap, coerente con il criterio del paper:
    # valore heatmap + termine di esplorazione alpha * sqrt(log(S+1)/(N+1)).
def _select_heatmap_candidate(a, candidates, H_work, chosen_times, idx, rng, alpha, total_actions):
    
    cand = [b for b in candidates.get(a, []) if b != a]
    if not cand:
        return None

    ia = idx[a]
    scores = []
    for b in cand:
        ib = idx[b]
        explore = 0.0
        if alpha > 0:
            explore = alpha * math.sqrt(math.log(total_actions + 2.0) / (chosen_times[ia, ib] + 1.0))
        score = max(float(H_work[ia, ib]) + explore, 1e-12)
        scores.append(score)

    ssum = sum(scores)
    r = rng.random() * ssum
    acc = 0.0
    for b, sc in zip(cand, scores):
        acc += sc
        if r <= acc:
            chosen_times[ia, idx[b]] += 1
            return b

    b = cand[-1]
    chosen_times[ia, idx[b]] += 1
    return b

# Sposta b immediatamente dopo a
def _move_relocate_after(tour, a, b, root):
    if a == b or a not in tour or b not in tour:
        return list(tour)
    t = list(tour)
    t.remove(b)
    pos_a = t.index(a)
    t.insert(pos_a + 1, b)
    return _rotate_tour_to_root(t, root)


# Mossa tipo 2-opt che prova a rendere a→b un arco del tour.
# È una versione orientata/adattata: rivaluto sempre il costo completo
def _move_two_opt_make_edge(tour, a, b, root):
    if a == b or a not in tour or b not in tour:
        return list(tour)

    rot = _rotate_tour_to_start(list(tour), a)
    pos_b = rot.index(b)
    if pos_b == 1:
        return _rotate_tour_to_root(rot, root)
    if pos_b <= 0:
        return _rotate_tour_to_root(rot, root)

    cand = rot[:1] + list(reversed(rot[1:pos_b + 1])) + rot[pos_b + 1:]
    return _rotate_tour_to_root(cand, root)

# Scambia due nodi, lasciando poi root in prima posizione 
def _move_swap_nodes(tour, a, b, root):
    
    if a == b or a not in tour or b not in tour:
        return list(tour)
    t = list(tour)
    ia, ib = t.index(a), t.index(b)
    t[ia], t[ib] = t[ib], t[ia]
    return _rotate_tour_to_root(t, root)


def _random_tour(nodes, root, rng):
    rest = [v for v in nodes if v != root]
    rng.shuffle(rest)
    return [root] + rest

#  Ricerca locale guidata dalla heatmap
def _utsp_paper_style_local_search(tour_seed, H_decode, nodes, root, avg_dist):
    if not tour_seed or len(tour_seed) <= 3:
        return list(tour_seed), float("inf"), {"actions": 0, "restarts": 0, "improvements": 0}

    rng = random.Random(UTSP_LS_RANDOM_SEED)
    idx = {v: k for k, v in enumerate(nodes)}
    n = len(nodes)
    M = max(1, min(UTSP_LS_M, n - 1))

    H_work = np.array(H_decode, dtype=float, copy=True)
    chosen_times = np.zeros_like(H_work, dtype=float)
    candidates = _build_heatmap_candidates(H_work, nodes, M, avg_dist=avg_dist)

    current = _rotate_tour_to_root(tour_seed, root)
    current_cost = _tour_cost_on_dist(current, avg_dist)

    if UTSP_LS_APPLY_INITIAL_2OPT:
      current, current_cost = _or_opt_descent_directed(current, avg_dist, root)

    best = list(current)
    best_cost = current_cost

    total_actions = 0
    restarts = 0
    improvements = 0
    t0 = time.time()

    while total_actions < UTSP_LS_MAX_ACTIONS and restarts <= UTSP_LS_MAX_RESTARTS:
        best_round = None
        best_round_cost = current_cost
        best_round_added_edges = []

        for _ in range(UTSP_LS_ACTIONS_PER_ROUND):
            total_actions += 1
            if total_actions > UTSP_LS_MAX_ACTIONS:
                break

            a = rng.choice(nodes)
            b = _select_heatmap_candidate(
                a, candidates, H_work, chosen_times, idx, rng,
                UTSP2_LS_ALPHA, total_actions
            )
            if b is None:
                continue

            # Piccolo insieme di mosse locali. La candidate list da H decide cosa provare;
            # la bontà viene misurata sul costo medio reale.
            proposals = [
                _move_relocate_after(current, a, b, root),
                _move_two_opt_make_edge(current, a, b, root),
                _move_swap_nodes(current, a, b, root),
            ]

            # Profondità K: per n piccoli uso mosse composte relocate-after ripetute,
            # guidate da candidati successivi della heatmap.
            if UTSP_LS_K > 2:
                comp = list(current)
                aa = a
                for _depth in range(min(UTSP_LS_K, 4)):
                    bb = _select_heatmap_candidate(
                        aa, candidates, H_work, chosen_times, idx, rng,
                        UTSP2_LS_ALPHA, total_actions
                    )
                    if bb is None:
                        break
                    comp = _move_relocate_after(comp, aa, bb, root)
                    aa = bb
                proposals.append(comp)

            cur_edges = set(_tour_edges(current))
            for cand in proposals:
                if len(set(cand)) != n:
                    continue
                cand = _rotate_tour_to_root(cand, root)
                cand_cost = _tour_cost_on_dist(cand, avg_dist)
                if cand_cost + 1e-9 < best_round_cost:
                    cand_edges = set(_tour_edges(cand))
                    added = list(cand_edges - cur_edges)
                    best_round = cand
                    best_round_cost = cand_cost
                    best_round_added_edges = added

        if best_round is not None:
            before = max(current_cost, 1e-9)
            gain = current_cost - best_round_cost
            current = best_round
            current_cost = best_round_cost
            improvements += 1

            # Backpropagation/update stile paper: aumenta peso degli archi che hanno
            # prodotto miglioramento.
            inc = UTSP_LS_BETA * (math.exp(max(gain, 0.0) / before) - 1.0)
            if inc > 0 and best_round_added_edges:
                for i, j in best_round_added_edges:
                    if i in idx and j in idx and i != j:
                        H_work[idx[i], idx[j]] += inc
                candidates = _build_heatmap_candidates(H_work, nodes, M, avg_dist=avg_dist)

            if current_cost + 1e-9 < best_cost:
                best = list(current)
                best_cost = current_cost
        else:
            restarts += 1
            current = _random_tour(nodes, root, rng)
            current_cost = _tour_cost_on_dist(current, avg_dist)
            if UTSP_LS_APPLY_INITIAL_2OPT:
               current, current_cost = _or_opt_descent_directed(current, avg_dist, root)
            if current_cost + 1e-9 < best_cost:
                best = list(current)
                best_cost = current_cost

    info = {
        "actions": total_actions,
        "restarts": restarts,
        "improvements": improvements,
        "seconds": time.time() - t0,
        "final_heatmap_max": float(H_work.max()) if H_work.size else 0.0,
    }
    return best, best_cost, info

# Stampa quanto A è diversa dalla sua trasposta
def _matrix_asymmetry_stats(A, name):
    
    A = np.array(A, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        print(f"  Asimmetria {name}: matrice non quadrata, salto diagnostica.")
        return

    n = A.shape[0]
    mask = ~np.eye(n, dtype=bool)
    diff = A - A.T
    abs_diff = np.abs(diff[mask])
    abs_vals = np.abs(A[mask])

    denom = float(np.mean(abs_vals)) + 1e-12
    print(
        f"  Asimmetria {name}: "
        f"mean|A-A.T|={float(np.mean(abs_diff)):.6e}  "
        f"p90={float(np.percentile(abs_diff, 90)):.6e}  "
        f"max={float(np.max(abs_diff)):.6e}  "
        f"rel_mean={float(np.mean(abs_diff))/denom:.6f}"
    )


def _build_mean_dist_from_results(results, scenario_ids, nodes):
    avg = {i: {} for i in nodes}
    for i in nodes:
        for j in nodes:
            if i == j:
                avg[i][j] = 0.0
            else:
                avg[i][j] = float(np.mean([results[sid]["scenario_dist"][i][j] for sid in scenario_ids]))
    return avg
 # Tour greedy nearest-neighbor come seed per la local search   
def _greedy_tour(nodes, root, dist):
    unvisited = set(nodes) - {root}
    tour, cur = [root], root
    while unvisited:
        nxt = min(unvisited, key=lambda j: dist[cur].get(j, float("inf")))
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return tour

# Orginale: Costruisce la matrice di adiacenza per un singolo scenario
def _build_adj_single(D, n, device, dist_scale, temperature):
    dist_norm = D / max(dist_scale, 1e-9)
    return torch.exp(-dist_norm / max(temperature, 1e-9))  # (n, n)
#def _build_adj_single(D, n, device, dist_scale, temperature, k=3, min_weight=0.2): 
#    """
#    Costruisce la matrice di adiacenza per un singolo scenario.
#    Usa la stessa normalizzazione (dist_scale, temperature) del training,
#    in modo che la GNN veda input nella stessa scala.
#    D: (n, n) tensor delle distanze reali dello scenario.
 
#    Applica la stessa garanzia k-NN di _build_adj_with_knn_guarantee,
#    così ogni nodo ha almeno k vicini con peso >= min_weight anche in inferenza.
#    """
#    dist_norm = D / max(dist_scale, 1e-9)
    # Riusa la funzione di training: aggiunge dimensione batch (1, n, n) poi la rimuove
#    adj = _build_adj_with_knn_guarantee(
#        dist_norm.unsqueeze(0), max(temperature, 1e-9), k=k, min_weight=min_weight
#    )
#    return adj.squeeze(0)  # (n, n)


# Forward pass della GNN su un singolo scenario nuovo
def _gnn_heatmap_single(model, xy, adj_single, device): 
    model.eval()
    with torch.no_grad():
        T_out = model(xy.unsqueeze(0), adj_single.unsqueeze(0), device)  # (1,n,n)
        H_t   = compute_heatmap(T_out)
    H = H_t.squeeze(0).cpu().numpy().copy()
    return H

# Distanze effettive per il secondo stadio della LS: aggiungo la penale C[i,j] agli archi di I non prenotat
def _effective_dist_for_ls(dist_s, nodes, I, x_ls, C): 
    from tsp_utils import canon_edge as _ce
    x_set = {_ce(i, j) for (i, j) in x_ls}
    eff   = {i: dict(dist_s[i]) for i in nodes}
    for (i, j) in I:
        if _ce(i, j) not in x_set:
            pen      = get_edge_value(C, i, j)
            eff[i][j] = dist_s[i][j] + pen
            eff[j][i] = dist_s[j][i] + pen
    return eff

# costo reale secondo stadio con percorrenze vere e penale degli archi
def _ls_cost_with_penalty(tour, dist_s, nodes, I, x_ls, C): 
    from tsp_utils import canon_edge as _ce
    x_set = {_ce(i, j) for (i, j) in x_ls}
    I_set = {_ce(i, j) for (i, j) in I}
    arcs  = _tour_edges(tour)
    tc    = tour_cost(tour, dist_s)
    pc    = sum(
        get_edge_value(C, i, j)
        for (i, j) in arcs
        if _ce(i, j) in I_set and _ce(i, j) not in x_set
    )
    return tc, pc

#Esegue il secondo stadio local search per ogni scenario
def _run_ls_on_scenarios(
    model, xy, nodes, root, I, p, C,
    results_scenarios, scenario_ids, scenario_probs,
    H_list_precomputed, dist_scale, temperature, device, x_ls, label="train",
    apply_penalties=True,
):
    n = len(nodes)
    reserv = sum(get_edge_value(p, i, j) for (i, j) in x_ls) if apply_penalties else 0.0
    costs, tc_dict, pc_dict, tours, solutions = {}, {}, {}, {}, {}

    t0_ls = time.time()
    for k, sid in enumerate(scenario_ids):
        t0_sid = time.time()
        dist_s = results_scenarios[sid]["scenario_dist"]

        # Heatmap scenario-specifica
        if H_list_precomputed is not None:
            H_t = H_list_precomputed[k]             # (1, n, n) tensor training/batch
            H_s = H_t.detach().squeeze(0).cpu().numpy().copy()
        else:
            D = torch.zeros(n, n, device=device)
            for ii, i in enumerate(nodes):
                for jj, j in enumerate(nodes):
                    if i != j:
                        D[ii, jj] = float(dist_s[i][j])
            # originale
            adj_s = _build_adj_single(D, n, device, dist_scale, temperature)

            # nuova versione alternativa, non attiva
            # adj_s = _build_adj_robust_floor(
            #     D.unsqueeze(0) / max(dist_scale, 1e-9),
            #     tau=UTSP2_TEMP_SCALE, eps=0.02, q_scale=0.90, clip_max=3.0,
            # ).squeeze(0)
            H_s = _gnn_heatmap_single(model, xy, adj_s, device)

        # Prima LS: se apply_penalties=False uso distanze pure.
        # Seconda LS: se apply_penalties=True uso prenotazioni fissate e multe attive.
        if apply_penalties:
            dist_eff = _effective_dist_for_ls(dist_s, nodes, I, x_ls, C)
        else:
            dist_eff = {i: dict(dist_s[i]) for i in nodes}

        seed = _greedy_tour(nodes, root, dist_eff)
        tour_s, _, ls_info = _utsp_paper_style_local_search(
            seed, H_s, nodes, root, dist_eff
        )

        # Costo finale coerente con la fase.
        # - pre-booking: solo costo di percorrenza, nessuna multa;
        # - post-booking: percorrenza vera + multe sugli archi in I non prenotati + costo prenotazione.
        if apply_penalties:
            tc, pc = _ls_cost_with_penalty(tour_s, dist_s, nodes, I, x_ls, C)
            x_used = list(x_ls)
        else:
            tc = tour_cost(tour_s, dist_s)
            pc = 0.0
            x_used = []

        arcs_s = _tour_edges(tour_s)

        elapsed_sid = time.time() - t0_sid
        costs[sid]   = reserv + tc + pc
        tc_dict[sid] = tc
        pc_dict[sid] = pc
        tours[sid]   = tour_s
        solutions[sid] = {
            "arcs":             arcs_s,
            "y_used":           arcs_s,
            "tour":             list(tour_s),
            "tour_cost":        tc,
            "total_cost":       costs[sid],
            "penalty_paid":     pc,
            "reservation_paid": reserv,
            "x_used":           x_used,
        }

        phase = "post" if apply_penalties else "pre"
        print(f"    [{label}|{phase}] s={sid}: tc={tc:.4f}  pc={pc:.4f}  "
              f"tot={costs[sid]:.4f}  t={elapsed_sid:.2f}s  tour={tour_s}")

    elapsed_ls = time.time() - t0_ls
    n_scen = len(scenario_ids)
    print(f"  [{label}] Local search completata: {n_scen} scenari in "
          f"{elapsed_ls:.2f}s  ({elapsed_ls/n_scen:.2f}s/scenario)")
    # Report per batch solo per le diagnostiche di train.
    # Nel test gli scenari sono un unico blocco decisionale; non stampo quindi
    # "Batch 1/15" o simili, che sarebbe fuorviante.
    cost_list = [costs[sid] for sid in scenario_ids]
    if str(label).startswith("train_"):
        n_batches_report = len(cost_list) // UTSP_BATCH_SIZE
        if n_batches_report > 1:
            for b_idx in range(n_batches_report):
                chunk = cost_list[b_idx*UTSP_BATCH_SIZE : (b_idx+1)*UTSP_BATCH_SIZE]
                print(f"    [{label}] Batch {b_idx+1}/{n_batches_report}: "
                      f"media={sum(chunk)/len(chunk):.4f}  "
                      f"min={min(chunk):.4f}  max={max(chunk):.4f}")
    if scenario_probs is not None:
        mean = sum(scenario_probs[sid] * costs[sid] for sid in scenario_ids)
    else:
        mean = sum(costs.values()) / len(costs)

    return costs, tc_dict, pc_dict, tours, solutions, mean

def _heatmap_numpy_from_H_list(H_list):
    H_sum = None
    for H in H_list:
        arr = H.detach().squeeze(0).cpu().numpy()
        H_sum = arr.copy() if H_sum is None else H_sum + arr
    return H_sum / max(len(H_list), 1)

#Costruisco la heatmap aggregata pesata    
def _heatmap_bar_from_H_list(H_list, scenario_ids, scenario_probs):
    H_bar = None

    for sid, H in zip(scenario_ids, H_list):
        p_omega = float(scenario_probs[sid])
        H_omega = H.detach().squeeze(0).cpu().numpy()

        if H_bar is None:
            H_bar = p_omega * H_omega
        else:
            H_bar += p_omega * H_omega

    return H_bar


def _evaluate_fixed_tour_on_scenarios(tour, results, scenario_ids):
    costs = {}
    arcs = _tour_edges(tour) if tour else []
    solutions = {}
    for sid in scenario_ids:
        dist = results[sid]["scenario_dist"]
        cost = tour_cost(tour, dist) if tour else None
        costs[sid] = cost
        solutions[sid] = {
            "arcs": arcs,
            "y_used": arcs,
            "tour": list(tour),
            "tour_cost": cost,
            "total_cost": cost,
            "penalty_paid": 0.0,
            "reservation_paid": 0.0,
            "x_used": [],
        }
    mean_cost = float(np.mean([c for c in costs.values() if c is not None])) if costs else None
    return costs, solutions, mean_cost


def _aggregate_test_instances(exp_name, istanza_metrics):
    """
    Media e deviazione standard delle metriche di test su tutte le istanze.
    NOTA: prima versione minimale, solo per sbloccare _run_local_search_branch.
    """
    import numpy as np
    keys = ["UTSP_LS_test", "PI_test", "PI_pren_test", "STO_test", "EEV_test",
            "gap_ls_sto", "gap_ls_eev", "gap_ls_pi"]
    agg = {"n_istanze": len(istanza_metrics)}
    for k in keys:
        vals = [m[k] for m in istanza_metrics if m.get(k) is not None and np.isfinite(m[k])]
        agg[f"{k}_mean"] = float(np.mean(vals)) if vals else float("nan")
        agg[f"{k}_std"] = float(np.std(vals)) if vals else float("nan")
    return agg

def _run_heatmap_local_search(nodes, E, root, env, results, scenario_ids, H_list):
    H_raw = _heatmap_numpy_from_H_list(H_list)
    H_decode = H_raw.copy()

    _matrix_asymmetry_stats(H_raw, "H_raw UTSP 2-stage")
    _matrix_asymmetry_stats(H_decode, "H_decode UTSP 2-stage")

    avg_dist = _build_mean_dist_from_results(results, scenario_ids, nodes)

    print("\n  Decodifica tour da heatmap con Gurobi ...")
    decoded = _decode_tour_gurobi(H_decode, nodes, E, root, env)
    if isinstance(decoded, dict):
        tour_seed = decoded.get("tour", [])
        arcs_seed = decoded.get("arcs", [])
        seed_cost = tour_cost(tour_seed, avg_dist) if tour_seed else None
    else:
        seed_cost, arcs_seed = decoded
        tour_seed = []

    if not tour_seed and arcs_seed:
        succ = {i: j for (i, j) in arcs_seed}
        tour_seed = [root]
        cur = root
        for _ in range(len(nodes)):
            nxt = succ.get(cur)
            if nxt is None or nxt == root:
                break
            tour_seed.append(nxt)
            cur = nxt
        seed_cost = tour_cost(tour_seed, avg_dist) if tour_seed else None

    print(f"  Tour iniziale heatmap: {tour_seed}")
    print(f"  Costo su distanza media training: {seed_cost if seed_cost is not None else 'N/A'}")

    print("\n  Local search UTSP guidata da H ...")
    tour_ls, ls_cost_avg, ls_info = _utsp_paper_style_local_search(
        tour_seed, H_decode, nodes, root, avg_dist
    )
    arcs_ls = _tour_edges(tour_ls) if tour_ls else []

    print(f"  Tour UTSP-LS: {tour_ls}")
    print(f"  Costo UTSP-LS su distanza media training: {ls_cost_avg:.4f}")
    print(f"  Info local search: {ls_info}")

    return {
        "H_raw": H_raw,
        "H_decode": H_decode,
        "avg_dist": avg_dist,
        "tour_seed": tour_seed,
        "arcs_seed": arcs_seed,
        "seed_cost_avg": seed_cost,
        "tour_ls": tour_ls,
        "arcs_ls": arcs_ls,
        "ls_cost_avg": ls_cost_avg,
        "ls_info": ls_info,
    }

def _compute_bookings_from_tours(tours, scenario_ids, nodes, I, p, C):
    """
    Prenotazioni ottimali post-LS: prenoto arco (i,j) ∈ I se la frequenza
    d'uso nei tour supera la soglia analitica p/C.
    """
    from tsp_utils import canon_edge as _ce

    I_set = {_ce(i, j) for (i, j) in I}
    n_scenarios = len(scenario_ids)

    usage_count = {edge: 0 for edge in I_set}
    for sid in scenario_ids:
        tour = tours[sid]
        arcs = [(tour[k], tour[(k + 1) % len(tour)]) for k in range(len(tour))]
        for (a, b) in arcs:
            ce = _ce(a, b)
            if ce in I_set:
                usage_count[ce] += 1

    x_opt = []
    print("    Prenotazioni post-LS (f > p/C):")
    for edge in sorted(I_set):
        i, j = edge
        f = usage_count[edge] / n_scenarios
        p_val = get_edge_value(p, i, j)
        C_val = get_edge_value(C, i, j)
        soglia = p_val / C_val if C_val > 0 else float('inf')
        book = f > soglia
        if book:
            x_opt.append(edge)
        flag = "✓ PRENOTA" if book else "✗ no"
        print(f"      {{{i},{j}}}  f={f:.3f}  soglia={soglia:.3f}  {flag}")
    print(f"    Totale: {len(x_opt)}/{len(I_set)}")
    return x_opt




def _scenario_mean(costs, scenario_ids, scenario_probs=None):
    """Media robusta dei costi sugli scenari indicati."""
    vals = []
    weights = []
    for sid in scenario_ids:
        val = costs.get(sid)
        if val is None:
            continue
        vals.append(float(val))
        if scenario_probs is not None:
            weights.append(float(scenario_probs.get(sid, 0.0)))

    if not vals:
        return float("nan")

    if scenario_probs is None:
        return float(np.mean(vals))

    wsum = float(np.sum(weights))
    if wsum <= 0:
        return float(np.mean(vals))
    return float(np.dot(vals, weights) / wsum)


def _compute_exact_free_costs_from_results(results, scenario_ids):
    """Costi PI scenario per scenario, usando exact_free prodotto dal generatore."""
    out = {}
    for sid in scenario_ids:
        rec = results.get(sid, {})
        exact = rec.get("exact_free", {}) if isinstance(rec, dict) else {}
        length = exact.get("length", exact.get("cost", None))
        out[sid] = float(length) if length is not None else None
    return out


def _extract_arcs_from_exact_free(exact):
    """Estrae gli archi della soluzione PI da exact_free in modo tollerante."""
    if not isinstance(exact, dict):
        return []

    for key in ("arcs", "y_used", "edges"):
        arcs = exact.get(key)
        if arcs:
            return list(arcs)

    tour = exact.get("tour", exact.get("path", None))
    if tour:
        return _tour_edges(list(tour))

    sol = exact.get("solution", None)
    if isinstance(sol, dict):
        for key in ("arcs", "y_used", "edges"):
            arcs = sol.get(key)
            if arcs:
                return list(arcs)
        tour = sol.get("tour", sol.get("path", None))
        if tour:
            return _tour_edges(list(tour))

    return []


def _compute_pi_with_booking_costs_local(results, scenario_ids, I, p):
    """
    Calcola PI+pren scenario per scenario sugli stessi scenari passati.

    Interpretazione: soluzione perfect-information dello scenario + costo di
    prenotazione per gli archi di I che compaiono nel tour PI. Questo evita che
    il valore di test venga preso da un insieme di scenari diverso o da chiavi
    non allineate.
    """
    from tsp_utils import canon_edge as _ce

    I_set = {_ce(i, j) for (i, j) in I}
    out = {}
    for sid in scenario_ids:
        rec = results.get(sid, {})
        exact = rec.get("exact_free", {}) if isinstance(rec, dict) else {}
        length = exact.get("length", exact.get("cost", None))
        if length is None:
            out[sid] = None
            continue

        arcs = _extract_arcs_from_exact_free(exact)
        used_I = {_ce(i, j) for (i, j) in arcs if _ce(i, j) in I_set}
        booking_cost = sum(get_edge_value(p, i, j) for (i, j) in used_I)
        out[sid] = float(length) + float(booking_cost)

    return out


def _variance_decomposition_by_batch(costs, scenario_ids, batch_id):
    """
    Decomposizione varianza totale = within + between con varianze di popolazione.
    È la decomposizione più comoda per controllare uniformità tra batch.
    """
    groups = {}
    for sid in scenario_ids:
        if sid not in costs or costs[sid] is None:
            continue
        b = batch_id.get(sid, None) if batch_id is not None else None
        groups.setdefault(b, []).append(float(costs[sid]))

    all_vals = np.array([v for vals in groups.values() for v in vals], dtype=float)
    if all_vals.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "var_total": float("nan"),
            "var_within": float("nan"),
            "var_between": float("nan"),
            "batch_stats": [],
        }

    grand_mean = float(all_vals.mean())
    n_tot = int(all_vals.size)
    var_total = float(np.mean((all_vals - grand_mean) ** 2))

    var_within = 0.0
    var_between = 0.0
    batch_stats = []
    for b in sorted(groups, key=lambda x: (-1 if x is None else x)):
        vals = np.array(groups[b], dtype=float)
        n_b = int(vals.size)
        mean_b = float(vals.mean())
        var_b = float(np.mean((vals - mean_b) ** 2)) if n_b else float("nan")
        w_b = n_b / n_tot
        var_within += w_b * var_b
        var_between += w_b * ((mean_b - grand_mean) ** 2)
        batch_stats.append({
            "batch": b,
            "n": n_b,
            "mean": mean_b,
            "var": var_b,
            "std": float(np.sqrt(var_b)) if np.isfinite(var_b) else float("nan"),
            "min": float(vals.min()) if n_b else float("nan"),
            "max": float(vals.max()) if n_b else float("nan"),
        })

    return {
        "n": n_tot,
        "mean": grand_mean,
        "var_total": var_total,
        "var_within": float(var_within),
        "var_between": float(var_between),
        "batch_stats": batch_stats,
    }


def _append_utsp_train_cost_diagnostics(
    exp_name, scenario_ids, costs_train, tc_train, pc_train, tours_train,
    train_batch_id=None, x_ls=None, split_label="train",
):
    """
    Appende al file già creato da plot_cost_distributions le diagnostiche UTSP
    sul training completo: costo scenario per scenario e varianza within/between.
    """
    def fmt(x):
        if x is None:
            return "N/A"
        try:
            return f"{float(x):.6f}"
        except Exception:
            return str(x)

    stats = _variance_decomposition_by_batch(costs_train, scenario_ids, train_batch_id or {})
    lines = [
        "",
        "=" * 80,
        "DIAGNOSTICA UTSP TRAIN COMPLETA",
        "=" * 80,
        f"Scenari UTSP train considerati = {stats['n']}",
        f"Prenotazioni x_ls = {sorted(x_ls or [])}",
        "",
        "DECOMPOSIZIONE VARIANZA COSTI UTSP TRAIN",
        f"  media globale      = {fmt(stats['mean'])}",
        f"  varianza totale    = {fmt(stats['var_total'])}",
        f"  varianza within    = {fmt(stats['var_within'])}",
        f"  varianza between   = {fmt(stats['var_between'])}",
    ]

    total = stats["var_total"]
    if total and np.isfinite(total) and abs(total) > 1e-12:
        lines += [
            f"  quota within       = {100.0 * stats['var_within'] / total:.4f}%",
            f"  quota between      = {100.0 * stats['var_between'] / total:.4f}%",
        ]

    lines += [
        "",
        "STATISTICHE PER BATCH",
        f"  {'batch':>8} | {'n':>6} | {'mean':>12} | {'var':>12} | {'std':>12} | {'min':>12} | {'max':>12}",
    ]
    for row in stats["batch_stats"]:
        b = row["batch"] if row["batch"] is not None else "NA"
        lines.append(
            f"  {str(b):>8} | {row['n']:>6} | {fmt(row['mean']):>12} | "
            f"{fmt(row['var']):>12} | {fmt(row['std']):>12} | "
            f"{fmt(row['min']):>12} | {fmt(row['max']):>12}"
        )

    lines += [
        "",
        "COSTI UTSP TRAIN SCENARIO PER SCENARIO",
        f"  {'scenario':>8} | {'batch':>8} | {'total':>12} | {'percorrenza':>12} | {'multa':>12} | tour",
    ]
    for sid in scenario_ids:
        b = (train_batch_id or {}).get(sid, "NA")
        lines.append(
            f"  {str(sid):>8} | {str(b):>8} | {fmt(costs_train.get(sid)):>12} | "
            f"{fmt(tc_train.get(sid)):>12} | {fmt(pc_train.get(sid)):>12} | "
            f"{tours_train.get(sid, [])}"
        )

    lines.append("=" * 80)

    grafici_dir = os.path.join(OUTPUT_DIR, "grafici")
    os.makedirs(grafici_dir, exist_ok=True)
    stats_file = os.path.join(grafici_dir, f"{exp_name}_cost_distributions_{split_label}_stats.txt")
    with open(stats_file, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  → Diagnostica UTSP train completa aggiunta a: {stats_file}")
def _build_heatmaps_for_scenarios(model, xy, nodes, results, scenario_ids, dist_scale, temperature, device):
    """
    Costruisce una heatmap per ogni scenario indicato, in un unico passaggio batchato.
    Serve per train/test diagnostico: la local search usa sempre la heatmap dello scenario.
    """
    n = len(nodes)
    if not scenario_ids:
        return []

    dist_tensors = []
    for sid in scenario_ids:
        sd = results[sid]["scenario_dist"]
        D = torch.zeros(n, n, device=device)
        for ii, i_node in enumerate(nodes):
            for jj, j_node in enumerate(nodes):
                if i_node != j_node:
                    D[ii, jj] = float(sd[i_node][j_node])
        dist_tensors.append(D)

    dist_stack = torch.stack(dist_tensors, dim=0)
    adj_stack = torch.exp(-(dist_stack / max(dist_scale, 1e-9)) / max(temperature, 1e-9))
    xy_tile = xy.unsqueeze(0).expand(len(scenario_ids), -1, -1).contiguous()

    model.eval()
    with torch.no_grad():
        T_batch = model(xy_tile, adj_stack, device)
        H_list = [compute_heatmap(T_batch[k:k+1]) for k in range(len(scenario_ids))]
    return H_list


def _test_scenario_cache_path():
    return os.path.join(TEST_SCENARIO_CACHE_DIR, "test_scenarios_cache.pkl")


def _test_scenario_cache_key(I, frequent_arcs, base_seed, scenario_kwargs=None):
    # NOTA: stesso schema di state_key già usato in scenarios.find_frequent_arcs.
    # Se I, frequent_arcs o base_seed cambiano, il PI di uno scenario_id può
    # essere diverso (I/frequent_arcs entrano nella scelta degli archi extra
    # perturbati), quindi la cache va invalidata e si riparte da zero.
    # L'impronta della sorgente (vento vs sintetico) è altrettanto essenziale:
    # vedi _scenario_source_fingerprint.
    I_set = {canon_edge(i, j) for (i, j) in I}
    freq_set = {canon_edge(i, j) for (i, j) in frequent_arcs}
    return (
        base_seed,
        tuple(sorted(I_set)),
        tuple(sorted(freq_set)),
        _scenario_source_fingerprint(scenario_kwargs),
    )


def _load_test_scenario_cache(state_key):
    path = _test_scenario_cache_path()
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        cache = pickle.load(f)
    if cache.get("key") != state_key:
        print("  Cache scenari di test trovata ma con parametri diversi (I/frequent_arcs/seed): riparto da zero")
        return {}
    print(f"  Cache scenari di test: {len(cache['results'])} scenari già risolti (riuso, niente Gurobi)")
    return cache["results"]


def _save_test_scenario_cache(state_key, results_by_sid):
    path = _test_scenario_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump({"key": state_key, "results": results_by_sid}, f)
    os.replace(tmp_path, path)


def _scenario_source_fingerprint(scenario_kwargs):
    """
    Identifica UNIVOCAMENTE come vengono generate le perturbazioni.

    # NOTA: indispensabile per la correttezza della cache. build_wind_perturbation
    # è deterministica dal campo di vento e IGNORA base_seed, mentre
    # build_perturbation (sintetica) dipende da base_seed e ignora il vento.
    # Senza questa impronta, una cache costruita con vento verrebbe riusata per
    # scenari sintetici (o con un file .nc diverso) restituendo PI/STO/EEV
    # semplicemente sbagliati, in silenzio.
    """
    wind = (scenario_kwargs or {}).get("wind")
    if wind is None:
        return ("synthetic",)
    h = hashlib.md5()
    h.update(np.ascontiguousarray(wind["u100"]).tobytes())
    h.update(np.ascontiguousarray(wind["v100"]).tobytes())
    return ("wind", int(wind["n_times"]), h.hexdigest())


def _sto_eev_cache_path():
    return os.path.join(TEST_SCENARIO_CACHE_DIR, "test_sto_eev_cache.pkl")


def _sto_eev_cache_key(I, frequent_arcs, base_seed, x_sto, x_ev, scenario_kwargs=None):
    I_set = {canon_edge(i, j) for (i, j) in I}
    freq_set = {canon_edge(i, j) for (i, j) in frequent_arcs}
    x_sto_set = tuple(sorted({canon_edge(i, j) for (i, j) in x_sto}))
    x_ev_set = tuple(sorted({canon_edge(i, j) for (i, j) in x_ev}))
    return (
        base_seed,
        tuple(sorted(I_set)),
        tuple(sorted(freq_set)),
        x_sto_set,
        x_ev_set,
        _scenario_source_fingerprint(scenario_kwargs),
    )


def _load_sto_eev_cache(state_key):
    path = _sto_eev_cache_path()
    if not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        cache = pickle.load(f)
    if cache.get("key") != state_key:
        print("  Cache STO/EEV trovata ma con parametri diversi (I/frequent_arcs/seed/x_sto/x_ev): riparto da zero")
        return {}
    print(f"  Cache STO/EEV: {len(cache['results'])} scenari già risolti (riuso, niente Gurobi)")
    return cache["results"]


def _save_sto_eev_cache(state_key, results_by_sid):
    path = _sto_eev_cache_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump({"key": state_key, "results": results_by_sid}, f)
    os.replace(tmp_path, path)



def _load_pool_costs():
    """Carica i costi del pool (test_pool_cache.pkl) se esiste, per scenario_id.
    Ritorna {} se il pool non c'è: in quel caso STO/EEV si risolvono con Gurobi
    come prima (nessun cambiamento di comportamento)."""
    import os as _os
    import pickle as _pickle
    path = _os.path.join(TEST_SCENARIO_CACHE_DIR, "test_pool_cache.pkl")
    if not _os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            return _pickle.load(f).get("results", {})
    except Exception:
        return {}


def _validate_policies_cached(
    nodes, E, I, p, C, root, env,
    x_sto, x_ev, results_by_sid, scenario_ids,
    base_seed=TEST_SCENARIO_SEED,
    frequent_arcs=None,
    scenario_kwargs=None,
):
    """
    Equivalente cacheato di evaluation.validate_policies, ristretto a
    STO_val/EEV_val/sto_costs/eev_costs (quello che serve al test sweep).

    NOTA IMPORTANTE: x_sto e x_ev qui sono policy di prenotazione GIÀ
    decise da Experiment B (fisse, non variabili di ottimizzazione). Il
    costo di ogni scenario si ottiene fissando x e risolvendo solo il
    second-stage (solve_reservation_tsp) — è quindi indipendente da quali
    altri scenari sono nello stesso blocco/istanza di test, esattamente
    come il PI. Diverso invece da solve_stochastic (Experiment B), che
    DECIDE x congiuntamente su un blocco intero e lì sì che il risultato
    dipende dal blocco. Riusa results_by_sid (già generato per il PI):
    niente doppia generate_scenarios.
    """
    frequent_arcs = frequent_arcs or []
    reservation_sto = sum(get_edge_value(p, i, j) for (i, j) in x_sto)
    reservation_ev = sum(get_edge_value(p, i, j) for (i, j) in x_ev)

    state_key = _sto_eev_cache_key(I, frequent_arcs, base_seed, x_sto, x_ev, scenario_kwargs)
    cache = _load_sto_eev_cache(state_key)

    missing_ids = [sid for sid in scenario_ids if sid not in cache]

    # Aggancio al POOL: se test_pool_cache.pkl contiene STO/EEV già risolti
    # per questi scenario_id (stesso seme, stessi scenari), li prendiamo da lì
    # invece di ririsolverli con Gurobi. È esattamente ciò che serve per
    # valutare UTSP sugli STESSI scenari su cui girano WS/STO/EEV del pool.
    if missing_ids:
        pool = _load_pool_costs()
        if pool:
            agganciati = 0
            for sid in list(missing_ids):
                pr = pool.get(sid)
                if pr and pr.get("STO", {}).get("cost") is not None \
                       and pr.get("EEV", {}).get("cost") is not None:
                    cache[sid] = {"sto_cost": pr["STO"]["cost"],
                                  "eev_cost": pr["EEV"]["cost"]}
                    agganciati += 1
            if agganciati:
                print(f"  STO/EEV: {agganciati} scenari agganciati dal pool "
                      f"(nessun Gurobi)")
                _save_sto_eev_cache(state_key, cache)
                missing_ids = [sid for sid in scenario_ids if sid not in cache]

    if missing_ids:
        print(f"  STO/EEV: {len(missing_ids)} scenari da risolvere con Gurobi "
              f"({len(scenario_ids) - len(missing_ids)} già in cache)")
        for sid in missing_ids:
            sd = results_by_sid[sid]["scenario_dist"]
            r_sto = solve_reservation_tsp(
                nodes, E, I, sd, root, p, C, env,
                fixed_reservations=list(x_sto), output_flag=0,
                model_name=f"val_sto_{sid}",
            )
            r_ev = solve_reservation_tsp(
                nodes, E, I, sd, root, p, C, env,
                fixed_reservations=list(x_ev), output_flag=0,
                model_name=f"val_ev_{sid}",
            )
            sto_tc = r_sto["tour_cost"] if r_sto["tour_cost"] is not None else 0.0
            sto_pc = r_sto["penalty_paid"] if r_sto["penalty_paid"] is not None else 0.0
            eev_tc = r_ev["tour_cost"] if r_ev["tour_cost"] is not None else 0.0
            eev_pc = r_ev["penalty_paid"] if r_ev["penalty_paid"] is not None else 0.0
            cache[sid] = {
                "sto_cost": reservation_sto + sto_tc + sto_pc,
                "eev_cost": reservation_ev + eev_tc + eev_pc,
            }
        _save_sto_eev_cache(state_key, cache)

    sto_costs = {sid: cache[sid]["sto_cost"] for sid in scenario_ids}
    eev_costs = {sid: cache[sid]["eev_cost"] for sid in scenario_ids}
    n = len(scenario_ids)
    STO_val = sum(sto_costs.values()) / n
    EEV_val = sum(eev_costs.values()) / n
    print(f"  STO_val = {STO_val:.4f}  EEV_val = {EEV_val:.4f}  (media su {n} scenari, cache STO/EEV)")

    return {"STO_val": STO_val, "EEV_val": EEV_val, "sto_costs": sto_costs, "eev_costs": eev_costs}


def generate_test_scenario_blocks(
    nodes, E, base_dist, I, frequent_arcs, root, env, p, C,
    scenario_ids, scenario_kwargs,
    dim_istanza_test, n_istanze_test=None,
    base_seed=TEST_SCENARIO_SEED,
):
    """
    Divide scenario_ids in blocchi consecutivi di dim_istanza_test scenari,
    ognuno usato come istanza di test indipendente.
    Scarta l'eventuale blocco finale incompleto.

    # NOTA: base_seed è lo stesso per ogni blocco, così le perturbazioni di
    # uno scenario_id sono riproducibili indipendentemente da come si affetta
    # scenario_ids in blocchi. Per lo stesso motivo il PI di ogni scenario_id
    # è cacheabile indipendentemente dal blocco/DIM/n_istanze: lo risolviamo
    # con Gurobi una volta sola e lo riusiamo per ogni combinazione successiva
    # (anche tra run diversi, anche tra BATCH_X diversi), salvando dopo ogni
    # blocco così un crash a metà non fa perdere il lavoro già fatto.

    Ritorna una lista di tuple (results, block_ids, scenario_probs).
    """
    if dim_istanza_test < 1:
        raise ValueError("dim_istanza_test deve essere >= 1")

    scenario_ids = list(scenario_ids)
    blocks = [
        scenario_ids[i:i + dim_istanza_test]
        for i in range(0, len(scenario_ids), dim_istanza_test)
    ]
    if blocks and len(blocks[-1]) < dim_istanza_test:
        blocks.pop()

    if n_istanze_test is not None:
        if n_istanze_test > len(blocks):
            raise ValueError(
                f"richieste {n_istanze_test} istanze ma solo {len(blocks)} "
                f"disponibili con dim_istanza_test={dim_istanza_test} "
                f"su {len(scenario_ids)} scenario_ids"
            )
        blocks = blocks[:n_istanze_test]

    state_key = _test_scenario_cache_key(I, frequent_arcs, base_seed, scenario_kwargs)
    cache = _load_test_scenario_cache(state_key)

    if TEST_SKIP_PI:
        print("  TEST_SKIP_PI=1: scenari generati SENZA risolvere il PI (solo perturbazioni + scenario_dist)")

    istanze = []
    for block_idx, block_ids in enumerate(blocks):
        missing_ids = [sid for sid in block_ids if sid not in cache]
        if missing_ids:
            print(f"  [blocco {block_idx+1}/{len(blocks)}] {len(missing_ids)} scenari da generare "
                  f"({len(block_ids) - len(missing_ids)} già in cache)")
            new_results, _, _total_random_uses = generate_scenarios(
                missing_ids, nodes, E, base_dist, I, frequent_arcs,
                N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC,
                base_seed,
                root=root, env=env, p=p, C=C,
                solve_pi=not TEST_SKIP_PI,
                **scenario_kwargs,
            )
            cache.update(new_results)
            _save_test_scenario_cache(state_key, cache)

        # Fill-in dei PI mancanti: entry generate in un run precedente con
        # TEST_SKIP_PI=1 hanno exact_free vuoto. Se ora il PI serve, lo
        # risolviamo sulla scenario_dist GIÀ memorizzata: stesso scenario,
        # stesso id, stesso ordine — il quarto modello si aggancia ai tre
        # già calcolati senza rigenerare nulla.
        if not TEST_SKIP_PI:
            to_fill = [
                sid for sid in block_ids
                if cache[sid].get("exact_free", {}).get("length") is None
            ]
            if to_fill:
                print(f"  [blocco {block_idx+1}/{len(blocks)}] fill-in PI per {len(to_fill)} scenari già in cache")
                for sid in to_fill:
                    cache[sid]["exact_free"] = solve_exact_tsp(
                        nodes, E, cache[sid]["scenario_dist"], root, env,
                        fixed_arcs=[], fixed_edges_undir=[], output_flag=0,
                        time_limit=PI_TIME_LIMIT, mip_gap=PI_MIP_GAP,
                    )
                _save_test_scenario_cache(state_key, cache)

        results = {sid: cache[sid] for sid in block_ids}
        scenario_probs = {sid: 1.0 / len(block_ids) for sid in block_ids}
        istanze.append((results, block_ids, scenario_probs))
    return istanze


def _write_utsp_pipeline_stats_file(
    exp_name,
    train_ids, train_probs, train_batch_id,
    train_pre_costs, train_pre_tc, train_pre_pc, train_pre_tours,
    train_post_costs, train_post_tc, train_post_pc, train_post_tours,
    train_PI, train_PI_pren, train_UTSP, x_train,
    test_ids, test_probs,
    test_pre_costs, test_pre_tc, test_pre_pc, test_pre_tours,
    test_post_costs, test_post_tc, test_post_pc, test_post_tours,
    test_PI, test_PI_pren, test_UTSP, x_test,
    history, temperature,
):
    """
    Scrive nel file *_cost_distributions_train_stats.txt la diagnostica coerente
    con la pipeline UTSP: train rete su batch, test unico da 300 scenari.
    """
    def fmt(x):
        if x is None:
            return "N/A"
        try:
            return f"{float(x):.6f}"
        except Exception:
            return str(x)

    pre_stats = _variance_decomposition_by_batch(train_pre_costs, train_ids, train_batch_id or {})
    post_stats = _variance_decomposition_by_batch(train_post_costs, train_ids, train_batch_id or {})

    def add_var_block(lines, title, stats):
        lines += [
            "",
            title,
            f"  n scenari          = {stats['n']}",
            f"  media globale      = {fmt(stats['mean'])}",
            f"  varianza totale    = {fmt(stats['var_total'])}",
            f"  varianza within    = {fmt(stats['var_within'])}",
            f"  varianza between   = {fmt(stats['var_between'])}",
        ]
        total = stats["var_total"]
        if total and np.isfinite(total) and abs(total) > 1e-12:
            lines += [
                f"  quota within       = {100.0 * stats['var_within'] / total:.4f}%",
                f"  quota between      = {100.0 * stats['var_between'] / total:.4f}%",
            ]
        lines += [
            "",
            f"  {'batch':>8} | {'n':>6} | {'mean':>12} | {'var':>12} | {'std':>12} | {'min':>12} | {'max':>12}",
        ]
        for row in stats["batch_stats"]:
            b = row["batch"] if row["batch"] is not None else "NA"
            lines.append(
                f"  {str(b):>8} | {row['n']:>6} | {fmt(row['mean']):>12} | "
                f"{fmt(row['var']):>12} | {fmt(row['std']):>12} | "
                f"{fmt(row['min']):>12} | {fmt(row['max']):>12}"
            )
        return lines

    lines = [
        "=" * 90,
        "DIAGNOSTICA PIPELINE UTSP",
        "=" * 90,
        "",
        "Schema corretto usato dal codice:",
        "  1. TRAIN RETE: scenari UTSP train divisi in batch.",
        "  2. DIAGNOSTICA TRAIN: local search scenario-specifica su tutti gli scenari train.",
        "  3. TEST UTSP: N_VALIDATION_SCENARIOS scenari presi tutti insieme, senza batching decisionale.",
        "  4. TEST PRE-BOOKING: local search senza prenotazioni e senza multe.",
        "  5. DECISIONE TEST: frequenza d'uso degli archi in I e soglia f > p/C.",
        "  6. TEST POST-BOOKING: local search sugli stessi scenari con costi aggiornati.",
        "",
        "Nota: i valori STO/EEV del modello B/Gurobi non vengono confusi con il train UTSP,",
        "perché appartengono al campione e ai parametri dell'esperimento B/Gurobi, non agli scenari della rete.",
        "",
        "TRAIN UTSP",
        f"  scenari train rete = {len(train_ids)}",
        f"  batch size train   = {UTSP_BATCH_SIZE}",
        f"  numero batch train = {len(set((train_batch_id or {}).values())) if train_batch_id else 'NA'}",
        f"  x_train da frequenze train = {sorted(x_train or [])}",
        f"  PI train           = {fmt(train_PI)}",
        f"  PI+pren train      = {fmt(train_PI_pren)}",
        f"  UTSP train post    = {fmt(train_UTSP)}",
        "",
        "TEST UTSP",
        f"  scenari test       = {len(test_ids)}",
        f"  seed test          = {TEST_SCENARIO_SEED}",
        f"  x_test da frequenze test = {sorted(x_test or [])}",
        f"  PI test            = {fmt(test_PI)}",
        f"  PI+pren test       = {fmt(test_PI_pren)}",
        f"  UTSP test post     = {fmt(test_UTSP)}",
        "",
        "TRAINING GNN",
        f"  Epoche             = {UTSP2_EPOCHS}",
        f"  Temperatura T      = {temperature:.6f}",
        f"  Loss iniziale      = {history['loss'][0]:.6f}",
        f"  Loss finale        = {history['loss'][-1]:.6f}",
        f"  Loss minima        = {min(history['loss']):.6f} (ep {int(np.argmin(history['loss'])) + 1})",
    ]

    add_var_block(lines, "VARIANZA TRAIN PRE-BOOKING", pre_stats)
    add_var_block(lines, "VARIANZA TRAIN POST-BOOKING", post_stats)

    lines += [
        "",
        "COSTI TRAIN SCENARIO PER SCENARIO",
        f"  {'scenario':>8} | {'batch':>8} | {'pre_total':>12} | {'post_total':>12} | {'post_perc':>12} | {'post_multa':>12} | tour post",
    ]
    for sid in train_ids:
        b = (train_batch_id or {}).get(sid, "NA")
        lines.append(
            f"  {str(sid):>8} | {str(b):>8} | {fmt(train_pre_costs.get(sid)):>12} | "
            f"{fmt(train_post_costs.get(sid)):>12} | {fmt(train_post_tc.get(sid)):>12} | "
            f"{fmt(train_post_pc.get(sid)):>12} | {train_post_tours.get(sid, [])}"
        )

    lines += [
        "",
        "COSTI TEST SCENARIO PER SCENARIO",
        f"  {'scenario':>8} | {'pre_total':>12} | {'post_total':>12} | {'post_perc':>12} | {'post_multa':>12} | tour post",
    ]
    for sid in test_ids:
        lines.append(
            f"  {str(sid):>8} | {fmt(test_pre_costs.get(sid)):>12} | "
            f"{fmt(test_post_costs.get(sid)):>12} | {fmt(test_post_tc.get(sid)):>12} | "
            f"{fmt(test_post_pc.get(sid)):>12} | {test_post_tours.get(sid, [])}"
        )

    lines.append("=" * 90)

    grafici_dir = os.path.join(OUTPUT_DIR, "grafici")
    os.makedirs(grafici_dir, exist_ok=True)
    stats_file = os.path.join(grafici_dir, f"{exp_name}_cost_distributions_train_stats.txt")
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  → Diagnostica pipeline UTSP scritta in: {stats_file}")




def _write_utsp_test_only_stats_file(
    exp_name, scenario_ids, x_test,
    test_pre_costs, test_pre_tc, test_pre_pc, test_pre_tours,
    test_post_costs, test_post_tc, test_post_pc, test_post_tours,
    PI_test, PI_pren_test, UTSP_LS_test, STO_test, EEV_test,
    gap_ls_sto, gap_ls_eev, gap_ls_pi,
    history, temperature, dist_scale,
    idx=0, n_istanze=1, base_exp_name=None,
):
    def fmt(x):
        if x is None:
            return "N/A"
        try:
            return f"{float(x):.6f}"
        except Exception:
            return str(x)

    # blocco di questa istanza; l'header globale è scritto solo alla prima
    lines = [
        "#" * 90,
        f"# ISTANZA {idx}",
        "#" * 90,
        "=" * 90,
        "DIAGNOSTICA UTSP TEST-ONLY",
        "=" * 90,
        "",
        "Modalità:",
        "  TESI_UTSP_TEST_ONLY=1",
        "  training GNN caricato da OUTPUT_DIR/train/",
        "  nessuna rigenerazione degli scenari train UTSP",
        "",
        f"Scenari test       = {len(scenario_ids)}",
        f"Seed test          = {TEST_SCENARIO_SEED}",
        f"x_test             = {sorted(x_test or [])}",
        f"Temperatura T      = {fmt(temperature)}",
        f"dist_scale         = {fmt(dist_scale)}",
        "",
        "RISULTATI TEST",
        f"  PI test          = {fmt(PI_test)}",
        f"  PI+pren test     = {fmt(PI_pren_test)}",
        f"  UTSP test        = {fmt(UTSP_LS_test)}",
        f"  STO test         = {fmt(STO_test)}",
        f"  EEV test         = {fmt(EEV_test)}",
        f"  Gap UTSP vs STO  = {fmt(gap_ls_sto)}%",
        f"  Gap UTSP vs EEV  = {fmt(gap_ls_eev)}%",
        f"  Gap UTSP vs PI   = {fmt(gap_ls_pi)}%",
        "",
        "TRAINING CARICATO",
        f"  Loss iniziale    = {fmt(history['loss'][0]) if history and 'loss' in history and history['loss'] else 'N/A'}",
        f"  Loss finale      = {fmt(history['loss'][-1]) if history and 'loss' in history and history['loss'] else 'N/A'}",
        "",
        "COSTI TEST SCENARIO PER SCENARIO",
        f"  {'scenario':>8} | {'pre_total':>12} | {'post_total':>12} | {'post_perc':>12} | {'post_multa':>12} | tour post",
    ]

    for sid in scenario_ids:
        lines.append(
            f"  {str(sid):>8} | {fmt(test_pre_costs.get(sid)):>12} | "
            f"{fmt(test_post_costs.get(sid)):>12} | {fmt(test_post_tc.get(sid)):>12} | "
            f"{fmt(test_post_pc.get(sid)):>12} | {test_post_tours.get(sid, [])}"
        )

    lines.append("=" * 90)

    # UN SOLO file per cella (tutte le istanze accodate), sotto grafici/.
    # base_exp_name è il nome della cella SENZA l'indice istanza.
    grafici_dir = os.path.join(OUTPUT_DIR, "grafici")
    os.makedirs(grafici_dir, exist_ok=True)
    base = base_exp_name or exp_name
    stats_file = os.path.join(grafici_dir, f"{base}_test_all_instances.txt")
    mode = "w" if idx == 0 else "a"      # prima istanza sovrascrive, poi accoda
    with open(stats_file, mode, encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if idx == n_istanze - 1:
        print(f"  → Diagnostica test-only ({n_istanze} istanze) in: {stats_file}")


def _run_utsp_test_only_branch(
    nodes, coords, E, root, env, res_B,
    model, xy, dist_scale, temperature, device, base_dist,
    scenario_kwargs=None, exp_name="espB_UTSP_LS", history=None,
):
    """
    Testa un modello già allenato su una o più istanze di
    DIM_ISTANZA_TEST scenari (N_ISTANZE_TEST istanze in totale), senza
    rieseguire training. Ogni istanza usa PI cacheato per scenario_id
    (vedi generate_test_scenario_blocks) e uno STO/EEV risolto ex novo
    (non cacheabile tra istanze/DIM diversi: dipende dal blocco intero).
    """
    print("\n" + "=" * 70)
    print("ESPERIMENTO B — UTSP TEST-ONLY DA TRAINING SALVATO")

    scenario_kwargs = dict(scenario_kwargs or {})
    I = res_B["I"]
    p = res_B["p"]
    C = res_B["C"]
    frequent_arcs = res_B["frequent_arcs"]

    print(f"  scenari test UTSP disponibili = {len(TEST_SCENARIO_IDS_UTSP)}  seed={TEST_SCENARIO_SEED}")
    print(f"  dim_istanza_test = {DIM_ISTANZA_TEST}  n_istanze_test = {N_ISTANZE_TEST}")

    istanze_test = generate_test_scenario_blocks(
        nodes, E, base_dist, I, frequent_arcs, root, env, p, C,
        scenario_ids=TEST_SCENARIO_IDS_UTSP,
        scenario_kwargs=scenario_kwargs,
        dim_istanza_test=DIM_ISTANZA_TEST,
        n_istanze_test=N_ISTANZE_TEST,
    )
    print(f"  istanze di test generate = {len(istanze_test)}")

    istanza_metrics = []
    istanza_outputs = []

    for idx, (results_test, scenario_ids_test, scenario_probs_test) in enumerate(istanze_test):
        multi = len(istanze_test) > 1
        label_suffix = f"test_only_i{idx}" if multi else "test_only"
        exp_name_i = f"{exp_name}_{label_suffix}" if multi else f"{exp_name}_test_only"

        H_test = _build_heatmaps_for_scenarios(
            model, xy, nodes, results_test, scenario_ids_test, dist_scale, temperature, device
        )

        print(f"\n  [Istanza {idx}] pre-booking: LS senza prenotazioni e senza multe ...")
        (test_pre_costs, test_pre_tc, test_pre_pc,
         test_pre_tours, test_pre_solutions, UTSP_test_pre) = _run_ls_on_scenarios(
            model, xy, nodes, root, I, p, C,
            results_test, scenario_ids_test, scenario_probs_test,
            H_list_precomputed=H_test,
            dist_scale=dist_scale, temperature=temperature,
            device=device, x_ls=[], label=f"{label_suffix}_pre_booking",
            apply_penalties=False,
        )

        x_test = _compute_bookings_from_tours(test_pre_tours, scenario_ids_test, nodes, I, p, C)
        reserv_test = sum(get_edge_value(p, i, j) for (i, j) in x_test)
        print(f"  [Istanza {idx}] costo prenotazione deciso: {reserv_test:.4f}")

        print(f"  [Istanza {idx}] post-booking: LS sugli stessi scenari con costi aggiornati ...")
        (test_post_costs, test_post_tc, test_post_pc,
         test_post_tours, test_post_solutions, UTSP_LS_test) = _run_ls_on_scenarios(
            model, xy, nodes, root, I, p, C,
            results_test, scenario_ids_test, scenario_probs_test,
            H_list_precomputed=H_test,
            dist_scale=dist_scale, temperature=temperature,
            device=device, x_ls=x_test, label=f"{label_suffix}_post_booking",
            apply_penalties=True,
        )

        pi_test_d = _compute_exact_free_costs_from_results(results_test, scenario_ids_test)
        PI_test = _scenario_mean(pi_test_d, scenario_ids_test, scenario_probs_test)
        pi_pren_test_d = _compute_pi_with_booking_costs_local(results_test, scenario_ids_test, I, p)
        PI_pren_test = _scenario_mean(pi_pren_test_d, scenario_ids_test, scenario_probs_test)

        # NOTA: STO/EEV non sono cacheabili tra istanze o tra DIM diversi,
        # a differenza del PI: dipendono dal blocco intero di scenari
        # (first-stage x comune a tutto il blocco), quindi ogni istanza va
        # risolta con Gurobi ex novo. Sono il costo dominante dello sweep.
        # NOTA: STO/EEV sono cacheati per scenario_id esattamente come il PI
        # (vedi _validate_policies_cached): x_sto/x_ev sono policy già fisse,
        # quindi il costo di ogni scenario non dipende dal blocco. Una volta
        # risolti per un dato scenario_id, restano validi per qualunque
        # combinazione DIM/n_istanze che lo includa.
        try:
            test_bench = _validate_policies_cached(
                nodes, E, I, p, C, root, env,
                res_B["x_used_sto"], res_B["x_ev"],
                results_test, scenario_ids_test,
                base_seed=TEST_SCENARIO_SEED,
                frequent_arcs=frequent_arcs,
                scenario_kwargs=scenario_kwargs,
            )
            STO_test = test_bench.get("STO_val", float("nan"))
            EEV_test = test_bench.get("EEV_val", float("nan"))

            if "eev_costs" in test_bench and "sto_costs" in test_bench:
                plot_cost_distributions(
                    test_bench["eev_costs"], test_bench["sto_costs"], test_post_costs,
                    exp_name_i, "test_only",
                )
        except Exception as exc:
            print(f"  Attenzione: _validate_policies_cached istanza {idx} non riuscita: {exc}")
            test_bench = {}
            STO_test = float("nan")
            EEV_test = float("nan")

        gap_ls_sto = (UTSP_LS_test - STO_test) / abs(STO_test) * 100 if STO_test and np.isfinite(STO_test) else float("nan")
        gap_ls_eev = (UTSP_LS_test - EEV_test) / abs(EEV_test) * 100 if EEV_test and np.isfinite(EEV_test) else float("nan")
        gap_ls_pi = (UTSP_LS_test - PI_test) / abs(PI_test) * 100 if PI_test and np.isfinite(PI_test) else float("nan")

        print(f"\n  [Istanza {idx}] RIEPILOGO ({len(scenario_ids_test)} scenari)")
        print(f"    PI={PI_test:.4f} PI+pren={PI_pren_test:.4f} "
              f"UTSP={UTSP_LS_test:.4f} STO={STO_test:.4f} EEV={EEV_test:.4f}")
        print(f"    Gap vs STO={gap_ls_sto:+.2f}% vs EEV={gap_ls_eev:+.2f}% vs PI={gap_ls_pi:+.2f}%")

        _write_utsp_test_only_stats_file(
            exp_name=exp_name_i,
            scenario_ids=scenario_ids_test,
            x_test=x_test,
            test_pre_costs=test_pre_costs,
            test_pre_tc=test_pre_tc,
            test_pre_pc=test_pre_pc,
            test_pre_tours=test_pre_tours,
            test_post_costs=test_post_costs,
            test_post_tc=test_post_tc,
            test_post_pc=test_post_pc,
            test_post_tours=test_post_tours,
            PI_test=PI_test,
            PI_pren_test=PI_pren_test,
            UTSP_LS_test=UTSP_LS_test,
            STO_test=STO_test,
            EEV_test=EEV_test,
            gap_ls_sto=gap_ls_sto,
            gap_ls_eev=gap_ls_eev,
            gap_ls_pi=gap_ls_pi,
            history=history or {"loss": []},
            temperature=temperature,
            dist_scale=dist_scale,
            idx=idx,
            n_istanze=len(istanze_test),
            base_exp_name=f"{exp_name}_test_only",
        )

        istanza_metrics.append({
            "idx": idx, "n_scenari": len(scenario_ids_test),
            "UTSP_LS_test": UTSP_LS_test, "PI_test": PI_test, "PI_pren_test": PI_pren_test,
            "STO_test": STO_test, "EEV_test": EEV_test,
            "gap_ls_sto": gap_ls_sto, "gap_ls_eev": gap_ls_eev, "gap_ls_pi": gap_ls_pi,
        })
        istanza_outputs.append({
            "results_test": results_test, "scenario_ids_test": scenario_ids_test,
            "x_test": x_test, "test_bench": test_bench,
            "costs_test": test_post_costs, "tc_test": test_post_tc, "pc_test": test_post_pc,
            "tours_test": test_post_tours, "solutions_test": test_post_solutions,
            "costs_test_pre": test_pre_costs, "tc_test_pre": test_pre_tc, "pc_test_pre": test_pre_pc,
            "tours_test_pre": test_pre_tours, "UTSP_test_pre": UTSP_test_pre,
        })

    agg_test = _aggregate_test_instances(exp_name, istanza_metrics)
    print("\n" + "─" * 65)
    print(f"RIEPILOGO AGGREGATO SU {agg_test['n_istanze']} ISTANZE")
    for k in ("UTSP_LS_test", "PI_test", "STO_test", "EEV_test", "gap_ls_sto", "gap_ls_eev", "gap_ls_pi"):
        print(f"  {k}: media={agg_test[f'{k}_mean']:.4f}  std={agg_test[f'{k}_std']:.4f}")
    print("─" * 65)

    agg_path = os.path.join(OUTPUT_DIR, "grafici", f"{exp_name}_test_only_aggregate.txt")
    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
    with open(agg_path, "w", encoding="utf-8") as f:
        f.write(f"n_istanze={agg_test['n_istanze']}  dim_istanza_test={DIM_ISTANZA_TEST}\n")
        for k in ("UTSP_LS_test", "PI_test", "PI_pren_test", "STO_test", "EEV_test",
                   "gap_ls_sto", "gap_ls_eev", "gap_ls_pi"):
            f.write(f"{k}_mean={agg_test[f'{k}_mean']:.6f}  {k}_std={agg_test[f'{k}_std']:.6f}\n")

    # Retrocompatibilità: chi si aspetta un solo x_test/UTSP_LS_test riceve l'ultima istanza.
    last = istanza_outputs[-1]
    last_m = istanza_metrics[-1]

    return {
        "x_test": last["x_test"],
        "results_test": last["results_test"],
        "scenario_ids_test": last["scenario_ids_test"],
        "costs_test_pre": last["costs_test_pre"],
        "costs_test_post": last["costs_test"],
        "tours_test_post": last["tours_test"],
        "UTSP_test_pre": last["UTSP_test_pre"],
        "UTSP_LS_test": last_m["UTSP_LS_test"],
        "PI_test": last_m["PI_test"],
        "PI_pren_test": last_m["PI_pren_test"],
        "STO_test": last_m["STO_test"],
        "EEV_test": last_m["EEV_test"],
        "gap_ls_sto": last_m["gap_ls_sto"],
        "gap_ls_eev": last_m["gap_ls_eev"],
        "gap_ls_pi": last_m["gap_ls_pi"],
        "istanze_test": istanza_metrics,
        "istanze_test_output": istanza_outputs,
        "aggregato_test": agg_test,
    }



def _append_test_instance_block(exp_name, idx, n_istanze, x_test,
                                scenario_ids, test_pre_costs, test_post_costs,
                                test_post_tc, test_post_pc, test_post_tours,
                                PI_test, UTSP_LS_test, STO_test, EEV_test,
                                gap_ls_sto, gap_ls_eev, gap_ls_pi):
    """Scrive il blocco TEST di UNA istanza, accodandolo a un UNICO file per
    cella (grafici/<exp>_test_all_instances.txt). Sostituisce i ~300 file
    per-istanza: un solo file, leggibile in un colpo da full_report.
    Formato allineato a FIELD_RE / X_RE / SCEN_RE di full_report: NON cambiare
    le etichette 'UTSP test =', 'x_test =', né lo schema colonne."""
    def fmt(x):
        if x is None:
            return "N/A"
        try:
            return f"{float(x):.6f}"
        except Exception:
            return str(x)

    lines = [
        "#" * 90,
        f"# ISTANZA {idx}",
        "#" * 90,
        f"x_test = {sorted(x_test or [])}",
        "RISULTATI TEST",
        f"  PI test   = {fmt(PI_test)}",
        f"  UTSP test = {fmt(UTSP_LS_test)}",
        f"  STO test  = {fmt(STO_test)}",
        f"  EEV test  = {fmt(EEV_test)}",
        f"  Gap UTSP vs STO = {fmt(gap_ls_sto)}%",
        f"  Gap UTSP vs EEV = {fmt(gap_ls_eev)}%",
        f"  Gap UTSP vs PI  = {fmt(gap_ls_pi)}%",
        "COSTI TEST SCENARIO PER SCENARIO",
        f"  {'scenario':>8} | {'pre_total':>12} | {'post_total':>12} | "
        f"{'post_perc':>12} | {'post_multa':>12} | tour post",
    ]
    for sid in scenario_ids:
        lines.append(
            f"  {str(sid):>8} | {fmt(test_pre_costs.get(sid)):>12} | "
            f"{fmt(test_post_costs.get(sid)):>12} | {fmt(test_post_tc.get(sid)):>12} | "
            f"{fmt(test_post_pc.get(sid)):>12} | {test_post_tours.get(sid, [])}"
        )

    grafici_dir = os.path.join(OUTPUT_DIR, "grafici")
    os.makedirs(grafici_dir, exist_ok=True)
    stats_file = os.path.join(grafici_dir, f"{exp_name}_test_all_instances.txt")
    mode = "w" if idx == 0 else "a"
    with open(stats_file, mode, encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if idx == n_istanze - 1:
        print(f"  → Diagnostica test ({n_istanze} istanze) in: {stats_file}")


def _run_local_search_branch(
    nodes, coords, E, root, env, res_B,
    results, scenario_ids, scenario_probs, H_list,
    PI_train, STO_train, EEV_train,
    history, temperature,
    model, xy, dist_scale, device, base_dist,
    scenario_kwargs=None, exp_name="espB_UTSP_LS",
    results_B=None, scenario_ids_B=None, scenario_probs_B=None,
    train_batch_id=None,
):
    print("\n" + "=" * 70)
    print("ESPERIMENTO B — UTSP HEATMAP + LOCAL SEARCH")

    scenario_kwargs = dict(scenario_kwargs or {})
    I = res_B["I"]
    p = res_B["p"]
    C = res_B["C"]
    frequent_arcs = res_B["frequent_arcs"]

    results_B = results if results_B is None else results_B
    scenario_ids_B = scenario_ids if scenario_ids_B is None else scenario_ids_B
    scenario_probs_B = scenario_probs if scenario_probs_B is None else scenario_probs_B
    train_batch_id = train_batch_id or {}

    print("\n  TRAIN UTSP: diagnostica su tutti gli scenari usati dalla rete")
    print(f"  scenari train UTSP = {len(scenario_ids)}")
    print(f"  batch size rete    = {UTSP_BATCH_SIZE}")

    # Heatmap dello scenario anche in train. Se H_list viene passato ed è allineato, lo uso;
    # altrimenti ricalcolo in modo sicuro sugli scenari UTSP train.
    if H_list is not None and len(H_list) == len(scenario_ids):
        H_train = H_list
    else:
        H_train = _build_heatmaps_for_scenarios(
            model, xy, nodes, results, scenario_ids, dist_scale, temperature, device
        )

    print("\n  Train pre-booking: LS senza prenotazioni e senza multe ...")
    (train_pre_costs, train_pre_tc, train_pre_pc,
     train_pre_tours, train_pre_solutions, UTSP_train_pre) = _run_ls_on_scenarios(
        model, xy, nodes, root, I, p, C,
        results, scenario_ids, scenario_probs,
        H_list_precomputed=H_train,
        dist_scale=dist_scale, temperature=temperature,
        device=device, x_ls=[], label="train_pre_booking",
        apply_penalties=False,
    )

    x_train = _compute_bookings_from_tours(train_pre_tours, scenario_ids, nodes, I, p, C)
    reserv_train = sum(get_edge_value(p, i, j) for (i, j) in x_train)
    print(f"  Costo prenotazione deciso da train: {reserv_train:.4f}")

    print("\n  Train post-booking: LS con prenotazioni e multe ...")
    (train_post_costs, train_post_tc, train_post_pc,
     train_post_tours, train_post_solutions, UTSP_LS_train) = _run_ls_on_scenarios(
        model, xy, nodes, root, I, p, C,
        results, scenario_ids, scenario_probs,
        H_list_precomputed=H_train,
        dist_scale=dist_scale, temperature=temperature,
        device=device, x_ls=x_train, label="train_post_booking",
        apply_penalties=True,
    )

    pi_train_d = _compute_exact_free_costs_from_results(results, scenario_ids)
    PI_train_eval = _scenario_mean(pi_train_d, scenario_ids, scenario_probs)
    pi_pren_train_d = _compute_pi_with_booking_costs_local(results, scenario_ids, I, p)
    PI_pren_train = _scenario_mean(pi_pren_train_d, scenario_ids, scenario_probs)
    
    
    print("\n" + "─" * 65)
    print("RIEPILOGO UTSP — TRAIN")
    print(f"  x_train = {sorted(x_train)}")
    print(f"  TRAIN UTSP ({len(scenario_ids)} scenari, batch size {UTSP_BATCH_SIZE})")
    print(f"    PI train       = {PI_train_eval:.4f}")
    print(f"    PI+pren train  = {PI_pren_train:.4f}")
    print(f"    UTSP train     = {UTSP_LS_train:.4f}")
    
    print("\n  TEST UTSP: scenari presi dal test set comune")
    print(f"  scenari test UTSP disponibili = {len(TEST_SCENARIO_IDS_UTSP)}")

    istanze_test = generate_test_scenario_blocks(
        nodes, E, base_dist, I, frequent_arcs, root, env, p, C,
        scenario_ids=TEST_SCENARIO_IDS_UTSP,
        scenario_kwargs=scenario_kwargs,
        dim_istanza_test=DIM_ISTANZA_TEST,
        n_istanze_test=N_ISTANZE_TEST,
    )
    print(f"  istanze di test generate = {len(istanze_test)} (dim={DIM_ISTANZA_TEST})")

    istanza_metrics = []
    istanza_outputs = []

    for idx, (results_test, scenario_ids_test, scenario_probs_test) in enumerate(istanze_test):
        multi = len(istanze_test) > 1
        label_suffix = f"test_i{idx}" if multi else "test"
        exp_name_i = f"{exp_name}_{label_suffix}" if multi else exp_name

        H_test = _build_heatmaps_for_scenarios(
            model, xy, nodes, results_test, scenario_ids_test, dist_scale, temperature, device
        )

        print(f"\n  [Istanza {idx}] pre-booking: LS senza prenotazioni e senza multe ...")
        (test_pre_costs, test_pre_tc, test_pre_pc,
         test_pre_tours, test_pre_solutions, UTSP_test_pre) = _run_ls_on_scenarios(
            model, xy, nodes, root, I, p, C,
            results_test, scenario_ids_test, scenario_probs_test,
            H_list_precomputed=H_test,
            dist_scale=dist_scale, temperature=temperature,
            device=device, x_ls=[], label=f"{label_suffix}_pre_booking",
            apply_penalties=False,
        )

        x_test = _compute_bookings_from_tours(test_pre_tours, scenario_ids_test, nodes, I, p, C)
        reserv_test = sum(get_edge_value(p, i, j) for (i, j) in x_test)
        print(f"  [Istanza {idx}] costo prenotazione deciso: {reserv_test:.4f}")

        print(f"  [Istanza {idx}] post-booking: LS con prenotazioni e multe ...")
        (test_post_costs, test_post_tc, test_post_pc,
         test_post_tours, test_post_solutions, UTSP_LS_test) = _run_ls_on_scenarios(
            model, xy, nodes, root, I, p, C,
            results_test, scenario_ids_test, scenario_probs_test,
            H_list_precomputed=H_test,
            dist_scale=dist_scale, temperature=temperature,
            device=device, x_ls=x_test, label=f"{label_suffix}_post_booking",
            apply_penalties=True,
        )

        pi_test_d = _compute_exact_free_costs_from_results(results_test, scenario_ids_test)
        PI_test = _scenario_mean(pi_test_d, scenario_ids_test, scenario_probs_test)
        pi_pren_test_d = _compute_pi_with_booking_costs_local(results_test, scenario_ids_test, I, p)
        PI_pren_test = _scenario_mean(pi_pren_test_d, scenario_ids_test, scenario_probs_test)

        try:
            test_bench = validate_policies(
                nodes, E, base_dist, root, env, I, p, C,
                res_B["x_used_sto"], res_B["x_ev"],
                frequent_arcs, len(scenario_ids_test),
                N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC,
                exp_name=exp_name_i,
                scenario_ids_val=scenario_ids_test,
                validation_seed=TEST_SCENARIO_SEED,
                **scenario_kwargs,
            )
            STO_test = test_bench.get("STO_val", float("nan"))
            EEV_test = test_bench.get("EEV_val", float("nan"))
        except Exception as exc:
            print(f"  Attenzione: validate_policies istanza {idx} non riuscita: {exc}")
            test_bench = {}
            STO_test = float("nan")
            EEV_test = float("nan")

        gap_ls_sto = (UTSP_LS_test - STO_test) / abs(STO_test) * 100 if STO_test and np.isfinite(STO_test) else float("nan")
        gap_ls_eev = (UTSP_LS_test - EEV_test) / abs(EEV_test) * 100 if EEV_test and np.isfinite(EEV_test) else float("nan")
        gap_ls_pi = (UTSP_LS_test - PI_test) / abs(PI_test) * 100 if PI_test and np.isfinite(PI_test) else float("nan")

        print(f"\n  [Istanza {idx}] RIEPILOGO ({len(scenario_ids_test)} scenari)")
        print(f"    PI={PI_test:.4f} PI+pren={PI_pren_test:.4f} "
              f"UTSP={UTSP_LS_test:.4f} STO={STO_test:.4f} EEV={EEV_test:.4f}")
        print(f"    Gap vs STO={gap_ls_sto:+.2f}% vs EEV={gap_ls_eev:+.2f}% vs PI={gap_ls_pi:+.2f}%")

        # UN SOLO blocco leggero per istanza, accodato al file unico della cella.
        # (Le diagnostiche pesanti train/pipeline si scrivono UNA volta, fuori dal
        #  loop: erano identiche a ogni istanza e duplicavano ~50MB per cella.)
        _append_test_instance_block(
            exp_name=exp_name, idx=idx, n_istanze=len(istanze_test),
            x_test=x_test, scenario_ids=scenario_ids_test,
            test_pre_costs=test_pre_costs, test_post_costs=test_post_costs,
            test_post_tc=test_post_tc, test_post_pc=test_post_pc,
            test_post_tours=test_post_tours,
            PI_test=PI_test, UTSP_LS_test=UTSP_LS_test,
            STO_test=STO_test, EEV_test=EEV_test,
            gap_ls_sto=gap_ls_sto, gap_ls_eev=gap_ls_eev, gap_ls_pi=gap_ls_pi,
        )

        istanza_metrics.append({
            "idx": idx, "n_scenari": len(scenario_ids_test),
            "UTSP_LS_test": UTSP_LS_test, "PI_test": PI_test, "PI_pren_test": PI_pren_test,
            "STO_test": STO_test, "EEV_test": EEV_test,
            "gap_ls_sto": gap_ls_sto, "gap_ls_eev": gap_ls_eev, "gap_ls_pi": gap_ls_pi,
        })
        istanza_outputs.append({
            "results_test": results_test, "scenario_ids_test": scenario_ids_test,
            "x_test": x_test, "test_bench": test_bench,
            "costs_test": test_post_costs, "tc_test": test_post_tc, "pc_test": test_post_pc,
            "tours_test": test_post_tours, "solutions_test": test_post_solutions,
            "costs_test_pre": test_pre_costs, "tc_test_pre": test_pre_tc, "pc_test_pre": test_pre_pc,
            "tours_test_pre": test_pre_tours,
        })

    agg_test = _aggregate_test_instances(exp_name, istanza_metrics)

    # NOTA: per i grafici comparativi sul campione B e per i valori scalari
    # "piatti" restituiti sotto (retrocompatibilità con chi si aspetta un solo
    # x_test/UTSP_LS_test), uso l'ultima istanza generata. Con una sola istanza
    # (comportamento di default) coincide esattamente con prima.
    results_test = istanza_outputs[-1]["results_test"]
    scenario_ids_test = istanza_outputs[-1]["scenario_ids_test"]
    x_test = istanza_outputs[-1]["x_test"]
    test_post_costs = istanza_outputs[-1]["costs_test"]
    test_post_tc = istanza_outputs[-1]["tc_test"]
    test_post_pc = istanza_outputs[-1]["pc_test"]
    test_post_tours = istanza_outputs[-1]["tours_test"]
    test_post_solutions = istanza_outputs[-1]["solutions_test"]
    test_pre_costs = istanza_outputs[-1]["costs_test_pre"]
    test_pre_tc = istanza_outputs[-1]["tc_test_pre"]
    test_pre_pc = istanza_outputs[-1]["pc_test_pre"]
    test_pre_tours = istanza_outputs[-1]["tours_test_pre"]
    test_bench = istanza_outputs[-1]["test_bench"]
    PI_test = istanza_metrics[-1]["PI_test"]
    PI_pren_test = istanza_metrics[-1]["PI_pren_test"]
    STO_test = istanza_metrics[-1]["STO_test"]
    EEV_test = istanza_metrics[-1]["EEV_test"]
    UTSP_LS_test = istanza_metrics[-1]["UTSP_LS_test"]
    gap_ls_sto = istanza_metrics[-1]["gap_ls_sto"]
    gap_ls_eev = istanza_metrics[-1]["gap_ls_eev"]
    gap_ls_pi = istanza_metrics[-1]["gap_ls_pi"]

    plot_cost_distributions(
        res_B["eev_costs"], res_B["stoch_costs"], train_post_costs,
        exp_name, "train",
    )
    _append_utsp_train_cost_diagnostics(
        exp_name, scenario_ids, train_post_costs, train_post_tc, train_post_pc,
        train_post_tours, train_batch_id=train_batch_id, x_ls=x_train, split_label="train",
    )

    if "eev_costs" in test_bench and "sto_costs" in test_bench:
        plot_cost_distributions(
            test_bench["eev_costs"], test_bench["sto_costs"], test_post_costs,
            exp_name, "test",
        )

    # Grafici comparativi con STO/EEV solo sul campione B originale, perché quelle
    # soluzioni sono state prodotte dall'esperimento B e non dai 3000 scenari UTSP.
    print("\n  Grafici comparativi sul campione B originale ...")
    (costs_B_plot, tc_B_plot, pc_B_plot,
     tours_B_plot, solutions_B_plot, UTSP_LS_B_plot) = _run_ls_on_scenarios(
        model, xy, nodes, root, I, p, C,
        results_B, scenario_ids_B, scenario_probs_B,
        H_list_precomputed=None,
        dist_scale=dist_scale, temperature=temperature,
        device=device, x_ls=x_test, label="B_original_plot",
        apply_penalties=True,
    )

    genera_grafici_utsp(
        exp_name=exp_name,
        nodes=nodes,
        coords=coords,
        scenario_ids=scenario_ids_B,
        results=results_B,
        eev_costs=res_B["eev_costs"],
        eev_solutions=res_B["eev_solutions"],
        stoch_costs=res_B["stoch_costs"],
        stoch_solutions=res_B["stoch_solutions"],
        utsp_costs=costs_B_plot,
        utsp_solutions=solutions_B_plot,
        x_ev=res_B["x_ev"],
        x_sto=res_B["x_used_sto"],
        x_utsp=x_test,
        utsp_label="UTSP local search",
        save=True,
    )

    return {
        "x_train": x_train,
        "x_test": x_test,
        "costs_train_pre": train_pre_costs,
        "tc_train_pre": train_pre_tc,
        "pc_train_pre": train_pre_pc,
        "tours_train_pre": train_pre_tours,
        "costs_train": train_post_costs,
        "tc_train": train_post_tc,
        "pc_train": train_post_pc,
        "tours_train": train_post_tours,
        "solutions_train": train_post_solutions,
        "UTSP_LS_train": UTSP_LS_train,
        "results_test": results_test,
        "scenario_ids_test": scenario_ids_test,
        "costs_test_pre": test_pre_costs,
        "tc_test_pre": test_pre_tc,
        "pc_test_pre": test_pre_pc,
        "tours_test_pre": test_pre_tours,
        "costs_test": test_post_costs,
        "tc_test": test_post_tc,
        "pc_test": test_post_pc,
        "tours_test": test_post_tours,
        "solutions_test": test_post_solutions,
        "UTSP_LS_test": UTSP_LS_test,
        "PI_train": PI_train_eval,
        "PI_pren_train": PI_pren_train,
        "PI_test": PI_test,
        "PI_pren_test": PI_pren_test,
        "STO_test": STO_test,
        "EEV_test": EEV_test,
        "gap_ls_sto": gap_ls_sto,
        "gap_ls_eev": gap_ls_eev,
        "gap_ls_pi": gap_ls_pi,
        "costs_B_plot": costs_B_plot,
        "solutions_B_plot": solutions_B_plot,
        "UTSP_LS_B_plot": UTSP_LS_B_plot,
        "istanze_test": istanza_metrics,     # metriche scalari per istanza
        "istanze_test_output": istanza_outputs,  # dettaglio completo per istanza
        "aggregato_test": agg_test,          # media/std su tutte le istanze
    }

def _save_utsp_ls_summary(
    exp_name, scenario_ids, results,
    x_ls, costs_train, tc_train, pc_train, tours_train,
    UTSP_LS_train, UTSP_LS_val,
    PI_train, STO_train, EEV_train,
    PI_pren_train,
    PI_val, STO_val, EEV_val,
    PI_pren_val,
    gap_ls_sto, gap_ls_eev, gap_ls_pi,
    history, temperature,
):
    """
    Riepilogo sintetico. Per compatibilità mantengo i nomi degli argomenti storici,
    ma nel nuovo flusso `scenario_ids/costs_train` rappresentano il TEST UTSP
    post-booking, non il training della rete.
    """
    def fmt(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return "N/A" if x is None else str(x)

    lines = [
        "=" * 65,
        "RIEPILOGO UTSP PIPELINE",
        "=" * 65,
        "",
        f"Prenotazioni finali del test x_test : {sorted(x_ls)}",
        "",
        "TRAIN UTSP / RETE",
        f"  PI train UTSP       = {fmt(PI_train)}",
        f"  PI+pren train UTSP  = {fmt(PI_pren_train)}",
        f"  UTSP train post     = {fmt(UTSP_LS_train)}",
        "",
        "BENCHMARK GUROBI/B ORIGINALE",
        f"  STO train B         = {fmt(STO_train)}",
        f"  EEV train B         = {fmt(EEV_train)}",
        "  Nota: questi non sono calcolati sui 3000 scenari UTSP della rete.",
        "",
        f"TEST UTSP ({len(scenario_ids)} scenari, blocco unico)",
        f"  {'Scen':>4} | {'PI':>10} | {'UTSP_post':>10} [perc, multa] | tour",
    ]

    for sid in scenario_ids:
        exact = results.get(sid, {}).get("exact_free", {}) if isinstance(results.get(sid, {}), dict) else {}
        pi = exact.get("length", exact.get("cost", None))
        lines.append(
            f"  {sid:>4} | {fmt(pi):>10} | {fmt(costs_train.get(sid)):>10} "
            f"[{fmt(tc_train.get(sid))}, {fmt(pc_train.get(sid))}] | "
            f"{tours_train.get(sid, [])}"
        )

    lines += [
        "",
        f"  UTSP test      = {fmt(UTSP_LS_val)}",
        f"  PI test        = {fmt(PI_val)}",
        f"  PI+pren test   = {fmt(PI_pren_val)}",
        f"  STO test       = {fmt(STO_val)}",
        f"  EEV test       = {fmt(EEV_val)}",
        "",
        f"  Gap UTSP test vs STO = {gap_ls_sto:+.4f}%",
        f"  Gap UTSP test vs EEV = {gap_ls_eev:+.4f}%",
        f"  Gap UTSP test vs PI  = {gap_ls_pi:+.4f}%",
        "",
        "TRAINING GNN",
        f"  Epoche         = {UTSP2_EPOCHS}",
        f"  Temperatura T  = {temperature:.6f}",
        f"  Loss iniziale  = {history['loss'][0]:.5f}",
        f"  Loss finale    = {history['loss'][-1]:.5f}",
        f"  Loss minima    = {min(history['loss']):.5f} "
        f"(ep {int(np.argmin(history['loss'])) + 1})",
        "=" * 65,
    ]

    text = "\n".join(lines)
    print("\n" + text)
    fname = os.path.join(OUTPUT_DIR, f"risultati_{exp_name}.txt")
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\n  → Salvato: {fname}")
