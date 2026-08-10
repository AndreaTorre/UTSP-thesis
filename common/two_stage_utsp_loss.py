# -*- coding: utf-8 -*-
"""Loss surrogata a due stadi + costruzione heatmap + decodifica booking.

Rispetto alla versione precedente: identica nei valori (verificato a ~1e-10),
ma booking/penalty e la heatmap sono vettorializzati invece che a loop Python
con `.item()` (che sincronizzava la GPU a ogni cella)."""

import torch
from tsp_utils import canon_edge, get_edge_value

# ── Pesi di default della loss ───────────────────────────────────────
DEFAULT_LAMBDA1      = 10.0   # row-wise
DEFAULT_LAMBDA2      = 10.0   # no-self-loop
DEFAULT_LAMBDA_E     = 1.0    # consistency (opzionale, off di default)
DEFAULT_LAMBDA_D     = 1.0    # asymmetry
DEFAULT_ALPHA        = 5.0    # saturazione booking/penalty
DEFAULT_LAMBDA_B_DIV = 1.0    # lambda_b = |Omega| / divisore


# ── Tensori statici (una volta sola, poi riusati a ogni forward) ──────

def build_I_tensors(I, nodes, p, C, device):
    """Maschere e costi per gli archi in I.
    I_mask: True su (i,j) e (j,i) per ogni {i,j}∈I. p_mat/C_mat: costi (0 fuori da I)."""
    n   = len(nodes)
    idx = {v: k for k, v in enumerate(nodes)}
    I_mask = torch.zeros(n, n, dtype=torch.bool,    device=device)
    p_mat  = torch.zeros(n, n, dtype=torch.float32, device=device)
    C_mat  = torch.zeros(n, n, dtype=torch.float32, device=device)

    for edge in I:
        i, j   = canon_edge(*edge)
        ii, jj = idx[i], idx[j]
        p_val  = p.get((i, j), p.get((j, i), 0.0))
        C_val  = C.get((i, j), C.get((j, i), 0.0))
        for a, b in ((ii, jj), (jj, ii)):          # I è non orientato
            I_mask[a, b] = True
            p_mat[a, b]  = p_val
            C_mat[a, b]  = C_val

    return I_mask, p_mat, C_mat, idx


def build_dist_tensor(scenario_dist, nodes, device):
    """Matrice (n,n) delle distanze di scenario (diagonale nulla).
    NOTA: unica implementazione — utsp.py e local_search.py la importano invece
    di ripetere lo stesso doppio for."""
    D = [[scenario_dist[i][j] if i != j else 0.0 for j in nodes] for i in nodes]
    return torch.tensor(D, dtype=torch.float32, device=device)


def normalize_dist_tensor(D, mode="mean_positive"):
    """Divide D per la scala scelta (media/mediana delle distanze positive fuori diagonale)."""
    if mode == "none":
        return D, 1.0

    n = D.size(-1)
    mask = ~torch.eye(n, dtype=torch.bool, device=D.device)
    if D.dim() == 3:
        mask = mask.unsqueeze(0).expand_as(D)
    vals = D[mask]
    vals = vals[vals > 0]
    if vals.numel() == 0:
        return D, 1.0

    scale = float(vals.mean()) if mode == "mean_positive" else float(torch.median(vals))
    scale = max(scale, 1e-9)
    return D / scale, scale


# ── Heatmap ──────────────────────────────────────────────────────────

def compute_heatmap(T_omega):
    """H = T · Z · Tᵀ con Z shift ciclico (Sylvester).
    Equivalente al loop sulle posizioni, in una sola bmm (verificato a 1e-15)."""
    return torch.bmm(T_omega, torch.roll(T_omega.transpose(1, 2), -1, 1))


def compute_H_bar(H_list, scenario_probs):
    """H̄ = Σ_ω p_ω H^ω."""
    H_bar = torch.zeros_like(H_list[0])
    for H_omega, p_w in zip(H_list, scenario_probs):
        H_bar = H_bar + p_w * H_omega
    return H_bar


# ── Termini della loss ───────────────────────────────────────────────
# Row-wise e no-self-loop sono identici al paper; distance è per-scenario;
# booking/asymmetry/penalty sono l'estensione a due stadi.

def _loss_row_wise(T_list, scenario_probs):
    loss = torch.tensor(0.0, device=T_list[0].device)
    for T_omega, p_w in zip(T_list, scenario_probs):
        row_sums = T_omega.sum(dim=2)
        loss = loss + p_w * ((row_sums - 1.0) ** 2).sum(dim=1).mean()
    return loss


def _loss_self_loop(H_list, scenario_probs):
    loss = torch.tensor(0.0, device=H_list[0].device)
    for H_omega, p_w in zip(H_list, scenario_probs):
        diag = torch.diagonal(H_omega, dim1=1, dim2=2).sum(dim=1)
        loss = loss + p_w * diag.mean()
    return loss


def _loss_distance(H_list, dist_list, scenario_probs):
    loss = torch.tensor(0.0, device=H_list[0].device)
    for H_omega, D_omega, p_w in zip(H_list, dist_list, scenario_probs):
        if D_omega.dim() == 2:
            D_omega = D_omega.unsqueeze(0)
        loss = loss + p_w * (H_omega * D_omega).sum(dim=(1, 2)).mean()
    return loss


def _loss_asymmetry(H_list, scenario_probs):
    """Penalizza l'uso simultaneo dei due versi: Σ_ω p_ω Σ H_ij H_ji."""
    loss = torch.tensor(0.0, device=H_list[0].device)
    for H_omega, p_w in zip(H_list, scenario_probs):
        loss = loss + p_w * (H_omega * H_omega.transpose(1, 2)).sum(dim=(1, 2)).mean()
    return loss


def _loss_consistency(H_list, H_bar, I_mask, scenario_probs):
    """Dispersione delle heatmap di scenario attorno a H̄, ristretta a I."""
    I_float = I_mask.float().unsqueeze(0)
    loss = torch.tensor(0.0, device=H_list[0].device)
    for H_omega, p_w in zip(H_list, scenario_probs):
        diff = ((H_omega - H_bar) ** 2) * I_float
        loss = loss + p_w * diff.sum(dim=(1, 2)).mean()
    return loss


def _booking_penalty(H_bar, p_mat, C_mat, I_upper, alpha):
    """Termini booking e penalty in un colpo solo, sulle coppie non orientate di I.

        S_ij  = H̄_ij + H̄_ji           (= Σ_ω p_ω (H^ω_ij + H^ω_ji))
        σ_ij  = 1 - exp(-alpha·S_ij)
        book  = Σ_{i<j∈I} p_ij·σ_ij
        pen   = Σ_{i<j∈I} C_ij·(1-σ_ij)·S_ij

    NOTA: il fattore |Omega| che qui era hardcoded è uscito: ora è lambda_b,
    calcolato in two_stage_utsp_loss come len(H_list)/lambda_b_div."""
    S     = (H_bar + H_bar.transpose(-1, -2)).squeeze(0)     # (n, n)
    sigma = 1.0 - torch.exp(-alpha * S)
    L_book = (p_mat * sigma)[I_upper].sum()
    L_pen  = (C_mat * (1.0 - sigma) * S)[I_upper].sum()
    return L_book, L_pen


# ── Loss principale ──────────────────────────────────────────────────

def two_stage_utsp_loss(
    T_list, dist_list, I_mask,
    p_mat, C_mat, scenario_probs,
    alpha=DEFAULT_ALPHA,
    lambda1=DEFAULT_LAMBDA1,
    lambda2=DEFAULT_LAMBDA2,
    lambda_e=DEFAULT_LAMBDA_E,
    include_entropy=False,
    lambda_d=DEFAULT_LAMBDA_D,
    lambda_b_div=DEFAULT_LAMBDA_B_DIV,
    include_penalty=False,
    return_components=False,
):
    """L = λ1·row + λ2·diag + dist + λ_b·book + λ_d·asym [+ pen] [+ λ_e·cons]."""
    device = T_list[0].device
    H_list = [compute_heatmap(T) for T in T_list]
    H_bar  = compute_H_bar(H_list, scenario_probs)

    # Maschera triangolare-superiore di I: ogni corridoio {i,j} conta una volta.
    n = I_mask.size(0)
    ar = torch.arange(n, device=device)
    I_upper = I_mask & (ar[:, None] < ar[None, :])

    L_row  = _loss_row_wise(T_list, scenario_probs)
    L_diag = _loss_self_loop(H_list, scenario_probs)
    L_dist = _loss_distance(H_list, dist_list, scenario_probs)
    L_asym = _loss_asymmetry(H_list, scenario_probs)
    L_book, L_pen_full = _booking_penalty(H_bar, p_mat, C_mat, I_upper, alpha)

    # Il costo di prenotazione è di primo stadio (pagato una volta), gli altri
    # termini sono attese pesate con p_ω: il fattore |Omega| al numeratore li
    # riporta alla stessa scala. Il divisore è il parametro da tarare.
    lambda_b = len(H_list) / float(lambda_b_div)

    loss = lambda1 * L_row + lambda2 * L_diag + L_dist + lambda_b * L_book + lambda_d * L_asym

    L_pen = L_pen_full if include_penalty else torch.tensor(0.0, device=device)
    if include_penalty:
        loss = loss + L_pen

    L_cons = torch.tensor(0.0, device=device)
    if include_entropy:
        L_cons = _loss_consistency(H_list, H_bar, I_mask, scenario_probs)
        loss = loss + lambda_e * L_cons

    if return_components:
        return loss, {
            "total":       float(loss.detach().cpu().item()),
            "row_wise":    float(L_row.detach().cpu().item()),
            "self_loop":   float(L_diag.detach().cpu().item()),
            "distance":    float(L_dist.detach().cpu().item()),
            "booking":     float(L_book.detach().cpu().item()),
            "lambda_b":    float(lambda_b),
            "consistency": float(L_cons.detach().cpu().item()),
            "asymmetry":   float(L_asym.detach().cpu().item()),
            "penalty":     float(L_pen.detach().cpu().item()),
        }
    return loss


# ── Decodifica della politica di primo stadio (diagnostica) ──────────

def decode_booking_policy(H_list, I, nodes, scenario_probs, p, C):
    """Prenota {i,j} se f_H = H̄_ij + H̄_ji supera la soglia analitica p/C
    (critical ratio del newsvendor). Nessun alpha né threshold liberi."""
    idx = {v: k for k, v in enumerate(nodes)}
    with torch.no_grad():
        H_agg = compute_H_bar(H_list, scenario_probs).squeeze(0)

    x_reserved, x_scores = [], {}
    for i, j in I:
        ii, jj = idx[i], idx[j]
        f_H = float(H_agg[ii, jj].item() + H_agg[jj, ii].item())
        x_scores[(i, j)] = f_H
        if f_H > get_edge_value(p, i, j) / get_edge_value(C, i, j):
            x_reserved.append((i, j))
    return x_reserved, x_scores


# ── Diagnostica ──────────────────────────────────────────────────────

def format_loss_components(components, epoch=None):
    prefix = f"Ep {epoch:>4}" if epoch is not None else "Loss"
    return (
        f"{prefix} | total={components['total']:.4f} "
        f"| dist={components['distance']:.4f} "
        f"| book={components['booking']:.4f}x{components.get('lambda_b', 1.0):.1f} "
        f"| cons={components['consistency']:.4f} "
        f"| asym={components['asymmetry']:.4f} "
        f"| pen={components['penalty']:.4f} "
        f"| row={components['row_wise']:.4f} "
        f"| diag={components['self_loop']:.4f}"
    )


def check_booking_coverage(x_scores, I, p, C):
    """Per ogni arco di I: score stimato, soglia p/C, decisione. Stessa regola del decode."""
    lines = ["Decisioni di prenotazione NN (regola f > p/C):"]
    n_booked = 0
    for i, j in sorted(I):
        score  = x_scores.get((i, j), x_scores.get((j, i), 0.0))
        C_val  = get_edge_value(C, i, j)
        soglia = get_edge_value(p, i, j) / C_val if C_val > 0 else float("inf")
        book   = score > soglia
        lines.append(f"  {{{i},{j}}}  f={score:.4f}  soglia={soglia:.4f}  "
                     f"{'✓ PRENOTA' if book else '✗ non prenota'}")
        n_booked += book
    lines.append(f"  Totale prenotazioni: {n_booked}/{len(I)}")
    return "\n".join(lines)
