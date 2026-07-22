# -*- coding: utf-8 -*-
import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# NOTA: ripulito. I parametri UTSP_LS_* e TEST_SCENARIO_* sono usati da
# local_search.py, non qui. UTSP2_ALPHA_LOSS/ALPHA_DECODE non erano usati da
# nessuna parte: vale solo UTSP2_LS_ALPHA.
from config import (
    OUTPUT_DIR, TRAIN_OUTPUT_DIR, N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, UTSP_BATCH_SIZE,
    TRAIN_SCENARIO_IDS_UTSP, DROP_LAST_TRAIN_BATCH, UTSP_TRAINING_SEED,
    UTSP2_HIDDEN, UTSP2_NLAYERS, UTSP2_EPOCHS, UTSP2_LR, UTSP2_STEP_LR, UTSP2_LOG_FREQ,
    UTSP2_LAMBDA1, UTSP2_LAMBDA2, UTSP2_LAMBDA_D, UTSP2_LAMBDA_E,
    UTSP2_TEMP_MODE, UTSP2_TEMP_SCALE, UTSP2_TEMP_FIXED,
    UTSP2_DIST_SCALE_MODE, UTSP2_INCLUDE_PENALTY, UTSP2_INCLUDE_ENTROPY, UTSP2_LS_ALPHA,
)
from tsp_utils import get_edge_value
from scenarios import generate_scenario_batches
from evaluation import (
    plot_utsp_heatmap, plot_utsp_graph_weights, plot_utsp_random_scenario_graphs,
)
from local_search import _run_local_search_branch, _run_utsp_test_only_branch, _heatmap_numpy_from_H_list
from two_stage_utsp_loss import (
    two_stage_utsp_loss,
    build_I_tensors,
    normalize_dist_tensor,
    decode_booking_policy,
    compute_heatmap,
    format_loss_components,
    check_booking_coverage,
)

_leaky = F.leaky_relu

# Numero di scenari usati nella fase di test UTSP.
# Separato da N_TRAINING_SCENARIOS_UTSP e da N_VALIDATION_SCENARIOS,
# perché il test della pipeline UTSP deve essere un blocco unico: di default usa N_VALIDATION_SCENARIOS.
#UTSP_TEST_SCENARIOS = int(os.environ.get("UTSP_TEST_SCENARIOS", str(N_VALIDATION_SCENARIOS)))
#UTSP_TEST_SEED = int(os.environ.get("UTSP_TEST_SEED", str(VALIDATION_SEED)))

# Gestione artefatti training UTSP.
# - default: training normale + salvataggio automatico
# - TESI_REUSE_UTSP_TRAIN=1: carica modello già salvato, se compatibile
# - TESI_UTSP_TEST_ONLY=1: carica modello già salvato e fa solo nuovi test
UTSP_REUSE_TRAIN = os.environ.get("TESI_REUSE_UTSP_TRAIN", "0").strip() == "1"
UTSP_TEST_ONLY = os.environ.get("TESI_UTSP_TEST_ONLY", "0").strip() == "1"
UTSP_TRAIN_NAME = os.environ.get("TESI_UTSP_TRAIN_NAME", "").strip()


# Prende il grafo (matrice di adiacenza W) e "propaga" le feature dei nodi attraverso i vicini.
# Ad ogni passo, ogni nodo aggrega le informazioni dei suoi vicini, pesate per il grado (la normalizzazione D). 
# L'ordine 3 significa che fai 3 passi di propagazione, quindi ogni nodo "vede" fino a 3 salti di distanza. Ritorna i risultati di ogni passo.
def _gcn_diffusion(W, order, feature, device):
    I_n = torch.eye(W.size(1), device=device).unsqueeze(0).expand(W.size(0), -1, -1)
    A   = W + I_n
    deg = torch.sum(A, 2, keepdim=True)
    D   = torch.pow(deg.clamp(min=1e-9), -0.5)
    res, x = [], feature
    for _ in range(order):
        x = D * x; x = torch.bmm(A, x); x = D * x
        res.append(x)
    return res


#Fa qualcosa di simile ma con una logica diversa: invece di passi discreti, fa una diffusione "lenta" (media pesata 0.5/0.5) 
#per 16 iterazioni e cattura le differenze tra scale diverse (buf[0]-buf[1], ecc.).
# È come una wavelet: cattura struttura locale a diverse risoluzioni del grafo.
def _scattering_diffusion(W, feature):
    deg, D = torch.sum(W, 2, keepdim=True).clamp(min=1e-9), None
    D   = torch.pow(deg, -1)
    buf, x = [], feature
    for i in range(16):
        x = 0.5 * x + 0.5 * torch.bmm(W, D * x)
        if i in [0, 1, 3, 7]:
            buf.append(x)
    return buf[0]-buf[1], buf[1]-buf[2], buf[2]-buf[3], buf[3]-feature*0


#Combina i due tipi di diffusione sopra tramite attention: per ogni nodo calcola quanto è rilevante ciascuna delle 6 rappresentazioni
# (2 da GCN + 4 da scattering) e le combina con pesi appresi. Poi passa attraverso due layer lineari. È il cuore della GNN
class _SCTConv(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.a       = nn.Parameter(torch.zeros(2 * hidden_dim, 1))
    
    def forward(self, X, adj, device):
        h_A, h_A2, _ = _gcn_diffusion(adj, 3, X, device)
        h_A, h_A2    = _leaky(h_A), _leaky(h_A2)
        s1, s2, s3, s4 = _scattering_diffusion(adj, X)
        s1, s2, s3, s4 = [torch.abs(s) for s in (s1, s2, s3, s4)]
        parts    = [h_A, h_A2, s1, s2, s3, s4]
        a_inputs = torch.stack([torch.cat([X, p], dim=2) for p in parts], dim=1)
        e        = torch.matmul(F.relu(a_inputs), self.a).squeeze(-1)
        attn     = F.softmax(e, dim=1).unsqueeze(-1)
        h_prime  = (attn * torch.stack(parts, dim=1)).sum(dim=1) # prima era mean
        return _leaky(self.linear2(_leaky(self.linear1(h_prime))))
    

class UTSP_GNN(nn.Module):
    def __init__(self, n_nodes, hidden_dim, n_layers):
        super().__init__()
        #self.bn0     = nn.BatchNorm1d(2) # pensavo fosse utile per le distanze normalizzate ma non cambia nulla, quindi non lo uso
        self.in_proj = nn.Linear(2, hidden_dim)
        self.convs   = nn.ModuleList([_SCTConv(hidden_dim) for _ in range(n_layers)])
        self.mlp1    = nn.Linear(hidden_dim * (1 + n_layers), hidden_dim)
        self.mlp2    = nn.Linear(hidden_dim, n_nodes)
        self.softmax = nn.Softmax(dim=1)
        
# ORIGINALE
#    def forward(self, xy, adj, device):
#      B, N, _ = xy.shape
#      x       = self.bn0(xy.reshape(B * N, 2)).reshape(B, N, 2)
#      x       = _leaky(self.in_proj(x))
#      hidden  = x
#      for conv in self.convs:
#          x = conv(x, adj, device)
#          hidden = torch.cat([hidden, x], dim=-1)
#      return self.softmax(self.mlp2(_leaky(self.mlp1(hidden))))
      
    #NUOVA
    def forward(self, xy, adj, device):
        B, N, _ = xy.shape
        #x       = self.bn0(xy.reshape(B * N, 2)).reshape(B, N, 2)
        x       = _leaky(self.in_proj(xy))
        hidden  = x
        for conv in self.convs:
            x = conv(x, adj, device)
            hidden = torch.cat([hidden, x], dim=-1)
        logits = self.mlp2(_leaky(self.mlp1(hidden)))              # (B, N, N)
        return self.softmax(logits)                                # no mask diagonale

# Normalizzo le coordinate in [0,1]
def _normalize_coords(nodes, coords, device): 
    xs = np.array([coords[v][0] for v in nodes], dtype=np.float32)
    ys = np.array([coords[v][1] for v in nodes], dtype=np.float32)
    xs = (xs - xs.min()) / (xs.max() - xs.min() + 1e-9)
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-9)
    xy = torch.from_numpy(np.stack([xs, ys], axis=1)).float().to(device)
    return xy   # (n, 2)

# calcolo temp per la adj
def _compute_temperature(dist_stack, mode, scale, fixed): 
    if mode == "fixed":
        return float(fixed)
    n    = dist_stack.size(-1)
    mask = ~torch.eye(n, dtype=torch.bool, device=dist_stack.device)
    mask = mask.unsqueeze(0).expand_as(dist_stack)
    vals = dist_stack[mask]
    vals = vals[vals > 0]
    if vals.numel() == 0:
        return float(fixed)
    T = float(torch.median(vals).item()) * float(scale)
    return max(T, 1e-9)

#Costruisco xy e le matrici di distanza normalizzate per tutti gli scenari.
#xy_tile    : (K, n, 2)  — coordinate normalizzate, replicate per ogni scenario
#dist_raw   : list di K tensori (n, n) — distanze reali (non normalizzate)
#dist_model : (K, n, n)  — distanze normalizzate per GNN/loss
#dist_scale :   scala usata per la normalizzazione
#temperature:  temperatura T per il kernel gaussiano
def _build_input_tensors(scenario_ids, results, nodes, coords, device): 
    K = len(scenario_ids)
    n = len(nodes)
    xy = _normalize_coords(nodes, coords, device)  # (n, 2)
    xy_tile = xy.unsqueeze(0).expand(K, -1, -1).contiguous() # (K, n, 2)

    # Distanze reali per ogni scenario
    dist_raw = []
    for sid in scenario_ids:
        sd = results[sid]["scenario_dist"]
        D  = torch.zeros(n, n, device=device)
        for ii, i in enumerate(nodes):
            for jj, j in enumerate(nodes):
                if i != j:
                    D[ii, jj] = float(sd[i][j])
        dist_raw.append(D)

    dist_stack = torch.stack(dist_raw, dim=0)  # (K, n, n)

    # Normalizzazione interna UTSP
    dist_model, dist_scale = normalize_dist_tensor(
        dist_stack, mode=UTSP2_DIST_SCALE_MODE
    )

    temperature = _compute_temperature(
        dist_model, UTSP2_TEMP_MODE, UTSP2_TEMP_SCALE, UTSP2_TEMP_FIXED
    )

    return xy_tile, dist_raw, dist_model, dist_scale, temperature
    


#altra possibile formulazione AL MOMENTO NON LA STO USANDO PERCHE QUELLA DEL PAPER FUNZIONA
#tau : temperatura del kernel dopo riscalatura robusta
# eps : peso minimo morbido per ogni arco fuori diagonale
# q_scale : quantile usato come scala di riga
# clip_max : massimo valore normalizzato prima del kernel
def _build_adj_robust_floor( dist_stack, tau=UTSP2_TEMP_SCALE,
    eps=0.02, q_scale=0.90, clip_max=3.0, ): 

    K, n, _ = dist_stack.shape
    device = dist_stack.device

    D_scaled = torch.zeros_like(dist_stack)

    for s in range(K):
        for i in range(n):
            row = dist_stack[s, i].clone()
            row[i] = float("inf")

            vals = row[torch.isfinite(row)]
            vals = vals[vals > 0]

            if vals.numel() == 0:
                scale_i = torch.tensor(1.0, device=device)
            else:
                scale_i = torch.quantile(vals, q_scale).clamp(min=1e-9)

            D_scaled[s, i] = dist_stack[s, i] / scale_i

    D_scaled = torch.clamp(D_scaled, min=0.0, max=clip_max)
    adj = torch.exp(-D_scaled / max(tau, 1e-9))
    # soglia minima morbida: ogni arco fuori diagonale resta visibile
    adj = eps + (1.0 - eps) * adj
    return adj




def _json_safe(obj):
    """Converte oggetti numpy/torch in tipi serializzabili JSON."""
    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu()
        if obj.numel() == 1:
            return float(obj.item())
        return obj.tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, set):
        return sorted(_json_safe(v) for v in obj)
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _canon_edge_local(i, j):
    return (i, j) if i <= j else (j, i)


def _edge_metadata(I, p, C):
    rows = []
    for i, j in sorted({_canon_edge_local(i, j) for (i, j) in I}):
        rows.append({
            "edge": [i, j],
            "p": float(get_edge_value(p, i, j)),
            "C": float(get_edge_value(C, i, j)),
        })
    return rows


def _artifact_exp_name(exp_name):
    return UTSP_TRAIN_NAME if UTSP_TRAIN_NAME else str(exp_name)


def _utsp_train_dir(exp_name):
    name = _artifact_exp_name(exp_name)
    # NOTA: TRAIN_OUTPUT_DIR, non OUTPUT_DIR — con TESI_TEST_OUTPUT_SUBDIR
    # l'output è deviato in test/IS_*_DIM_*, ma i checkpoint vivono sempre
    # in <batch_dir>/train/<nome>. I due coincidono fuori dal test sweep.
    train_dir = os.path.join(TRAIN_OUTPUT_DIR, "train", name)
    os.makedirs(train_dir, exist_ok=True)
    return train_dir


def _utsp_train_paths(exp_name):
    train_dir = _utsp_train_dir(exp_name)
    return {
        "dir": train_dir,
        "model": os.path.join(train_dir, "utsp_model.pt"),
        "history": os.path.join(train_dir, "utsp_history.json"),
        "metadata": os.path.join(train_dir, "utsp_metadata.json"),
    }


def _save_utsp_train_artifact(
    exp_name, model, history, nodes, scenario_ids,
    temperature, dist_scale, I, p, C,
):
    paths = _utsp_train_paths(exp_name)

    metadata = {
        "exp_name": str(exp_name),
        "artifact_name": _artifact_exp_name(exp_name),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "output_dir": str(OUTPUT_DIR),
        "n_nodes": int(len(nodes)),
        "nodes": list(nodes),
        "n_training_scenarios": int(len(scenario_ids)),
        "utsp_batch_size": int(UTSP_BATCH_SIZE),
        "utsp_training_seed": int(UTSP_TRAINING_SEED),
        "utsp2_hidden": int(UTSP2_HIDDEN),
        "utsp2_nlayers": int(UTSP2_NLAYERS),
        "utsp2_epochs": int(UTSP2_EPOCHS),
        "utsp2_lr": float(UTSP2_LR),
        "temperature": float(temperature),
        "dist_scale": float(dist_scale),
        "I_edges": _edge_metadata(I, p, C),
    }

    payload = {
        "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "history": _json_safe(history),
        "metadata": metadata,
    }

    torch.save(payload, paths["model"])

    with open(paths["history"], "w", encoding="utf-8") as f:
        json.dump(_json_safe(history), f, indent=2)

    with open(paths["metadata"], "w", encoding="utf-8") as f:
        json.dump(_json_safe(metadata), f, indent=2)

    print("\n  Artefatto training UTSP salvato:")
    print(f"    model    = {paths['model']}")
    print(f"    history  = {paths['history']}")
    print(f"    metadata = {paths['metadata']}")


def _assert_utsp_artifact_compatible(metadata, nodes, I, p, C):
    errors = []

    if int(metadata.get("n_nodes", -1)) != len(nodes):
        errors.append(
            f"n_nodes salvato={metadata.get('n_nodes')} corrente={len(nodes)}"
        )

    if list(metadata.get("nodes", [])) != list(nodes):
        errors.append("lista nodi diversa")

    if int(metadata.get("utsp2_hidden", -1)) != int(UTSP2_HIDDEN):
        errors.append("UTSP2_HIDDEN diverso")

    if int(metadata.get("utsp2_nlayers", -1)) != int(UTSP2_NLAYERS):
        errors.append("UTSP2_NLAYERS diverso")

    saved_edges = metadata.get("I_edges", [])
    current_edges = _edge_metadata(I, p, C)
    if saved_edges != current_edges:
        errors.append("I/p/C diversi rispetto al modello salvato")

    if errors:
        raise ValueError(
            "Artefatto UTSP incompatibile:\n  - " + "\n  - ".join(errors)
        )


def _load_utsp_train_artifact(exp_name, nodes, coords, I, p, C, device):
    paths = _utsp_train_paths(exp_name)

    if not os.path.exists(paths["model"]):
        raise FileNotFoundError(paths["model"])

    payload = torch.load(paths["model"], map_location=device)
    metadata = payload.get("metadata", {})
    _assert_utsp_artifact_compatible(metadata, nodes, I, p, C)

    model = UTSP_GNN(len(nodes), UTSP2_HIDDEN, UTSP2_NLAYERS).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    if os.path.exists(paths["history"]):
        with open(paths["history"], "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = payload.get("history", {"loss": [float("nan")], "components": []})

    if "loss" not in history or not history["loss"]:
        history["loss"] = [float("nan")]

    xy = _normalize_coords(nodes, coords, device)
    temperature = float(metadata["temperature"])
    dist_scale = float(metadata["dist_scale"])

    print("\n  Artefatto training UTSP caricato:")
    print(f"    model       = {paths['model']}")
    print(f"    temperature = {temperature:.6f}")
    print(f"    dist_scale  = {dist_scale:.6f}")
    print(f"    n_nodes     = {metadata.get('n_nodes')}")

    return model, history, xy, temperature, dist_scale, metadata



# Addestro la GNN su batch di scenari
def _train_utsp_2stage(
    nodes, coords, scenario_ids, results, scenario_probs,
    I_mask, p_mat, C_mat, device,
    batches=None,
): 
    K = len(scenario_ids)
    n = len(nodes)

    # Costruisco tensori per TUTTI gli scenari (normalizzazione globale)
    xy_tile, dist_raw, dist_model, dist_scale, temperature = _build_input_tensors(
        scenario_ids, results, nodes, coords, device
    )
    probs_t = torch.tensor(
        [scenario_probs[sid] for sid in scenario_ids],
        dtype=torch.float32, device=device
    )

    print(f"[SEED_CHECK] torch.initial_seed()={torch.initial_seed()}")
    print(f"[SEED_CHECK] np.random state[0]={np.random.get_state()[1][0]}")
   
    # Seed locale dedicato all'inizializzazione GNN, indipendente dal flusso globale
    torch.manual_seed(UTSP_TRAINING_SEED)
    model = UTSP_GNN(n, UTSP2_HIDDEN, UTSP2_NLAYERS).to(device)
    
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = optim.Adam(model.parameters(), lr=UTSP2_LR)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=UTSP2_STEP_LR, gamma=0.8
    )

    # Adiacenza per ogni scenario: adj_k = exp(-D^ω_norm / T)
    adj_stack = torch.exp(-dist_model / temperature)  # (K, n, n)

    # ── Costruzione batch slices ──────────────────────────────────────
    if batches is not None:
        # Mappa scenario_id → indice nel tensore globale
        sid_to_idx = {sid: k for k, sid in enumerate(scenario_ids)}
        batch_slices = []
        for batch in batches:
            b_sids = batch["scenario_ids"]
            b_indices = [sid_to_idx[sid] for sid in b_sids]
            b_probs = torch.tensor(
                [batch["scenario_probs"][sid] for sid in b_sids],
                dtype=torch.float32, device=device,
            )
            batch_slices.append({
                "indices": b_indices,
                "probs_t": b_probs,
                "K": len(b_sids),
            })
        n_batches = len(batch_slices)
    else:
        # Nessun batching: un unico batch con tutti gli scenari
        batch_slices = [{
            "indices": list(range(K)),
            "probs_t": probs_t,
            "K": K,
        }]
        n_batches = 1

    K_batch = batch_slices[0]["K"]
    print(f"\n  Training UTSP 2-stage | device={device} | n_params={n_par:,}")
    print(f"\n  Training UTSP 2-stage | device={device} | n_params={n_par:,}")
    print(f"  GNN({n}→{UTSP2_HIDDEN}×{UTSP2_NLAYERS}) | "
          f"K_total={K} ({n_batches} batch × {K_batch}) | T={temperature:.4f} | scale={dist_scale:.4f}")
    print(f"  Epoche={UTSP2_EPOCHS}  lr={UTSP2_LR}  "
          f"λ1={UTSP2_LAMBDA1}  λ2={UTSP2_LAMBDA2}  λe={UTSP2_LAMBDA_E}  ")

    history = {"loss": [], "components": []}
    best_loss, best_state = float("inf"), None
    t0 = time.time()

    for epoch in range(1, UTSP2_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        epoch_comps = None

        for bs in batch_slices:
            idx = bs["indices"]
            K_b = bs["K"]
            xy_b   = xy_tile[idx]       # (K_b, n, 2)
            adj_b  = adj_stack[idx]     # (K_b, n, n)
            dist_b = dist_model[idx]    # (K_b, n, n)

            T_batch = model(xy_b, adj_b, device)                  # (K_b, n, n)
            T_list    = [T_batch[k:k+1] for k in range(K_b)]
            dist_list = [dist_b[k:k+1]  for k in range(K_b)]

            loss, comps = two_stage_utsp_loss(
                T_list, dist_list, I_mask, p_mat, C_mat, bs["probs_t"],
                alpha=UTSP2_LS_ALPHA, lambda1=UTSP2_LAMBDA1, lambda2=UTSP2_LAMBDA2,
                lambda_e=UTSP2_LAMBDA_E, lambda_d=UTSP2_LAMBDA_D,
                include_penalty=UTSP2_INCLUDE_PENALTY,
                include_entropy=UTSP2_INCLUDE_ENTROPY,
                return_components=True,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_comps = comps       # ultimo batch come riferimento

        scheduler.step()

        avg_loss = epoch_loss / n_batches
        history["loss"].append(avg_loss)
        history["components"].append(epoch_comps)

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % UTSP2_LOG_FREQ == 0 or epoch == 1:
            marker = " ★" if abs(avg_loss - best_loss) < 1e-9 else ""
            print(f"  {format_loss_components(epoch_comps, epoch)} "
                  f"avg={avg_loss:.5f}{marker}")

    elapsed = time.time() - t0
    print(f"\n{'═'*65}")
    print(f"  TRAINING COMPLETATO")
    print(f"  Tempo totale          = {elapsed:.2f}s  ({elapsed/UTSP2_EPOCHS*1000:.1f}ms/epoca)")
    print(f"  Miglior loss          = {best_loss:.5f}  (ep {int(np.argmin(history['loss']))+1})")
    print(f"  Loss iniziale→finale  = {history['loss'][0]:.5f} → {history['loss'][-1]:.5f}")
    print(f"  Riduzione loss        = {(history['loss'][0]-best_loss)/history['loss'][0]*100:.1f}%")
    print(f"{'═'*65}")

    model.load_state_dict(best_state)
    model.eval()
    return model, history, adj_stack, dist_model, xy_tile, probs_t, temperature,  dist_scale
    
    
def _decode_policy(model, adj_stack, xy_tile, I, nodes, I_mask, probs_t, device):
    K = adj_stack.size(0)
    model.eval()
    with torch.no_grad():
        T_batch = model(xy_tile, adj_stack, device)        # (K, n, n)

    H_list = [compute_heatmap(T_batch[k:k+1]) for k in range(K)]

    x_reserved, x_scores = decode_booking_policy(
    H_list, I, nodes, I_mask, probs_t, alpha=UTSP2_LS_ALPHA )
    return x_reserved, x_scores, H_list, T_batch 
    
# Diagnostica numerica su T e H dopo il passaggio nella GNN per possibili problemi     
def debug_T_H(T_batch, H_list, name="TRAIN"): 

    with torch.no_grad():
        T = T_batch.detach().cpu()
        H = torch.stack([h.squeeze(0).detach().cpu() for h in H_list], dim=0)

        Tm = T.mean(dim=0).numpy()
        Hm = H.mean(dim=0).numpy()

        def stats(A, label):
            A = np.asarray(A)
            n = A.shape[0]
            mask = ~np.eye(n, dtype=bool)
            vals = A[mask]

            print(f"\n[{name}] {label}")
            print(f"  min fuori diagonale   = {vals.min():.6e}")
            print(f"  max fuori diagonale   = {vals.max():.6e}")
            print(f"  media fuori diagonale = {vals.mean():.6e}")
            print(f"  std fuori diagonale   = {vals.std():.6e}")
            print(f"  p50 fuori diagonale   = {np.percentile(vals, 50):.6e}")
            print(f"  p90 fuori diagonale   = {np.percentile(vals, 90):.6e}")
            print(f"  p99 fuori diagonale   = {np.percentile(vals, 99):.6e}")
            print(f"  somma righe min/max   = {A.sum(axis=1).min():.6f} / {A.sum(axis=1).max():.6f}")
            print(f"  somma colonne min/max = {A.sum(axis=0).min():.6f} / {A.sum(axis=0).max():.6f}")
            print(f"  diagonale somma       = {np.trace(A):.6e}")

        stats(Tm, "T medio")
        stats(Hm, "H medio")  

# Stampo la matrice di adiacenza media con valori numerici
def _print_adj_matrix(adj_stack, nodes, label="Matrice adiacenza (media scenari)"):
    adj_avg = adj_stack.detach().cpu().numpy().mean(axis=0)
    n = len(nodes)
    sep = "─" * (9 * n + 14)
    print(f"\n{'═'*65}")
    print(f"  {label}")
    print(sep)
    header = f"  {'i→j':>6} |" + "".join(f" {str(v):>7}" for v in nodes)
    print(header)
    print(sep)
    for ii, i in enumerate(nodes):
        row = f"  {str(i):>6} |" + "".join(
            f" {adj_avg[ii, jj]:>7.4f}" if ii != jj else f" {'—':>7}"
            for jj in range(n)
        )
        print(row)
    print(sep)
    # Statistiche sintetiche
    mask = ~np.eye(n, dtype=bool)
    vals = adj_avg[mask]
    print(f"  min={vals.min():.4f}  max={vals.max():.4f}  "
          f"media={vals.mean():.4f}  std={vals.std():.4f}")
    print(f"{'═'*65}")

#Entropia e copertura della heatmap media
def _print_heatmap_diagnostics(H_list, nodes, scenario_ids, label="Heatmap diagnostica"): 
    H_avg = None
    for H in H_list:
        arr = H.detach().squeeze(0).cpu().numpy()
        H_avg = arr.copy() if H_avg is None else H_avg + arr
    H_avg /= max(len(H_list), 1)

    n = len(nodes)
    mask = ~np.eye(n, dtype=bool)
    vals = H_avg[mask]
    vals_pos = vals[vals > 1e-9]

    # Entropia di Shannon normalizzata
    if vals_pos.sum() > 0:
        p = vals_pos / vals_pos.sum()
        entropy = float(-np.sum(p * np.log(p + 1e-12)))
        max_entropy = float(np.log(len(vals_pos)))
        entropy_norm = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        entropy, entropy_norm = 0.0, 0.0

    # Copertura: % archi con H > soglie
    thresholds = [0.05, 0.10, 0.20]
    print(f"\n{'═'*65}")
    print(f"  {label}")
    print(f"  Entropia Shannon         = {entropy:.4f}")
    print(f"  Entropia normalizzata    = {entropy_norm:.4f}  "
          f"(1.0=uniforme, 0.0=concentrata)")
    print(f"  H_avg: min={vals.min():.4f}  max={vals.max():.4f}  "
          f"media={vals.mean():.4f}")
    for thr in thresholds:
        n_above = int((vals > thr).sum())
        pct = 100.0 * n_above / len(vals)
        print(f"  Archi con H > {thr:.2f}        = {n_above}/{len(vals)} ({pct:.1f}%)")
    print(f"{'═'*65}")
  
    


 

# per far andare l esperimento intero
def run_esperimento_B_UTSP(
    nodes, coords, base_dist, E, root, env, res_B,
    mode="local_search", scenario_kwargs=None, exp_name="espB_UTSP_LS",
):
    """
    Esegue UTSP solo in modalità local search.

    Parametri specifici del generatore di scenari, per esempio quelli di CVETT
    legati al vento, vanno passati in scenario_kwargs. Se scenario_kwargs=None,
    il comportamento resta quello PERT.
    """
    scenario_kwargs = dict(scenario_kwargs or {})
    wind_train = scenario_kwargs.pop("wind_train", None)
    wind_test = scenario_kwargs.pop("wind_test", None)
    
    scenario_kwargs_train = dict(scenario_kwargs)
    scenario_kwargs_test = dict(scenario_kwargs)
    
    if wind_train is not None:
        scenario_kwargs_train["wind"] = wind_train
    
    if wind_test is not None:
        scenario_kwargs_test["wind"] = wind_test
    
    # Compatibilità con PERT o vecchio CVETT:
    # se arriva ancora scenario_kwargs={"wind": wind}, usa lo stesso vento per train e test.
    if "wind" in scenario_kwargs:
        scenario_kwargs_train["wind"] = scenario_kwargs["wind"]
        scenario_kwargs_test["wind"] = scenario_kwargs["wind"]

    # NOTA (fix di correttezza): generate_scenarios attiva la perturbazione da
    # vento SOLO se riceve sia `wind` che `coords` (condizione
    # `coords is not None and wind is not None`); altrimenti ricade
    # silenziosamente su build_perturbation, cioè la perturbazione SINTETICA.
    # Finora `coords` non veniva mai messo in scenario_kwargs, quindi in CVETT
    # il training e il test della rete giravano su scenari sintetici mentre
    # Experiment B (che passa coords+wind esplicitamente) generava i benchmark
    # STO/EEV/PI su scenari da vento: benchmark e rete su due distribuzioni
    # diverse, quindi numeri non confrontabili. Qui lo aggiungiamo, ma solo
    # dove c'è davvero il vento — per PERT resta assente e la perturbazione
    # sintetica è quella giusta.
    if "wind" in scenario_kwargs_train:
        scenario_kwargs_train["coords"] = coords
    if "wind" in scenario_kwargs_test:
        scenario_kwargs_test["coords"] = coords

    mode_norm = (mode or "local_search").lower().replace(" ", "_")
    if mode_norm != "local_search":
        raise ValueError("Questo utsp comune supporta solo mode='local_search'.")

    print("\n" + "=" * 70)
    print(f"ESPERIMENTO B — UTSP LOCAL SEARCH | exp_name={exp_name}")
    if scenario_kwargs:
        print(f"  Parametri scenario extra: {sorted(scenario_kwargs.keys())}")

    needed = ["I", "p", "C", "b", "results", "scenario_probs",
              "frequent_arcs", "PI", "STO", "EEV",
              "stoch_costs", "eev_costs", "x_ev", "x_used_sto",
              "eev_solutions", "stoch_solutions"]
    missing = [k for k in needed if k not in res_B]
    if missing:
        raise KeyError(f"res_B manca delle chiavi: {missing}")

    I = res_B["I"]
    p = res_B["p"]
    C = res_B["C"]
    b = res_B["b"]
    frequent_arcs = res_B["frequent_arcs"]

    # Scenari già generati da Esperimento B: servono per benchmark, grafici e riepilogo.
    results_B = res_B["results"]
    scenario_probs_B = res_B["scenario_probs"]
    scenario_ids_B = list(results_B.keys())

    PI_train = res_B["PI"]
    STO_train = res_B["STO"]
    EEV_train = res_B["EEV"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if UTSP_TEST_ONLY:
        print("\n  Modalità TESI_UTSP_TEST_ONLY=1: carico training salvato e genero solo nuovi test.")
        model, history, xy_single, temperature, dist_scale, train_metadata = _load_utsp_train_artifact(
            exp_name, nodes, coords, I, p, C, device
        )
        ls_out = _run_utsp_test_only_branch(
            nodes=nodes,
            coords=coords,
            E=E,
            root=root,
            env=env,
            res_B=res_B,
            model=model,
            xy=xy_single,
            dist_scale=dist_scale,
            temperature=temperature,
            device=device,
            base_dist=base_dist,
            scenario_kwargs=scenario_kwargs_test,
            exp_name=exp_name,
            history=history,
        )
        return {
            "model": model,
            "history": history,
            "local_search": ls_out,
            "loaded_train_artifact": train_metadata,
        }

    # ── Scenari UTSP training: N_TRAINING_SCENARIOS_UTSP / UTSP_BATCH_SIZE batch ──
    batches_utsp = generate_scenario_batches(
    nodes, E, base_dist, I, frequent_arcs,
    N_EXTRA_ARCS, MEAN_FRAC, SIGMA_FRAC, UTSP_TRAINING_SEED,
    root, env, p, C,
    scenario_ids=TRAIN_SCENARIO_IDS_UTSP,
    batch_size=UTSP_BATCH_SIZE,
    drop_last=DROP_LAST_TRAIN_BATCH,
    **scenario_kwargs_train,)

    # Combino tutti i risultati per calcoli globali (dist_scale, training GNN, decode, ecc.).
    # Rimappo gli id degli scenari su indici globali 0..K-1: in questo modo
    # evito collisioni tra batch diversi se il generatore riusa id locali.
    results_utsp = {}
    scenario_ids_utsp = []
    scenario_probs_utsp = {}
    train_batch_id_utsp = {}
    batches_utsp_global = []
    global_sid = 0

    for b_idx, batch in enumerate(batches_utsp):
        new_batch = dict(batch)
        new_ids = []
        new_results = {}
        new_probs = {}

        for local_sid in batch["scenario_ids"]:
            sid = global_sid
            global_sid += 1

            new_ids.append(sid)
            new_results[sid] = batch["results"][local_sid]
            new_probs[sid] = batch["scenario_probs"][local_sid]

            results_utsp[sid] = batch["results"][local_sid]
            scenario_ids_utsp.append(sid)
            train_batch_id_utsp[sid] = b_idx + 1

        new_batch["scenario_ids"] = new_ids
        new_batch["results"] = new_results
        new_batch["scenario_probs"] = new_probs
        batches_utsp_global.append(new_batch)

    batches_utsp = batches_utsp_global

    # Le probabilità globali servono per riepiloghi/diagnostiche sull'intero
    # training UTSP. La loss continua invece a usare le probabilità interne
    # di ciascun batch, passate in batches_utsp.
    scenario_probs_utsp = {
        sid: 1.0 / max(len(scenario_ids_utsp), 1)
        for sid in scenario_ids_utsp
    }

    n = len(nodes)

    I_mask, p_mat, C_mat, node_idx = build_I_tensors(I, nodes, p, C, device)

    _, _, _, dist_scale, _ = _build_input_tensors(
        scenario_ids_utsp, results_utsp, nodes, coords, device
    )

    p_mat = p_mat / dist_scale
    C_mat = C_mat / dist_scale

    assert I_mask.sum() > 0, f"I_mask è vuota! Controlla build_I_tensors e la lista I passata"
    print(f"Archi in I: {I_mask.sum().item() // 2} coppie non orientate")
    print(f"p_mat nonzero: {(p_mat > 0).sum().item()} celle")
    print(f"p_mat range: [{p_mat[p_mat>0].min().item():.4f}, {p_mat.max().item():.4f}]")
    print(f"dist_scale: {dist_scale:.4f}")

    print(f"\n  Setup: |I|={len(I)}  n_nodi={n}  "
          f"K_train_UTSP={len(scenario_ids_utsp)} ({len(batches_utsp)} batch × {UTSP_BATCH_SIZE})  "
          f"device={device}")
    print(f"  Scenari B per grafici/benchmark = {len(scenario_ids_B)}")
    print(f"  I = {I}")

    train_loaded = False

    if UTSP_REUSE_TRAIN:
        try:
            model, history, xy_single, temperature, dist_scale, train_metadata = _load_utsp_train_artifact(
                exp_name, nodes, coords, I, p, C, device
            )

            # Ricostruisco tensori train solo per diagnostiche/plot del flusso completo.
            xy_tile, _, dist_model, _, _ = _build_input_tensors(
                scenario_ids_utsp, results_utsp, nodes, coords, device
            )
            adj_stack = torch.exp(-dist_model / max(float(temperature), 1e-9))
            probs_t = torch.tensor(
                [scenario_probs_utsp[sid] for sid in scenario_ids_utsp],
                dtype=torch.float32, device=device,
            )
            train_loaded = True
        except FileNotFoundError as exc:
            print(f"\n  Artefatto UTSP non trovato: {exc}")
            print("  Eseguo training normale e salvo il nuovo artefatto.")
            train_loaded = False

    if not train_loaded:
        (model, history, adj_stack, dist_model,
         xy_tile, probs_t, temperature, dist_scale) = _train_utsp_2stage(
            nodes,
            coords,
            scenario_ids_utsp,
            results_utsp,
            scenario_probs_utsp,
            I_mask,
            p_mat,
            C_mat,
            device,
            batches=batches_utsp,
        )

        _save_utsp_train_artifact(
            exp_name=exp_name,
            model=model,
            history=history,
            nodes=nodes,
            scenario_ids=scenario_ids_utsp,
            temperature=temperature,
            dist_scale=dist_scale,
            I=I,
            p=p,
            C=C,
        )

    # Decode mantenuto solo come diagnostica della heatmap; la policy Gurobi non viene più usata.
    x_utsp, x_scores, H_list, T_batch = _decode_policy(
        model, adj_stack, xy_tile, I, nodes, I_mask, probs_t, device,
    )

    debug_T_H(T_batch, H_list, name="dopo training UTSP")

    # Diagnostica adiacenza e heatmap
    _print_adj_matrix(adj_stack, nodes, label="Matrice adiacenza GNN (media scenari training)")
    _print_heatmap_diagnostics(H_list, nodes, scenario_ids_utsp,
                               label="Heatmap diagnostica (media scenari training)")

    reservation_utsp = sum(get_edge_value(p, i, j) for (i, j) in x_utsp)

    # Visualizzazioni POST-RETE: heatmap media e grafo pesato medio
    _H_avg = _heatmap_numpy_from_H_list(H_list)
    _H_decode_vis = _H_avg.copy()
    
    plot_utsp_heatmap(
        f"{exp_name}_heatmap", nodes, _H_decode_vis,
        title_suffix=f"(media {len(H_list)} scenari)",
    )
    plot_utsp_graph_weights(
        f"{exp_name}_heatmap", nodes, coords, _H_avg,
        title_suffix=f"(media {len(H_list)} scenari)", I=I,
    )
    
    # POST-RETE: 5 grafi pesati scenario-specifici
    plot_utsp_random_scenario_graphs(
        exp_name, nodes, coords, scenario_ids_utsp, H_list, I,
        n_samples=5, seed=UTSP_TRAINING_SEED,
    )
    
    print(f"\n{check_booking_coverage(x_scores, I)}")
    print(f"\n  Costo prenotazione UTSP diagnostico : {reservation_utsp:.4f}")
    
    # Visualizzazione dell'adiacenza dopo il kernel
    _adj_avg = adj_stack.detach().cpu().numpy().mean(axis=0)
    
    plot_utsp_heatmap(
        f"{exp_name}_adj_kernel",
        nodes,
        _adj_avg,
        title_suffix=f"(adj_stack media su {adj_stack.shape[0]} scenari — dopo kernel)",
    )
    
    plot_utsp_graph_weights(
        f"{exp_name}_adj_kernel", nodes, coords, _adj_avg,
        title_suffix=f"(adj_stack media su {adj_stack.shape[0]} scenari — dopo kernel)", I=I,
    )

    output = {
        "model": model,
        "history": history,
        "x_utsp_diag": x_utsp,
        "x_scores": x_scores,
        "I_mask": I_mask,
    }

    ls_out = _run_local_search_branch(
    nodes, coords, E, root, env, res_B,
    results_utsp, scenario_ids_utsp, scenario_probs_utsp,
    H_list=None,  # heatmap ricalcolata scenario per scenario
    PI_train=PI_train,
    STO_train=STO_train,
    EEV_train=EEV_train,
    history=history,
    temperature=temperature,
    model=model,
    xy=xy_tile[0],  # (n, 2) — coordinate normalizzate
    dist_scale=dist_scale,
    device=device,
    base_dist=base_dist,
    scenario_kwargs=scenario_kwargs_test,
    exp_name=exp_name,
    results_B=results_B,
    scenario_ids_B=scenario_ids_B,
    scenario_probs_B=scenario_probs_B,
    train_batch_id=train_batch_id_utsp,)
    
    output.update({"local_search": ls_out})

    return output

