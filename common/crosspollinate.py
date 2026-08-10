#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crosspollinate.py — intervento causale ("impollinazione incrociata") sul canale
cross-scenario del modello UTSP.

Idea: per un'istanza A e una donatrice B (stesso DIM), si calcolano le heatmap e le
prenotazioni di A DUE volte — una col consenso VERO di A, una col consenso di B
iniettato al posto del suo. Se il cross-scenario è usato causalmente, le heatmap si
spostano e le prenotazioni cambiano. Il CONTROLLO (donatore = A stessa) deve dare
spostamento ~0: se non lo è, il meccanismo di iniezione è rotto e lo script te lo dice.

Va usato solo sul modello CON cross (il no-cross non ha consenso da scambiare).

Riusa: _load_utsp_train_artifact (utsp), _build_heatmaps_for_scenarios (local_search,
batchato sui K scenari ⇒ consenso attivo), decode_booking_policy (regola f>p/C), e il
hook xpoll_mode aggiunto in utsp.py.

Uso:
  TESI_EXPERIMENT=PERT TESI_N_NODES=15 TESI_BATCH_SWEEP=20 \
  TESI_UTSP_TRAIN_NAME=espB_UTSP_LS python crosspollinate.py --dim 30 --pairs 40
"""
import argparse
import os
import pickle
import statistics as st

import numpy as np
import torch
torch.set_num_threads(1)   # NOTA: n=15 è minuscolo; con molti thread torch passa
                           # più tempo a sincronizzare che a calcolare (10× più lento).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import load_data
from config import TEST_SCENARIO_CACHE_DIR
from utsp import _load_utsp_train_artifact, xpoll_mode
from local_search import _build_heatmaps_for_scenarios
from two_stage_utsp_loss import decode_booking_policy


def _hmaps(model, xy, nodes, results, ids, dist_scale, temp, device, mode="own", donor=None):
    """Heatmap dell'istanza `ids`. mode='own' → consenso vero; mode='inject' → prima
    registra il consenso di `donor`, poi lo inietta in `ids`."""
    if mode == "own":
        xpoll_mode("off")
        return _build_heatmaps_for_scenarios(model, xy, nodes, results, ids, dist_scale, temp, device)
    xpoll_mode("record")   # registra il consenso della donatrice, layer per layer
    _build_heatmaps_for_scenarios(model, xy, nodes, results, donor, dist_scale, temp, device)
    xpoll_mode("inject")   # ...e lo riusa nella passata del target
    H = _build_heatmaps_for_scenarios(model, xy, nodes, results, ids, dist_scale, temp, device)
    xpoll_mode("off")
    return H


def _stack(H_list):
    return np.stack([H.squeeze(0).detach().cpu().numpy() for H in H_list], 0)   # (K,n,n)


def _rel_shift(A, B):
    """||A-B||_F / ||A||_F per scenario."""
    num = np.linalg.norm(A - B, axis=(1, 2))
    den = np.maximum(np.linalg.norm(A, axis=(1, 2)), 1e-12)
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=30, help="scenari per istanza (K)")
    ap.add_argument("--pairs", type=int, default=40, help="numero di coppie (target, donatrice)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodes, coords, base_dist, E, root = load_data()

    resB = pickle.load(open(os.path.join(TEST_SCENARIO_CACHE_DIR, "res_B_cached.pkl"), "rb"))
    I, p, C = resB["I"], resB["p"], resB["C"]

    exp_name = os.environ.get("TESI_UTSP_TRAIN_NAME", "espB_UTSP_LS")
    model, _, xy, temperature, dist_scale, meta = _load_utsp_train_artifact(
        exp_name, nodes, coords, I, p, C, device)

    cache_path = os.path.join(TEST_SCENARIO_CACHE_DIR, "test_scenarios_cache.pkl")
    cache = pickle.load(open(cache_path, "rb"))
    results = cache.get("results", {})
    sids = sorted(results.keys())
    K = a.dim
    blocks = [sids[i * K:(i + 1) * K] for i in range(len(sids) // K)]
    if len(blocks) < 2:
        print(f"Servono ≥2 istanze di dim {K}: trovati {len(sids)} scenari in cache "
              f"({len(blocks)} blocchi). Riduci --dim o genera più scenari.")
        return

    rng = np.random.default_rng(a.seed)
    probs = [1.0 / K] * K
    lines = []
    def P(*x):
        s = " ".join(str(t) for t in x); print(s); lines.append(s)

    P("=" * 84)
    P(f"CROSS-POLLINATION (intervento sul canale cross-scenario) — modello '{exp_name}'")
    P(f"DIM={K}  coppie={a.pairs}  istanze disponibili={len(blocks)}  device={device}")
    P("=" * 84)

    sh_for, sh_self, jac, flips, dbook = [], [], [], [], []
    t0 = __import__("time").time()
    for kp in range(a.pairs):
        ia, ib = rng.choice(len(blocks), size=2, replace=False)
        A, B = blocks[ia], blocks[ib]

        H_own_l = _hmaps(model, xy, nodes, results, A, dist_scale, temperature, device, "own")
        H_for_l = _hmaps(model, xy, nodes, results, A, dist_scale, temperature, device, "inject", donor=B)
        H_self_l = _hmaps(model, xy, nodes, results, A, dist_scale, temperature, device, "inject", donor=A)

        H_own, H_for, H_self = _stack(H_own_l), _stack(H_for_l), _stack(H_self_l)
        sh_for.append(float(_rel_shift(H_own, H_for).mean()))
        sh_self.append(float(_rel_shift(H_own, H_self).mean()))

        x_own, _ = decode_booking_policy(H_own_l, I, nodes, probs, p, C)
        x_for, _ = decode_booking_policy(H_for_l, I, nodes, probs, p, C)
        so, sf = set(x_own), set(x_for)
        jac.append(len(so & sf) / len(so | sf) if (so | sf) else 1.0)
        flips.append(len(so ^ sf))
        dbook.append(len(sf) - len(so))
        print(f"  [coppia {kp+1:>3}/{a.pairs}] shift_for={sh_for[-1]:.4f} "
              f"shift_self={sh_self[-1]:.4f} flips={flips[-1]}  ({__import__('time').time()-t0:.1f}s)",
              flush=True)

    def ms(v):
        return f"{st.fmean(v):.4f} ± {st.pstdev(v):.4f}"

    P("\n1. CONTROLLO DEL MECCANISMO (donatore = A stessa)")
    P(f"   shift heatmap self-injection : {ms(sh_self)}   (deve essere ≈ 0)")
    ok = st.fmean(sh_self) < 1e-5
    P("   → meccanismo OK: l'iniezione riproduce le heatmap vere."
      if ok else
      "   ⚠ shift self ≠ 0: iniezione NON fedele (ordine/numero delle medie non combacia). "
      "Sospendi le conclusioni finché non è ~0.")

    P("\n2. EFFETTO DELL'IMPOLLINAZIONE (donatore = altra istanza B)")
    P(f"   shift heatmap consenso-foreign : {ms(sh_for)}")
    ratio = st.fmean(sh_for) / max(st.fmean(sh_self), 1e-12)
    P(f"   rapporto foreign/self          : {ratio:.1f}×  (quanto il consenso 'sbagliato' muove l'output)")

    P("\n3. EFFETTO SULLE PRENOTAZIONI (regola f>p/C)")
    frac_flip = sum(1 for f in flips if f > 0) / len(flips)
    P(f"   Jaccard prenotazioni (own vs foreign) : {ms(jac)}   (1 = identiche)")
    P(f"   corridoi che cambiano per coppia      : {ms([float(f) for f in flips])}")
    P(f"   coppie con ≥1 cambio                   : {100*frac_flip:.0f}%")
    P(f"   Δ n. prenotazioni (foreign - own)      : {ms([float(d) for d in dbook])}")

    P("\nLettura:")
    if not ok:
        P("  Prima sistema il controllo (punto 1).")
    elif st.fmean(sh_for) < 1e-4 and frac_flip < 0.02:
        P("  Il consenso d'istanza NON conta: iniettarne uno estraneo non muove né heatmap né")
        P("  prenotazioni. Coerente con un modello in cui cross ≈ no-cross (attention sui cross ~0).")
    else:
        P("  Il consenso d'istanza è usato CAUSALMENTE: cambiarlo sposta le heatmap e fa cambiare")
        P("  le prenotazioni. È l'evidenza che il miglioramento del cross non è correlazione: la")
        P("  rete sfrutta davvero la struttura degli ALTRI scenari dell'istanza per decidere.")

    # ── grafici ──────────────────────────────────────────────────────────
    outdir = a.out or os.path.join(os.path.dirname(TEST_SCENARIO_CACHE_DIR), "report")
    os.makedirs(outdir, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hist(sh_self, bins=20, alpha=.6, label="self (controllo)")
    ax[0].hist(sh_for, bins=20, alpha=.6, label="foreign (impollinazione)")
    ax[0].set_title("Spostamento heatmap per coppia"); ax[0].set_xlabel("||ΔH||/||H||")
    ax[0].set_ylabel("coppie"); ax[0].legend(); ax[0].grid(True, ls="--", alpha=.4)
    ax[1].hist(flips, bins=range(0, max(flips) + 2), alpha=.8, color="#C0392B")
    ax[1].set_title("Corridoi con prenotazione cambiata (foreign vs own)")
    ax[1].set_xlabel("n. corridoi cambiati"); ax[1].set_ylabel("coppie"); ax[1].grid(True, ls="--", alpha=.4)
    fig.suptitle(f"Cross-pollination — {exp_name}  DIM={K}", fontweight="bold")
    fig.tight_layout()
    fig_path = os.path.join(outdir, f"crosspollination_dim{K}.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight"); plt.close(fig)

    rep_path = os.path.join(outdir, f"crosspollination_dim{K}.txt")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  → Report: {rep_path}\n  → Grafico: {fig_path}")


if __name__ == "__main__":
    main()
