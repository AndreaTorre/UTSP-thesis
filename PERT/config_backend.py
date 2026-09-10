# -*- coding: utf-8 -*-
"""
Parametri specifici dell'esperimento PERT (perturbazioni sintetiche).

N_NODES arriva già iniettato da common/config.py. WIND_NC_PATH_TRAIN e
WIND_NC_PATH_TEST arrivano iniettati anche qui per uniformità con
CVETT/config_backend.py, ma PERT non li usa.

ROOT_DIR, OUTPUT_DIR, DATA_FILE_PRIMARY/FALLBACK, INSTANCE_TAG restano
in common/config.py: qui non si ricalcola nessun path.
"""

import os

# ── Parametri dipendenti dall'istanza ────────────────────────────
if N_NODES == 15:
    K_MEDOID_NODES        = [70, 101, 84]
    MAX_KMEDOID_I_ARCS    = 7
    KMEDOID_ARCS_PER_NODE = 5
    STO_TIME_LIMIT        = 600          # ~10 min
    STO_MIP_GAP            = 0.005

elif N_NODES == 25:
    K_MEDOID_NODES        = [16, 12, 30, 44, 19]
    MAX_KMEDOID_I_ARCS    = 14
    KMEDOID_ARCS_PER_NODE = 5
    STO_TIME_LIMIT        = 21600
    STO_MIP_GAP            = 0.005

elif N_NODES == 40:
    K_MEDOID_NODES        = [34, 26, 20, 10, 7, 51, 18, 6, 33, 14]
    MAX_KMEDOID_I_ARCS    = 25
    KMEDOID_ARCS_PER_NODE = 5
    STO_TIME_LIMIT        = 43200        # 12h
    STO_MIP_GAP            = 0.005

else:
    raise ValueError(
        f"Istanza non configurata: nodi_{N_NODES}. "
        f"Aggiungi un blocco elif N_NODES == {N_NODES}."
    )

# Override da env di STO_TIME_LIMIT / STO_MIP_GAP (probe e run a 40 nodi:
# tarati sulla curva gap-tempo, non a naso). Default = valori per-taglia sopra.
if os.environ.get("TESI_STO_TIME_LIMIT"):
    STO_TIME_LIMIT = float(os.environ["TESI_STO_TIME_LIMIT"])
if os.environ.get("TESI_STO_MIP_GAP"):
    STO_MIP_GAP = float(os.environ["TESI_STO_MIP_GAP"])

# ── Parametri generali ────────────────────────────────────────────

N_TRAINING_SCENARIOS = 8
DO_VALIDATION = True
N_VALIDATION_SCENARIOS = 300
SCENARIO_IDS = list(range(1, N_TRAINING_SCENARIOS + 1))


PI_TIME_LIMIT  = 300     # secondi, per singolo scenario PI
PI_MIP_GAP     = 0.08

EEV_TIME_LIMIT = 300     # secondi, per singolo solve (medione o second stage)
EEV_MIP_GAP    = 0.08



ESPERIMENTI_DA_ESEGUIRE = ["B", "B_UTSP"]

# Perturbazioni sintetiche
N_EXTRA_ARCS = 30
MEAN_FRAC = 0.40
SIGMA_FRAC = 0.20

# Costi first-stage / penalità
PRENOTAZIONE_FRAC = 0.25
PENALTY_FRAC = 0.50

# Archi frequenti usati nella generazione scenari di B
N_CALIBRATION_SCENARIOS = 15
N_FREQUENT_ARCS = 6
MIN_FREQ_FREQUENT = 0.80

# Per la calibrazione basta un tour buono, non l'ottimo esatto:
# time limit per scenario + gap accettabile, per non restare bloccati.
CALIB_TIME_LIMIT = 30      # secondi per singolo scenario di calibrazione
CALIB_MIP_GAP = 0.05

# ── Iperparametri UTSP 2-stage ───────────────────────────────────
UTSP2_HIDDEN = 64
UTSP2_NLAYERS = 2
UTSP2_LR = 1e-3
UTSP2_EPOCHS = 50
UTSP2_STEP_LR = 10
UTSP2_LOG_FREQ = 5

# Loss weights 
 
UTSP2_LAMBDA1    = 5.0
UTSP2_LAMBDA2    = 1.0
UTSP2_LAMBDA_D   = 3.0
UTSP2_LAMBDA_E   = 0.5
UTSP2_LAMBDA_B_DIV = 1.0   # lambda_B = |Omega| / divisore.  1.0 = comportamento attuale.
UTSP2_TEMP_SCALE = 0.5 

UTSP2_ALPHA_LOSS = 1.0   # saturazione booking/penalty in training


# Temperatura kernel gaussiano adj = exp(-d/T)
UTSP2_TEMP_MODE = "median"
UTSP2_TEMP_FIXED = 1.0

# Normalizzazione distanze interna alla GNN/loss
UTSP2_DIST_SCALE_MODE = "mean_positive"

UTSP2_INCLUDE_PENALTY = os.getenv("TESI_UTSP_INCLUDE_PENALTY", "1").strip() == "1"
UTSP2_INCLUDE_ENTROPY = True

# ── Modalità ed esecuzione UTSP ──────────────────────────────────
# "policy"       = x_utsp + secondo stadio Gurobi
# "local_search" = H_avg + decodifica + local search UTSP
# "both"         = entrambe le valutazioni con un solo training
UTSP_RUN_MODE = "local_search"

UTSP_BATCH_SIZE = int(os.environ.get("TESI_BATCH_SWEEP", 30))
N_ISTANZE_TARGET = 100
N_TRAINING_SCENARIOS_UTSP = UTSP_BATCH_SIZE * N_ISTANZE_TARGET

# NOTA: PERT genera perturbazioni sintetiche, non ha un tetto di
# osservazioni come CVETT (vento ERA5) — quindi qui gli id scenario sono
# semplicemente range, non un campionamento senza reinserimento da un
# pool finito.
#
# ATTENZIONE ai semi: build_perturbation inizializza il generatore con
# Random(base_seed + scenario_id) — una SOMMA. Due stream con semi diversi
# si sovrappongono con offset: (seed_a + s) == (seed_b + t) produce scenari
# IDENTICI. Con il vecchio default TEST_SCENARIO_SEED=99 e training a
# GLOBAL_SEED=42 su id 1..3000, i primi 2943 scenari di test coincidevano
# byte-per-byte con gli scenari di training 58..3000: contaminazione totale
# del test set. Il default 1_000_000 tiene l'intervallo dei semi effettivi
# di test (1_000_001..1_030_000) disgiunto da training (43..3042),
# calibrazione (seed 30) ed Experiment B (seed 42, id 1..8).
TRAIN_SCENARIO_IDS_UTSP = list(range(1, N_TRAINING_SCENARIOS_UTSP + 1))
DROP_LAST_TRAIN_BATCH = os.environ.get("TESI_DROP_LAST_TRAIN_BATCH", "1").strip() == "1"

# Scenari di test comune, usati da _run_utsp_test_only_branch /
# _run_local_search_branch e affettati in istanze da DIM_ISTANZA_TEST /
# N_ISTANZE_TEST (iniettati da common/config.py). Il default copre esattamente
# il caso peggiore N_ISTANZE_TEST x DIM_ISTANZA_TEST; per un test sweep con
# combinazioni più grandi, alza TESI_N_TEST_SCENARIOS_UTSP di conseguenza.
# I due pool: intervalli effettivi (base+id) disgiunti da train (43..3042).
#   validation → 500_001..(500_000+N)   test → 1_000_001..(1_000_000+N)
# EVAL_SPLIT (iniettato da common/config.py) sceglie quale valuta la rete.
NETWORK_TEST_SEED = int(os.environ.get("TESI_TEST_SCENARIO_SEED", 1_000_000))
NETWORK_VAL_SEED  = int(os.environ.get("TESI_VAL_SCENARIO_SEED",    500_000))

N_TEST_SCENARIOS_UTSP = int(os.environ.get(
    "TESI_N_TEST_SCENARIOS_UTSP",
    DIM_ISTANZA_TEST * N_ISTANZE_TEST,
))
TEST_SCENARIO_IDS_UTSP = list(range(1, N_TEST_SCENARIOS_UTSP + 1))


UTSP_LS_MAX_ACTIONS = 5000
UTSP_LS_ACTIONS_PER_ROUND = 120
UTSP_LS_MAX_RESTARTS = 80
UTSP_LS_M = 8
UTSP_LS_K = 15
UTSP_LS_BETA = 10.0

UTSP_LS_APPLY_INITIAL_2OPT = True
UTSP2_LS_ALPHA   = 0.05   # esplorazione UCB nella local search  

# NOTA: PERT non definisce GRID_SEARCH (a differenza di CVETT).
# grid_search.py va quindi eseguito solo sotto TESI_EXPERIMENT=CVETT
# finché non si decide una griglia anche per PERT.