# -*- coding: utf-8 -*-
"""
Parametri specifici dell'esperimento CVETT (perturbazioni da vento ERA5).

N_NODES, WIND_NC_PATH_TRAIN, WIND_NC_PATH_TEST arrivano già iniettati
da common/config.py: qui si usano così come sono, solo per contare le
osservazioni e campionare gli scenari. ROOT_DIR, OUTPUT_DIR,
DATA_FILE_PRIMARY/FALLBACK, INSTANCE_TAG restano in common/config.py:
qui non si ricalcola nessun path (prima venivano ridefiniti da capo,
con un default hardcoded proprio — rimosso: unica fonte di verità ora
è common/config.py).
"""

import os
import random as _random
import h5py


# ============================================================
# FUNZIONI DI SUPPORTO
# ============================================================

def _env_int(name, default):
    return int(os.getenv(name, str(default)))


def _env_float(name, default):
    return float(os.getenv(name, str(default)))


def _env_bool(name, default):
    value = os.getenv(name, str(default)).lower()
    return value in {"1", "true", "yes", "y", "si", "s"}


def _count_wind_observations(nc_path, time_name):
    with h5py.File(nc_path, "r") as ds:
        if time_name in ds:
            return int(ds[time_name].shape[0])

        if "u100" in ds:
            return int(ds["u100"].shape[0])

        raise ValueError(
            f"Non trovo né '{time_name}' né 'u100' nel file: {nc_path}. "
            f"Variabili disponibili: {list(ds.keys())}"
        )


def _sample_scenario_ids(max_id, n_scenarios, seed, label):
    if n_scenarios <= 0:
        raise ValueError(
            f"{label}: il numero di scenari deve essere positivo. "
            f"Valore attuale: {n_scenarios}."
        )

    if n_scenarios > max_id:
        raise ValueError(
            f"{label}: richiesti {n_scenarios} scenari, "
            f"ma ne sono disponibili solo {max_id}."
        )

    rng = _random.Random(seed)

    return sorted(
        rng.sample(
            range(1, max_id + 1),
            n_scenarios,
        )
    )


# ============================================================
# FILE CVETT TRAIN / TEST
# ============================================================

PERTURBATION_MODE = "wind_nc"

WIND_U_VAR = "u100"
WIND_V_VAR = "v100"
WIND_LAT_NAME = "latitude"
WIND_LON_NAME = "longitude"
WIND_TIME_NAME = "valid_time"

N_WIND_OBSERVATIONS_TRAIN = _env_int(
    "TESI_N_WIND_OBSERVATIONS_TRAIN",
    _count_wind_observations(WIND_NC_PATH_TRAIN, WIND_TIME_NAME),
)

N_WIND_OBSERVATIONS_TEST = _env_int(
    "TESI_N_WIND_OBSERVATIONS_TEST",
    _count_wind_observations(WIND_NC_PATH_TEST, WIND_TIME_NAME),
)

if N_WIND_OBSERVATIONS_TRAIN <= 0:
    raise ValueError(
        f"Il file train non contiene osservazioni valide: "
        f"{N_WIND_OBSERVATIONS_TRAIN}."
    )

if N_WIND_OBSERVATIONS_TEST <= 0:
    raise ValueError(
        f"Il file test non contiene osservazioni valide: "
        f"{N_WIND_OBSERVATIONS_TEST}."
    )


# ============================================================
# ESPERIMENTI DA ESEGUIRE
# ============================================================

ESPERIMENTI_DA_ESEGUIRE = ["B", "B_UTSP"]


# ============================================================
# SEMI
# ============================================================

GLOBAL_SEED = _env_int("TESI_GLOBAL_SEED", 42)

# Seme comune per tutte le estrazioni dal file train:
# STO, EEV, PI e UTSP.
TRAIN_SCENARIO_SEED = _env_int("TESI_TRAIN_SCENARIO_SEED", 50)

CALIBRATION_SCENARIO_SEED = TRAIN_SCENARIO_SEED
UTSP_TRAINING_SCENARIO_SEED = TRAIN_SCENARIO_SEED

# Seme comune per tutte le valutazioni sul file test.
TEST_SCENARIO_SEED = _env_int("TESI_TEST_SCENARIO_SEED", 99)

FINAL_SCENARIO_SEED = TEST_SCENARIO_SEED
UTSP_TEST_SCENARIO_SEED = TEST_SCENARIO_SEED

VALIDATION_SEED = _env_int("TESI_VALIDATION_SEED", 123)

UTSP_TRAINING_SEED = GLOBAL_SEED
UTSP_LS_RANDOM_SEED = _env_int("TESI_UTSP_LS_RANDOM_SEED", 12345)


# ============================================================
# SCENARI TRAIN PER STO / EEV / PI
# ============================================================

N_TRAINING_SCENARIOS = _env_int(
    "TESI_N_TRAINING_SCENARIOS",
    40,
)

SCENARIO_IDS = _sample_scenario_ids(
    max_id=N_WIND_OBSERVATIONS_TRAIN,
    n_scenarios=N_TRAINING_SCENARIOS,
    seed=TRAIN_SCENARIO_SEED,
    label="STO/EEV/PI train",
)

PI_TIME_LIMIT  = 300     # secondi, per singolo scenario PI
PI_MIP_GAP     = 0.08

EEV_TIME_LIMIT = 300     # secondi, per singolo solve (medione o second stage)
EEV_MIP_GAP    = 0.08


# ============================================================
# SCENARI DI VALIDAZIONE
# ============================================================

DO_VALIDATION = _env_bool("TESI_DO_VALIDATION", True)

N_VALIDATION_SCENARIOS = _env_int(
    "TESI_N_VALIDATION_SCENARIOS",
    300,
)

VALIDATION_SCENARIO_IDS = (
    _sample_scenario_ids(
        max_id=N_WIND_OBSERVATIONS_TRAIN,
        n_scenarios=N_VALIDATION_SCENARIOS,
        seed=VALIDATION_SEED,
        label="validazione",
    )
    if DO_VALIDATION
    else []
)


# ============================================================
# PARAMETRI VENTO
# ============================================================

WIND_MIN_FACTOR = _env_float("TESI_WIND_MIN_FACTOR", 0.70)
WIND_MAX_FACTOR = _env_float("TESI_WIND_MAX_FACTOR", 1.50)
WIND_TURB_SIGMA = _env_float("TESI_WIND_TURB_SIGMA", 0.08)
WIND_MAP_COORDS_TO_GRID = _env_bool("TESI_WIND_MAP_COORDS_TO_GRID", True)


# ============================================================
# PERTURBAZIONI SINTETICHE (compatibilità con l'interfaccia PERT)
# ============================================================

N_EXTRA_ARCS = _env_int("TESI_N_EXTRA_ARCS", 30)
MEAN_FRAC = _env_float("TESI_MEAN_FRAC", 0.40)
SIGMA_FRAC = _env_float("TESI_SIGMA_FRAC", 0.20)


# ============================================================
# COSTI FIRST-STAGE / PENALITÀ
# ============================================================

PRENOTAZIONE_FRAC = _env_float("TESI_PRENOTAZIONE_FRAC", 0.25)
PENALTY_FRAC = _env_float("TESI_PENALTY_FRAC", 0.50)


# ============================================================
# PARAMETRI GUROBI / MODELLO TWO-STAGE
# ============================================================

N_CALIBRATION_SCENARIOS = _env_int("TESI_N_CALIBRATION_SCENARIOS", 30)
N_FREQUENT_ARCS = _env_int("TESI_N_FREQUENT_ARCS", 6)
MIN_FREQ_FREQUENT = _env_float("TESI_MIN_FREQ_FREQUENT", 0.80)

CALIB_TIME_LIMIT = _env_int("TESI_CALIB_TIME_LIMIT", 30)
CALIB_MIP_GAP = _env_float("TESI_CALIB_MIP_GAP", 0.05)

if N_CALIBRATION_SCENARIOS > N_WIND_OBSERVATIONS_TRAIN:
    raise ValueError(
        f"Richiesti {N_CALIBRATION_SCENARIOS} scenari di calibrazione, "
        f"ma il file train ne contiene solo {N_WIND_OBSERVATIONS_TRAIN}."
    )


# ============================================================
# PARAMETRI DIPENDENTI DALL'ISTANZA
# ============================================================

if N_NODES == 15:
    K_MEDOID_NODES = [70, 101, 84]
    MAX_KMEDOID_I_ARCS = 7
    KMEDOID_ARCS_PER_NODE = 5
    STO_TIME_LIMIT = _env_int("TESI_STO_TIME_LIMIT", 10800)
    STO_MIP_GAP = _env_float("TESI_STO_MIP_GAP", 0.0001)

elif N_NODES == 25:
    K_MEDOID_NODES = [16, 12, 30, 44, 19]
    MAX_KMEDOID_I_ARCS = 14
    KMEDOID_ARCS_PER_NODE = 5
    STO_TIME_LIMIT = _env_int("TESI_STO_TIME_LIMIT", 10800)
    STO_MIP_GAP = _env_float("TESI_STO_MIP_GAP", 0.0001)

elif N_NODES == 40:
    K_MEDOID_NODES = [34, 26, 20, 10, 7, 51, 18, 6, 33, 14]
    MAX_KMEDOID_I_ARCS = 25
    KMEDOID_ARCS_PER_NODE = 5
    STO_TIME_LIMIT = _env_int("TESI_STO_TIME_LIMIT", 43200)
    STO_MIP_GAP = _env_float("TESI_STO_MIP_GAP", 0.005)

else:
    raise ValueError(f"Istanza non configurata: nodi_{N_NODES}")


# ============================================================
# PARAMETRI TRAIN RETE UTSP
# ============================================================

UTSP_BATCH_SIZE = _env_int("TESI_UTSP_BATCH_SIZE", 30)

# Numero di istanze di training della rete.
# Questo resta fisso quando confronti batch size diversi.
N_UTSP_TRAIN_INSTANCES = _env_int(
    "TESI_N_UTSP_TRAIN_INSTANCES",
    100,
)

# Numero totale di scenari usati dalla rete nel singolo esperimento.
# Cambia con UTSP_BATCH_SIZE perché ogni istanza contiene batch_size scenari.
N_TRAINING_SCENARIOS_UTSP = N_UTSP_TRAIN_INSTANCES * UTSP_BATCH_SIZE

TRAIN_SCENARIO_IDS_UTSP = _sample_scenario_ids(
    max_id=N_WIND_OBSERVATIONS_TRAIN,
    n_scenarios=N_TRAINING_SCENARIOS_UTSP,
    seed=UTSP_TRAINING_SCENARIO_SEED,
    label="UTSP train",
)

N_TRAINING_BATCHES = N_UTSP_TRAIN_INSTANCES


# ============================================================
# PARAMETRI TEST COMUNE
# ============================================================

N_TEST_SCENARIOS = _env_int(
    "TESI_N_TEST_SCENARIOS",
    N_WIND_OBSERVATIONS_TEST,
)

TEST_SCENARIO_IDS = _sample_scenario_ids(
    max_id=N_WIND_OBSERVATIONS_TEST,
    n_scenarios=N_TEST_SCENARIOS,
    seed=TEST_SCENARIO_SEED,
    label="test comune",
)

# Alias per compatibilità con codice UTSP esistente.
N_TEST_SCENARIOS_UTSP = N_TEST_SCENARIOS
TEST_SCENARIO_IDS_UTSP = TEST_SCENARIO_IDS



DROP_LAST_TRAIN_BATCH = _env_bool("TESI_DROP_LAST_TRAIN_BATCH", True)
DROP_LAST_TEST_BATCH = _env_bool("TESI_DROP_LAST_TEST_BATCH", False)

if DROP_LAST_TRAIN_BATCH and N_TRAINING_SCENARIOS_UTSP % UTSP_BATCH_SIZE != 0:
    raise ValueError(
        f"Training non divisibile in batch completi: "
        f"{N_TRAINING_SCENARIOS_UTSP} scenari, "
        f"batch_size={UTSP_BATCH_SIZE}."
    )

if DROP_LAST_TEST_BATCH and N_TEST_SCENARIOS_UTSP % UTSP_BATCH_SIZE != 0:
    raise ValueError(
        f"Test non divisibile in batch completi: "
        f"{N_TEST_SCENARIOS_UTSP} scenari, "
        f"batch_size={UTSP_BATCH_SIZE}."
    )

N_TEST_BATCHES = (
    N_TEST_SCENARIOS_UTSP + UTSP_BATCH_SIZE - 1
) // UTSP_BATCH_SIZE


# ============================================================
# IPERPARAMETRI RETE UTSP 2-STAGE
# ============================================================

UTSP2_HIDDEN = _env_int("TESI_UTSP2_HIDDEN", 64)
UTSP2_NLAYERS = _env_int("TESI_UTSP2_NLAYERS", 2)
UTSP2_LR = _env_float("TESI_UTSP2_LR", 1e-3)
UTSP2_EPOCHS = _env_int("TESI_UTSP2_EPOCHS", 50)
UTSP2_STEP_LR = _env_int("TESI_UTSP2_STEP_LR", 10)
UTSP2_LOG_FREQ = _env_int("TESI_UTSP2_LOG_FREQ", 5)

# Configurazione loss con aggregazione .sum
UTSP2_LS_ALPHA = _env_float("TESI_UTSP2_LS_ALPHA", 0.01)
UTSP2_LAMBDA1 = _env_float("TESI_UTSP2_LAMBDA1", 5.0)
UTSP2_LAMBDA2 = _env_float("TESI_UTSP2_LAMBDA2", 1.0)
UTSP2_LAMBDA_D = _env_float("TESI_UTSP2_LAMBDA_D", 3.0)
UTSP2_LAMBDA_E = _env_float("TESI_UTSP2_LAMBDA_E", 0.5)
UTSP2_TEMP_SCALE = _env_float("TESI_UTSP2_TEMP_SCALE", 0.5)

UTSP2_ALPHA_LOSS = _env_float("TESI_UTSP2_ALPHA_LOSS", 0.6)
UTSP2_ALPHA_DECODE = _env_float("TESI_UTSP2_ALPHA_DECODE", 4.0)

UTSP2_TEMP_MODE = os.getenv("TESI_UTSP2_TEMP_MODE", "median")
UTSP2_TEMP_FIXED = _env_float("TESI_UTSP2_TEMP_FIXED", 1.0)

UTSP2_DIST_SCALE_MODE = os.getenv(
    "TESI_UTSP2_DIST_SCALE_MODE",
    "mean_positive",
)

UTSP2_INCLUDE_PENALTY = _env_bool("TESI_UTSP_INCLUDE_PENALTY", True)   # NOTA: era TESI_UTSP2_*, non coincideva con PERT
UTSP2_INCLUDE_ENTROPY = _env_bool("TESI_UTSP2_INCLUDE_ENTROPY", False)


# ============================================================
# GRIGLIA DI RICERCA
# ============================================================

GRID_SEARCH = {
    "UTSP2_LAMBDA1": [5.0, 10.0, 20.0],
    "UTSP2_LAMBDA2": [0.5, 1.0, 3.0],
    "UTSP2_LAMBDA_D": [1.0, 3.0, 5.0],
}


# ============================================================
# MODALITÀ DI VALUTAZIONE UTSP
# ============================================================

# "policy"       = x_utsp + secondo stadio Gurobi
# "local_search" = H_avg + decodifica + local search UTSP
# "both"         = entrambe le valutazioni con un solo training
UTSP_RUN_MODE = os.getenv("TESI_UTSP_RUN_MODE", "local_search")


# ============================================================
# LOCAL SEARCH UTSP
# ============================================================

UTSP_LS_MAX_ACTIONS = _env_int("TESI_UTSP_LS_MAX_ACTIONS", 5000)
UTSP_LS_ACTIONS_PER_ROUND = _env_int("TESI_UTSP_LS_ACTIONS_PER_ROUND", 120)
UTSP_LS_MAX_RESTARTS = _env_int("TESI_UTSP_LS_MAX_RESTARTS", 80)
UTSP_LS_M = _env_int("TESI_UTSP_LS_M", 8)
UTSP_LS_K = _env_int("TESI_UTSP_LS_K", 15)
UTSP_LS_BETA = _env_float("TESI_UTSP_LS_BETA", 10.0)

UTSP_LS_APPLY_INITIAL_2OPT = _env_bool(
    "TESI_UTSP_LS_APPLY_INITIAL_2OPT",
    True,
)


# ============================================================
# PARAMETRI DRONE / MODELLO ENERGETICO
# ============================================================

DRONE_U  = 12.0          # ground speed nominale [m/s]
DRONE_A0 = 168.49        # W
DRONE_A2 = 1.66375e-2    # W s^2 m^-2
DRONE_A3 = 9.242625e-3   # W s^3 m^-3