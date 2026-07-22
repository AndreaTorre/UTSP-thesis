import os
import importlib.util
from pathlib import Path

# ============================================================
# CONFIG UNICO UTSP
# ============================================================
# Tutto si controlla da variabili d'ambiente:
#
#   TESI_EXPERIMENT=PERT oppure CVETT
#   TESI_N_NODES=15 oppure 25 oppure 40
#   TESI_ROOT_DIR=...            (opzionale, default sotto)
#
# Esempio:
#   TESI_EXPERIMENT=PERT TESI_N_NODES=40 python main.py --only B
#   TESI_EXPERIMENT=CVETT TESI_N_NODES=25 python main.py --only B_UTSP_LS
#
# Questo file è l'UNICA fonte di verità per i path. config_backend.py
# (uno diverso per PERT e per CVETT, stesso nome file) riceve i path
# già pronti e si occupa solo di ciò che cambia tra i due esperimenti:
# iperparametri, costanti dipendenti da N_NODES, parametri vento.
# ============================================================

ROOT_DIR = Path(os.getenv("TESI_ROOT_DIR", "/home/atorre/UTSP/unione/git/UTSP"))

EXPERIMENT = os.getenv("TESI_EXPERIMENT", "PERT").upper()
N_NODES = int(os.getenv("TESI_N_NODES", "40"))

if EXPERIMENT not in {"PERT", "CVETT"}:
    raise ValueError(f"TESI_EXPERIMENT non valido: {EXPERIMENT}. Usa PERT oppure CVETT.")
if N_NODES not in {15, 25, 40}:
    raise ValueError(f"TESI_N_NODES non valido: {N_NODES}. Usa 15, 25 oppure 40.")

INSTANCE_TAG = str(N_NODES)
IS_PERT = EXPERIMENT == "PERT"
IS_CVETT = EXPERIMENT == "CVETT"

DATA_DIR = ROOT_DIR / "data"
PERT_DATA_DIR = DATA_DIR / "pert"
CVETT_DATA_DIR = DATA_DIR / "cvett"

DATA_FILE_PRIMARY = str(PERT_DATA_DIR / f"nodi_{N_NODES}.json")
DATA_FILE_FALLBACK = DATA_FILE_PRIMARY

EXPERIMENT_DIR = ROOT_DIR / EXPERIMENT
OUTPUT_DIR = str(EXPERIMENT_DIR / f"RISULTATI_{N_NODES}")

BASE_OUTPUT_DIR = OUTPUT_DIR
UTSP_VARIANT = os.getenv("TESI_UTSP_VARIANT", "").strip()
if UTSP_VARIANT:
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, UTSP_VARIANT)

# Cartella condivisa per la cache degli scenari di test (PI/perturbazioni).
# NOTA: fissa apposta, non segue batch_sweep né la sottocartella di test:
# il PI di uno scenario_id non dipende dal batch size della rete né da
# quante istanze/DIM stai testando, quindi la cache va condivisa da tutti.
TEST_SCENARIO_CACHE_DIR = os.path.join(BASE_OUTPUT_DIR, "pkl")

WIND_NC_PATH_TRAIN = str(CVETT_DATA_DIR / "cvett_train.nc")
WIND_NC_PATH_TEST = str(CVETT_DATA_DIR / "cvett_test.nc")

# Alias vecchi, tenuti solo per compatibilità con codice che importa
# ancora questi nomi (puntano entrambi al file di train).
ERA5_NC_PATH_TRAIN = WIND_NC_PATH_TRAIN
ERA5_NC_PATH_TEST = WIND_NC_PATH_TEST
WIND_NC_PATH = WIND_NC_PATH_TRAIN
ERA5_NC_PATH = ERA5_NC_PATH_TRAIN

ROOT_DIR = str(ROOT_DIR)
EXPERIMENT_DIR = str(EXPERIMENT_DIR)

for _subdir in ("output", "grafici", "checkpoint", "pkl"):
    os.makedirs(os.path.join(OUTPUT_DIR, _subdir), exist_ok=True)

# ============================================================
# Backend specifico dell'esperimento (solo iperparametri)
# ============================================================
# PERT/config_backend.py e CVETT/config_backend.py hanno lo stesso
# nome file, quindi vanno caricati per percorso esplicito invece che
# con un semplice `import`.
#
# I tre valori sotto vengono iniettati nel modulo PRIMA di eseguirlo:
# il backend li trova già come variabili definite. Niente ricalcolo
# di ROOT_DIR/OUTPUT_DIR/DATA_FILE_* nel backend, niente passaggio
# per os.environ, niente sezione di "override" dopo.

_backend_path = os.path.join(ROOT_DIR, EXPERIMENT, "config_backend.py")
_spec = importlib.util.spec_from_file_location("_config_backend", _backend_path)
_backend = importlib.util.module_from_spec(_spec)

_backend.N_NODES = N_NODES
_backend.WIND_NC_PATH_TRAIN = WIND_NC_PATH_TRAIN
_backend.WIND_NC_PATH_TEST = WIND_NC_PATH_TEST

# TESI_DIM_ISTANZA_TEST / TESI_N_ISTANZE_TEST: dimensione e numero delle
# istanze di test-sweep. Iniettate nel backend PRIMA di eseguirlo perché
# N_TEST_SCENARIOS_UTSP (definito nel backend) deve poterne dipendere,
# così basta un solo posto dove cambiare la dimensione del pool di test.
DIM_ISTANZA_TEST = int(os.getenv("TESI_DIM_ISTANZA_TEST", "20"))
N_ISTANZE_TEST = int(os.getenv("TESI_N_ISTANZE_TEST", "100"))
_backend.DIM_ISTANZA_TEST = DIM_ISTANZA_TEST
_backend.N_ISTANZE_TEST = N_ISTANZE_TEST

_spec.loader.exec_module(_backend)

for _name in dir(_backend):
    if _name.isupper():
        globals()[_name] = getattr(_backend, _name)
        
_penalty_env = os.getenv("TESI_UTSP_INCLUDE_PENALTY")
if _penalty_env is not None:
    UTSP2_INCLUDE_PENALTY = _penalty_env.strip().lower() in {"1", "true", "yes"}

UTSP2_AGGREGATION = os.getenv("TESI_UTSP_AGGREGATION", "sum").strip().lower()
if UTSP2_AGGREGATION not in {"sum", "mean"}:
    raise ValueError(f"TESI_UTSP_AGGREGATION non valido: {UTSP2_AGGREGATION}")

# ============================================================
# Batch sweep satellite (opzionale)
# ============================================================
# TESI_BATCH_SWEEP, se settata, sposta l'output sotto
# OUTPUT_DIR/batch_sweep/BATCH_<size>/ e forza UTSP_BATCH_SIZE
# a quel valore. Se non settata, nessun cambiamento.

_variant = os.getenv("TESI_VARIANT")
if _variant is not None:
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, "variants", _variant)
    TEST_SCENARIO_CACHE_DIR = os.path.join(OUTPUT_DIR, "pkl")
    for _subdir in ("output", "grafici", "checkpoint", "pkl"):
        os.makedirs(os.path.join(OUTPUT_DIR, _subdir), exist_ok=True)

_batch_sweep = os.getenv("TESI_BATCH_SWEEP")
if _batch_sweep is not None:
    UTSP_BATCH_SIZE = int(_batch_sweep)
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, "batch_sweep", f"BATCH_{UTSP_BATCH_SIZE}")
    for _subdir in ("output", "grafici", "checkpoint", "pkl"):
        os.makedirs(os.path.join(OUTPUT_DIR, _subdir), exist_ok=True)

# ============================================================
# Sottocartella di test sweep (opzionale)
# ============================================================
# TESI_TEST_OUTPUT_SUBDIR, se settata, sposta l'output sotto
# OUTPUT_DIR/<valore>, tipicamente "test/IS_<n>_DIM_<dim>". Usata insieme
# a TESI_UTSP_TEST_ONLY=1 per isolare ogni combinazione (IS, DIM) del
# test sweep dentro la cartella del suo BATCH_X, senza toccare train/.
# NOTA: TEST_SCENARIO_CACHE_DIR resta quello calcolato sopra, prima di
# questa mutazione: la cache PI è condivisa da tutte le combinazioni.

# I checkpoint di training vivono SEMPRE in <batch_dir>/train/<nome>, anche
# quando l'output del test è deviato nella sottocartella: TRAIN_OUTPUT_DIR
# congela il valore PRIMA della mutazione. Senza questo, in modalità
# test-only _load_utsp_train_artifact cercherebbe il modello dentro
# test/IS_*_DIM_*/train/... che non esiste (FileNotFoundError).
TRAIN_OUTPUT_DIR = OUTPUT_DIR

# TESI_TEST_SKIP_PI=1: genera gli scenari di test SENZA risolvere il PI
# (solve_exact_tsp), che non serve per confrontare STO/EEV/UTSP. Le entry
# in cache restano complete di perturbazioni e scenario_dist, con
# exact_free vuoto: un run successivo con TESI_TEST_SKIP_PI=0 riempie i
# PI mancanti sugli STESSI scenari, senza rigenerare né spostare nulla.
TEST_SKIP_PI = os.getenv("TESI_TEST_SKIP_PI", "0").strip() == "1"

_test_subdir = os.getenv("TESI_TEST_OUTPUT_SUBDIR")
if _test_subdir is not None:
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, _test_subdir)
    for _subdir in ("output", "grafici", "checkpoint", "pkl"):
        os.makedirs(os.path.join(OUTPUT_DIR, _subdir), exist_ok=True)

# Alcuni script (grid_search.py, run_single.py) lanciano sotto-processi
# che devono ereditare l'OUTPUT_DIR risolto qui, batch sweep incluso.
os.environ["TESI_OUTPUT_DIR"] = OUTPUT_DIR
