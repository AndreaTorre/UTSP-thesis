# PERT — perturbazioni sintetiche

Regime di incertezza sintetico: per ogni scenario, un sottoinsieme di archi
(`I` ∪ archi frequenti ∪ estrazione casuale di `N_EXTRA_ARCS`) riceve una
perturbazione gaussiana direzionale (`MEAN_FRAC`, `SIGMA_FRAC`) sulla distanza
base. Tutto è deterministico dato `(base_seed, scenario_id)`.

## ⚠️ Architettura dei semi — leggere prima di toccare qualunque seme

`build_perturbation` inizializza il generatore con `Random(base_seed + scenario_id)`:
una **somma**. Due stream con semi diversi producono scenari **identici** con
offset: `(seed_a + s) == (seed_b + t)` ⇒ stesso scenario. Gli stream attivi:

| Stream | Seme | Id | Semi effettivi |
|--------|------|-----|----------------|
| Calibrazione archi frequenti | 30 | 1..N_calib | 31..30+N |
| Experiment B (STO/EEV) | 42 | 1..8 | 43..50 |
| Training UTSP | 42 | 1..3000 | 43..3042 |
| **Test UTSP** | **1_000_000** | 1..30000 | 1_000_001..1_030_000 |

Il seme di test è 1.000.000 proprio per stare fuori da ogni altro intervallo:
col vecchio default (99) i primi 2943 scenari di test coincidevano
byte-per-byte con scenari di training. Se aggiungi un nuovo stream, scegli un
seme il cui intervallo `[seme+1, seme+max_id]` sia disgiunto da tutti quelli
in tabella.

## File

- `config_backend.py` — tutti i parametri PERT: semi, perturbazioni, costi
  `p`/`C`, limiti Gurobi, iperparametri GNN e local search, dimensioni test.
  Riceve `N_NODES`, `DIM_ISTANZA_TEST`, `N_ISTANZE_TEST` iniettati da
  `common/config.py` prima dell'esecuzione.
- `main.py` — entry point (chiamato da `common/main.py`): carica/costruisce
  `res_B_cached.pkl` e lancia `run_esperimento_B_UTSP`.
- `gurobi_parallelo.py` — fasi della pipeline parallela (setup/pi/eev/sto/assemble).
- `run_tutto.sh <15|25|40>` — sottomette l'intera pipeline B su SLURM con dipendenze.
- `collect_pert_results.sh`, `analyze_batch_sweep.py` — aggregazione risultati.

## Pipeline Experiment B (parallela)

```
setup ──► pi (8 TSP liberi) ──┐
     ├──► eev (medione + 8)  ─┼──► assemble ──► RISULTATI_N/pkl/res_B_cached.pkl
     └──► sto (two-stage)    ─┘
```

```bash
cd PERT && bash run_tutto.sh 15    # oppure 25, 40
```

STO a 40 nodi può richiedere ore (vedi `STO_TIME_LIMIT`). I job esportano
`TESI_EXPERIMENT=PERT` e `TESI_N_NODES` automaticamente.

## Training e test UTSP

```bash
# training, uno per batch size (checkpoint in batch_sweep/BATCH_X/train/)
cd ../common && bash run_batch_sweep.sh PERT 25 "20 30 40 50 55 60 65 70"

# test sweep su tutti i checkpoint esistenti (via sbatch, vedi README root)
bash run_test_sweep.sh PERT 25
```

Il test sweep di default **salta il PI** (`TESI_TEST_SKIP_PI=1`): confronta
STO/EEV/UTSP. Per aggiungere il PI dopo, sugli stessi scenari:

```bash
TESI_TEST_SKIP_PI=0 bash run_test_sweep.sh PERT 25
```

Il fill-in risolve solo i PI mancanti sulla `scenario_dist` già in cache —
zero rigenerazioni, aggancio per scenario_id garantito.

## Output

```
RISULTATI_N/
├── pkl/            res_B_cached.pkl, cache scenari test, cache STO/EEV
├── output/         log SLURM
├── grafici/        confronti PI/STO/EEV/UTSP
├── checkpoint/     checkpoint intermedi Experiment B
└── batch_sweep/BATCH_X/
    ├── train/<nome>/   utsp_model.pt, history, metadata
    └── test/IS_<n>_DIM_<d>/   risultati di ogni combinazione del test sweep
```
