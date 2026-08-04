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
  `common/config.py` prima dell'esecuzione. **Nessun valore definito qui è
  definitivo**: `common/config.py` applica dopo il layer `TESI_P_<NOME>`.
- `main.py` — entry point (chiamato da `common/main.py`): carica/costruisce
  `res_B_cached.pkl` e lancia `run_esperimento_B_UTSP`.
- `gurobi_parallelo.py` — fasi della pipeline parallela (setup/pi/eev/sto/assemble).
  Usa `@contextmanager gurobi_env()` per rilasciare il token anche su eccezione.
- `run_tutto.sh <15|25|40>` — sottomette l'intera pipeline B su SLURM.
- `collect_pert_results.sh` — aggregazione risultati.
- `plot_graph_totale_scenari.py` — figure per la tesi, non pipeline.

I wrapper `run_b.sh` e `run_exp.sh` sono stati rimossi: `common/submit_b.sh` e
`common/submit_exp.sh` accettano l'esperimento come argomento.

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
cd ../common
source env.sh
export TESI_EXPERIMENT=PERT TESI_N_NODES=15 TESI_BATCH_SWEEP=20
export TESI_N_ISTANZE_TEST=100 TESI_DIM_ISTANZA_TEST=20

python main.py --only B_UTSP_LS       # interattivo: training + test
bash submit_exp.sh PERT 15            # oppure come job
```

Solo test, riusando il modello già addestrato:

```bash
export TESI_UTSP_TEST_ONLY=1 TESI_TEST_OUTPUT_SUBDIR=IS100_DIM20
python main.py --only B_UTSP_LS
unset TESI_UTSP_TEST_ONLY
```

È anche il meccanismo per tarare i parametri di local search
(`TESI_P_UTSP2_LS_ALPHA`, `TESI_P_UTSP_LS_M`, `TESI_P_UTSP_LS_K`, …) senza
riaddestrare: minuti invece di ore.

Il test sweep può saltare il PI (`TESI_TEST_SKIP_PI=1`) — con WS disponibile è
il bound giusto e il PI diventa opzionale. Per aggiungerlo dopo, sugli stessi
scenari:

```bash
TESI_TEST_SKIP_PI=0 bash run_test_sweep.sh PERT 25
```

Il fill-in risolve solo i PI mancanti sulla `scenario_dist` già in cache: zero
rigenerazioni, aggancio per `scenario_id` garantito.

## Grid search

```bash
cd ../common && source env.sh

# 1. scalda la cache WS con una run singola (evita che le prime combo
#    in parallelo risolvano gli stessi MIP)
export TESI_BATCH_SWEEP=20 TESI_N_ISTANZE_TEST=100 TESI_DIM_ISTANZA_TEST=20
bash submit_exp.sh PERT 15

# 2. misura quanto dura UNA combinazione a cache calda
python grid_search.py --run-one 1 --exp PERT --nodes 15 --batch 20

# 3. dimensiona il chunk: CHUNK_SIZE ≈ 1440 × 0.8 / T_minuti
export CHUNK_SIZE=<...> MAX_PARALLEL=6 TIME=1-00:00:00 TIMEOUT=10800
DRY=1 bash grid_submit.sh PERT 15 all
bash grid_submit.sh PERT 15 all

# 4. analisi
python grid_analyze.py --exp PERT --nodes 15 --batch 20
python grid_analyze.py --all --top 8
```

Il chunk deve stare dentro il wall time. Sbagliare per difetto costa poco: un
chunk troncato perde solo le combo non ancora scritte, e rilanciare lo stesso
comando riprende da dove si era fermato.

Serve `RISULTATI_N/pkl/res_B_cached.pkl`: la grid non ricalcola STO/EEV/PI, e
`grid_submit.sh` salta con un avviso le taglie che non ce l'hanno.

## Output

```
RISULTATI_N/
├── pkl/            res_B_cached.pkl, cache scenari/WS/STO/EEV, pool
├── output/         log SLURM
├── checkpoint/     checkpoint intermedi Experiment B
├── report/         .txt e .csv di risultato
├── grafici/        .png
├── modello/        utsp_model.pt, history, metadata
└── batch_sweep/BATCH_X/
    ├── report/ grafici/ modello/
    └── <TESI_TEST_OUTPUT_SUBDIR>/report/ …
```

Solo `report/`, `grafici/` e `modello/` seguono le mutazioni di `OUTPUT_DIR`.
`pkl/`, `output/` e `checkpoint/` restano a livello `RISULTATI_N` e sono
condivisi da tutte le run della stessa taglia.

I risultati della grid vivono fuori, in
`grid_search/PERT/NODI_<n>/BATCH_<b>/combo_XXXX/`, con la stessa struttura
per-run.

## Cosa leggere nei report

| File | Contenuto |
|------|-----------|
| `report/esperimento_B_espB.txt` | riepilogo Experiment B |
| `report/validazione_*.txt` | validazione out-of-sample |
| `report/utsp_ls_espB_UTSP_LS.txt` | riepilogo train/test UTSP |
| `report/*_test_all_instances.txt` | una riga per istanza di test |
| `report/*_test_aggregate.txt` | medie e deviazioni su tutte le istanze |
| `report/controllo_prenotazioni_*.txt` | heatmap vs tour, arco per arco |

Il numero da citare è quello **aggregato**, non l'ultima istanza. La `std` che
compare accanto è la dispersione **fra istanze**: dice quanto il risultato è
stabile al variare del campione di test.
