# common/ — logica condivisa PERT/CVETT

Unica fonte di verità: qui vive tutto il codice; `PERT/` e `CVETT/` contengono
solo `config_backend.py`, `main.py` e script di lancio.

## Mappa dei moduli

| File | Ruolo |
|------|-------|
| `config.py` | Legge `TESI_EXPERIMENT`/`TESI_N_NODES`, risolve percorsi e OUTPUT_DIR, inietta i parametri nel `config_backend.py` dell'esperimento e lo esegue. Gestisce le mutazioni di OUTPUT_DIR per batch sweep e test sweep. |
| `common.py` | `load_data` (JSON nodi), `load_env` (ambiente Gurobi), semi, utilità directory. |
| `tsp_utils.py` | Archi canonici, ricostruzione tour, costi. |
| `scenarios.py` | `build_perturbation` (sintetica), `generate_scenarios` (con `solve_pi=`), `find_frequent_arcs` (calibrazione con checkpoint), `generate_scenario_batches`. |
| `wind_perturbation.py` | Caricamento NetCDF ERA5, proiezione vento sugli archi, `build_wind_perturbation` (deterministica, scenario_id = indice temporale). |
| `gurobi_models.py` | `solve_exact_tsp` (PI, MTZ), `solve_reservation_tsp` (x fissabile), `solve_stochastic` (two-stage, salva status/gap/bound), `build_I_from_medoid_outgoing_nodes`. Tutti `Threads=1, Seed=42` per riproducibilità. |
| `experiment_B.py` | Orchestra il benchmark: I → calibrazione → scenari → EEV → STO → riepiloghi. Checkpoint ripristinabili. Produce `res_B`. |
| `two_stage_utsp_loss.py` | Loss a 7 termini (row-wise, self-loop, distanza, booking sigmoide, penalità, consistency, asimmetria) + costruzione heatmap e decodifica booking. |
| `utsp.py` | GNN (GCN + 4 canali scattering, pesi appresi), training su batch di scenari, salvataggio/caricamento checkpoint con controllo compatibilità, `run_esperimento_B_UTSP`. |
| `local_search.py` | LS guidata da heatmap, generazione istanze di test con cache, `_validate_policies_cached` (STO/EEV per scenario), rami train/test-only, riepiloghi. |
| `evaluation.py` | EEV sul medione, `validate_policies`, statistiche, tutti i grafici. |
| `main.py` | Entry point comune: stampa la configurazione risolta e delega a `<EXP>/main.py`. |

Script: `run_batch_sweep.sh` (training per batch size), `run_test_sweep.sh`
(test di tutti i checkpoint su più DIM), `submit_b.sh`/`submit_exp.sh`
(sottomissione SLURM parametrica), `collect_results.py`.

## Variabili d'ambiente (riferimento)

| Variabile | Effetto | Default |
|-----------|---------|---------|
| `TESI_EXPERIMENT` | PERT o CVETT | obbligatoria |
| `TESI_N_NODES` | 15, 25, 40 | obbligatoria |
| `TESI_ROOT_DIR` | override della root del progetto | path narval |
| `TESI_BATCH_SWEEP=<X>` | `UTSP_BATCH_SIZE=X` e OUTPUT_DIR → `batch_sweep/BATCH_X/` | off |
| `TESI_UTSP_TEST_ONLY=1` | carica il checkpoint, salta il training | off |
| `TESI_UTSP_TRAIN_NAME` | nome del training da caricare | `espB_UTSP_LS` |
| `TESI_TEST_OUTPUT_SUBDIR` | OUTPUT_DIR → `<...>/test/IS_x_DIM_y` (i checkpoint restano in `TRAIN_OUTPUT_DIR`) | off |
| `TESI_DIM_ISTANZA_TEST` | scenari per istanza di test | 20 |
| `TESI_N_ISTANZE_TEST` | numero istanze di test | 100 |
| `TESI_N_TEST_SCENARIOS_UTSP` | ampiezza pool scenari test (PERT) | IS×DIM |
| `TESI_TEST_SCENARIO_SEED` | seme scenari test (PERT: vedi tabella semi nel README PERT) | 1_000_000 |
| `TESI_TEST_SKIP_PI=1` | genera scenari senza risolvere il PI; `=0` fa il fill-in dei PI mancanti | 0 (1 nello sweep) |
| `TESI_DROP_LAST_TRAIN_BATCH` | scarta l'ultimo batch di training incompleto | 1 |

## Il contratto `res_B`

`experiment_B.py` → `utsp.py` comunica con un dict contenente almeno:
`I, b, p, C, results, scenario_probs, frequent_arcs, PI, STO, EEV, x_ev,
x_used_sto, stoch_costs, eev_costs, eev_solutions, stoch_solutions`.
È cacheato in `RISULTATI_N/pkl/res_B_cached.pkl` (path fisso, non segue le
mutazioni di OUTPUT_DIR).

## Sistema di cache dei test

Tutte in `RISULTATI_N/pkl/` (deliberatamente **prima** delle mutazioni
batch/test di OUTPUT_DIR: condivise da ogni batch e combinazione):

- `test_scenarios_cache.pkl` — per scenario_id: perturbazione, `scenario_dist`,
  `exact_free` (PI; vuoto se generato con skip-PI, riempito dal fill-in).
- `test_sto_eev_cache.pkl` — per scenario_id: costo delle policy STO/EEV
  (reservation-TSP a `x` fisso ⇒ indipendente dal blocco).

Le chiavi includono: seme, `I`, archi frequenti, sorgente scenari (impronta
md5 del campo vento, o "synthetic") e — per STO/EEV — le policy `x_sto`/`x_ev`.
Cambia uno di questi ⇒ la cache riparte da zero, mai dati sporchi. Scrittura
atomica (tmp + rename), salvataggio incrementale per blocco: un crash a metà
non perde il lavoro fatto.

**Vincolo di concorrenza**: le cache non sono progettate per scritture
parallele. Un solo job di sweep per (esperimento, N) alla volta; job su grafi
diversi (15 vs 25) sono indipendenti e possono girare insieme.
