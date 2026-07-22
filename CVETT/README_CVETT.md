# CVETT — perturbazioni da vento ERA5

Regime di incertezza fisico: la perturbazione di ogni arco deriva dalla
proiezione del campo di vento ERA5 (componenti `u100`/`v100`) lungo l'arco.
`build_wind_perturbation` è **deterministica dal vento**: nessun generatore
casuale, `scenario_id` è l'indice (1-based) dell'istante temporale nel file
NetCDF. Id fuori range ⇒ errore esplicito.

## Separazione train/test: fisica, non per seme

A differenza di PERT, train e test usano **due file `.nc` distinti**
(`WIND_NC_PATH_TRAIN`, `WIND_NC_PATH_TEST`): nessuna possibilità di
contaminazione per costruzione. I limiti del test set derivano dal numero di
osservazioni del file (`N_WIND_OBSERVATIONS_TEST`): se chiedi più scenari di
quante osservazioni esistono, il config va in errore — non genera dati sporchi.
Se serve un test set più grande, serve un file `.nc` più lungo.

## ⚠️ Storia importante: il bug `coords` (risolto)

Fino al fix, `generate_scenarios` attivava il ramo vento solo se riceveva sia
`wind` sia `coords` — e `coords` non veniva mai passato dalla pipeline UTSP.
Risultato: Experiment B generava i benchmark su scenari **da vento**, mentre
training e test della rete giravano su scenari **sintetici**. Qualunque
risultato o checkpoint CVETT prodotto prima del fix è quindi **non valido** e
va rigenerato (riallenando la rete). I checkpoint non contengono ancora la
sorgente degli scenari nei metadati: non fidarsi di checkpoint CVETT di cui
non si conosce la data rispetto al fix.

## File

- `config_backend.py` — parametri CVETT: semi (`TRAIN_SCENARIO_SEED=50`,
  `TEST_SCENARIO_SEED=99` — qui i semi contano solo per il fallback sintetico,
  il vento non li usa), conteggio osservazioni dai file `.nc`, campionamento
  scenari, costi, Gurobi, GNN, local search. Molti valori sovrascrivibili da
  env (`TESI_GLOBAL_SEED`, `TESI_N_WIND_OBSERVATIONS_TRAIN/TEST`, ...).
- `config_backend_pre_split_dati.py` — versione storica pre-split dei dati
  (solo riferimento, non usata dalla pipeline).
- `main.py` — entry point: carica il campo vento da `ERA5_NC_PATH`, usa
  `run_esperimento_B_wind(...)` e passa `wind_train`/`wind_test` alla parte UTSP.
- `gurobi_parallelo.py` — fasi pipeline B (variante CVETT).
- `run_b.sh <N>` / `run_exp.sh <N>` — lancio Experiment B / esperimento
  completo via `common/submit_b.sh` e `common/submit_exp.sh`.
- `run_tutto.sh <N>` — pipeline B parallela con dipendenze SLURM.
- `plot_mappa_vento_tsp_montreal.py` — script per le figure della tesi
  (mappa vento + tour), non fa parte della pipeline.

## Comandi

```bash
cd CVETT
bash run_b.sh 15      # solo Experiment B
bash run_exp.sh 15    # esperimento completo (B + UTSP)
```

Training per batch e test sweep: stessi comandi di PERT
(`common/run_batch_sweep.sh CVETT <N> ...`, `common/run_test_sweep.sh CVETT <N>`),
ma **solo dopo aver riallenato i checkpoint post-fix**. Prima del test sweep
verifica che il file vento di test copra `N_ISTANZE_TEST × DIM_ISTANZA_TEST`
osservazioni.

## Output

Identico a PERT: `RISULTATI_N/{pkl,output,grafici,checkpoint,batch_sweep/...}`.
Le cache in `pkl/` includono nell'identità l'hash del campo di vento: cambiare
file `.nc` invalida automaticamente la cache.
