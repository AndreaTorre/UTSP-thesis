# CVETT — perturbazioni da vento ERA5

Regime di incertezza fisico: la perturbazione di ogni arco deriva dalla
proiezione del campo di vento ERA5 (componenti `u100`/`v100`) lungo l'arco.
`build_wind_perturbation` è **deterministica dal vento**: nessun generatore
casuale, `scenario_id` è l'indice (1-based) dell'istante temporale nel file
NetCDF. Id fuori range ⇒ errore esplicito.

> ⚠️ **CVETT è cambiato**, anche se nessuna modifica è stata scritta
> pensando a lui. Tutta la logica sta in `common/`, quindi loss, decodifica
> delle prenotazioni, WS, layout degli output e grid search valgono
> identici qui. In più `config_backend.py` e `main.py` sono stati modificati
> direttamente. Vedi "Cosa è cambiato" in fondo.

## Separazione train/test: fisica, non per seme

A differenza di PERT, train e test usano **due file `.nc` distinti**
(`ERA5_NC_PATH_TRAIN`, `ERA5_NC_PATH_TEST`): nessuna possibilità di
contaminazione per costruzione. I limiti del test set derivano dal numero di
osservazioni del file (`N_WIND_OBSERVATIONS_TEST`): se chiedi più scenari di
quante osservazioni esistono, il config va in errore — non genera dati sporchi.
Se serve un test set più grande, serve un file `.nc` più lungo.

## ⚠️ Storia importante: il bug `coords` (risolto)

Fino al fix, `generate_scenarios` attivava il ramo vento solo se riceveva sia
`wind` sia `coords` — e `coords` non veniva mai passato dalla pipeline UTSP.
Risultato: Experiment B generava i benchmark su scenari **da vento**, mentre
training e test della rete giravano su scenari **sintetici**. Qualunque
risultato o checkpoint CVETT prodotto prima del fix è **non valido** e va
rigenerato riallenando la rete. I checkpoint non contengono la sorgente degli
scenari nei metadati: non fidarsi di checkpoint CVETT di cui non si conosce la
data rispetto al fix.

## File

- `config_backend.py` — parametri CVETT: semi (`TRAIN_SCENARIO_SEED=50`,
  `TEST_SCENARIO_SEED=99` — contano solo per il fallback sintetico, il vento
  non li usa), conteggio osservazioni dai file `.nc`, campionamento scenari,
  costi, Gurobi, GNN, local search, costanti del drone (`DRONE_U`, `DRONE_A0`,
  `DRONE_A2`, `DRONE_A3`) usate da `common/wind_perturbation.py`.
- `main.py` — entry point: carica i campi vento train/test, usa
  `run_esperimento_B_wind(...)` e passa `wind_train`/`wind_test` alla parte UTSP.
- `gurobi_parallelo.py` — fasi pipeline B (variante CVETT).
- `run_tutto.sh <N>` — pipeline B parallela con dipendenze SLURM.
- `collect_cvett_results.sh` — aggregazione risultati.
- `plot_mappa_vento_tsp_montreal.py` — figure per la tesi (mappa vento + tour),
  non pipeline.

I wrapper `run_b.sh` e `run_exp.sh` sono stati rimossi: usa
`common/submit_b.sh CVETT <N>` e `common/submit_exp.sh CVETT <N>`, che
accettano l'esperimento come argomento.

## Comandi

```bash
cd /home/atorre/UTSP/unione/git/UTSP/common
source env.sh
export TESI_EXPERIMENT=CVETT TESI_N_NODES=15

bash submit_b.sh CVETT 15      # solo Experiment B
bash submit_exp.sh CVETT 15    # B_UTSP_LS (richiede res_B_cached.pkl)
```

Training per batch, test sweep e grid search: stessi comandi di PERT, con
`CVETT` al posto di `PERT`. Prima verifica che il file vento di test copra
`N_ISTANZE_TEST × DIM_ISTANZA_TEST` osservazioni.

```bash
bash grid_submit.sh CVETT 15 all
python grid_analyze.py --exp CVETT --nodes 15 --batch 20
```

## Output

Identico a PERT:

```
RISULTATI_N/
├── pkl/ output/ checkpoint/     condivisi dalla taglia
├── report/ grafici/ modello/    per-run, seguono OUTPUT_DIR
└── batch_sweep/BATCH_X/…
```

Le cache in `pkl/` includono nell'identità l'md5 del campo di vento: cambiare
file `.nc` invalida automaticamente la cache. È la stessa proprietà che in PERT
è garantita dal seme.

---

## Cosa è cambiato

**Ereditato da `common/`** (nessuna azione richiesta, ma cambia i risultati):

- Il termine booking della loss è ora pesato da `λ_B = |Ω| / UTSP2_LAMBDA_B_DIV`
  invece del fattore `|Ω|` hardcoded.
- `alpha` della loss viene da `UTSP2_ALPHA_LOSS`, non più da `UTSP2_LS_ALPHA`.
- `decode_booking_policy` usa `H̄` e la soglia analitica `p/C`; `alpha` e
  `threshold=0.8` sono spariti come iperparametri.
- Nuovo confronto diagnostico heatmap vs tour → `report/controllo_prenotazioni_*.txt`
  e metriche `book_accordo`, `book_corr`, `book_delta_cost`.
- WS calcolato e cacheato accanto a STO/EEV; nuove metriche `WS_test` e
  `gap_ls_ws`, che è ora l'obiettivo primario.
- Entrambi i rami di test usano `_validate_policies_cached`: gli scenari non
  vengono più rigenerati né i PI ricalcolati a ogni run.
- Layout output: `report/`, `grafici/`, `modello/` (era `train/<nome>/`).

**Modifiche dirette a CVETT:**

- `config_backend.py`: aggiunto `UTSP2_LAMBDA_B_DIV`; la variabile della
  penalty è stata rinominata da `TESI_UTSP2_INCLUDE_PENALTY` a
  `TESI_UTSP_INCLUDE_PENALTY`, lo stesso nome usato da PERT e dagli script —
  prima i due rami leggevano variabili diverse.
- `main.py`: `CACHE_PATH` non è più un percorso assoluto costruito da
  `os.environ['TESI_N_NODES']` ma `TEST_SCENARIO_CACHE_DIR`, allineato a PERT;
  il `pickle.dump` è rientrato dentro l'`else` (prima riscriveva la cache **due
  volte** anche quando l'aveva appena letta); rimosso il ramo `if only == "B"`
  duplicato.
- `common/wind_perturbation.py`: le costanti `DRONE_*` erano usate senza essere
  importate — `NameError` su ogni chiamata. Ora l'import è esplicito e protetto,
  così il ramo PERT che importa il modulo senza usarlo non si rompe.

**Da verificare prima di rilanciare CVETT**, dato che non è ancora stato
eseguito nella nuova struttura:

```bash
cd /home/atorre/UTSP/unione/git/UTSP/common
source env.sh
TESI_EXPERIMENT=CVETT TESI_N_NODES=15 python - <<'PY'
import config as c, wind_perturbation as wp
print(c.EXPERIMENT, c.N_NODES)
print("train:", c.ERA5_NC_PATH_TRAIN)
print("test :", c.ERA5_NC_PATH_TEST)
print("out  :", c.OUTPUT_DIR)
print("cache:", c.TEST_SCENARIO_CACHE_DIR)
print("DRONE_U:", wp.DRONE_U)
w = wp.load_wind_field(c.ERA5_NC_PATH_TEST)
print("istanti nel file di test:", w["n_times"])
PY
```

`DRONE_U` non deve essere `None` e il numero di istanti deve coprire
`N_ISTANZE_TEST × DIM_ISTANZA_TEST`.
