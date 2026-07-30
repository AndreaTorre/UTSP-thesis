# common/ — logica condivisa PERT/CVETT

Unica fonte di verità: qui vive tutto il codice; `PERT/` e `CVETT/` contengono
solo `config_backend.py`, `main.py` e gli script di lancio specifici.

## Mappa dei moduli

| File | Ruolo |
|------|-------|
| `env.sh` | Ambiente comune: determina la root dalla posizione del file, carica moduli, attiva il venv, imposta `PYTHONPATH`. Va sourceato come prima riga di ogni script — sostituisce il blocco `module load` + `source venv` che era duplicato in una dozzina di file. |
| `config.py` | Legge `TESI_EXPERIMENT`/`TESI_N_NODES`, risolve percorsi e `OUTPUT_DIR`, inietta i parametri nel `config_backend.py` dell'esperimento e lo esegue. Gestisce le mutazioni di `OUTPUT_DIR` (batch sweep, variant, test sweep, override) e il layer di override `TESI_P_*`. |
| `common.py` | `load_data` (JSON nodi), `load_env` (ambiente Gurobi), semi, `out_path(filename, subdir=None)`. |
| `tsp_utils.py` | Archi canonici, ricostruzione tour, costi. |
| `scenarios.py` | `build_perturbation` (sintetica), `generate_scenarios` (con `solve_pi=`), `find_frequent_arcs` (calibrazione con checkpoint), `generate_scenario_batches`. |
| `wind_perturbation.py` | Caricamento NetCDF ERA5, proiezione vento sugli archi, `build_wind_perturbation` (deterministica, `scenario_id` = indice temporale). |
| `gurobi_models.py` | `solve_exact_tsp` (PI, MTZ), `solve_reservation_tsp` (x fissabile o **libera** ⇒ WS), `solve_stochastic` (two-stage), `build_I_from_medoid_outgoing_nodes`. Tutti `Threads=1, Seed=42`. |
| `experiment_B.py` | Orchestra il benchmark: I → calibrazione → scenari → EEV → STO → riepiloghi. Checkpoint ripristinabili. Produce `res_B`. |
| `two_stage_utsp_loss.py` | Loss a 7 termini + costruzione heatmap + decodifica booking. |
| `utsp.py` | GNN (GCN + 4 canali scattering, pesi appresi), training su batch di scenari, salvataggio/caricamento checkpoint, `run_esperimento_B_UTSP`. |
| `local_search.py` | LS guidata da heatmap, generazione istanze di test con cache, `_validate_policies_cached` (WS/STO/EEV per scenario), `compare_bookings_heatmap_vs_tours`, rami train/test-only, riepiloghi. |
| `evaluation.py` | EEV sul medione, `validate_policies` (usata **solo** da Experiment B), statistiche, grafici. |
| `main.py` | Entry point comune: stampa la configurazione risolta e delega a `<EXP>/main.py`. |

**Grid search** (vedi sezione dedicata): `grid_search.py`, `grid_submit.sh`,
`grid_analyze.py`.

**Script di lancio**: `submit_b.sh <EXP> <N>` e `submit_exp.sh <EXP> <N>`
(sottomissione SLURM parametrica, accettano l'esperimento come argomento — i
vecchi wrapper `PERT/run_b.sh`, `PERT/run_exp.sh`, `CVETT/run_b.sh`,
`CVETT/run_exp.sh` sono stati rimossi), `run_batch_sweep.sh`,
`run_test_sweep.sh`, `run_train_variant.sh`, `run_pi.sh`, `run_ws.sh`,
`run_pool.sh <N>`.

**Analisi**: `collect_test_sweep.py`, `analyze_batch_sweep.py`,
`anova_train_test.py` (ANOVA a due vie; `f_pvalue` e `size_label` sono riusate
da `grid_analyze.py`), `full_report.py`, `presentation_report.py`.
`tools/audit_imports.py` è utilità di manutenzione, non pipeline.

---

## La loss a due stadi

```
L = λ1·L_row + λ2·L_diag + L_dist + λ_B·L_book + λ_d·L_asym + λ_e·L_cons + L_pen
```

Due cose non ovvie:

**`λ_B = |Ω| / UTSP2_LAMBDA_B_DIV`**, con `|Ω| = len(H_list)` letto dentro la
loss. La cardinalità degli scenari sta al numeratore perché il costo di
prenotazione è di primo stadio (pagato una volta) mentre gli altri termini sono
attese pesate con `p_ω`. Il divisore è il parametro da tarare; `1.0` riproduce
il comportamento storico, quando il fattore `|Ω|` era hardcoded dentro
`_loss_booking`. Così il peso segue automaticamente il batch e i `BATCH_*`
restano confrontabili.

**`alpha` (`UTSP2_ALPHA_LOSS`) governa la saturazione** di booking e penalty,
che usano `attivazione` e `1 − attivazione` e devono quindi condividerlo. Con
`alpha = 0` l'attivazione è identicamente nulla: il termine booking sparisce e
la penalty resta al massimo. Non è un parametro innocuo.

`UTSP2_LS_ALPHA` è un'**altra** cosa: l'esplorazione UCB della local search.
Confonderli è già costato un baco.

## Decodifica delle prenotazioni

`decode_booking_policy(H_list, I, nodes, scenario_probs, p, C)` stima la
frequenza d'uso di ogni tratta di `I` come

```
f_H(i,j) = H̄[i,j] + H̄[j,i]        con H̄ = Σ_ω p_ω H^ω
```

e prenota quando `f_H > p_ij / C_ij` — il *critical ratio*, cioè la soglia
analitica del newsvendor. Non ci sono `alpha` né `threshold` liberi:
un decisore con parametri tarabili renderebbe infalsificabile l'affermazione
"la heatmap prenota bene". `check_booking_coverage` usa la stessa regola.

Nella versione attuale **questa politica è solo diagnostica**: la prenotazione
di primo stadio effettivamente valutata viene da `_compute_bookings_from_tours`,
che misura la frequenza sui tour della local search e applica la stessa soglia
`p/C`. `compare_bookings_heatmap_vs_tours` confronta le due — stessa grandezza,
stessa regola, quindi l'unica variabile è la qualità della heatmap. Produce
`report/controllo_prenotazioni_*.txt` e le metriche `book_accordo`,
`book_corr`, `book_delta_cost`.

> Il costo della politica heatmap è calcolato a **instradamento fisso**: i tour
> sono stati ottimizzati conoscendo la politica della LS. Misura l'effetto
> diretto su prenotazioni e multe, non la rirotta.

## Benchmark: WS, non PI

| Sigla | Significato | Ruolo |
|-------|-------------|-------|
| **WS** | wait-and-see: primo e secondo stadio ottimi insieme, scenario per scenario | **bound corretto** (`WS ≤ STO ≤ EEV`) |
| STO | politica stocastica a due stadi, `x` comune | riferimento |
| EEV | politica dello scenario medio | riferimento superiore |
| PI | TSP libero, ignora `I`, `p`, `C` | quanta parte del costo è puro routing |

WS è `solve_reservation_tsp(..., fixed_reservations=None)`, calcolato dentro
`_validate_policies_cached` accanto a STO/EEV e cacheato per `scenario_id`; se
esiste `test_pool_cache.pkl` viene letto da lì invece di essere ricalcolato.

Il PI non è un bound per un problema a due stadi con prenotazioni. Se non
serve, `TESI_TEST_SKIP_PI=1` lo salta; la cache resta valida e un run
successivo con `=0` fa il fill-in.

Metriche di test: `WS_test`, `STO_test`, `EEV_test`, `PI_test`,
`UTSP_LS_test`, e i gap `gap_ls_ws` (primario), `gap_ls_sto`, `gap_ls_eev`,
`gap_ls_pi`.

---

## Grid search

```bash
python grid_search.py --list   --batch 20            # quante combinazioni
python grid_search.py --run-one   IDX --exp PERT --nodes 15 --batch 20
python grid_search.py --run-chunk K   --exp PERT --nodes 15 --batch 20
python grid_search.py --collect       --exp PERT --nodes 15 --batch 20

bash grid_submit.sh PERT 15 20      # una tripla
bash grid_submit.sh PERT 15 all     # tutti i batch
DRY=1 bash grid_submit.sh PERT all all

python grid_analyze.py --exp PERT --nodes 15 --batch 20
python grid_analyze.py --all --top 8
```

`GRID` e `FIXED` stanno in cima a `grid_search.py`. Ogni chiave è il nome
**esatto** di una costante di `config.py`.

Struttura prodotta:

```
grid_search/<EXP>/NODI_<n>/BATCH_<b>/
├── combos.json        manifesto indice → parametri (riproducibilità)
├── logs/              stdout/stderr SLURM
├── combo_0000/        output completo + result.json
├── ...
└── grid_summary.csv   aggregato, ordinato per gap_ls_ws
```

Scelte di progetto:

- **Un processo per combinazione.** `config` legge l'ambiente all'import;
  cambiarlo nello stesso processo richiederebbe `importlib.reload` fragili. Un
  crash o un OOM non porta giù le combo successive.
- **Ripresa automatica.** `run_chunk` salta le combo con `result.json` a
  `status: ok`: rilanciare lo stesso comando riprende da dove si era fermato.
- **Job array**, non job-che-lancia-job: la sottomissione avviene dal login
  node e non brucia wall time.
- Un job `collect` in `afterany` scrive il CSV.

`grid_analyze.py` non si limita a ordinare: su un fattoriale completo calcola
l'**effetto marginale** di ogni parametro (ANOVA a una via, η², p) e verifica
se la configurazione migliore è distinguibile dal rumore fra istanze. Con
`--all` controlla anche se la classifica regge da una foglia all'altra.

> Prima di lanciare, verifica che ogni parametro della griglia arrivi davvero
> alla loss: in questo progetto ne sono stati trovati **quattro** silenziosamente
> inerti (`UTSP2_LAMBDA_B`, `UTSP2_AGGREGATION`, `UTSP2_ALPHA_LOSS`,
> `UTSP2_ALPHA_DECODE`). Due valori estremi, la loss deve cambiare.

---

## Variabili d'ambiente

| Variabile | Effetto | Default |
|-----------|---------|---------|
| `TESI_EXPERIMENT` | PERT o CVETT | obbligatoria |
| `TESI_N_NODES` | 15, 25, 40 | obbligatoria |
| `TESI_ROOT_DIR` | override della root | dedotta dalla posizione di `config.py` |
| `TESI_P_<NOME>=<val>` | sovrascrive **qualunque** costante di config, convertita nel tipo dell'originale | — |
| `TESI_OUTPUT_OVERRIDE` | sostituisce `OUTPUT_DIR` (usata dalla grid) | off |
| `TESI_BATCH_SWEEP=<X>` | `UTSP_BATCH_SIZE=X`, `OUTPUT_DIR → batch_sweep/BATCH_X/` | off |
| `TESI_VARIANT` | `OUTPUT_DIR → variants/<nome>/` | off |
| `TESI_UTSP_TEST_ONLY=1` | carica il checkpoint, salta il training | off |
| `TESI_UTSP_TRAIN_NAME` | nome del training da caricare | `espB_UTSP_LS` |
| `TESI_TEST_OUTPUT_SUBDIR` | `OUTPUT_DIR → <...>/<subdir>` (i checkpoint restano in `TRAIN_OUTPUT_DIR`) | off |
| `TESI_DIM_ISTANZA_TEST` | scenari per istanza di test | 20 |
| `TESI_N_ISTANZE_TEST` | numero istanze di test | 100 |
| `TESI_N_TEST_SCENARIOS_UTSP` | ampiezza pool scenari test | IS×DIM |
| `TESI_TEST_SCENARIO_SEED` | seme scenari test (vedi tabella semi nel README PERT) | 1_000_000 |
| `TESI_TEST_SKIP_PI=1` | genera scenari senza risolvere il PI; `=0` fa il fill-in | 0 |
| `TESI_UTSP_INCLUDE_PENALTY` | termine penalty nella loss (**stesso nome** in PERT e CVETT) | 1 |
| `TESI_DROP_LAST_TRAIN_BATCH` | scarta l'ultimo batch di training incompleto | 1 |

`TESI_P_<NOME>` con un nome inesistente solleva `ValueError`: un refuso nella
griglia non deve produrre run "riuscite" girate coi valori di default.

---

## Il contratto `res_B`

`experiment_B.py` → `utsp.py` comunica con un dict contenente almeno:
`I, b, p, C, results, scenario_probs, frequent_arcs, PI, STO, EEV, x_ev,
x_used_sto, stoch_costs, eev_costs, eev_solutions, stoch_solutions`.
Cacheato in `RISULTATI_N/pkl/res_B_cached.pkl` — path fisso, non segue le
mutazioni di `OUTPUT_DIR`.

## Cache dei test

Tutte in `RISULTATI_N/pkl/`, **prima** delle mutazioni di `OUTPUT_DIR`:
condivise da ogni batch, variante e combinazione della grid.

- `test_scenarios_cache.pkl` — per scenario_id: perturbazione, `scenario_dist`,
  `exact_free` (PI; vuoto se generato con skip-PI).
- `test_sto_eev_cache.pkl` — per scenario_id: costi WS, STO, EEV.
- `test_pool_cache.pkl` — pool precalcolato (PI/STO/EEV/WS), opzionale.

Le chiavi includono seme, `I`, archi frequenti, sorgente scenari (md5 del campo
vento o `"synthetic"`) e — per STO/EEV — le policy `x_sto`/`x_ev`. Cambia uno
di questi ⇒ la cache riparte da zero, mai dati sporchi. Scrittura atomica
(tmp + rename), salvataggio incrementale per blocco.

`ws_cost` viene aggiunto alle entry esistenti con un secondo passaggio, senza
invalidare quanto già calcolato.

> **Concorrenza**: le cache non reggono scritture parallele. Un solo job di
> sweep per (esperimento, N) alla volta; taglie diverse sono indipendenti.
> Prima di lanciare la grid conviene "scaldare" la cache WS con una run
> singola, o le prime combo in parallelo risolvono gli stessi MIP.

## Layout degli output

**Per-run** — seguono `OUTPUT_DIR`, quindi anche `TESI_OUTPUT_OVERRIDE`:

```
report/     tutti i .txt e .csv di risultato
grafici/    solo .png
modello/    utsp_model.pt, utsp_history.json, utsp_metadata.json
```

**Condivise** a livello `RISULTATI_N/`:

```
output/     log SLURM
checkpoint/ ripresa di Experiment B
pkl/        cache
```

`out_path(filename, subdir)` crea la sottocartella: non concatenare la
sottocartella dentro il nome del file.
