# Roadmap: llama.cpp-vulkan-moe su RX 6800

Aggiornamento: **2026-10-07**. Documento di lavoro per `Cartaz/llama.cpp-vulkan-moe`, branch di riferimento `experiment/moe-trace-strides`; incrementi sperimentali R17-R37 su branch separati. Snapshot del codice prima di questa roadmap: `c679dc22012ee9281db57336ff8fd17a93bc54b2`.

Obiettivo: inferenza locale affidabile e veloce su RX 6800 16 GB, Ryzen 7 5700X3D e 32 GB RAM, CachyOS/RADV. Prima una sequenza; poi 2, 4 e 8 richieste da agenti. PP, TG, TTFT, latenza completa, throughput aggregato e throughput per richiesta sono risultati distinti.

## Come leggere le sigle

Le sigle non sono quattro liste da eseguire in successione. **S** identifica una strategia/funzionalita', **R** un incremento o esperimento registrato, **M** una milestone complessiva e **T** un protocollo di verifica. Gli incrementi R sviluppano e verificano le strategie S: R27 e R28, per esempio, avanzano S06. Una strategia puo' richiedere piu' incrementi e un incremento puo' verificare prerequisiti di piu' strategie. L'ordine dipende dai colli di bottiglia e dalle dipendenze, non dal dover terminare tutti gli R prima degli S.

Percorso corrente: S01/S02/S03 misurazione e routing; S06 budget/policy offline; S04/S05 pool GPU esatto, remap e gestione slot; poi S07/S08 fallback ibrido e S10/S11 transfer. M1/M2 rimangono parziali finche' mancano corpus esteso, costo miss CPU e lifetime esteso; R29 verifica il prototipo pool sui kernel GPU reali; R30 integra un layer nel modello con gate breve OFF/ON/B1 e A/B a placement fissato; M3 richiede ancora T2 esteso e placement utile al decode normale.

## Stato leggibile in un minuto

| Voce | Stato al 2026-10-07 |
| --- | --- |
| Esecuzione benchmark | **Ripresa autorizzata esplicitamente dall'utente il 2026-10-07. R20-R28:127processi modello; R29 solo fixture; R30 aggiunge17processi; R31 aggiunge16processi; R32 aggiunge27processi; R33 aggiunge25processi; R34 aggiunge19processi, totale231; R35 aggiunge20fixture operatori validi e0processi modello; R36 aggiunge5processi modello, totale236, e20fixture numerici/2controlli sintetici; R37 aggiunge37processi modello/7437vettori, totale273, diagnostica precisione valida ma screen di fedelta' FAIL; B4 breve none PP+37.32%, TG/p95 equivalenti; R28 corpus12nuovi prompt e riserva quote layer validati sul pilot.** |
| Baseline originale | llama.cpp v0.5.0, `7fe450e19305b828c199d602c23a8337aaa1f03b`, senza modifiche |
| Modello di riferimento | Ornith-1.5-35B-Q4_K_M.gguf; identita' completa nella sezione baseline |
| Cache GPU persistente di esperti | **R33 aggiunge placement mirato per un layer, OFF di default**: B1 GPU residente/targetOFF/poolON/repeat raw identici sul breve; pool32/54MiB, hit62.44%, API weight bytes TG-62.47%.12processi GPU: TG+2.30%[+1.12,+3.54] vs targetOFF INCONCLUDENTE; TG-10.40% vs layer residente REGRESSIONE. Nessuna promozione CPU/GPU: screen numerico FAIL |
| CPU routing compatto | Implementato, opzionale e OFF di default; riduzione del workspace verificata, incremento PP/TG non dimostrato |
| CPU traversal degli esperti attivi | Implementato, opzionale e OFF di default; incremento PP/TG non dimostrato |
| Caricamento `none` / Vulkan_Host | R22 build9375 PP+33.80%; R26 dentroB4f498 PP+37.32%[36.92,37.38], TG/p95 equivalenti, GTT0.627->9.566..9.646GiB/init5.528->8.729s. A/B distinti sul breve; nessun ranking tra versioni |
| Ubatch 2048 | Forte beneficio su un replay 4096; **candidato**, non configurazione universalmente validata |
| Qualita' risposte | R12/R13 conservati; R23 B1 lungo non ripetibile; R26 coppiaB4lungo bit-identica ma breve29/129vettori diversi daB1. Nessun nuovo score semantico/equivalenza generale; R33 decode CPU/GPU oltre i limiti numerici preregistrati pur con201argmax uguali; R34 warm/cold/clear esatti, gate/up/down/gate+up ripetibili ma screen FAIL e0/1/1/2argmax diversi; B1 lungo ripetuto cambia4argmax |
| Sweep configurazione | ncmoe 0..40 completato e shortlist verificata; coordinate successive parziali; interazioni e matrice contesti da completare |
| Profilo provvisorio dello sweep | `-ncmoe 12 -tb 8`, `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`, librerie DEFAULT congelate; **non vincitore finale** |
| Fondamenta S01/S02 | Tooling iniziale IMPLEMENTATO su `experiment/moe-profile-foundations`: manifest live con verifica freeze, entropia/riuso/finestre e stime cache in byte; R21 valida quattro prompt; R27/R28 planner statico con riserva,12nuovi prompt/sei famiglie; T2completo/remap/runtime cache incompleti |
| Fondamenta S03 | R20: join host PP/TG sul modello VALIDATO, costo profiler TG -6.07%, logger timestamp corretto e correlato, accounting memoria sul replay breve; timeline normale calibrata e contesti lunghi incompleti |
| Prossima modifica di ricerca | S09/T0: R34 esclude warmup/clear come condizioni necessarie sul breve e riproduce differenze con ciascuna proiezione B1; R35 conferma pipeline Q8_1 gate/up e F32down nel fixture; input CPUQ8_K su GPUF32 riduce RMS16-21mila volte. R36 cattura600record reali con osservatoreOFF/ON eB1/fork esatti; replay di pesi/ID/attivazioni ai passi1/100/200: precisioneCPUQ8_K su stessaGPUF32 riduce RMS45-52mila volte. R37 completa precisione layer17 sul modello:37processi, repeat/OFF/copy/debug/profile/sanitizer esatti; tutti screen rispettoCPUoriginale FAIL, roundtripCPU non neutro. Prossimo: controllo ReBAR e primo intermedio divergente, datiCPU quantizzati reali prima di placement o merge ibrido. Ripetibilita' B1 lunga ancora ERRORE_BASELINE. R33 rende attivo il pool senza soglia globale1 ma non dimostra speedup pratico contro targetOFF; residente piu' rapido a maggiore allocazione. Sweep capacity S04 soltanto nel perimetro GPU esatto. Corpus, pressione KV, partizione/warm-start e concurrency restano aperti |

Questo documento registra la ripresa richiesta dall'utente il 2026-10-07; gli archivi interrotti R13/R15 restano conservati. I problemi riprodotti anche sulla v0.5.0 originale sono registrati come `ERRORE_BASELINE`, come richiesto dall'utente: non sono automaticamente regressioni del fork e non bloccano tutta la ricerca. Una misura con NaN, output nullo o fallback CPU involontario resta invalida per il ranking prestazionale.

## Come aggiornare stato e verdetto

Ogni esperimento conserva ID, baseline, commit, opzioni, workload, misure, costi e artefatti. Aggiornare questa roadmap quando si aggiunge una implementazione o si conclude un A/B. Un risultato nuovo non cancella tentativi falliti o misure precedenti.

| Stato di lavoro | Significato |
| --- | --- |
| PROPOSTO | Idea con protocollo; nessuna implementazione o misura locale |
| PRONTO | Dati/protocollo o prototipo preparati; esecuzione non conclusa |
| IMPLEMENTATO | Codice presente; non implica beneficio o validazione end-to-end |
| TEST_PARZIALI | Esistono misure, ma manca una parte del protocollo |
| VALIDATO | Gate previsti superati sul workload dichiarato; specificare il perimetro |
| INCONCLUDENTE | Variabilita', controlli insufficienti o confondenti impediscono un verdetto |
| SCARTATO | Tentativo archiviato/revertito con motivo e riferimento |
| PAUSA | Esecuzione sospesa; mantenere checkpoint e risultati |
| ERRORE_BASELINE | Difetto riprodotto nel riferimento originale alle condizioni dichiarate |

Il **verdetto** e' separato dallo stato. Per PP, TG, latenza, RAM, VRAM e qualita' usare `MIGLIORAMENTO`, `REGRESSIONE`, `NESSUN_CAMBIAMENTO` o `NON_MISURATO`; usare `INCONCLUDENTE` quando i dati non distinguono le alternative, `MISTO` quando obiettivi diversi peggiorano/migliorano. `NESSUN_CAMBIAMENTO` richiede un intervallo di equivalenza prestabilito; non significa semplicemente "non abbiamo trovato significativita'". Nelle tabelle seguenti "beneficio non dimostrato" resta inconcludente.

## Baseline e identita' riproducibili

Non esiste un unico numero di baseline valido per tutti i contesti. Registrare una baseline per ciascun workload, processo e configurazione. Il risultato piu' veloce di giorni diversi non e' un controllo A/B.

| ID | Riferimento | Uso |
| --- | --- | --- |
| B0 | Originale v0.5.0 `7fe450e19305b828c199d602c23a8337aaa1f03b`, parametri applicativi di default | Esperienza utente e difetti condivisi; registrare fit, placement e default effettivi |
| B1 | Stessa v0.5.0, parametri fissati e fit OFF | Controllo principale del fork; identico modello, token, KV, batch, placement, sampling e hardware |
| B2 | Fork allo stesso SHA del candidato, tutte le nuove opzioni OFF | Isolare una patch; B1 separato verifica l'effetto cumulativo del fork |
| B3 | Migliore configurazione precedente che ha superato il protocollo, SHA e librerie congelati | Misurare il progresso reale, evitando di confrontare ogni novita' solo con un default lento |
| B4 | Upstream recente, SHA fissato, build e runtime equivalenti | Verificare novita' upstream separatamente; non sostituire retroattivamente B0/B1 |
| B5 | Strata o altri motori, versione e modello documentati | Confronto esterno; stesso modello solo se supportato, altrimenti rapporto qualita'/memoria/latenza, senza speedup diretto sul nostro fork |

B3 non e' ancora un nuovo profilo finale approvato dallo sweep. La configurazione ncmoe12 resta provvisoria; non va rinominata "default".

Identita' storica delle misure RX 6800 del 4-6 ottobre:

- GPU: RX 6800 16 GB, `RADV NAVI21`, RDNA2; CPU: Ryzen 7 5700X3D; RAM: 32 GB; sistema: CachyOS.
- Kernel `7.2.9-1-cachyos`; Mesa/vulkan-radeon `26.2.4-1`; GCC `16.2.1+r23+gd564253eb6c8-1`; CMake `4.4.4-1.1`; Ninja `1.13.2-3.1`. Per nuove misure rilevare nuovamente le versioni.
- Build Release/native/Ninja/Vulkan. Conservare `CMakeCache.txt`, comandi, patch sorgente se dirty, SHA di eseguibile e di **ogni libreria effettivamente caricata**. Non basta la stringa versione del binario.
- Modello `Ornith-1.5-35B-Q4_K_M.gguf`, 21,713,462,848 byte; SHA-256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`.
- Profilo GPU corrente richiesto dall'utente: core 2600 MHz, memoria 1075 MHz, offset -100 mV, cap 186 W. Mantenerlo; non sono autorizzati esperimenti di tensione/clock. Le misure storiche precedenti riportano il proprio profilo e non vanno equiparate senza controllo.

Profilo fisso usato nei confronti lunghi iniziali, da passare a **entrambi** i binari di B1/B2:

```text
-ngl 99 -ncmoe 18 -t 8 -tb 8 -c 8192 -b 2048 -ub 512
-fa on -ctk q8_0 -ctv q8_0 --fit off --load-mode mmap
```

I flag sperimentali letti per presenza nell'ambiente devono essere rimossi per OFF: per esempio `GGML_VK_FA_Q8_SYNC=0` attiva comunque la barriera. Registrare esplicitamente variabili assenti e presenti.

Questo e' un riferimento documentato, non il default dell'applicazione. Replay brevi con contesto 1024 e batch 512 sono workload diversi. `--load-mode none`, ub2048, compact, active e barriera Q8 appartengono a candidati identificati separatamente. Per nuovi A/B congelare anche variabili `GGML_*`, `RADV_*`, `VK_*`, `LD_LIBRARY_PATH`, affinita' CPU, stato dei processi concorrenti e placement effettivo. Nel profilo dello sweep il fit puo' abortire per override gia' impostati: verificare i valori applicati nei log, non dedurre il placement dagli argomenti.

## Registro delle implementazioni e dei risultati gia' ottenuti

Le percentuali sono osservazioni sul workload indicato, non previsioni su tutti i prompt. I documenti collegati contengono comandi, ordine dei processi, metadata e limiti; gli archivi locali conservano dati grezzi non tutti pubblicati.

| ID | Implementazione / confronto | Stato | Baseline e risultato verificato | Verdetto e limite |
| --- | --- | --- | --- | --- |
| R01 | CPU routing compatto; `0a5bc4dc93516e4223a2ae4f0a2dcc4e25aa8646` | VALIDATO sul perimetro | B2 OFF/ON: workspace totale 34,171,480 -> 750,176 byte nel caso 256 esperti/top8/2048 token; 217 casi deterministici, sanitizer e 955 operatori per variante | **MIGLIORAMENTO workspace CPU**; PP/TG INCONCLUDENTE; non risparmia pesi o VRAM. [Report](docs/development/moe-compact-routing.md) |
| R02 | Lettura tracing con stride; pubblicato `9149912a398e1bf4e78d63ac11657e6ae7642276` | VALIDATO | Corretto accesso al view di `ggml_argsort_top_k`; 9 casi mirati CPU/Vulkan, vecchio callback fallisce gli stessi casi | **MIGLIORAMENTO correttezza del profiler**; rigenerare vecchi trace multi-token. [Tracing](examples/moe-trace/README.md) |
| R03 | Replay completo upstream / compact OFF / ON | VALIDATO sul replay | 244 prompt + 128 continuazione fissa, 128,133,120 byte di logits identici; SHA `b7bb28a86e623daf75f436f9bddb786cee6c43d8a28d1b85333f3f25d9cbb662`; compact ON/OFF PP -0.15%, TG +0.83% con soli 2 processi/variante | Fedelta' verificata sul replay; **incremento velocita' non dimostrato**. [Report](docs/development/moe-replay-rx6800.md) |
| R04 | Traversal CPU degli esperti attivi, <=8 token; `10b0cade93078154852809eb8006b0e9c5d6a66b` | VALIDATO sul perimetro | B2 active OFF/ON, compact ON; 361 casi + 955 operatori/variante e replay identico. PP +0.13%, TG +0.004%; workspace 750,176 -> 751,212 byte | **INCONCLUDENTE prestazioni**, regimi fra processi dominanti; OFF di default. [Report](docs/development/moe-replay-rx6800.md) |
| R05 | LRU fredda/calda e static prefill, simulazione | TEST_PARZIALI | Trace 244+128, layer CPU 0..17: 32 slot/layer cold 55.96%, warm 56.91%, static 47.42%; 64 slot 72.18/74.85/65.54% | Copertura logica del callback storico; **R21 trova10.68% set diversi dal percorso normale: non usare per scegliere policy normali**. Cache GPU/TG NON_MISURATI. [Tutti i tagli](docs/development/moe-replay-rx6800.md) |
| R06 | Configurazione `--load-mode mmap` vs `none` | VALIDATO per PP limitato | Stesso binario: PP512 286.91 -> 365.32 (+27.3%); replay244 PP173.919 ->239.793 (+37.88%). Sul replay TG25.799 ->22.844 (-11.45%), con regimi fra processi | **MIGLIORAMENTO PP**, TG INCONCLUDENTE con possibile costo; **REGRESSIONE startup/RAM**: ~4 ->12 s, GTT ~0.9 ->9.6-9.8 GiB, MemAvailable ~23.3 ->17.4 GiB. [Report](docs/development/moe-host-transfer-rx6800.md) |
| R07 | Full-copy prefill senza nuova coda/overlap | SCARTATO, patch revertita | OFF/ON/ON/OFF, PP512: 366.012 ->317.689 (-13.20%); logits completi identici, 2,979,840 byte | **REGRESSIONE PP**; evitare semplice copia di tutti gli esperti. Overlap resta un esperimento diverso. [Report](docs/development/moe-host-transfer-rx6800.md) |
| R08 | `RADV_PERFTEST=nogttspill` | SCARTATO allo screening | Singolo screen PP79.15/TG4.87 contro controllo PP165.59/TG22.07 | **REGRESSIONE nello screen**, entita' non certificata da A/B esteso; non attivare come raccomandazione. [Report](docs/development/moe-host-transfer-rx6800.md) |
| R09 | Barriera opzionale FA Q8 K/V, `GGML_VK_FA_Q8_SYNC=1`; `ce3ba740ee565300b58a87601cf4c36d5a26489c` | VALIDATO su casi Q8 dichiarati; generale parziale | Replay4096/ub512, mmap/none, 3 ripetizioni e riferimento all-barrier; 362 casi FA OFF/ON. Replay breve PP -0.59%, TG -0.15% | Workaround circoscritto; **beneficio di velocita' non dimostrato**, causa dei difetti non risolta. Con barriera ON, none migliora PP lungo257.424 ->323.838 (+25.80%). [Report](docs/development/moe-prefill-sync-rx6800.md) |
| R10 | Barriera ampia, serializzazione e disabilitazione fusion | SCARTATO | Barriera ampia non risolve F16; controllo con 3,228,160 valori non finiti. Serializzazione: hash ripetibili ma output tutti zero. Disable fusion: non finiti | **Correttezza fallita**; nessun punteggio prestazionale valido. Un hash stabile da solo non basta. [Report](docs/development/moe-prefill-sync-rx6800.md) |
| R11 | Ubatch512 vs2048, none + Q8 guard ON | TEST_PARZIALI | Replay4096: PP324.403 ->766.919 (+136.41%, 2.36x); VRAM13.41 ->15.48 GiB; GTT~9.76 GiB. 2 seed, 128 token generati: fork=upstream allo stesso ubatch; fra ubatch cambia dal token5 in entrambi | **MIGLIORAMENTO PP replay**, **REGRESSIONE headroom VRAM**; TG/qualita' generale non certificati. [Report](docs/development/moe-large-batch-generation-rx6800.md) |
| R12 | Qualita' originale64, ub512 vs2048 | TEST_PARZIALI; gate FALLITO | Primary60/64 entrambi; retrieval16/16 vs16/16, aritmetica15/16 vs14/16, logica13/16 vs15/16, codice16/16 vs15/16. Secondary61/64 vs60/64. Ub2048: 3 fallimenti numerici; primo ub512 fallito, rerun completo finito | **Nessuna equivalenza dimostrata**. Il pareggio non compensa regressioni semantiche o NaN. Protocollo storico chiedeva output a ogni chunk fisico; nuovo helper usa batch logico/output finale come server |
| R13 | Rivalidazione nativa con helper corretto | PAUSA, incompleta | Ub512: 74 chiamate, primary60/64, zero NaN, 6 duplicati con token identici. Ub2048: interrotto e conservato | **Qualita' comparativa NON_VALIDATA**; non completare senza ripresa richiesta. Non sostituire retroattivamente R12 |
| R14 | Originale default vs fork default, prefisso512+PP3294 | ERRORE_BASELINE | Entrambi non finiti al token3805 con F16 K/V, logical2048/physical512. Auto-fit sceglie placement un po' diverso | **Non dimostrata una regressione del fork**; matched arguments, non identico placement. Escludere tempi invalidi, conservare riproduttore |
| R15 | Sweep startup, ncmoe e coordinate CPU/fit | PAUSA / TEST_PARZIALI | 41 ncmoe screen conclusi + shortlist lunga; placement/fit/margine e thread generazione provati, thread batch incompleti; profilo ncmoe12/tb8 provvisorio. Interazioni/contesti non conclusi | **Nessun nuovo vincitore finale**; NON_MISURATO per matrice finale1/2/4/8 e qualita' |
| R16 | Controlli dopo ripresa del 6 ottobre | INCONCLUDENTE per TG | Stesso profilo provvisorio, depth512/PP512/TG32: PP368.36 e368.13; TG24.42 e33.33 in due processi; endpoint corrispondenti identici | PP ripetibile nello screen; TG varia molto fra processi. Controllo finale coordinate04 interrotto: nessun vincitore thread adottato |
| R17 | Tooling offline S01/S02: manifest live/freeze e analyzer esteso | IMPLEMENTATO; test offline superati | Compatibilita' di tutti i campi per-layer precedenti sul trace corretto salvato, 40 layer per prefill/decode/all; suite senza modello con oracle di riuso indipendente e snapshot del processo Python | PP/TG/TTFT/qualita' **NON_MISURATO**; nessuna campagna ripresa, cache GPU assente. [Report e comandi](docs/development/moe-profile-foundations-rx6800.md) |
| R18 | S03 iniziale: scope scheduler host e payload copie opt-in | IMPLEMENTATO / TEST_PARZIALI | 32 valutazioni operatore per invocazione, CPU/RX6800, output scalare esatto e identico OFF/ON/controllo; byte/padding/stride/fallback verificati; sanitizer CPU superati | PP/TG/TTFT/overhead **NON_MISURATO**. Logger GPU esistente fallisce anche nel controllo per un caso parallel/view; seriale passa. Timeline, fasi e budget completo restano incompleti. [Report e metadata](docs/development/moe-scheduler-profile-rx6800.md) |
| R19 | S03: intervalli espliciti replay PP/TG e join scheduler | IMPLEMENTATO / TEST_PARZIALI | 32/32 associazioni sintetiche esatte; warmup separato, tempi ms e payload per fase; 5 test CPU/RX6800/sanitizer per suite, output invariato; replay compilato senza modello | Correlazione runtime sul modello, overhead e prestazioni **NON_MISURATI**; timeline GPU/budget ancora incompleti. [Report e metadata](docs/development/moe-phase-profile-rx6800.md) |
| R20 | S03 sul modello: correttezza, overhead, timestamp e memoria | VALIDATO sul replay / TEST_PARZIALI S03 | 30 processi singoli; 6 dump logits completi finiti/nonzero e identici; OFF/ON 8 processi/variante: PP +0.13%, TG -6.07%; B1/fork-OFF 4 processi/variante entro margini 3% PP/TG e5% p95; 55/2432 graph GPU associati a PP/decode128 | **REGRESSIONE TG da strumentazione**, PP equivalente; nessun guadagno del fork OFF dimostrato. Memoria breve/diagnostica GPU misurata, timeline normale e contesti lunghi incompleti. [Report e metadata](docs/development/moe-model-profile-rx6800.md) |
| R21 | S02/S06: collector normale, corpus e richieste complete | VALIDATO pilot / TEST_PARZIALI M2 | 19 processi; quattro prompt train/held-out, generator/B1/collector con logits identici; CPU18/top8 completi. A490.43MiB LRU completa0.52..1.39%, statico train->held-out0%; callback storico altera logits anche con engine B1 | **ERRORE_BASELINE nel callback diagnostico**, causa aperta. Simulazione logica, nessuna cache runtime/TG gain. [Report e metadata](docs/development/moe-cache-plan-rx6800.md) |
| R22 | S10 prerequisito: replica load-mode sul bottleneck PP | VALIDATO replay breve | Build9375, 11 processi; mmap/none4processi/variante: PP168.824->225.892 (+33.80%, CI32.26..35.17), TG28.853->28.837 entro3%, p95 entro5%; rawB1 identici | **MIGLIORAMENTO PP / MISTO memoria-avvio**: GTT residente0.62->9.57GiB, init5.02->9.89s. Config gia' esistente, non nuova cache/overlap. [Report](docs/development/moe-host-transfer-recheck-rx6800.md) |
| R23 | T4: prompt code-review3107+128, replay ub512 | ERRORE_BASELINE ripetibilita' / gate FAILED | Sei dump completi finiti/nonzero; due run B1 differiscono in133/135vettori dal terzo chunk, maxabs3.43184 e2argmax PP diversi; fork anche variabile | Nessun processo di speed ranking; causa non stabilita, nessuna regressione fork o fix driver dimostrati. [Report](docs/development/moe-long-prefill-recheck-rx6800.md) |
| R24 | Riduzione R23: prefix B1 1024/1536+32 | TEST_PARZIALI, coppie identiche | Quattro processi originali, due/prefix; tutti34/35vettori finiti/nonzero e bit-identici dentro coppia; un launch CRLF rifiutato prima del load e conservato | Il solo terzo chunk non riproduce il difetto lungo; root cause aperta, nessun ranking velocita'. [Report](docs/development/moe-chunk-repeatability-rx6800.md) |
| R25 | Controllo helper reset dati dopo warmup | IMPLEMENTATO, default compatibile; gate lungo FAILED | Helper79c7778 MOE_REPLAY_CLEAR_DATA=1; quattro run3107+128 finiti/nonzero, coppia B1 diff134/135vettori gia' dal secondo chunk. Assente/0/vuoto: tre run244+128 identiciB1; sei CLIcheck senza modello | True clearing **non risolve** ripetibilita' originale; opt-in diagnostico, nessun fix/TG gain/default change. [Report](docs/development/moe-state-reset-rx6800.md) |
| R26 | Controllo upstream B4 separato + A/B load-mode breve | VALIDATO replay breve / TEST_PARZIALI B4 | Upstream f498f864 pulito;14processi, sei dump finiti/nonzero, coppie B4 brevi/lunghe identiche. Breve29/129diversi daB1; mmap/none4processi/modalita': PP179.537->246.548,+37.324%[36.922,37.383], TG/p95 equivalenti | **MIGLIORAMENTO PP / MISTO memoria-avvio**: GTT residente0.627->9.566..9.646GiB, init5.528->8.729s. Nessun fix generale/ranking cross-version/promozione. [Report](docs/development/moe-upstream-control-rx6800.md) |
| R27 | S06 planner statico TRAIN budget in byte/quote layer | IMPLEMENTATO, VALIDATO offline pilot | Tre policy a cap228.867/490.430/1013.555MiB, due TRAIN/due HELDOUT normali R21;18testPASS. Cap490.43bundle completa3.819%/2.344%, ma15layer senza slot; uniforme0% | Copertura logica, zero nuovi processi modello; nessuna cache runtime, risparmio PCIe o TG gain. [Report](docs/development/moe-layer-budget-rx6800.md) |
| R28 | S02/S06: corpus sei famiglie e riserva minima per layer | IMPLEMENTATO, VALIDATO pilot / T2 PARZIALE | 36processi,12prompt nuovi/6TRAIN+6HELDOUT;65vettori per caso identici generator/B1/collector;20testPASS/defaultR27identico.19/21allocazioni fattibili.490.43MiB bundle riserva0/4/8: completa4.456/4.181/2.459%, layerquotezero16/0/0 | Copertura logica con tradeoff; min8non entra228.87MiB; nessun token all-hit su18layer, cache runtime/TG gain/score semantico assenti. [Report](docs/development/moe-layer-reservations-rx6800.md) |
| R29 | S04/S05: prototipo pool GPU persistente e remap esatto | IMPLEMENTATO, VALIDATO su fixture / modello NON_MISURATO | fa4afd46,48combinazioni/384richieste;256grafi residenti+128fallback per processo;1152vettori finiti/nonzero e2captureGPUidentiche;LRU,lease,destructor e sanitizer mirato PASS, default9375 raw identico | Input duplicati top8/33token falliscono anche B1, ERRORE_BASELINE sintetico; fallbackCPU esplicito. Nessuna integrazione scheduler modello, speedup o nuova concurrency. [Report](docs/development/moe-expert-pool-rx6800.md) |
| R30 | S04/S05: pool opzionale nel scheduler, layer17/stream1 | IMPLEMENTATO, VALIDATO breve / T2 PARZIALE | 17processi,903vettori raw finiti/nonzero e identici nei controlli abbinati;22fixture/132richieste e ASAN/UBSAN/leak PASS; pool32slot54MiB, hit62.01%, payload TG layer17-62.04%, miss p50/p95 0.356/0.725ms | PP-0.45%[-2.31,+0.99], TG+1.08%[+0.73,+1.44], p95-0.90%[-1.79,-0.07]: NESSUN_CAMBIAMENTO nelle fasce +/-3%/+/-5%; soglia1 identica OFF/ON/B1, non confronto al default. TG normale CPU non usa cache; arena completa invariata, +54MiB allocati; M3parziale. [Report](docs/development/moe-scheduler-pool-rx6800.md) |
| R31 | S01/S02: coppia benchmark breve/lungo e PP multi-chunk | IMPLEMENTATO, VALIDATO riferimenti / T2 PARZIALE | 512PP+200TG/c1024 e10000PP+200TG/c12288;16processi,1684vettori raw finiti/nonzero, coldB1/warmB1/forkOFF bitwise identici;11test PASS e5052hash timing coerenti | Baseline PP 259.04/246.76t/s, TG 22.34/16.52t/s (breve/lungo),4processi per caso/3rep dopo warmup, CI95 nel report. Non e' uno speedup e non risolve i precedenti workload instabili; poolON lungo/concurrency/corpus aperti. [Report](docs/development/moe-short-long-rx6800.md) |
| R32 | S14: allocazione pool/remap al primo accesso utile | IMPLEMENTATO, VALIDATO breve / lungo ERRORE_BASELINE | 23fixture scheduler/138richieste/414vettori,48core/384richieste/1152vettori,ASAN/UBSAN/leak e11test PASS;27processi,3368vettori raw finiti/nonzero; breve normale/attivo bitwise identici | Pool54MiB e remap evitati nei bypass; arena498.52MiB invariata, resident VRAM non diminuisce di54MiB. A/B8processi breve: PP-9.53%[-27.73,+0.71],TG+7.24%[-0.54,+21.86],p95-3.73%[-13.85,+2.94], tutti INCONCLUDENTE; processo06 conservato. Lungo B1 non ripetibile nella stessa campagna, nessun ranking. [Report](docs/development/moe-pool-lazy-rx6800.md) |
| R33 | S04/S09: offload mirato layer17 e controllo residente | IMPLEMENTATO, VALIDATO GPU breve / screen CPU-GPU FAIL / lungo ERRORE_BASELINE | 33fixture placement/113richieste/339vettori C++,23scheduler/414raw,ASAN/UBSAN/leak e11test PASS;25processi/1847raw finiti/nonzero; cinque GPU controls bitwise identiciB1;999hit/601miss,API-62.47% | 12timingGPU: pool vs target PP+0.50% NESSUN_CAMBIAMENTO,TG+2.30%[+1.12,+3.54] INCONCLUDENTE; vs residente PP-8.75%/TG-10.40% REGRESSIONE. CPU/GPU max_abs2.232,RMS0.611,KL0.0267 oltre limiti; lungoB1 cambia4argmax fra repeat. Nessun rankingCPU/default o lungo. [Report](docs/development/moe-targeted-placement-rx6800.md) |
| R34 | S09/T0: cold replay e placement per proiezione B1 | IMPLEMENTATO, VALIDATO controlli brevi / screen CPU-GPU FAIL | Helper MOE_REPLAY_WARMUP=0; librerie R33 invariate,18CLI PASS;19processi/3819raw finiti/nonzero; sei coppie cold esatte, warm/cold/clear CPU/full esatti e cinque controlli default compatibiliR33 | Gate/up/down/gate+up differiscono gia' al primoTG, RMS0.061..0.068; tutti screen FAIL,0/1/1/2argmax diversi. Warmup/reset non necessari sul breve; causa kernel non isolata, nessun ranking/fix lungo/promozione ibrida. [Report](docs/development/moe-placement-numerics-rx6800.md) |
| R35 | S09/T0: Q4_K isolato e precisione attivazioni | IMPLEMENTATO, VALIDATO fixture / modello ancora aperto | Estesa fixture esistente,18processi numerici/1296vettori attuali/2592oracoli,7repeat eB1/fork/debug esatti; pipeline registrati;2ASAN/UBSAN/leak e11test legacy PASS | InputCPUQ8_K su GPUF32 riduce RMS16-21mila volte rispetto aCPU/GPUauto; residuali~1e-7 non bitwise. Pesi/attivazioni sintetici, nessun fix logits/modello/ranking. Primo helper fallito conservato come HARNESS_ERROR. [Report](docs/development/moe-operator-numerics-rx6800.md) |
| R36 | S09/T0: cattura e replay MoE con dati reali | IMPLEMENTATO, VALIDATO osservatore breve/operatori / fix modello aperto | 5processi/1005logits esatti B1/fork/OFF/ON;2capture identiche da600record/2000input-vector;3pass reali1/100/200,18controlli/2ASAN,2synthetic,6negative e11legacy PASS | Stessa pipelineGPUF32 con inputCPUQ8_K riduce RMS45-52mila volte, residui~1e-7/1e-8. Solo traiettoriaCPU/un layer; nessun fix end-to-end, ranking o qualita' semantica. HARNESS_ERROR del verifier conservato. [Report](docs/development/moe-real-operator-replay-rx6800.md) |
| R37 | S09/T0: precisione ingressi layer17 sul modello completo | IMPLEMENTATO, VALIDATO diagnostica / SCARTATO come fix fedelta' | 34processi preregistrati+3CPU follow-up,7437logits finiti/nonzero;12repeat,4compat,copy,F32noop,3DEBUG,profile e2sanitizer esatti;4fixture/11legacy PASS | Q8_Kall maxRMS0.543 vs GPUoriginale0.611, tutti screen FAIL; CPUroundtrip diverge gia' al primoTG e CPU/GPUtrasformate piu' avanti. Nessun ranking/fix lungo/qualita'. [Report](docs/development/moe-model-input-precision-rx6800.md) |

R12-R16 derivano dagli archivi locali del 5-6 ottobre e dallo stato dello sweep; alcuni report precedenti sulla repo descrivono ancora un pilot o una rivalidazione in corso. Questa snapshot aggiorna **lo stato**, senza fingere che quei report storici siano gia' stati riscritti.

Due eventi vanno mantenuti separati: il precedente crollo PP durante alcuni screen CPU e' stato attribuito dall'utente a un gioco in Firefox che occupava la GPU; quelle righe non sono controlli puliti. In una successiva esecuzione senza device Vulkan disponibile il fallback CPU ha causato pressione RAM e systemd-oomd ha terminato il gruppo dell'app. Non e' una prova di difetto del kernel MoE. I guardrail locali ora verificano RX6800 prima del caricamento, riserva MemAvailable>=6144 MiB e scope separato con MemoryMax24G/MemorySwapMax2G. Sono protezioni del runner locale, **non nuove ottimizzazioni gia' pubblicate nel backend**.

Archivi da preservare, a partire dalla root del checkout locale:

```text
risultati/2026-10-04-validation/
risultati/2026-10-04-replay/
risultati/2026-10-04-active/
risultati/2026-10-04-runtime/
risultati/2026-10-04-generation/
risultati/2026-10-04-generation-large/
risultati/2026-10-05-quality64/
risultati/2026-10-05-context-sweep/
```

Checkpoint/prove della pausa nella workspace di Vulkan MoE Lab: `moe-session/context-sweep/STATUS.json`, `moe-session/context-sweep/resume-20261006/paused-by-user.json`, `resume-control-review.json`, `historical-contention-note.json` e `INCIDENT.md`. Non sono percorsi garantiti in un clone della repo. Pubblicare riepiloghi e manifest piccoli quando si conclude un esperimento; non inserire pesi o dump gigabyte in Git.

## Cosa prendere dai riferimenti esterni

### Video e branch Codacus: trasferimento e overlap nel prefill

[Video Codacus](https://www.youtube.com/watch?v=VytSYCDhWQ0) e [video Cloud Codes](https://www.youtube.com/watch?v=_Jdjq6pgIRg) descrivono lo stesso lavoro; non sono due repliche. Le patch primarie mostrano registrazione dei pesi host, copia anticipata dei tensori su stream CUDA separato e tre slot con correzione delle lifetime. Non implementano una cache persistente guidata dal router. Il +64.5% dichiarato e' PP su RTX3060/Qwen3.6, non TG o un risultato RX6800. Le tre modifiche e i loro costi vanno isolate: R06 ha gia' sondato il percorso host, R07 ha scartato la copia totale senza nuovo overlap; S11 resta da provare. [Branch e benchmark](https://github.com/thecodacus/llama.cpp/tree/fable5/prefetch-experts), [host registration](https://github.com/thecodacus/llama.cpp/commit/20f5994bfeb91d24da328077c4b6095998cc9888), [overlap](https://github.com/thecodacus/llama.cpp/commit/1163cb34939fe4a9cb07aec034c5954144497ae9), [tre slot/lifetime](https://github.com/thecodacus/llama.cpp/commit/5f83fbbe7c668c59912a1fe09e86a0ef580406c4).

### Strata: residenza selettiva e lavoro CPU/GPU

Revisione letta: [`82f46a8c8f475f001ad76d92f58f4a4f8ffb0253`](https://github.com/Niko1221/Strata/tree/82f46a8c8f475f001ad76d92f58f4a4f8ffb0253). Il codice usa profili di routing, quote per layer, slot residenti e percorsi distinti per hit GPU/miss CPU. I candidati qui sotto prendono queste idee come disegni da adattare, non come codice CUDA/HIP da incollare in Vulkan. [Cache header](https://github.com/Niko1221/Strata/blob/82f46a8c8f475f001ad76d92f58f4a4f8ffb0253/include/strata/core/expert_cache.hpp), [cache implementation](https://github.com/Niko1221/Strata/blob/82f46a8c8f475f001ad76d92f58f4a4f8ffb0253/src/core/expert_cache.cpp), [profili](https://github.com/Niko1221/Strata/blob/82f46a8c8f475f001ad76d92f58f4a4f8ffb0253/tools/make_profile.py).

Il dato ten-fold riportato nel cache header, 0.6447 hit a 4105 slot, riguarda il suo modello/workload. E' un esempio utile di validazione su trace non usati per scegliere gli slot; non predice il nostro hit rate. Nel nostro trace breve static prefill perde contro LRU a 32/64 slot: testare sia statico sia adattivo, senza scegliere la policy sul test set.

Ulteriori piste del codice Strata sono prefetch da router lookahead, complement RAM dei pesi residenti GPU, CPU SIMD/prefetch, budget fra KV ed esperti e decoding speculativo. S17/S18/S20/S21/S30 le rendono esperimenti separati. Il lookahead scalda dati: non deve sostituire gli ID scelti dal router reale. [Expert source](https://github.com/Niko1221/Strata/blob/82f46a8c8f475f001ad76d92f58f4a4f8ffb0253/include/strata/core/expert_source.hpp), [CPU IQ AVX2](https://github.com/Niko1221/Strata/blob/82f46a8c8f475f001ad76d92f58f4a4f8ffb0253/src/kernels/cpu/iq_avx2.cpp).

Le prestazioni pubblicate di Strata non sono un A/B locale: modelli, quantizzazioni, RAM, backend e spesso sistema operativo differiscono. La [segnalazione #816](https://github.com/Niko1221/Strata/issues/816) usa RX6800/5700X3D ma **64 GB e Windows11/HIP**, un modello IQ2_XS e contesto128K; disabilitare lo stream separato del shared expert migliora TG nella segnalazione. E' un motivo per misurare S12 in entrambe le direzioni, non una garanzia di 50-60 TG sul nostro sistema. La [documentazione AMD](https://github.com/Niko1221/Strata/blob/82f46a8c8f475f001ad76d92f58f4a4f8ffb0253/docs/AMD_HIP.md) distingue gfx1030, senza WMMA, dai percorsi RDNA3/4. I kernel Vulkan devono usare le feature effettivamente disponibili, non il nome architetturale di un kernel HIP.

Alcuni modelli Strata sono ridotti nel numero di esperti o hanno quantizzazione diversa: questo cambia il modello, non solo la cache. Tali confronti appartengono a S41/B5. Strata e' MIT; conservare attribuzione/licenza se si riusa codice, oltre a verificare compatibilita' tecnica. Il presente documento propone disegni, non un porting gia' completato. [Modelli](https://github.com/Niko1221/Strata/blob/82f46a8c8f475f001ad76d92f58f4a4f8ffb0253/docs/MODELS.md), [licenza](https://github.com/Niko1221/Strata/blob/82f46a8c8f475f001ad76d92f58f4a4f8ffb0253/LICENSE).

### Upstream e letteratura: riferimenti, non risultati del fork

Ricerca upstream al 2026-10-06, snapshot `abeada335e2e78bd3fe63febafab7e900ce75810`. Prima di implementare, rileggere scheduler e backend sia alla baseline sia allo SHA di destinazione: una discussion non prova che la funzione sia gia' integrata. Il [pool persistente RFC #28248](https://github.com/ggml-org/llama.cpp/discussions/28248) suggerisce remap degli ID verso slot e kernel esistenti; il [cache ibrido RFC #24528](https://github.com/ggml-org/llama.cpp/discussions/24528) motiva l'alternativa CPU-miss. L'[issue #25859](https://github.com/ggml-org/llama.cpp/issues/25859) riguarda le attese H2D del prefill; l'[issue #20757](https://github.com/ggml-org/llama.cpp/issues/20757) descrive cache a livelli. [Scheduler allo SHA letto](https://github.com/ggml-org/llama.cpp/blob/abeada335e2e78bd3fe63febafab7e900ce75810/ggml/src/ggml-backend.cpp).

L'[issue Vulkan #25195](https://github.com/ggml-org/llama.cpp/issues/25195), chiusa, documenta hazard di riuso dei buffer di trasferimento su altre GPU/driver. E' materiale per verificare sincronizzazione e lifetime, **non una causa dimostrata dei nostri NaN**. I [contatori DRM](https://docs.kernel.org/gpu/drm-usage-stats.html) distinguono allocazione e residenza: una differenza VRAM/GTT non misura da sola byte/s PCIe.

[Fiddler](https://arxiv.org/abs/2402.07033) e [MoE-Infinity](https://arxiv.org/abs/2401.14361) sono fonti per offload, profili e prefetch; [PowerInfer](https://arxiv.org/abs/2312.12456) tratta sparsita' delle attivazioni e ispira placement, ma non e' un cache MoE esatto gia' pronto. I loro speedup non vanno trasferiti a RDNA2/RADV.

## Ranking: dove investire prima

Ordine decisionale per **questa macchina**, non classifica universale. Priorita' P0 = rendere interpretabile il prossimo A/B; P1 = nucleo con potenziale su singolo stream; P2 = ottimizzare dopo una misura dei colli di bottiglia; P3 = dipendente da modello/capacita' o costoso. Confidenza descrive l'evidenza, non una probabilita' inventata di successo.

| Ordine | Strategie | Potenziale principale | Evidenza / costo / rischio | Prima condizione di avanzamento |
| ---: | --- | --- | --- | --- |
| 0 | S01-S03 | Misure corrette e spiegazione PP/TG | R20 verifica replay; costo profiler TG misurato, nessun speedup promesso | Controlli puliti, budget e tempi per fase |
| 1 | S04-S06 | Cache reale e placement esatti; nel decode R20 gli esperti CPU non sono caricati sulla GPU | Hit logici locali; costo medio; lifetime/remap da validare | Pool minimo, stesso output, byte H2D e TG misurati |
| 2 | S07-S08 | Hit GPU + miss CPU, overlap senza trasferire pesi freddi | Strata/Fiddler come disegno; costo alto; riordino/sync rischiosi | Confronto contro pool tutto-GPU, non solo contro CPU |
| 3 | S09 | Placement e quote per layer da profilo | Dati tracing disponibili; costo medio; profili possono generalizzare male | Held-out + stessa VRAM + distribuzione quote |
| 4 | S10-S11 | Trasferimenti selettivi e overlap prefill | R06 positivo PP, R07 negativo; costo medio-alto | Dimostrare overlap e TTFT, rispettare RAM32GB |
| 5 | S13-S14 | Policy adattiva e protezione cache decode | LRU locale migliore di static breve; costo medio | Ridurre miss-cost, non solo alzare hit rate |
| 6 | S15-S16 | CPU/thread/AVX2, CPU miss piu' rapido | CPU e' parte del percorso; costo basso-medio | Tempi CPU esposti misurati, controlli fra processi |
| 7 | S19-S20 | RAM utile, caricamento e pinning limitati | Costo RAM none osservato; costo medio-alto | MemAvailable e GTT migliori a PP/TG comparabili |
| 8 | S21-S23 | KV/cache/context/batch bilanciati | Vantaggi e fallimenti locali; costo basso-medio | Nessun allargamento automatico di ubatch senza gate |
| 9 | S24-S27 | Kernel e riduzione overhead Vulkan | Evidenza da raccogliere sul kernel dominante; costo medio-alto | Timing GPU distingue memory/compute/submission |
| 10 | S28-S29 | TTFT e memoria per conversazioni/agent1-8 | Workload reale ancora non misurato; costo medio | Latenza per richiesta e correttezza dello stato |
| 11 | S17-S18 | Prefetch temporale/router lookahead | Solo ipotesi locale; costo medio-alto; rischio traffico extra | Precisione utile e byte sprecati con prefetch OFF/ON |
| 12 | S12 | Shared-expert su queue separata oppure seriale | Segnalazione Strata avverte possibile regressione; costo medio | Timeline e A/B RDNA2 prima di mantenere due queue |
| 13 | S30-S32 | Speculazione/prompt lookup/draft | Modello e rollback dipendenti; costo medio-alto | Token accettati/s, fedelta' sampling, KV/recurrent corretti |
| 14 | S33-S36 | Tuning controllato e aggiornamenti upstream/driver | Riduce configurazioni inutili; costo variabile | Un solo fattore per volta, SHA/driver fissati |
| 15 | S37-S40 | Quantizzazione, three-tier, persistenza, motore esterno | Cambia capacita'/qualita' o richiede piu' memoria; costo alto | Controllo qualita' separato e vincolo 32GB |
| 16 | S41-S42 | Riduzione modello/sparsita'/GPU-driven routing | Rischio alto, modifica semantica o architettura; costo alto | Solo dopo aver esaurito ottimizzazioni conservative |

La prima cache deve essere semplice e misurabile. Non iniziare con cache + lookahead + nuovo kernel + speculative decoding insieme. Nel regime in cui tutti i pesi gia' entrano comodamente in VRAM, la cache puo' aggiungere solo overhead: includere quel controllo se il modello scelto lo consente.

## Catalogo delle strategie con esperimento A/B

Ogni strategia seguente e' **PROPOSTA**, salvo i prerequisiti e le prove R01-R33 esplicitamente citate. S01/S02 hanno il tooling iniziale implementato in R17, ma restano incompleti i gate runtime, il corpus e la verifica held-out. S03 ha scope host/contatori copie R18, intervalli/aggregazione PP/TG R19 e verifica sul modello R20 con timestamp diagnostici e memoria breve; timeline calibrata nel percorso normale e budget per contesti lunghi restano incompleti. Non e' dichiarata implementata perche' esiste in una fonte esterna. I protocolli T0-T7 sono definiti nella sezione successiva; gli A/B qui fissano la variabile e le metriche aggiuntive. Anche i gruppi nello stesso livello si provano uno alla volta.

### P0: fondamenta e modello dei costi

**S01 - Congelare baseline e runner riproducibile.** Collo: confondenti di default, librerie, fit e interferenza GPU. Utile su RX6800 per separare regimi di allocazione da effetti reali. Riusare replay/helper e manifest esistenti; aggiungere controllo librerie, device e processi prima del load. Costo: tempo di controllo, fuori dal timer. A/B T0/T1 B1 vs B2 senza feature; attribuire eventuali differenze prima di promuovere ottimizzazioni. Guardrail locali presenti; protocollo finale ancora da completare.

**S02 - Profilo routing rappresentativo e costo per layer.** Collo: si ottimizza senza conoscere working set e percorso CPU/transfer/GPU. Estendere analyzer strided corretto con entropia, reuse distance, finestre1/8/32/128 token, cambi di dominio, hit byte-weighted e fasi prefill/decode. Trace a parte dai timing, perche' il callback sincronizza. T2 su italiano, codice, tool, chat lunga e held-out; confrontare cold/warm/static/LRU con quote uguali in byte. Nessuno speedup da un hit rate simulato.

**Incremento R17 (2026-10-06).** `examples/moe-trace/manifest.py` cattura un processo gia' avviato fuori dai timer e puo' rifiutare drift rispetto a un manifest congelato della stessa variante. Non lancia inferenza; una libreria Vulkan mappata non dimostra esecuzione sulla GPU. L'analyzer aggiunge entropia, riuso atomico per top-k, finestre multiple e stime logiche in byte da metadata espliciti, senza alterare le metriche precedenti. Mancano corpus rappresentativo/held-out, misure di costo e validazione della campagna. [Definizioni, limiti e test](docs/development/moe-profile-foundations-rx6800.md).

**S03 - Timeline CPU/GPU e accounting memoria.** Collo: TG puo' dipendere da CPU, copia o sincronizzazione anziche' shader. Aggiungere timestamp/query Vulkan e marker CPU opt-in, byte di upload/readback, attese e picchi pesi/KV/recurrent/compute/cache. Non dedurre PCIe da GTT. Costo: overhead misurare OFF/ON; backend senza profiling come riferimento. T1/T3 con T0: rapporto durata reale/strumentata e percentuale di tempo esposto per componente.

**Incremento R18 (2026-10-06).** `GGML_SCHED_PROFILE=PATH` registra scope host e payload API delle copie, padding, attese e buffer riservati dallo scheduler; assente/vuoto e' OFF. Parser e fixture indipendente verificano output e contatori CPU/RX6800 senza modello. Riusato il logger timestamp Vulkan esistente: il caso parallel/view fallisce anche nel controllo, seriale passa. Nessuna nuova sincronizzazione GPU; nessuna stima implicita del traffico PCIe o assegnazione PP/TG. [Limiti, metadati e comandi](docs/development/moe-scheduler-profile-rx6800.md).

**Incremento R19 (2026-10-07).** `MOE_REPLAY_PROFILE=PATH` registra intervalli dichiarati prefill/decode, warmup incluso e distinto; `profile-summary.py --phases` associa scope contenuti e aggrega ms/byte per ripetizione/fase. La fase non viene dedotta dal numero di token; scope ambigui o attraversanti confini sono rifiutati. Fixture CPU/RX6800 verifica 32 associazioni e output invariato; replay compilato, join sul modello non eseguito. [Limiti, metadati e comandi](docs/development/moe-phase-profile-rx6800.md).

**Incremento R20 (2026-10-07).** Ripresa autorizzata. Join sul modello e correttezza completa validati; overhead del profiler host -6.07% TG, PP equivalente nel margine3%. Logger Vulkan riusa correttamente un contesto gia' aperto da copie/eventi e produce marker per il join per fase; output invariato, 32 join sintetici e 55/2432 graph PP/decode128 reali. Misurati allocator, DRM per-client e VRAM globale; i timestamp sono diagnostici serializzati, non una timeline normale calibrata. [Report](docs/development/moe-model-profile-rx6800.md).

### P1: cache esperti e percorso ibrido

**S04 - Pool GPU persistente minimo, routing esatto.** Collo: pesi ricopiati quando gli esperti CPU sono eseguiti sulla GPU. R20 non misura tali upload nel decode a ncmoe18: una cache decode deve cambiare anche placement; richieste miste e costo miss CPU/upload sono parte del disegno. Slot `(layer,expert)` con triplet gate/up/down, upload dei soli miss e remap ID per i kernel `MUL_MAT_ID` esistenti. Su16GB puo' tenere il working set caldo senza spostare interi layer. Costo: VRAM sottratta a KV/compute e miss sincroni. T0/T2/T3 B2 cacheOFF vsON, capacita'0/16/32/64/96/128 per layer compatibilmente col budget; TG, H2D, evictions, cold/warm e casi tutti-miss/tutti-hit.

**Incremento R29 (2026-10-07).** Prototipo isolato con buffer persistenti RX6800 per triplette F32/Q4_K/Q6_K, miss-only upload, remap top-k, LRU per istanza layer e lease singola.48combinazioni/384richieste,1152vettori per processo, due captureGPUbit-identiche; gate scalareF32 e same-backendquant senza tolleranza. Duplicati entro top8 vanno in fallbackCPU dopo difetto sintetico riprodotto B1. Scheduler modello non integrato, costo miss/TG/concurrency NON_MISURATI. [Report](docs/development/moe-expert-pool-rx6800.md).

**Incremento R30 (2026-10-07).** Pool collegato allo scheduler per un layer, senza cambiare placement o router. Replay244+128: B1/OFF/ON raw identici, anche con soglia offload1 per il decode; min-batch normale32: PP oltre cap e TGCPU non usano il pool.32slot54MiB,635hit/389miss su128TG, byte pesi layer17-62.04%, attese p50 4.739ms. PP/TG/p95 equivalenti nelle fasce +/-3%/+/-5%. Solo breve/stream1; nessuna cache universale o risparmio arena. [Report](docs/development/moe-scheduler-pool-rx6800.md).

**S05 - LRU e fallback del pool al percorso originale.** Collo: una cache piena non deve fare thrashing o fallire l'allocazione. Iniziare con LRU deterministica per layer, pin degli slot in-flight e bypass quando il batch usa piu' esperti della capacita'. Riusare transfer originale per bypass; non cambiare il router. Costo: lookup e doppia rappresentazione degli ID. T0/T2: cacheOFF/LRU/bypass, ID ripetuti, capacita'1, pressione KV, batch oltre capacita', abort e ripresa; T3 quantifica costo dei miss.

**Incremento R21 (2026-10-07).** Collector normale opt-in senza nuovi readback/sync, join con fasi esplicite, filtro layer e statistiche all-hit/all-miss/mixed. Quattro prompt brevi train/held-out superano il gate raw B1; a15slot CPU18 il payload e490.43MiB e richieste complete LRU0.52..1.39%, non il37..43% di singole activation hit. Il callback storico diverge anche con engine originale; R05 e' conservato ma non validato per inferenza normale. M2 resta parziale: corpus esteso, remap/lifetime e costo miss. [Report](docs/development/moe-cache-plan-rx6800.md).

**S06 - Budget cache in byte e quote per layer.** Collo: quote uniformi possono sprecare memoria e i primi layer possono monopolizzare un pool globale. R27 implementa confronto offline uniforme/global-frequency/bundle; R28 aggiunge riserva minima4/8, cap in byte e piani TRAIN persistiti prima del parsingHELDOUT. Corpus12nuovi prompt, riserva4elimina quotezero ma complete4.456->4.181% a490MiB; nessuna cache runtime/TG gain. [Report](docs/development/moe-layer-reservations-rx6800.md). Budget dopo pesi densi, KV/recurrent, compute worst-case e margine desktop misurato. Confrontare uniforme, minima quota + distribuzione a beneficio marginale, globale con riserva per layer. Riusare planner, non nuovo allocator globale. T2/T3 a budget0/256/512/1024/2048 MiB solo se disponibili; stessa memoria, byte salvati/ms e p95 dei miss. Dipende S03/S04.

**S07 - Hit GPU, miss CPU, merge esatto.** Collo: trasferire un esperto freddo puo' costare piu' che calcolarlo su5700X3D. Separare attivazioni/ID hit e miss, GPU per residenti, CPU per altri, scatter nell'ordine originale top-k e stessa riduzione. Adattare scheduler/backend, riusare CPU SIMD; evitare diversa quantizzazione delle attivazioni fra i due percorsi. Costo: readback attivazioni, sincronizzazione e possibile rounding diverso. T0/T3 pool tutto-GPU vs ibrido, miss0/25/50/75/100%, stessi slot e pesi; logits completi, TTFT/TG/CPU busy, non solo hit rate.

**S08 - Overlap CPU miss e GPU hit, soglia calibrata.** Collo: CPU e GPU eseguiti in serie espongono il lato lento. Usare task indipendenti e un join per layer; poi selezionare CPU vs upload+GPU dal costo misurato, distinto per prefill/decode. Su singolo stream overlap utile solo se CPU e GPU hanno lavoro sufficiente. Costo: task/thread overhead e stima instabile. T3 S07 seriale vs overlap, poi soglia fissa vs calibrata; includere shared expert e dimensione batch, timeline e p95. Dipende S07 con T0 superato.

**S09 - Residenza statica da profilo e placement per layer.** Collo: ncmoe sceglie layer interi, non gli esperti utili. Prima controlli `ncmoe`/tensor override esistenti; poi caricare esperti caldi da profilo e lasciare freddi CPU. Strata suggerisce profili e quote, ma il nostro static prefill R05 e' debole: confrontare profilo corpus, prompt e adattivo separatamente. Costo: startup e mismatch dominio. T2/T3 ncmoe fisso vs statico vs S04, stesso budget VRAM e trace held-out; misurare fine-tuning del profilo senza leakage.

### P1/P2: trasferimenti e policy

**S10 - Upload selettivi raggruppati.** Collo: molte piccole copie/ID readback durante prefill. Usare l'unione reale degli ID, accorpare intervalli contigui, ordinare i trasferimenti e scatter/remap; contare padding e gap copiati. RADV puo' beneficiare di meno regioni, ma copiare tutto ha gia' perso R07. Costo: preprocessing CPU e bytes inutili. T1/T3 selettivo attuale vs intervalli vs full-copy archiviato come controllo, PP512/2048/4096, payload reale e numero di submit; T0 strided/quant.

**Incremento R22/R23 (2026-10-07).** R22 replica sullo stesso engine9375 il load-mode esistente: PP breve +33.80%, TG/p95 entro margini, raw identici B1; GTT residente circa9.57GiB e avvio +4.88s. Il tempo host upload scende ma attesa si sposta in input_wait: niente prova di banda fisica/overlap. R23 tenta il prompt lungo3107token: dal terzo chunk B1 stesso non ripete i logits (finiti/nonzero), quindi gate fallito e nessun ranking lungo. Bounded staging e pool richiedono protocolli propri; B3 non aggiornato. [Replica host](docs/development/moe-host-transfer-recheck-rx6800.md), [gate lungo](docs/development/moe-long-prefill-recheck-rx6800.md).

**S11 - Prefill asincrono con staging ring e queue.** Collo: H2D e compute serializzati, ispirazione principale del video. Prima dimostrare queue/features RADV disponibili; ring3 slot per gate/up/down o numero misurato, eventi/fence e lifetime esplicite. Copia anticipata completa e copia selettiva sono varianti distinte. Costo: piu' staging/VRAM/RAM, traffico e contention bandwidth. T0/T1/T3 stesso load mode, sync vs async, ring1/2/3, PP512/2048/4096 e TTFT; shader/timeline devono provare overlap, fallback senza UAF obbligatorio.

**S12 - Shared expert: seriale vs queue separata.** Collo: il shared expert puo' sovrapporsi al routed MoE o competere per risorse. Hook limitato al nodo shared, se presente nel modello; nessun assunto dal report HIP Strata. Costo: submit e bandwidth, possibile perdita su RDNA2. T0/T3 queueOFF/ON, hit/miss misti e batch piccoli/grandi, timeline shared/routed e TG; mantenere seriale se non utile. Dipende S03 e supporto modello.

**S13 - Policy di admission/eviction adattiva.** Collo: LRU soffre scansioni, LFU conserva esperti non piu' utili. Dopo S05 provare LFU con decay, SLRU/2Q, admission tipo TinyLFU e cost-weighted (byte/latency evitati) una alla volta. Costo: counters/metadati e CPU, non sempre piu' hit equivale a piu' TG. T2 corpus held-out prima, T3 solo migliori2 a stessa memoria: miss byte-weighted, tempo miss, evictions, adattamento dopo cambio dominio; includere overhead lookup.

**S14 - Cache distinta prefill/decode e warm-start.** Collo: un prefill grande puo' espellere tutto il working set decode. Provare bypass prefill, partizione protetta, warm LRU da prompt, profilo decayed del decode precedente e reintegro bounded. Costo: priming e cattiva previsione; la cache fredda puo' vincere sui prompt brevi. T2/T3 cold vs warm, PP244/512/4096/10000 e TG200 nei nuovi riferimenti R31, pause tra turni e cambio task; includere priming in TTFT/end-to-end. Dipende S04; statico R05 non basta a scegliere.

### P2: CPU, prefetch e memoria host

**S15 - Thread, affinita' e scheduling CPU.** Collo: banda DDR4/AVX2 e oversubscription fra miss, loader e agenti. Riusare `-t`/`-tb`, testare1/2/4/6/8/12/16 con CPU placement fisso e affinita' solo se utile. 5700X3D ha8 core fisici; non assumere SMT utile. Costo: maggiore uso CPU e peggiore p95 multi-request. T1/T3 balanced con controlli iniziali/finali; poi T6 thread budget globale. Riprendere i checkpoint R15, non reinterpretare screen disturbati da Firefox.

**S16 - Kernel CPU MoE, prefetch e repacking limitato.** Collo: lettura pesi/dispatch CPU dei miss. Con profiling scegliere Q4_K vector dot, traversal attivo, locality e software prefetch; distanze0/256/1024/2048 byte, repack solo degli esperti necessari. Riutilizzare kernel/operator bench; R01/R04 non hanno gia' dimostrato TG migliore. Costo: cache pollution, RAM duplicata, startup. T0 operatori e T3 modello, misura CPU bandwidth/tempo esposto; Strata IQ/Zen3 e' ispirazione, non prova per Q4_K.

**S17 - Prefetch da storia routing.** Collo: attesa sul prossimo miss. Usare n-gram, frequenza decayed o ultimo set per layer per riempire solo slot liberi; demand ha sempre precedenza. Costo: byte sprecati, eviction utile e contention. T2 precision/recall e lead-time; T3 OFF vs top1/top2/top4 a stesso budget, miss residuali e waste bytes. Routing reale invariato, slot prefetch in-flight protetti. Dipende cache e timing.

**S18 - Router lookahead e warming pagine.** Collo: miss su file/RAM o H2D scoperti tardi. Prima warming RAM con hint/readahead; eventuale worker CPU applica router successivo ad attivazioni disponibili come **previsione**, sul modello che lo consente. Non usare predizione come output router. Costo: CPU aggiuntiva e bassa accuratezza; sovrapposizione da dimostrare. T2/T3 no-lookahead vs one-layer, startup e decode separati, fallimenti predizione e waste; T0 verifica stesso percorso reale. Ispirazione expert_source Strata.

**S19 - Pinning host bounded e caricamento per segmenti.** Collo: `none` evita staging ma usa molti GiB di memoria host non facilmente recuperabile. Provare ring host pinned limitato, registrazione segmentata dei soli esperti hot, mmap+cache staging. Verificare API Vulkan/RADV disponibili; non assumere `cudaHostRegister` portabile. Costo: copie CPU supplementari, page faults e registration. T1/T3 mmap vs none vs cap256/512/1024 MiB, PP/TG, startup, MemAvailable/GTT/resident, T6 pressione agenti. Nessun risparmio RAM dedotto dal solo RSS.

**S20 - RAM complementare alla cache GPU.** Collo: copie CPU/repacked e residenti GPU possono duplicare pesi su32GB. Prima quantificare reclaim del file-backed mmap; poi pool host mantiene solo gli esperti non residenti e staging di scambio bounded. Non liberare l'unica copia prima del commit del trasferimento; sorgente file sempre recuperabile. Costo: fault/ricaricamento e passaggi di ownership. T0 eviction/in-flight/abort, T3 stessa cache con duplicazione vs complement, PSS/MemAvailable/GTT e miss p95; non partire da questo allocator complesso.

### P2: KV, batch e shader Vulkan

**S21 - Budget dinamico KV/cache e contesto.** Collo: KV e cache competono per VRAM; recurrent state del modello conta separatamente. Prima allocazioni pianificate per contesto/slots, poi pool segmented Vulkan che riduce cache al crescere KV; eventuale prestito prefill con restore misurato. Non copiare CUDA VMM senza equivalente. Costo: evictions e p95 di resize. T3/T4 budget fisso vs elastico, contesti crescenti e1/2/4/8slots, peak/waste e restore, T0 stato completo.

**S22 - Tipi KV e Flash Attention.** Collo: banda/capacita' attention. A/B F16/Q8_0 e altri tipi realmente supportati da build/modello, FAOFF/ON e guardQ8 OFF/ON separati. Risparmio KV puo' finanziare S04; non e' gratis in qualita'. Costo: errori numerici/conversioni. T0/T4 e T5, same-type upstream, layer norm/logits finiti; difetti condivisi=ERRORE_BASELINE. BarrieraQ8 R09 resta circoscritta, non correzione generale F16.

**S23 - Batch, ubatch e chunking applicativo.** Collo: utilization prefill vs VRAM e TTFT. Riusare parametri `-b`/`-ub` e helper con final-prompt-output; sweep ub128/256/512/1024/2048, batch512/1024/2048 a routing/placement fissi. Costo: compute buffer e continuazioni diverse, R11/R12. T1/T4/T5: PP, TTFT, memoria e complete risposte; upstream allo stesso batch e confronto fra batch distinti. Non scegliere2048 soltanto dal replay.

**S24 - MMVQ decode e MMQ prefill per RDNA2.** Collo: kernel quantizzati dominanti. Solo dopo S03, sperimentare workgroup/subgroup, tiles, vector load, shared memory e integer-dot se feature Vulkan accelerata realmente presente. Riusare shader dispatch con toggle e fallbacks. RDNA2 non riceve automaticamente il percorso WMMA RDNA3/4. Costo: register pressure, occupancy e layout vincoli. T0 operatori Q4_K/Q6_K strided, T1/T3 small/large batch, timestamp e TG; una modifica/kernel per A/B.

**S25 - Fusion gate/up/activation/down e grouped experts.** Collo: submit e traffico intermedio MoE. Prima grouped dispatch per esperti residenti e fuse gate/up+activation con pesi invariati; separare CPU-miss/GPU-hit. Costo: uso registri, diversa somma/rounding e compatibilita' quant. T0 indipendente, logits e T3 counts1/8/32 esperti attivi + TG/PP; non promuovere fusion con nonfinite R10. Dipende S24 solo se profiling la giustifica.

**S26 - Attention, recurrent/GDN/SSM e shared/dense.** Collo: dopo cache, il tempo puo' spostarsi fuori dal MoE. Profilare nodi realmente presenti in Ornith, poi tile/accumulo/fusion circoscritti con fallback. La cache non accelera questi nodi automaticamente. Costo: errori nello stato ricorrente, FA e contesti lunghi. T0/T4 kernel isolato + teacher-forced lungo + turni/server; T3 TG e TTFT, prima/dopo breakdown. Non assumere che un layer ricorrente sia un KV tradizionale.

**S27 - Submit, readback, descriptor e graph reuse.** Collo: overhead per token con piccoli kernel. Riusare command/descriptor e ridurre sync solo quando le dipendenze lo consentono; evitare readback inutili degli ID senza ripetere full-copy R07. Costo: hazard e output stale. T0 resize/abort/multi-stream con validation layers fuori timing, T3 graph reuse e async toggles singoli, GPU vs CPU duration. Gli screen di disabilitazione precedenti non hanno risolto il difetto lungo: non chiamarli fix.

### P2/P3: lavoro reale e speculazione

**S28 - Prefix cache e stato conversazione riusabile.** Collo: prompt ripetuti/tool history aumentano TTFT. Prima usare funzioni server esistenti, poi cache/persistenza dello stato completo KV+recurrent, identity modello/template/RoPE e prefisso token esatto. Costo: RAM e invalidazione; condivisione errata cambia risposte. T0/T7 fresh vs reuse su stesso prefisso, prefissi quasi uguali e cambio template; TTFT, peak RAM e output. Dipende supporto modello, non solo KV-copy.

**S29 - Scheduling e memoria per1/2/4/8 agenti.** Collo: prefill lungo puo' bloccare decode e ogni stream puo' competere per cache. Comparare continuous batching esistente, chunk prefill, quote/cache condivisa vs riserva per stream, backpressure e parking stato idle. Costo: fairness vs throughput e snapshot RAM. T6/T7: una richiesta lunga + richieste brevi, TTFT/p95 inter-token, TG per stream, aggregato, RAM e hot-set sharing. Ottimizzare1prima; non progettare cluster multiutente.

**S30 - MTP nativo, se disponibile.** Collo: un passo target/token e troppo caro. Verificare che checkpoint, formato GGUF e grafo implementino head MTP; Ornith non viene dichiarato compatibile senza verifica. Adattare verify/rollback di KV+recurrent con kernel target esistenti. Costo: VRAM head, draft compute e bassa acceptance. T0/T5/T7 OFF vs depth1/2/3, target tokens/s accettati, reject-cost, TTFT e distribuzione sampling. I guadagni Strata su altri modelli non sono previsti locali.

**S31 - Prompt lookup / n-gram speculative.** Collo: copie di codice/testo ripetuto richiedono decode completo. Riusare supporto esistente alla revisione scelta prima di patchare; verificare supporto recurrent e rollback. Costo: search/verification inutile su testo nuovo. T0/T5/T7 OFF vs draft1/3/5/8, code-edit/copy e chat non ripetitiva, acceptance e end-to-end. Candidato meno invasivo di un secondo modello se le primitive sono gia' disponibili.

**S32 - Draft model esterno o self-speculation.** Collo: TG target, alternativa a MTP. Scegliere draft piccolo compatibile col tokenizer e memoria32GB; self-speculation/layer skip e' una variante separata con exact target verification. Costo: modello aggiuntivo, VRAM sottratta a cache, acceptance/rollback. T0/T5/T7 target solo vs draft in CPU/GPU, stesso sampling e prompt, tokens accettati/s e costo totale. Nessun percorso approximate puo' essere descritto come output-identico senza prova.

### P2/P3: configurazioni e fonti di progresso indipendenti

**S33 - Autotuner offline per placement e configurazione.** Collo: un singolo ncmoe non e' ottimo per tutte le fasi. Completare sweep R15 con coordinate, interazioni selezionate e resweep finale; profili distinti single-stream/chat-lunga/4streams. Costo: molte prove e overfit al prompt; niente retuning nel mezzo di A/B di una patch. T1/T4/T6 hold-out, baseline B0 e B1 separate; objective latenza con budget RAM/VRAM, non solo somma PP+TG.

**S34 - Allocazione small-BAR, heap e margini.** Collo: placement host-visible/device-local puo' cambiare regime su visibleVRAM256MiB. Confrontare flag esistente `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM` e margini fit a stessi tensori; eventualmente allocator opt-in solo dopo identifying buffer. Costo: staging e VRAM piu' stretta. T0/T3/T4 OFF/ON ABBA con requested/resident e timeline; R16 non dimostra ancora causa ne' fix. Non cambiare BAR/BIOS/profilo GPU per questo esperimento.

**S35 - Aggiornamento upstream/cherry-pick mirato.** Collo: fix o kernel recenti possono rendere superflua una patch. Isolare clean B4, poi singolo cherry-pick in branch dedicato; congelare v0.5.0. Costo: API/semantica default e regressioni. T0/T1/T4/T7 stesso modello e valori effettivi; verificare diff e issue collegate, riportare miglioramento attribuito upstream. No rebase distruttivo dei build congelati.

**S36 - Mesa/RADV e compiler A/B controllati.** Collo: shader compile, scheduling e driver possono dominare. Prima confrontare pipeline/cache compile, flags native e versioni rilevate; eventuali driver alternativi in ambiente isolato con rollback, una sola variabile. Costo: setup e attribuzione complessa. T0/T1/T3 stesso shader/model/runtime e cache fredda/calda separate. Non usare CUDA/ROCm come misura di una patch Vulkan, e non cambiare tensione o clock.

### P3: capacita', modelli e architetture piu' impegnative

**S37 - Quantizzazione pesi e precisione per tensore.** Collo: dimensione esperti e banda DDR/PCIe/VRAM. A/B GGUF realmente disponibili Q4_K_M/Q3/IQ o precisione diversa dense/router/shared vs esperti, con kernel supportati. Piu' residenti puo' aiutare TG ma cambia qualita'; salvataggi cache devono includere identita' quant. Costo: quant/conversione e accuracy. T0 numerica, T5/T7 qualita', T3 bytes/cache/TG; risultati categoria modello, non ottimizzazione lossless sullo stesso GGUF.

**S38 - Terzo livello SSD/file tier.** Collo: modelli che non entrano in32GB. Readahead/madvise, staging bounded e RAM cache sopra mmap; NVMe solo se hardware effettivamente presente/documentato. Costo: latenze tail e traffico; per Ornith che entra in RAM probabilmente priorita' bassa. T3 cold page-cache/warm separati, minor/major faults, I/O bytes, p95 miss e1/2/4/8; non attribuire a GPU cache un beneficio dalla page-cache OS.

**S39 - Profili persistenti e cache state versionato.** Collo: cold-start e cambio conversazione perdono informazione utile. Serializzare frequenze/quote, non puntatori; modello hash, tensor layout, quant e versione schema; atomic write, validazione e fallback. Slot weights possono essere ricostruiti dal GGUF. Costo: startup/I/O e stale profile. T0 mismatch/corruzione, T2/T7 fresh vs restored, TTFT completo con priming e domain shift; dipende S09/S13. Cache conversazione S28 e' stato diverso dal profilo routing.

**S40 - Benchmark Strata/altro backend come riferimento esterno.** Collo: valutare se il fork e' la via piu' efficace per il workload reale. Nessuna installazione in questa task. In futuro verificare supporto su CachyOS32GB, modello/licenza e replay condivisibile; B5 con stessi pesi se possibile, altrimenti tabella separata qualita'/memoria/latenza. Costo: conversioni e incompatibilita' modello. T5/T7, PP/TG/TTFT e stato completo; non importare numeri Windows64GB come baseline locale.

**S41 - Pruning esperti, riduzione top-k o sparsita' approssimata.** Collo: pesi/compute ridotti oltre la cache esatta. E' modifica del modello; mai rinormalizzare/eliminare esperti silenziosamente nel percorso conservativo. Branch/modello dedicati e metadata di trasformazione, eventuale calibration/fine-tuning esplicito. Costo: accuracy e comportamento imprevedibile. T5 held-out esteso e T7, stesso budget output/task, vantaggio memoria/latency e regressioni per categoria. Priorita' bassa prima di S04-S09 lossless.

**S42 - Routing/remap GPU e architettura integralmente GPU-driven.** Collo: readback ID/dispatch CPU residui dopo cache efficace. Prima remap compute shader, poi indirect/grouped dispatch solo se estensioni e pipeline lo permettono; cache miss CPU e sincronizzazione restano necessari. Costo: grande cambiamento scheduler, contention atomiche e piu' debugging. T0 con CPU reference indipendente, T3 vs S04/S27 a stessi slot, readback/submission e TG. Avanzare solo con evidenza che l'overhead CPU limita davvero il singolo stream.

Multi-GPU, NUMA multi-socket, cluster/disaggregazione, RDMA e serving a centinaia di utenti sono fuori dalla priorita' di questa macchina. Non sostituiscono la verifica single-stream/RDNA2. Il catalogo copre le strategie pratiche individuate nelle fonti e nei dati attuali; nuove idee ricevono un ID, motivazione e A/B prima dell'implementazione.

## Invarianti per implementare una cache senza cambiare il modello

1. Gli ID reali scelti dal router, i pesi e i coefficienti restano invariati; il remap cambia solo dove si leggono gli stessi dati. Preservare ordine top-k, ripetizioni/broadcast, byte stride, quant blocks e ordine di riduzione o documentare ogni differenza numerica ammessa prima dei test.
2. Un esperto logico include tutti i tensori gate/up/down effettivi del modello; un hit parziale non e' un hit pronto al calcolo. Shared/dense non sono conteggiati come hit routed per gonfiare la metrica.
3. Nessun buffer o slot viene liberato/sovrascritto finche' un submit lo usa. Versione/generation dello slot, eventi e ownership delle sorgenti impediscono use-after-free, stale ID e write-after-write. Allocate-before-free e rollback della prenotazione in caso di errore.
4. Prefetch e cache admission non modificano il routing; il demand path resta valido anche con predizioni completamente errate. Quando budget/feature mancano, fallback esplicito al percorso originale e registrazione del fallback.
5. Budget include pesi densi/residenti, KV, stato recurrent, compute worst-case, staging, descriptor/metadati e margine desktop. Cache miss non deve creare allocazioni illimitate per token.
6. In 1/2/4/8 streams, cache condivisa puo' riusare solo pesi immutabili della stessa identita' modello/quant; stato conversazione e recurrent restano correttamente separati. Fairness e quote sono parte del risultato.
7. Feature separabile, OFF di default durante la ricerca, nessun cambio implicito al modello o ai default. Conservare il percorso precedente per A/B, fallback e bisect.

## Protocolli comuni e criteri di decisione

### T0 - Correttezza, provenance e condizioni valide

- Prima del load: verificare device **RX6800/Vulkan**, librerie con `ldd` e process maps quando necessario, modello/hash, build flags e valori runtime effettivi. Se Vulkan non e' disponibile, fermare il benchmark: nessun fallback CPU silenzioso.
- Stesso input di token e stessi valori runtime in B1/B2. Per cache/dispatch/transfer che devono preservare output, partire da raw logits e operatori cross-build/scalar indipendenti. Suite che condivide il codice di reference non basta da sola.
- Controllare **tutti** i valori di output finiti e non nulli prima di accettare fingerprint; includere reused plans, changed inputs, ID ripetuti, view strided, quant, cache full/empty, eviction in-flight, resize, cancel e errore allocazione. Validation layers/sanitizer fuori dalle misure di velocita'.
- Per kernel/precisione che cambiano floating point, dichiarare prima tolleranze assolute/relative e KL/top1/PPL teacher-forced; confrontare riferimento upstream allo stesso batch. Non fissare tolleranze dopo aver visto un fallimento. Semantic quality separata in T5.
- Seed/sampler fissati nelle generazioni, feed-back autonomo e KV/recurrent coerenti. Teacher-forced e generazione autonoma rispondono a domande diverse; usarli entrambi dove cambia calcolo/stato.
- Se difetto anche B0/B1: ERRORE_BASELINE con riproduttore e limiti. Se compare solo nel candidato a condizioni equivalenti: indagare/revertire quella feature. Non attribuire causa da una issue su altra GPU o da un probe che altera sync.
- Tenere fermi altri benchmark/giochi GPU. La riserva6GiB e lo scope24G/2G sono protezioni iniziali locali, non garanzie universali: monitorare MemAvailable, swap, cgroup e memoria driver; se insufficienti fermare e segnare NON_ESEGUIBILE a quel budget. Ogni benchmark ha il proprio gruppo, separato dal browser/app.

### T1 - Prestazioni PP/TG controllate

Workload minimi: replay244+128 e PP512/TG128; conferma su prompt applicativi3294/4096 e TG128/512 se il contesto lo consente. Screen PP512/TG32 serve solo a eliminare candidati palesemente peggiori. Prompt esatto e token file identificati da hash. Tutto a una sequenza inizialmente.

Congelare source, binari, librerie e metadata prima del primo processo. Ordine bilanciato OFF/ON/ON/OFF, almeno4 processi freschi per variante per decisioni finali;3 o piu' ripetizioni nello stesso processo, senza contarle come campioni di processi indipendenti. Se TG resta instabile come R16, estendere secondo un criterio preregistrato o restituire INCONCLUDENTE; non scartare i regimi scomodi. Warmup identico e dichiarato, cold-start misurato a parte.

Aggregare token/tempo, riportare processi singoli, mediana/dispersione e intervallo bootstrap a livello di processo. Prima dell'A/B fissare margine pratico, inizialmente3% per PP/TG e5% per p95, da rivedere **prima** di nuove campagne se rumore/budget lo richiedono. MIGLIORAMENTO/REGRESSIONE richiede effetto pratico e intervallo coerente; NESSUN_CAMBIAMENTO richiede intervallo dentro +/-margine. Altrimenti INCONCLUDENTE. Queste sono soglie del prossimo protocollo, non certificazioni retroattive di R01-R16.

Ripetere controllo identico a inizio/fine coordinate; >20% drift o endpoint diverso nella stessa configurazione invalida il confronto e richiede rescreen. Il20% e' allarme grossolano del runner, non autorizzazione a ignorare drift minore: margini del verdetto restano3/5%. Registrare temperature, clock/power effettivi, GPU busy, RAM/VRAM/GTT e processi concorrenti. Hashing/dump/profiling fuori dal timer dichiarato.

### T2 - Routing, policy e working set

Corpus separato in training/profilo e test held-out: italiano/chat, code-review/code-edit, JSON/tool, ragionamento e cambio dominio; almeno3 prompt per famiglia e3 seed/continuazioni per una prima validazione, ampliando prima della promozione generale. Distinguere trace teacher-forced (uguale percorso token) da generazioni autonome (percorso reale).

Misure per layer e fase: frequenze, working set, reuse distance, entropia, hit count e hit byte-weighted, costo priming, eviction, miss consecutivi, CPU fallback, H2D/D2H e waste prefetch. Ripetere cold, warm e turni consecutivi, a capacita' uniforme e budget globale uguali. Report con finestre1/8/32/128 token e quote16/32/48/64/96/128 solo se il budget fisico lo consente. La simulazione non include automaticamente trasferimenti, compute o sync: seleziona policy, non certifica TG.

### T3 - Cache/transfer e throughput reale

T0+T1, includendo benchmark isolato della copia host->device e kernel CPU/GPU, poi modello completo. Fasi di controllo: cacheOFF, tutti-hit, tutti-miss, cold-start e warm-state; a stesso budget e placement. Sweep miss0/25/50/75/100% sintetico separato dal routing naturale. Dimensioni di transfer corrispondenti a un esperto reale, triplet completo e unioni prefill; non solo copia lineare gigante.

Misurare timestamp submit/transfer/compute/join, bytes realmente trasferiti compreso padding, chiamate/copy regions, banda osservata, CPU cost e percentuale di overlap; anche upload/download/readback che il contatore hit non descrive. Riportare PP/TG/TTFT, MemAvailable/PSS/RSS, requested/resident VRAM/GTT, KV/recurrent/compute/cache e cgroup. Campionamento1Hz puo' perdere picchi: allocazioni del planner e high-watermark completano telemetria.

Promuovere una cache solo se migliora end-to-end o raggiunge un budget memoria utile a costo dichiarato. Hit rate piu' alto con TG peggiore e' una regressione prestazionale, non un successo da presentare senza costo.

### T4 - Profondita' contesto, KV e batch

Distinguere capacita' allocata e token effettivamente occupati. Profondita'0/512/2048/8192/16384/32768/65536/90000 dove modello, RoPE, RAM/VRAM e API lo consentono; la matrice puo' contenere NON_ESEGUIBILE invece di forzare un OOM. Aggiungere PP512/3294/4096 senza superare contesto e TG128/512. Contesti lunghi non si simulano solo con `-c` grande e prefisso vuoto.

Confrontare ub512/1024/2048 e KV F16/Q8 su stesse dimensioni dichiarate, poi migliori profili in T6. Capture init/fit, allocazioni, finite/zero checks e teacher-forced drift per ogni cella. Per modelli ibridi validare clear/restore/cancel dello stato recurrent insieme a KV. Conferma finale fuori dai prompt usati dall'autotuner.

### T5 - Qualita' e fedelta' del modello

R12 non passa e R13 e' incompleto: prima di una promozione generale serve un confronto corretto completo, **solo dopo ripresa autorizzata**. Riprendere corpus64 e protocollo applicativo n_batch/output finale, ma registrare nuova campagna, non sovrascrivere lo storico. Aggiungere piu' seed e task reali, held-out da profiling/tuning; stabilire margine di non inferiorita' e numerosita' prima della prova, ampliando campione se intervallo e' troppo largo.

Riportare primary e secondary grader, errori semantici per categoria, validita' JSON/tool, truncation e NaN separati. Non far compensare a un errore di formato corretto nel contenuto una regressione di codice/aritmetica. Valutazioni affiancate e review dei discordanti; riusare scorer verificato. Teacher-forced top1, KL e PPL completano il confronto ma non sostituiscono risposte corrette. [Metodo esterno Strata cache-parity](https://github.com/Niko1221/Strata/blob/82f46a8c8f475f001ad76d92f58f4a4f8ffb0253/bench/results/2026-09-27-cache-parity/README.md) come riferimento di metodologia, non gate gia' passato qui.

### T6 - Concorrenza1/2/4/8

Un solo processo server e richieste reali simultanee; registrare slots, batch effettivo e thread budget. **Non** interpretare un batch di token di una sequenza come8 agenti. Stesse trace/request arrivals per A/B, richieste corte/lunghe miste, cache fredda/calda e contesto occupato realistico. Non moltiplicare quattro processi che caricano ciascuno21GB per ottenere una misura di server a quattro slots.

Misure: tokens/s aggregati e per richiesta, TTFT, inter-token p50/p95, completion p50/p95, fairness, richieste fallite, peak RAM/VRAM e working set condiviso. Comparare cache comune vs quote minime per stream e chunk prefill. Controllo single-stream deve restare riportato; un miglioramento aggregato con forte perdita per stream e' MISTO. Budget32GB/16GB vincola le celle eseguibili.

### T7 - Agenti e latenza applicativa

Workload riproducibile: chat italiana multi-turn, code-review con prompt3-4K e risposta512token, code-edit con testo ripetuto, agente tool/JSON e1/2/4/8 task con prefix condiviso/non condiviso. Payload, template, seed, max output e attese degli strumenti fissati; distinguere tempi del modello da I/O tool e latenza intera.

Misurare cold process (load incluso), warm process/nuova conversazione e turni con state/prefix riusabile. Task completati correttamente/minuto, TTFT e latenza fino all'ultima risposta utile sono obiettivi principali; lunghezza del testo generato e raw TG vanno riportati per evitare falsi guadagni da risposte piu' corte o sbagliate. Confronto con llama-bench rimane separato.

## Milestone e ordine concreto di esecuzione

| Milestone | Stato | Lavoro / dipendenze | Definizione di completamento |
| --- | --- | --- | --- |
| M0 - Snapshot e basi | PARZIALE | R01-R16, B0/B1/B2, freeze libs, pausa registrata | Questa roadmap e fonti pubblicate; manifest piccoli delle prossime campagne sempre versionati |
| M1 - Misura senza confondenti | TEST_PARZIALI | R20 S03, R22 replica breve controllata; R23 fallisce ripetibilita' lunga anche B1; R26 screen B4 ripetibile ma diverso daB1; R15/R16 e difetti condivisi catalogati | T0/T1 su controlli puliti, break-down costi e budget; nessun vincitore scelto da TG instabile |
| M2 - Profili e disegno minimo | TEST_PARZIALI | R21/R28 corpus normale, R27/R28 planner quote layer e riserva minima; R29 fixture e R30 integrazione modello breve, costo miss/upload e attese misurati; R33 placement mirato e residente confrontati, screen numericoCPU/GPU FAIL; R34 warm/cold/clear e singole proiezioni ripetibili; R35 precisione attivazioni isolata su fixtureQ4_K; R36 replay reali/observerOFF-ON esatti; R37 precisione sul modello fallisce gate, roundtripCPU non neutro; primo intermedio divergente ancora aperto; T2esteso, pressione KV e fedelta' placement ancora aperti | Corpus/held-out, quote per layer e schema remap/lifetime; costo miss CPU/upload misurato |
| M3 - Cache GPU esatta | TEST_PARZIALI | R30 pool integrato per un layer/stream1, gate breve PASS e A/B registrato; R33 target attivo/32slot esatto a stessoGPU ma TG vs transfer INCONCLUDENTE e peggiore del residente; fedelta'CPU/T2esteso/M1-M2 aperti | T0 all-hit/all-miss/eviction, T2/T3 e verdict; feature separabile OFF di default |
| M4 - Ibrido e transfer | TEST_PARZIALI prerequisiti | R22 replica host positivo PP/costo memoria, R26 controllo dentroB4; S07/S08 e staging ring S10/S11 ancora proposti su branch distinti | Confronto contro M3, non solo vecchio default; timeline dimostra o smentisce overlap |
| M5 - Policy e memoria | PROPOSTO | S13/S14/S19/S21 e CPU se profilata dominante | Budget reale32GB, held-out, headroom e TG/TTFT migliori o tradeoff esplicito |
| M6 - Applicazione e agenti | PROPOSTO | Ripresa autorizzata; T5 completo, S28/S29, T4/T6/T7 ancora da eseguire | Qualita' dichiarata e matrice1/2/4/8, cold/warm, latenza e memoria; limiti pubblicati |
| M7 - Kernel/speculazione/modelli | PROPOSTO | S24-S27/S30-S42, solo bottleneck dimostrati | Una variante alla volta; benchmark e qualita' propri, nessuna fusione di speedup non confrontabili |
| M8 - Promozione profilo o integrazione | PROPOSTO | Milestone rilevanti superate | B3 aggiornato con SHA/hash/config, report replicabile e rollback; integrazione quando richiesta dal proprietario |

Checklist del lavoro dopo la ripresa autorizzata:

- [x] Baseline v0.5.0 e SHA del fork identificati.
- [x] Separati risultati locali, video CUDA e Strata HIP.
- [x] Documentati miglioramenti di workspace/PP, regressioni e risultati inconcludenti.
- [x] Pausa precedente conservata; ripresa benchmark autorizzata il 2026-10-07.
- [x] Implementato primo tooling offline S01/S02 e verifica senza modello; nessuna ripresa dei benchmark implicita.
- [x] Implementati scope host/contatori copie S03 opt-in; R20 completa il gate sul replay, timestamp diagnostici e memoria breve. Timeline normale e budget esteso restano da completare.
- [x] Implementati intervalli espliciti replay PP/TG; R20 valida correlazione sul modello e overhead, warmup distinto e output identico.
- [x] Controllo originale/fork a valori effettivi identici e dati hardware aggiornati in R20; logits completi finiti/nonzero e identici.
- [ ] Spiegare o contenere variabilita' fra processi prima di un vincitore CPU/placement.
- [x] Misurati costi host CPU/transfer, timestamp GPU diagnostici e memoria VRAM/KV/recurrent/compute sul replay244+128 in R20; contesti lunghi restano aperti.
- [x] R21: pilot routing train/held-out normale, logits identici B1, quote uniformi a byte uguali e richieste complete; callback storico con difetto condiviso archiviato.
- [x] R27: confrontate quote uniformi/globali/bundle a cap uguale, input/hash e split TRAIN/HELDOUT,18test offline; evidenziato rischio di concentrazione.
- [x] R28: sei famiglie/12prompt nuovi,36processi con parity completa; riserva4/8 e piani TRAIN persistiti prima di parsingHELDOUT;19/21allocazioni fattibili.
- [ ] Completare T2: almeno3prompt/famiglia e3continuazioni, turni/contesti estesi; estendere remap/lifetime/admission oltre il singolo layer e misurare costo miss CPU.
- [x] R29: prototipo S04/S05 con bufferGPU reali, remap/lease/LRU/fallback,48casi e controlli di compatibilita'; nessun gate modello o beneficio PP/TG dichiarato.
- [x] R30: S04/S05 nel modello, un layer/stream1 OFF di default; gate raw normale/min-batch1, costo miss/attese/memoria e8processi A/B. PP/TG/p95 equivalenti nelle fasce +/-3%/+/-5%; perimetro breve, M3parziale.
- [x] R31:512PP+200TG e10000PP+200TG, PP aggregato fra chunk; coldB1/warmB1/forkOFF raw identici,8processi timing puliti e riferimenti separati. R32 trova B1 lungo non ripetibile; poolON lungo e corpus T2 restano aperti.
- [x] R32: incremento S14 admission lazy prima di allocazione, budget combinato pesi/remap, gate breve normale/attivo e memoria. A/B breve INCONCLUDENTE; lungo ERRORE_BASELINE riprodotto anche B1 e conservato.
- [x] R33: placementGPU mirato layer17 senza soglia globale1, controllo residente esistente e gateGPU esatto;33fixture placement,12timingGPU/4profiler e25processi modello. TG pool vs transfer INCONCLUDENTE, vs residente REGRESSIONE; CPU/GPU numerical screen FAIL e lungoB1 non ripetibile.
- [x] R34: controllo cold nel helper,19processi brevi e18CLI; warmup/clear non cambiano rawCPU/full, sei coppie per-placement esatte; gate/up/down/gate+up screen FAIL e fino a2argmax diversi, causa kernel ancora aperta.
- [x] R35: operatoriQ4_K a forme reali con dati sintetici, riferimentiF64/F32/Q8_K e pipeline effettivi;18controlli/2sanitizer, bit-repeat eB1/fork/debug esatti. Precisione attivazioni domina nel fixture, residui~1e-7; logits reali aperti.
- [x] R36: osservatoreCPU layer17 OFF di default,5processi512PP+200TG raw esatti,600record/capture ripetibili; replay di pesi/ID/attivazioni reali1/100/200,20controlli numerici/2synthetic/6negative/11legacy PASS. Precisione ingressi domina; fix logits end-to-end ancora aperto.
- [x] R37: precisione mirata sul modello layer17,37processi/7437vettori,OFF/copy/repeat/debug/profile/sanitizer esatti; tutti screen end-to-end FAIL; roundtripCPU non neutro. Diagnostica opt-in, scartato come fix.
- [ ] S09/T0: dopo ReBAR, ripetere controlli congelati e localizzare il primo intermedio divergente con raw parity osservatore; catturare datiCPU quantizzati realmente usati; cattura/replay reale e gate osservatoreOFF/ON completatiR36; spiegare divergenza decodeCPU/GPU e qualificare fedelta'/qualita' prima di promuovere placement o merge S07/S08; sweep capacityS04 solo a stessoGPU validato. Partizione decode protetta, warm-start e corpus restano aperti.
- [x] R22: A/B mmap/none breve a build/config identici, raw gate,4processi/variante, PP/TG/p95 e memoria/startup separati.
- [x] R23: provato prompt3107+128; fallita ripetibilita' anche B1, sei dump finiti/nonzero e controlli conservati, nessun ranking lungo.
- [x] R24: screen B1 con prefix1024/1536+32, due processi ciascuno identici; non riproduce il3107, causa aperta. Token/prompt R22-R24 versionati.
- [x] R25: controllo reset dati nel solo helper; default/0/vuoto compatibili sul breve, true-clear non risolve B1 lungo. Fonte/helper e engine originali separati, gate fallito conservato.
- [x] R26: B4 separato, coppie brevi/lunghe bit-ripetibili;29/129vettori brevi diversi daB1. A/B mmap/none dentroB4: PP+37.32%, TG/p95 equivalenti, memoria/avvio registrati.
- [ ] Isolare history/shape e cambi upstream, estendere correttezza/qualita'; confermare PP/TG/TTFT e1/2/4/8 nei limiti autorizzati.
- [ ] Aggiornare B3 e registro con il verdetto, anche se REGRESSIONE o NESSUN_CAMBIAMENTO.

## Scheda da compilare per ogni nuovo tentativo

Usare un report piccolo `docs/development/moe-<id>-rx6800.md` e collegarlo nel registro. I comandi esatti devono essere copiabili dal report; una descrizione "stessi parametri" non basta. I nomi seguenti sono campi da riempire, non risultati gia' esistenti.

```yaml
id: S04-attempt-01
date_utc: YYYY-MM-DDTHH:MM:SSZ
status: PROPOSTO | IMPLEMENTATO | TEST_PARZIALI | VALIDATO | INCONCLUDENTE | SCARTATO | PAUSA
hypothesis: bottleneck, vantaggio previsto su RDNA2, costo previsto
baseline: B1/B2/B3 e SHA esatto, parametri e valori effettivi
candidate: SHA esatto, patch dirty hash se presente, un fattore cambiato
model: file, byte, sha256, tokenizer/template identity
build: compiler, cmake, ninja, flags, CMakeCache e binary/library hashes
system: kernel, mesa_radv, firmware, CPU/RAM/GPU e profilo energia
environment: tutte le variabili rilevanti e LD_LIBRARY_PATH
workload: hash token/prompt, fase, depth/context, batch/ubatch, KV/recurrent, seed/sampler, concurrency
commands: comando baseline e candidato completi, cwd e librerie
protocol: T0-T7 applicati, soglie preregistrate, ordine processi e warmup
correctness: raw-logits/fingerprint, finite/zero, top1/KL/PPL se applicabili
quality: primary/secondary, discordanti e numeric failures; NON_MISURATO se assente
metrics: PP, TG, TTFT, p50/p95, aggregate/per_stream e campioni di processi
cache: budget_bytes, slot_count, hit_count/byte_weighted, eviction, miss_cost, H2D/D2H/waste
memory: RAM_PSS/RSS/MemAvailable, swap/cgroup, VRAM/GTT_requested_resident, KV/recurrent/compute
controls: init/end, drift, attivita' GPU concorrente, fallback e tentativi invalidi
verdict: per metrica MIGLIORAMENTO/REGRESSIONE/NESSUN_CAMBIAMENTO/NON_MISURATO/INCONCLUDENTE
decision: mantenere_opt_in | promuovere | ripetere | revertire, e motivo
artifacts: report in repo, archivio grezzo, checksums, patch e checkpoint
```

Per il ranking successivo considerare **tempo di task corretto risparmiato per GiB e per complessita' introdotta**. Non sommare +PP di un modello, +TG di un altro backend e hit rate di una simulazione in un singolo "speedup". Mantenere nel registro anche un risultato nullo: evita di ripetere lavoro senza una nuova ipotesi verificabile.
