# R35: quantizzazione delle attivazioni negli operatori MoE

Data: 2026-10-07. S09/T0, branch experiment/moe-operator-numerics sopra R34, draft PR16 non integrato. Fixture finale locale6113dd7d53cfa7e6136967617d0efe8f6f0f00c2, remoto d059adb39d8f9d21cc6d29fd45267dd4a805667f, tree comune313578af903bb71639907910627e0c7d9c3a04c8. Le librerie normali sono immutate: B1 originale7fe450e19305b828c199d602c23a8337aaa1f03b, fork R33 locale61da28a2efc1c1a4ece4d68cd99bea5d3ad08314 / remoto d02dfb6d06ed8808418737a5b3051917d8f8537d, tree b09ab8bc3a41a6d135390dbe8b872345c40797e1.

## Domanda e modifica

R34 trova differenze al primo TG quando ciascuna proiezione del layer17 originale viene spostata CPU->GPU, anche senza warmup e con stato cancellato. R35 separa numericamente la conversione delle attivazioni e il calcolo dell'operatore. Estende scheduler-check/expert-pool-check, senza nuovi file tests/*, kernel, scheduler o modifiche dell'inferenza.

La nuova modalita' --operator-numerics esegue un solo MUL_MAT_ID per grafo, con pesi Q4_K sintetici, input/ID deterministici e forme reali gate/up/down. Gate/up: pesi[2048,512,256], input[2048,1,1]. Down: pesi[512,2048,256], input[512,8,1]. Top8,256 esperti, token1, tre pass con ingressi diversi e permutazione degli ID. Solo gli esperti0/1/7/31/63/127/191/255 sono popolati; gli altri sono blocchi zero mai selezionati. Tutti i pesi sono sintetici, non quelli di Ornith; non sono attivazioni catturate dal modello, routing naturale o un workload agenti.

Per ogni risultato si costruiscono due riferimenti scalari con pesi Q4_K dequantizzati e accumulo double: il primo usa gli input F32 originali; il secondo usa gli stessi input dopo la conversione CPU Q8_K e dequantizzazione. La conversione usa il trait CPU from_float attivo, la dequantizzazione la funzione esistente dequantize_row_q8_K. Il controllo MOE_OPERATOR_INPUT_Q8_K=1 fornisce proprio questi valori alla GPU come F32. E' un intervento sulla precisione nel solo fixture, non una nuova opzione di inferenza. Assente/vuoto/0 mantiene l'input originale; altri valori sono respinti. La modalita' usa8 thread CPU; i modi precedenti mantengono2.

MOE_OPERATOR_DATA_PREFIX salva pesi packed, ID, input originali e input effettivi. MOE_OPERATOR_ORACLE_PREFIX salva riferimenti float64, mentre --output-bin contiene risultati float32. Ordine: gate/up/down, poi pass0/1/2, slot0..7 e righe; vettori512/512/2048. Ogni vettore attuale e di riferimento e' finito e contiene almeno un valore nonzero prima della SHA. Il CSV distingue per pass max_abs/RMS rispetto a entrambi i riferimenti. Output e oracoli vengono validati anche esternamente; i numeri CSV coincidono con i confronti NumPy.

## Protocollo e provenance

18 processi numerici registrati prima dei raw: due processi per B1CPU, forkCPU, B1GPUauto, forkGPUauto, B1GPUF32, B1GPU-MMVQ forzato e B1GPUF32 con input Q8_K; poi quattro catture DEBUG corrispondenti ai modi GPU. Prima serie nell'ordine indicato, seconda in ordine inverso, poi auto/F32/MMVQ/Q8_K DEBUG. Non c'e' warmup del modello o generazione di token.

```text
check --operator-numerics --output-bin ACTUAL.bin [--vulkan]
```

Ambiente pulito e LD_LIBRARY_PATH della variante congelata. Auto: nessun flag MMVQ. F32: GGML_VK_DISABLE_MMVQ=1. MMVQ: GGML_VK_FORCE_MMVQ=1. Q8_K->F32: GGML_VK_DISABLE_MMVQ=1 e MOE_OPERATOR_INPUT_Q8_K=1. Un flag MMVQ globale isola l'operatore qui perche' il grafo contiene solo quello; sul modello completo cambierebbe altri operatori e non isolerebbe il layer17.

Helper compilato separatamente con header B1/fork, -std=c++17 -O3 -DNDEBUG e librerie ggml/base/cpu/vulkan copiate dal freeze R34, SHA invariata. Snapshot helper e header privati, comandi, cache e tutte le librerie caricate sono registrati. DEBUG riusa helper e librerie ggml/base/cpu del fork, sostituendo solo Vulkan con una build separata dello stesso codice e GGML_VULKAN_DEBUG=ON; logger gia' esistente. L'evidenza richiede dispatch effettivi, non pipeline soltanto create, e risultati bit-identici al controllo normale prima di usare il log.

RX6800/5700X3D/32GB, kernel7.2.9-1-cachyos, Mesa/RADV26.2.4-arch3.1, GCC16.2.1 20260810, CMake4.4.4, Ninja1.13.2, Release/native/Vulkan/OpenMP per i motori normali. Profilo utente2600/1075MHz,-100mV,186W invariato. Scope8G/swap0, riserva RAM6GiB e guard VRAM/device, timeout180s. Argomenti, UTC, mappe /proc, SHA driver/librerie, clock/cap e telemetria dei18 processi conservati. Nessun ranking dei tempi: generazione dati, dump, oracolo scalare e logging rendono questi controlli diversi da un benchmark operatori o PP/TG.

Gate registrato: ogni vettore finito/nonzero; tutti i pesi/input originali/ID e i due oracoli identici fra configurazioni; sette repeat esatti; B1/fork CPU e GPUauto esatti; quattro DEBUG esatti. Limiti per ogni pass: max_abs<=0.001 e relative_RMS<=0.0001 per CPU vs riferimentoQ8_K, GPUF32 vs riferimentoF32 e GPUF32 con inputQ8_K vs riferimentoQ8_K. Relative RMS e' RMS errore / RMS riferimento, non la metrica sui logits R33/R34. I limiti del modello restano invariati e non sono sostituiti da quelli dell'operatore.

## Risultati

18/18 controlli numerici PASS:1296 vettori attuali e2592 vettori oracolo, tutti finiti/nonzero. Le sette coppie sono bit-identiche; B1/fork CPU e GPUauto sono esatti. Pesi packed, ID e input originali coincidono in tutti18 processi, cosi' come entrambi gli oracoli. Il controllo Q8_K cambia soltanto gli input effettivi; shape, ID, pesi e riferimenti rimangono gli stessi.

Dispatch DEBUG, con raw esatto rispetto alle quattro configurazioni normali:

| Modo | Gate/up | Down |
| --- | --- | --- |
| Auto | quantize_q8_1_x4 + mul_mat_vec_id_q4_k_q8_1_f32 | mul_mat_vec_id_q4_k_f32 |
| MMVQ disattivato | mul_mat_vec_id_q4_k_f32 | mul_mat_vec_id_q4_k_f32 |
| MMVQ forzato | quantize_q8_1_x4 + mul_mat_vec_id_q4_k_q8_1_f32 | quantize_q8_1_x4 + mul_mat_vec_id_q4_k_q8_1_f32 |
| Input CPUQ8_K, MMVQ disattivato | mul_mat_vec_id_q4_k_f32 | mul_mat_vec_id_q4_k_f32 |

La previsione condizionata R34 e' confermata per questi grafi. La pipeline del modello completo non viene catturata in R35. Nessuna SiLU, moltiplicazione gate/up, somma esperti o fusion puo' spiegare lo scostamento iniziale del singolo operatore, perche' qui sono assenti.

I tre gate scalari PASS: CPU vs Q8_K max_abs8.858e-7/max relativeRMS2.220e-7; GPUF32 vs F32 max_abs4.821e-7/max relativeRMS1.207e-7; GPUF32 con inputQ8_K vs Q8_K max_abs7.604e-7/max relativeRMS1.880e-7. Residui piccoli di dequantizzazione/riduzione e arrotondamento rimangono; non e' equivalenza bitwise CPU/GPU.

RMS GPU vs CPU su tutte le righe dei tre pass per proiezione:

| Proiezione | GPUauto, input originale | GPUF32, input convertito CPUQ8_K | Riduzione RMS |
| --- | ---: | ---: | ---: |
| gate | 0.004785429 | 2.45477747e-07 | 19494x |
| up | 0.004771516 | 2.26436223e-07 | 21072x |
| down | 0.001842170 | 1.13232133e-07 | 16269x |

Il controllo a pipeline identico GPUF32, cambiando solo l'input originale in inputQ8_K, riduce RMS di circa14933/16278/16269 volte per gate/up/down. La tabella precedente confronta invece auto con input originali contro F32 con inputQ8_K e include anche il cambio di pipeline per gate/up.

Disattivare MMVQ da solo non allinea la CPU: gate/up RMS0.003666/0.003686, down0.001842 invariato. Forzare MMVQ lascia gate/up invariati e cambia down aRMS0.002347. Invece allineare la precisione delle attivazioni abbassa tutti gli errori di circa16-21mila volte. Nei grafi sintetici, la diversa rappresentazione delle attivazioni spiega la parte dominante dello scostamento rispetto ai piccoli residui di calcolo; non si puo' attribuire a un errore esclusivo del pool o risolvere il merge scegliendo solo il flag MMVQ.

## Controlli host e catture fallite

11 test precedenti parser/scheduler/profiler PASS sul helper finale. Due ulteriori processi --operator-numerics CPU/GPU con ASAN/UBSAN/leak PASS;144 vettori attuali e288 oracoli validi, quattro librerie ASAN congelate verificate sia nelle mappe sia negli init del loader. Sono controlli del codice host, non sanitizer del codice shader. I loro raw coincidono anche con i controlli Release CPU/GPUauto, ma non sono inclusi nei18 confronti numerici registrati.

Una prima esecuzione CPU del helper70dc363, precedente a tutti i dati accettati, termina con SEGV per il trait to_float nullo di Q8_K. E' HARNESS_ERROR, non ERRORE_BASELINE. Cattura completa, librerie e source congelati nel sibling operator-numerics-attempt1; nessun vettore accettato. Il riferimento e' corretto usando il dequantizzatore esistente prima di ripetere l'intero protocollo, senza rilassare i criteri. Inoltre il primo verificatore post-run ASAN confrontava il nome .so.0 del loader con il percorso canonico delle mappe: controllo corretto risolvendo i symlink e riusando la cattura CPU riuscita, senza un ulteriore processo. Entrambi i tentativi sono conservati.

## Limiti e prossimo incremento

R35 isola una causa dominante della differenza negli operatori sintetici: CPUQ8_K, VulkanQ8_1 per gate/up e VulkanF32 per down trattano diversamente le stesse attivazioni. Non e' ancora una dimostrazione quantitativa della causa dei logits R34: servono pesi e attivazioni reali, effetti della SiLU/merge/routing e propagazione fra layer. L'intervento CPUQ8_K->F32 aggiungerebbe conversione e possibili readback/trasferimenti sul modello; nessun costo o speedup e' stato misurato, nessuna patch di inferenza e' implementata.

Prossimo S09/T0: cattura opt-in delle attivazioni reali CPU layer17 senza callback del modello, con gate raw completo osservatoreOFF/ON; riprodurre singoli operatori con stessi pesi/ID/input e verificare quantizzazione, fusion e riduzione. Soltanto dopo un controllo di precisione mirato potra' essere provato sul modello, con screen R33/R34 invariato e gateGPU/pool separato, poi A/B PP/TG se valido. S07/S08 ibridi rimangono aperti e non sono lossless per definizione.

Nessun modello caricato in R35.20 processi nuovi della modalita' operator-numerics validi (18 numerici +2sanitizer), piu' un tentativo helper fallito. Totale cumulativo dei processi modello R20-R35 invariato a231. Nessuna nuova misura512/10000PP+200TG, sampled-quality, TTFT, concurrency o soluzione della ripetibilita' lunga B1; R32/R33 lungo rimane ERRORE_BASELINE.

## Artefatti e fonti

Archivio locale risultati/2026-10-07-operator-numerics/: protocollo e addendum sanitizer, sorgenti/header helper, comandi/build/cache, varianti frozenBASE/NEW/DEBUG/ASAN,20 dump attuali e40 oracoli,18 dataset packed/input/ID, CSV/log/debug dispatch, manifest/mappe/driver/telemetria, confronto e tentativi conservati. [Manifest versionato](moe-operator-numerics-validation.json) contiene protocollo, misure, SHA e librerie deduplicate; dataset/dump grandi restano locali.

Upstream ricontrollato al commit18b5f8b1862ebfe0f1c33d2355b81a54d2fec867: [selezione Vulkan](https://github.com/ggml-org/llama.cpp/blob/18b5f8b1862ebfe0f1c33d2355b81a54d2fec867/ggml/src/ggml-vulkan/ggml-vulkan.cpp), [quantize Q8_1](https://github.com/ggml-org/llama.cpp/blob/18b5f8b1862ebfe0f1c33d2355b81a54d2fec867/ggml/src/ggml-vulkan/vulkan-shaders/quantize_q8_1.comp), [Q8_K](https://github.com/ggml-org/llama.cpp/blob/18b5f8b1862ebfe0f1c33d2355b81a54d2fec867/ggml/src/ggml-quants.c). Source snapshot/SHA salvati; nessun cherry-pick upstream applicato ai motori. Le modifiche del proprietario e gli archivi precedenti rimangono intatti.
