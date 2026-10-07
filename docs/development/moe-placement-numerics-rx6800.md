# R34: decode CPU/GPU, warmup e singole proiezioni

Data: 2026-10-07. Incremento S09/T0 su experiment/moe-placement-numerics, sopra R33; draft PR15, non integrato. Helper locale aa90393e802271f7ee6804e85400dc371c318ba7, remoto 618037997cb6be88fb2fc4c60b9ad91be333284b, tree comune 6406f79cc3315acb622993493e7559bc931c237d. Le librerie di inferenza sono quelle congelate R33: B1 originale 7fe450e19305b828c199d602c23a8337aaa1f03b e fork locale 61da28a2efc1c1a4ece4d68cd99bea5d3ad08314 / remoto d02dfb6d06ed8808418737a5b3051917d8f8537d, tree b09ab8bc3a41a6d135390dbe8b872345c40797e1.

## Modifica e obiettivo

R33 trova una divergenza CPU/GPU nel decode anche con l'override residente della baseline originale. R34 separa storia del warmup e placement di gate/up/down prima di modificare kernel, quantizzazione o merge ibrido. Nessuna nuova ottimizzazione prestazionale e nessuna tolleranza rilassata.

Il solo helper moe-replay aggiunge MOE_REPLAY_WARMUP: assente, vuoto o 1 conserva il warmup precedente; 0 salta l'intero evaluate(rep=-1). Altri valori falliscono prima del caricamento del modello. L'opzione e il valore clear_data sono registrati nel log. Il warmup interno common/llama rimane disattivato come prima, callback OFF, ingresso teacher-forced e nessun sampling o feedback dai logits. Lo schema CSV e i raw delle ripetizioni non cambiano. MOE_REPLAY_CLEAR_DATA=1 rimane un controllo separato: llama_memory_clear prima di ogni replay, fuori dagli intervalli misurati.

Cold significa solo assenza del warmup del helper in un nuovo processo. Non significa cache del filesystem o shader cache RADV fredde. Le librerie engine non sono ricompilate: le sei librerie BASE/NEW sono copiate e verificate byte per byte rispetto al freeze R33. Il medesimo nuovo sorgente helper e' compilato due volte con GCC -std=c++17 -O3 -DNDEBUG, usando rispettivamente header B1 e fork e rpath $ORIGIN. Build commands, header roots, sorgente helper, phase-profile.h, CMakeCache e SHA sono conservati.

## Protocollo registrato e parametri

19 processi modello, uno alla volta, una ripetizione registrata per processo, 512 PP + 200 TG, c1024, step512. Cinque controlli warm: B1 CPU, fork CPU, B1 layer17 residente, fork placement mirato senza pool, fork placement mirato con pool17:32:128. Poi due processi cold per CPU/full/gate/up/down/gate+up; infine cold true-clear CPU e full. Ordine intercalato preregistrato in protocol.json prima dei raw. Il modello e i token sono gli stessi di R33, nessuna continuazione autonoma.

```text
-m /home/casa/Programmi/modelli/Ornith-1.5-35B-Q4_K_M.gguf
-ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512
-fa on -ctk q8_0 -ctv q8_0 --fit off --load-mode mmap --verbosity 4
```

Le varianti di placement per proiezione usano soltanto l'engine originale B1 e -ot PRIMA di -ncmoe, dato che vince il primo override corrispondente. Esempio full:

```text
-ot '^blk[.]17[.]ffn_(gate|up|down)_exps[.]weight$=Vulkan0'
```

Gate/up/down sostituiscono il gruppo con il solo nome; gate+up usa (gate|up). Il valore globale di offload resta 32. Il fork con target usa GGML_SCHED_EXPERT_GPU_LAYER=17 e il pool aggiunge GGML_SCHED_EXPERT_POOL=17:32:128. Il runner rimuove le altre variabili GGML/RADV/VK dall'ambiente ereditato. CPU compact/active OFF. Manifest per processo con argv, ambiente, configurazione effettiva, SHA di ogni libreria/driver caricati, mappe /proc, clock, cap e telemetria a 2 Hz.

Modello 21,713,462,848 byte, SHA256 ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f ricontrollata integralmente. RX6800/5700X3D/32GB, kernel7.2.9-1-cachyos, Mesa/RADV26.2.4-arch3.1, GCC16.2.1 20260810, CMake4.4.4, Ninja1.13.2. Librerie Release/native/Vulkan/OpenMP. Profilo utente invariato: core2600/mem1075 MHz, offset-100 mV, cap186 W. Scope memory24G/swap2G, riserva RAM6GiB, guard device/VRAM, timeout600s. Nessun arresto del guard.

Screen invariato R33: argmax agreement >=99%, max_abs <=0.5, massimo RMS per-vettore <=0.05, massimo symmetric KL per-vettore <=0.01. Ripetizioni della stessa configurazione devono essere bit-identiche. Finitezza e almeno un valore nonzero sono verificati per ogni vettore prima della SHA. Nessun ranking dai tempi: i dump raw e il cold replay non sono una campagna PP/TG o TTFT applicativo.

## Risultati

19/19 processi completati, 3819 vettori completi da248320 logits, tutti finiti/nonzero. 18 controlli CLI sui due helper PASS: assente/vuoto/0/1 accettati fino al modello deliberatamente mancante; 2/01/true/spazio0/-1 respinti prima del load. Nessun modello caricato dai controlli CLI.

I cinque controlli warm riproducono esattamente R33: CPU B1/fork SHA70a46c02e881eccb8f109aeda82cbde51990391b077edd3515c206814f7c0424; full B1/targetOFF/poolON SHA311b49e5aac3ca8604e8e1fe4c8a36127cba23d9bcb1dc4b5abb86574dd4cf95. Tutte le sei coppie cold sono bit-ripetibili, incluse le quattro proiezioni parziali. Warm vs cold e false-clear vs true-clear sono esatti sia per CPU sia per full GPU. Su questo workload, warmup e clear_data non sono condizioni necessarie per la divergenza.

Il solo vettore finale PP registrato e' identico in tutti i placement B1. La prima differenza e' gia' al primo decode, posizione512: RMS full0.067523, gate0.062101, up0.062651, down0.061372, gate+up0.068378, tutti oltre0.05. Questo non certifica l'uguaglianza di tutti i tensori intermedi del prefill. Tutti i200 vettori TG differiscono in ogni confronto CPU/proiezione.

| Proiezioni layer17 residenti GPU vs CPU | max_abs | Max RMS vettore | Max symmetric KL vettore | Argmax diversi | Screen |
| --- | ---: | ---: | ---: | ---: | --- |
| Tutte | 2.231972 | 0.611467 | 0.026661 | 0/201 | FAIL |
| Gate | 2.425610 | 0.727674 | 0.021214 | 0/201 | FAIL |
| Up | 2.489266 | 0.840937 | 0.018569 | 1/201 | FAIL |
| Down | 2.658100 | 0.695840 | 0.054675 | 1/201 | FAIL |
| Gate+up | 2.517686 | 0.473715 | 0.066560 | 2/201 | FAIL |

Lo screen FAIL e' numerico; non misura la qualita' semantica. Up/down cambiano un argmax e gate+up ne cambia due, pur restando dentro il requisito argmax>=99%; gli altri tre limiti falliscono. Full GPU mantiene tutti gli argmax ma fallisce anch'esso. Le metriche non si sommano fra proiezioni: gate+up ha RMS massimo minore del full ma KL massimo maggiore. Non c'e' evidenza per scegliere un unico colpevole o dichiarare il merge CPU/GPU lossless.

## Audit dei percorsi e ipotesi aperta

I tre tensori reali blk.17.ffn_gate/up/down_exps.weight sono tutti Q4_K, 144MiB ciascuno; gate/up hanno forma[2048,512,256], down[512,2048,256]. Non e' corretto attribuire a down una quantizzazione Q6_K solo dal nome del GGUF Q4_K_M.

Nel sorgente B1, MUL_MAT_ID CPU converte le attivazioni F32 nel vec_dot_type Q8_K per Q4_K. Vulkan ha percorsi Q8_1 e F32; la selezione MMVQ dipende da supporto integer dot, contiguita', forma e soglie della GPU. La RX6800 rilevata dichiara integerDotProduct4x8BitPackedSignedAccelerated=true. Con batch decode1, l'euristica AMD usa la soglia k>=2048: gate/up possono essere eleggibili, down con k512 no. Questa e' una previsione condizionata dal codice; R34 non cattura il nome del pipeline realmente eseguito ne' le attivazioni intermedie. Differenze di quantizzazione/riduzione/fusion e placement degli intermedi restano ipotesi da separare con fixture indipendenti, non una causa dimostrata dal solo intervento sul modello.

Sono stati ricontrollati il codice upstream corrente e i sorgenti locali. Snapshot upstream 18b5f8b1862ebfe0f1c33d2355b81a54d2fec867, SHA dei file salvati in source-audit.json: [Vulkan](https://github.com/ggml-org/llama.cpp/blob/18b5f8b1862ebfe0f1c33d2355b81a54d2fec867/ggml/src/ggml-vulkan/ggml-vulkan.cpp), [CPU](https://github.com/ggml-org/llama.cpp/blob/18b5f8b1862ebfe0f1c33d2355b81a54d2fec867/ggml/src/ggml-cpu/ggml-cpu.c). Nessun aggiornamento upstream e' applicato alle librerie della campagna.

## Implicazioni e prossima verifica

S09/T0 avanza: la divergenza e' ripetibile, compare senza warmup e con ciascuna proiezione sul motore originale. Il pool resta esatto rispetto al controllo GPU abbinato del breve; non e' dimostrato equivalente alla CPU. S07/S08 ibridi e promozione placement restano aperti; R33 continua a non dimostrare uno speedup pratico del pool32 rispetto al transfer.

Prossimo incremento: riusare la fixture operatori esistente per Q4_K con forme reali gate/up/down e input/ID identici, confronto CPU Q8_K e Vulkan con pipeline registrato; separare quantizzazione delle attivazioni, fusion e riduzione prima di una patch minima. Verificare prima le operazioni isolate e poi logits sul modello, stessi screen, gate indipendente GPU/pool. Flag globali MMVQ sul modello cambierebbero anche altri operatori: non isolano da soli il layer17.

Il lungo10000PP+200TG rimane ERRORE_BASELINE di R32/R33; non e' rieseguito o dichiarato risolto da questi19 processi brevi. Nessuna certificazione di output sampled, corpus semantico, TTFT completo o concurrency1/2/4/8. Totale cumulativo R20-R34:231 processi modello.

## Artefatti

Archivio locale immutabile dopo la campagna: risultati/2026-10-07-placement-numerics/. Comprende protocollo preregistrato, prepare/run/compare/report.py, due helper e librerie frozen, cache/compilazione, sorgenti upstream acquisiti, modello/input SHA,19 dump raw/CSV/log e manifest, mappe, telemetria/DRM e15 confronti per-vettore. [Manifest versionato](moe-placement-numerics-validation.json) include provenance, registrazione, controlli CLI, tutti gli argv/ambienti e SHA librerie deduplicate, raw e confronti. I dump grandi restano locali. I file modificati dal proprietario e gli archivi precedenti rimangono intatti.
