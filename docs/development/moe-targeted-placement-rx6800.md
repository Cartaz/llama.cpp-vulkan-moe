# R33: placement GPU mirato degli esperti

Data: 2026-10-07. Branch experiment/moe-targeted-expert-placement, sopra R32. Codice locale 61da28a2efc1c1a4ece4d68cd99bea5d3ad08314, remoto d02dfb6d06ed8808418737a5b3051917d8f8537d, tree comune b09ab8bc3a41a6d135390dbe8b872345c40797e1. Baseline B1 originale 7fe450e19305b828c199d602c23a8337aaa1f03b.

## Problema e modifica

Nel decode normale ncmoe18 lascia gli esperti sulla CPU e il pool R32 non viene usato. La soglia globale1 del controllo R30 sposta invece sulla GPU anche tutti gli altri esperti host: non isola il beneficio della cache di un layer. R33 aggiunge GGML_SCHED_EXPERT_GPU_LAYER=17: rende eleggibili per GPU soltanto le operazioni MUL_MAT_ID con pesi host WEIGHTS nominati esattamente blk.17.ffn_gate_exps.weight, blk.17.ffn_up_exps.weight e blk.17.ffn_down_exps.weight. La soglia globale rimane 32.

Lo scheduler continua a richiedere supporto dell'operazione dal backend e considera soltanto device GPU/IGPU per questa eccezione. Assegnazioni esplicite, pesi gia' residenti, operatori diversi, altri layer, esperti condivisi e --no-op-offload conservano il percorso precedente. L'opzione e' letta alla creazione dello scheduler, separata da GGML_SCHED_EXPERT_POOL e OFF se assente/vuota;0 seleziona il layer 0, non disabilita. Accetta cifre da 0 a 4096; configurazioni invalide danno warning e restano OFF. Non aggiunge dispatch/kernel, static hot-expert loading o una nuova API pubblica.

Ipotesi RDNA2: rendere attivo il pool per un solo layer puo' ridurre gli upload dei pesi senza spostare tutti gli esperti CPU. I costi sono compute GPU, trasferimenti di attivazioni/ID, attese, miss sincroni e memoria pool; hit rate alto non certifica throughput migliore. Il controllo resident usa l'override tensor gia' disponibile, con tutti gli esperti del layer residenti in GPU, come alternativa S09 a memoria maggiore.

## Validazione runtime

33 casi placement: 16 casi F32/Q4_K/Q6_K misti, token 1/33, broadcast e ID strided, sei richieste ciascuno; 17 controlli di configurazione, shared/unrelated layer e op_offload=false. 113 richieste e 339 vettori controllati in C++; 288 vettori dei casi di calcolo sono catturati separatamente, tutti finiti/nonzero prima della SHA. Due dump coincidono esattamente, SHA 0fa70171c843b1f2fb26dae79d2a65b43635aaa911835cc378ed4515c4d9b63f, 27,156,480 byte. Coincidono anche con il prefisso dei 16 casi R32.

I 23 casi / 138 richieste / 414 vettori scheduler precedenti restano identici al raw R32, SHA 59c0de5e1aa7b83b05b569b619157b0eefea87d11336a78a0a924e05b9449ac5. Oracle F32 e source ID preservation restano attivi. ASAN/UBSAN/leak placement CPU e GPU PASS; una cattura LD_DEBUG separata verifica SHA e percorsi delle quattro librerie ggml ASAN congelate. 11 test parser/riepilogo/runtime PASS. Nessuna fixture rappresenta da 1 a 8 richieste server.

## Protocollo sul modello

Ornith-1.5-35B Q4_K_M: 21,713,462,848 byte, SHA256 ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f ricontrollata. RX6800/5700X3D/32GB, kernel7.2.9-1-cachyos, Mesa/RADV26.2.4-arch3.1, GCC16.2.1 20260810, CMake4.4.4 e Ninja1.13.2. Release/native/Vulkan/OpenMP, CPU compact/active OFF. Source, binari, CMakeCache, librerie, input e modello congelati; maps e SHA di ogni libreria/driver caricati registrati per processo. Clock/cap utente invariati: 2600/1075 MHz, -100 mV, 186 W. Ambiente pulito, una sequenza/processo alla volta, scope 24G/swap2G, riserva 6 GiB, guard RX6800, telemetria 2 Hz, timeout 600 s. Nessun build o altro benchmark durante i timing.

Parametri esatti: -ngl 99 -ncmoe 18 -t 8 -tb 8 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off --load-mode mmap --verbosity 4. Breve 512 PP + 200 TG, c1024; lungo 10000 PP + 200 TG, c12288, 19 chunk da 512 + ultimo da 272. Helper replay con warmup intero e clear_data=false dichiarato; llama_decode+llama_synchronize, esclusi load/hash/raw I/O. Non TTFT applicativo o generazione autonoma.

Controllo GPU B1: -ot '^blk[.]17[.]ffn_(gate|up|down)_exps[.]weight$=Vulkan0' PRIMA di -ncmoe 18, perche' il loader usa il primo override corrispondente. Gate breve: B1/OFF normali esatti, poi B1 residente, fork residente, target host senza pool, target con pool e repeat; raw completo finito/nonzero prima della SHA.

Prima dei raw fissato anche uno screen CPU/GPU: argmax agreement >=99%, max_abs <=0.5, RMS per-vettore <=0.05, symmetric KL per-vettore <=0.01. E' uno screen numerico, non una certificazione della qualita' semantica. I criteri non sono rilassati dopo il risultato.

Il protocollo iniziale tre varianti CPU/TARGET/POOL e' conservato. Prima del primo timing e' aggiunto RESIDENT per confrontare l'alternativa esistente S09; dopo il fallimento dello screen CPU/GPU, un secondo emendamento conservato esclude tutti i ranking contro CPU e mantiene i soli controlli GPU esatti TARGET/POOL/RESIDENT. Nessun processo timing precedente o campione selezionato.

Timing effettivo: quattro processi freschi per ciascuno dei tre controlli GPU, tre rep dopo warmup, ordine TARGET/POOL/RESIDENT/RESIDENT/POOL/TARGET/POOL/RESIDENT/TARGET/TARGET/RESIDENT/POOL. Profiler, pool log e raw OFF. Mediana di 3 rep per processo, media di 4 mediane, bootstrap 20000, seed 33, CI95, margini +/-3% PP/TG e +/-5% p95; endpoint drift >20% invalida ranking. Tutti i campioni conservati. Confronti POOL vs TARGET e POOL vs RESIDENT, nessun CPU vs GPU.

Quattro catture profile CPU/TARGET/POOL/RESIDENT sono separate dai timing, con hash per-call confrontati col riferimento validato della stessa configurazione. CPU serve solo a osservare percorso/memoria, non al ranking. Scope host non sono kernel GPU isolati; payload API non e' traffico PCIe fisico. DRM total/resident VRAM/GTT sono picchi aggregati client a 2 Hz; RSS include mmap e scope memory.peak non misura tutta la RAM driver.

## Gate e risultati del timing

Breve normale B1/OFF: 201 vettori finiti/nonzero per processo, raw SHA 70a46c02e881eccb8f109aeda82cbde51990391b077edd3515c206814f7c0424, identico a R31. Cinque controlli GPU B1-resident/fork-resident/targetOFF/targetON/repeat sono bitwise identici, SHA 311b49e5aac3ca8604e8e1fe4c8a36127cba23d9bcb1dc4b5abb86574dd4cf95. La modifica cache/placement conserva esattamente il riferimento B1 con lo stesso compute GPU sul breve.

Screen CPU/GPU FAIL: prefill 512 identico, ma il decode ha max_abs 2.231972, massimo RMS per-vettore 0.611467 e symmetric KL 0.026661, oltre i limiti 0.5/0.05/0.01. Tutti i 201 argmax coincidono. Il cambio esiste anche con l'override GPU della baseline originale; non e' una regressione esclusiva della cache. Argmax invariato non sostituisce la distribuzione o il gate: nessuna tolleranza rilassata, nessun ranking/promozione contro CPU e nessuna certificazione semantica. Cause numeriche non isolate in questa campagna.

Lungo ERRORE_BASELINE: due B1 nuovi con stessi argomenti/librerie/driver hanno SHA d0c44f3b76e4b316677b62284f89ea151c38c21107fb27f981588b8dab73de7d e 0401575baab4a45afe732d96e654de2983e7f890c8305fb0d29fed2c2687f7a7. Tutti i 220 vettori finiti/nonzero per processo; 219/220 vettori diversi a partire dal secondo chunk, max_abs 9.763012/RMS 0.280608 e 4 argmax diversi fra le due esecuzioni. Rispetto a R31, 219 vettori e 5 argmax diversi. La causa non e' stabilita; target lungo e ranking lungo esclusi. Tutti i raw precedenti rimangono archiviati.

12 processi timing GPU / 36 rep / 7236 hash per-call coincidono col riferimento raw valido. Nessun endpoint PP/TG supera il drift 20% registrato: TARGET PP+0.75%/TG-1.94%, POOL PP-0.58%/TG-0.18%, RESIDENT PP+0.30%/TG+0.21%. Tutti i processi sono inclusi.

| Controllo GPU | PP token/s | TG token/s | TG p95 ms |
| --- | ---: | ---: | ---: |
| Host, no cache (TARGET) | 258.02 | 20.58 | 54.88 |
| Host, pool32 (POOL) | 259.31 | 21.06 | 53.20 |
| Layer residente (RESIDENT) | 284.17 | 23.50 | 48.70 |

| Confronto | Metrica | Delta% / CI95% | Verdetto |
| --- | --- | --- | --- |
| POOL vs TARGET | PP | +0.50 / [-0.50,+1.48] | NESSUN_CAMBIAMENTO, margine 3% |
| POOL vs TARGET | TG | +2.30 / [+1.12,+3.54] | INCONCLUDENTE, margine 3% |
| POOL vs TARGET | TG p95 | -3.05 / [-7.45,+1.27] | INCONCLUDENTE, margine 5% |
| POOL vs RESIDENT | PP | -8.75 / [-9.23,-8.34] | REGRESSIONE |
| POOL vs RESIDENT | TG | -10.40 / [-11.64,-9.21] | REGRESSIONE |
| POOL vs RESIDENT | TG p95 | +9.25 / [+3.57,+15.12] | INCONCLUDENTE, margine 5% |

La cache a 32 slot non dimostra un miglioramento pratico rispetto al trasferimento originale a pari compute GPU. E' piu' lenta del layer GPU completamente residente in PP/TG. Un intervallo TG positivo ma non interamente oltre il 3% non soddisfa il criterio di miglioramento registrato. Queste classificazioni non confrontano la CPU normale e non scelgono un profilo applicativo vincitore.

## Transfer e memoria

Il profiler vede upload degli esperti nel decode soltanto per il layer 17 selezionato; gli altri layer host conservano il compute CPU. Con pool, 200 TG / 600 proiezioni sono ready: 999 hit / 601 miss / 601 eviction, hit 62.4375%, 10 richieste completamente hit. API weight bytes senza cache 2,833,509,888; con cache 1,063,452,672: -62.4687%, inclusa la differenza di padding del percorso originale. Il controllo residente non carica pesi host durante TG. Remap 32 byte; pool 54 MiB. API bytes non misurano direttamente PCIe fisico.

Prima warmup PP: pool/remap 0/0 e bypass capacity. Prima TG: 54 MiB/32 byte; admission 12.845 ms include allocate/clear/upload. Dopo warmup il PP supera la capacity ma conserva il pool gia' allocato. Miss admission p50/p95=0.375/0.838 ms, somma admission 78.4 ms sui 200 TG; attesa di ripristino dei puntatori/synchronize p50/p95=4.661/4.843 ms, somma 937.739 ms. Quest'attesa puo' comprendere lavoro GPU gia' in volo: non misura solo overhead aggiunto o un kernel isolato e non si somma liberamente a scope annidati.

| Diagnostica breve | DRM total VRAM MiB | DRM resident VRAM MiB | DRM total GTT MiB | DRM resident GTT MiB |
| --- | ---: | ---: | ---: | ---: |
| CPU normale, solo diagnostica | 12110.930 | 11902.164 | 630.113 | 838.879 |
| GPU layer 17, no cache | 12110.930 | 11902.164 | 630.113 | 838.879 |
| GPU layer 17, pool32 | 12164.934 | 11902.168 | 630.113 | 892.879 |
| GPU layer 17 residente | 12576.215 | 12513.402 | 597.957 | 660.770 |

Il pool aggiunge 54 MiB di allocazione rispetto al target senza cache; resident VRAM cambia soltanto 4 KiB, mentre resident GTT aumenta 54 MiB. Non e' prova di 54 MiB fisici di VRAM in piu' o della residenza del singolo buffer. Il controllo interamente residente ha 432 MiB di pesi GPU in piu' nei log del modello, mentre il pool contiene 54 MiB; differenza dei picchi total VRAM residente-pool 411.281 MiB. Sono contatori client aggregati con possibili staging/migrazioni, non soltanto i pesi. Arena compute GPU 498.52 MiB in tutte le configurazioni, nessun risparmio di arena. Nessuno score combina picchi RAM/VRAM raccolti in istanti diversi.

## Decisione e limiti

R33 implementa il primo placement mirato S04/S09 senza cambiare soglia globale, e misura una cache realmente attiva con riferimento GPU originale. La cache32 non e' promossa come accelerazione del decode normale: il guadagno rispetto al trasferimento originale resta INCONCLUDENTE e il layer residente e' piu' rapido, a costo di maggiore allocazione GPU misurata. Lo screen numerico CPU/GPU fallisce: il placement non diventa default.

S09 resta parziale: non implementa split statico degli esperti caldi/freddi, non certifica qualita' semantica, pure GPU kernel time o un vincitore CPU/GPU. S07/S08 CPU miss fallback deve ancora rispettare quantizzazione e ordine di riduzione fra i due percorsi. Corpus/T2, stato lungo, pressione KV, partizione/warm-start e concurrency 1/2/4/8 rimangono aperti.

Prossimo incremento: S09/T0 isolare la divergenza del decode CPU/GPU con controlli quantitativi e riferimenti originali, prima di promuovere placement o merge ibrido; mantenere separata la diagnosi di ripetibilita' lunga. Uno sweep capacity S04 puo' procedere soltanto nel perimetro GPU esatto e con budget/timing preregistrati, senza dichiarare risolta la fedelta' alla CPU.

Campagna: 25 processi modello, 1847 vettori raw finiti/nonzero e 7236 hash timing corretti; quattro profiler separati aggiungono 804 hash per-call corretti. Totale R20-R33: 212 processi modello. [Manifest completo](moe-targeted-placement-validation.json), archivio locale risultati/2026-10-07-targeted-placement/. Nessun dato storico sovrascritto.

## Riprodurre i controlli brevi

Con gli input R31 versionati, la stessa build congelata e ambiente pulito, usare la command line sopra e un output nuovo per ogni processo. Comando del target con pool:

    env -i PATH=/usr/bin:/bin HOME="$HOME" XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" LC_ALL=C LD_LIBRARY_PATH=/absolute/path/frozen/NEW MOE_REPLAY_IN=/absolute/path/r31-short-512-200.csv MOE_REPLAY_REPS=3 GGML_SCHED_EXPERT_GPU_LAYER=17 GGML_SCHED_EXPERT_POOL=17:32:128 /absolute/path/frozen/NEW/replay -m /absolute/path/Ornith-1.5-35B-Q4_K_M.gguf -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off --load-mode mmap --verbosity 4 > fresh.csv 2> fresh.log

TARGET omette soltanto GGML_SCHED_EXPERT_POOL. RESIDENT omette entrambe le variabili e inserisce -ot '^blk[.]17[.]ffn_(gate|up|down)_exps[.]weight$=Vulkan0' prima di -ncmoe 18. Prima dei timing ripetere i raw con MOE_REPLAY_LOGITS_OUT e verificare ogni vettore, poi disattivare raw/profiler/logger. Guardrail, manifest, ordine e statistiche completi sono nel JSON; non confrontare semplicemente un processo di ciascun tipo.

Fonti primarie ricontrollate il 2026-10-07: [scheduler upstream](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp), [offload minimum issue18530](https://github.com/ggml-org/llama.cpp/issues/18530). Nessuno speedup su altre GPU e' assunto per RX6800.
