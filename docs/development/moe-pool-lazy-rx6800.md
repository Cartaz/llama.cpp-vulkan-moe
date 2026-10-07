# R32: allocazione lazy del pool esperti

Data: 2026-10-07. Branch `experiment/moe-pool-lazy-admission`, sopra R31. Codice locale `763bc575ebb58c2518e49f08e145ed1bf745451e`, remoto `03c89a74b4507c0730366293232ca824b9ba8d90`, tree comune `1a160d633d4dc33da224a773ef3da1661a6268df`. Baseline originale B1 `7fe450e19305b828c199d602c23a8337aaa1f03b`; controllo eager R30 `45bc2e846de4b311a1dafbc3e4cf7ed5a1850a72` / remoto `00d131fe421ab31a3dc8ae380ae65ac823a99b91`, tree `0e5b40c376172e7f7f49e18b91e629e6e6e100e3`.

## Problema e modifica

R30 allocava il pool durante la preparazione del grafo, prima di conoscere il routing. Con `17:32:128`, un PP che richiede piu' di 32 esperti fa bypass e il TG normale resta CPU; il buffer da 54 MiB rimane comunque allocato. Questo costo e' misurato su RX6800 nel report R30, con arena GPU completa invariata.

R32 separa metadati e buffer GPU: prepara le geometrie e verifica il budget, poi controlla ID, duplicati top-k e capacita' prima di allocare pool e remap. Riserva logicamente i byte previsti dei pesi anche mentre il buffer non esiste, cosi' il controllo del budget combinato non assegna al remap spazio gia' destinato al pool. Al primo accesso ammesso, alloca e azzera il pool; gli accessi successivi riusano buffer, slot e LRU. Il percorso diretto del pool mantiene l'allocazione eager come default; soltanto lo scheduler sperimentale richiede il modo lazy.

Il collo di bottiglia affrontato e' l'allocazione GPU inutile. Su una RX6800 da 16 GB evitare memoria inutilizzata puo' lasciare piu' margine, ma non implica uno speedup o una riduzione equivalente della VRAM fisicamente residente. I costi si spostano al primo accesso utile; la prima admission include allocazione/clear e il controllo aggiuntivo degli ID. Non cambiano kernel, placement, router, lease o politica LRU.

E' il primo incremento S14: una richiesta PP piccola che entra nel pool puo' ancora usarlo. Non aggiunge partizione decode protetta, priming, warm-start, rilascio del pool gia' usato o CPU fallback nuovo. Feature sempre OFF di default; nessuna nuova opzione pubblica.

## Validazione runtime

23 casi scheduler / 138 richieste / 414 vettori: F32, Q4_K, Q6_K misti, token1/33, broadcast, ID strided, reorder, eviction, cap/budget, callback e pipeline. I primi 22 casi conservano esattamente il raw R30; il nuovo caso `3:128:65` entra nel budget dei pesi ma non in quello del remap e deve rimanere non allocato. I cinque gruppi validamente configurati che non ammettono alcuna richiesta hanno pool/remap GPU di zero byte. I 16 gruppi attivi mantengono 20 miss/8 eviction ciascuno e zero upload dei pesi sugli all-hit.

48 casi del pool diretto / 384 richieste / 1,152 vettori sono ripetibili e coprono anche il percorso eager invariato. Il controllo management aggiunge richieste lazy invalide/capacity/duplicati senza allocazione, prima admission valida, slot inutilizzati zero e riuso all-hit. ASAN/UBSAN/leak: 23 casi scheduler GPU e 48 core CPU PASS; caricamento delle librerie ASAN congelate verificato separatamente. Gli 11 test parser/riepilogo/runtime passano.

SHA raw scheduler `59c0de5e1aa7b83b05b569b619157b0eefea87d11336a78a0a924e05b9449ac5`, 28,254,720 byte; core `81d6b37c29d62e4a1db98dde6dcfb6f12a0ec48c3a7375ce44b917bb4d62dcf9`, 78,807,040 byte. Ogni vettore e' verificato finito/nonzero prima di accettare la SHA. Oracle F32 indipendente e ID sorgente preservati nei controlli C++.

## Protocollo sul modello

Ornith-1.5-35B Q4_K_M, 21,713,462,848 byte, SHA256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`, ricontrollata per R32. RX6800 / Ryzen5700X3D / 32GB, CachyOS, kernel `7.2.9-1-cachyos`, Mesa/RADV `26.2.4-arch3.1`, GCC16.2.1 20260810, CMake4.4.4, Ninja1.13.2. Release/native/Vulkan/OpenMP, CPU compact/active OFF; CMakeCache e librerie congelati e registrati per variante.

Workload R31 identici: 512PP+200TG/c1024 e 10000PP+200TG/c12288. `-ngl 99 -ncmoe 18 -t 8 -tb 8 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off --load-mode mmap`, soglia offload default32, pool `17:32:128` per entrambi i controlli eager/lazy. Clock/cap utente invariati, ambiente pulito, telemetria 2 Hz, RX6800 prima del load, riserva RAM6GiB, scope24G/swap2G e timeout600s. Un processo modello alla volta; nessun build o altro test durante i timing.

Gate per caso normale: B1 fresco, nuovo fork OFF, eager ON, lazy ON e lazy ON ripetuto. Devono coincidere bitwise, anche con il riferimento raw R31. Gate attivo separato: breve512+200, soglia globale1 identica su B1 e due lazy ON; controlla il pool realmente usato, senza confrontare velocita' con il placement normale. Ogni dump verifica tutti i vettori finiti/nonzero prima della SHA; PP cattura l'ultimo token di ciascun chunk, TG tutti i passi.

Timing preregistrato: quattro processi per variante/caso, tre rep dopo warmup, ordine eager/lazy/lazy/eager/eager/lazy/lazy/eager. Somma di tutti i chunk PP e dei 200 passi TG, `llama_decode` + `llama_synchronize`; load/hash/raw I/O esclusi, non TTFT applicativo. Mediana per processo, media delle quattro mediane, bootstrap20000 a livello processo, seed32/CI95; equivalenza +/-3% PP/TG e +/-5% p95. Profiler/logger/raw disattivati. Quattro catture diagnostiche separate verificano pool, trasferimenti API e memoria client total/resident VRAM/GTT. Nessun confronto di speedup fra prompt/contesti diversi.

## Risultati

Breve normale PASS: cinque dump B1/OFF/eager/lazy/lazy-repeat hanno SHA `70a46c02e881eccb8f109aeda82cbde51990391b077edd3515c206814f7c0424`, come R31. Breve attivo con soglia1 PASS: B1 e due lazy ON hanno SHA `6467067b0ff47608a70ba4d3dd856c2da002a98e606994dd2e6ecc8fa033e2b9`. Vettori201 per processo; tutti finiti/nonzero prima della SHA.

Lungo ERRORE_BASELINE: il nuovo B1 produce SHA `d56599aa6099636285e20464b7694080297a74645dbad84dda083966f9eb8c2b`, diversa da R31 `416e6948b1229aaf8cca9b272bd0af5d5c5da9750d28ca80d67e466fe6734494`. Gli argomenti e tutte le SHA di librerie/driver caricati coincidono. Differiscono tutti i 220 vettori, 54,629,892 valori; max_abs3.71424/RMS0.108724, argmax invariati. Il B1 ripetuto nella stessa campagna produce un'altra SHA `7fbef8175703f1b90cd8841fc154ec391ab6962ff12b97ff2b7314d9c1acda98`: tutti i 220 vettori diversi dalla prima esecuzione, 54,629,910 valori e argmax invariati. Tutti i dump sono finiti/nonzero; non si allarga il criterio esatto.

Anche i quattro controlli del fork lungo non coincidono col B1 non ripetibile. Questo non permette di accettare la correttezza lunga o attribuire una regressione al fork. Il ranking lungo e' disattivato; le catture restano archiviate, senza cancellare i riferimenti R31 o riclassificare questi tempi come speedup. La causa non e' stabilita. L'amendamento del runner esclude soltanto il caso che ha fallito il gate, come previsto dal protocollo; ordine, statistiche e margini del breve restano invariati. Le catture lunghe con profiler/raw sono usate esclusivamente per osservare allocazioni.

| Metrica breve | Eager | Lazy | Delta% / CI95% | Verdetto |
| --- | --- | --- | --- | --- |
| PP token/s | 258.23 | 233.62 | -9.53 / [-27.73,+0.71] | INCONCLUDENTE |
| TG token/s | 22.44 | 24.07 | +7.24 / [-0.54,+21.86] | INCONCLUDENTE |
| TG p95 ms | 50.00 | 48.14 | -3.73 / [-13.85,+2.94] | INCONCLUDENTE |

Tutti gli 8 processi / 24 rep / 4,824 hash per-call sono corretti. Il processo `timing-06-lazy` e' molto diverso dagli altri: PP 163.39 t/s, TG 28.98 t/s, p95 40.63 ms. Sono rilevati clock GPU effettivi diversi nei campioni, mentre limiti clock/cap, argomenti, librerie e output coincidono. La causa del cambio di tempi non e' stabilita. Si conserva il campione; non si elimina post hoc e non si dichiara equivalenza, speedup o regressione. Gli altri processi non sostituiscono il verdetto preregistrato.

Lungo: nessuna serie di timing viene eseguita, per ERRORE_BASELINE. I tempi dei dump e dei profiler restano diagnostici, non classificano un vincitore.

| Caso | Buffer pool eager/lazy | Remap picco eager/lazy | DRM total VRAM eager/lazy MiB | DRM resident VRAM eager/lazy MiB | DRM resident GTT eager/lazy MiB |
| --- | --- | --- | --- | --- | --- |
| Breve valido | 54 MiB / 0 | 16 KiB / 0 | 12164.95 / 12110.93 | 11902.18 / 11902.16 | 892.88 / 838.88 |
| Lungo solo allocazioni | 54 MiB / 0 | 16 KiB / 0 | 12300.93 / 12246.91 | 11953.87 / 11957.35 | 994.96 / 940.96 |

Pool e remap rimangono a zero su tutte le chiamate normali lazy: PP fa bypass per capacita', TG resta CPU. L'arena compute GPU e' 498.52 MiB in tutte le varianti. Il total VRAM del client scende di circa 54 MiB, ma il resident VRAM breve cambia soltanto 16 KiB e nel lungo varia in direzione opposta; resident GTT scende di 54 MiB in entrambe le catture. Sono picchi client aggregati a 2 Hz, non prova della residenza di un singolo buffer o di 54 MiB di VRAM fisica liberata. RSS include mmap; scope memory.peak e RAM anonima non sono la stessa misura.

Il controllo attivo separato conferma la lazy admission: prima del TG il pool/remap e' 0/0; alla prima richiesta utile e' 54 MiB / 32 byte. Prima admission warmup 12.769 ms, inclusi allocazione/clear/upload; questo e' il tradeoff della prima richiesta, non un tempo TTFT applicativo. Dopo warmup,200 TG / 600 proiezioni, 993 hit / 607 miss (62.06%),607 eviction, 9 all-hit e 1,074,069,504 byte di upload API; miss admission p50 0.401ms. Tutti gli hash coincidono col gate attivo valido. Nessun confronto di velocita' con il placement normale.

Campagna R32: 27 processi modello, 3,368 vettori raw finiti/nonzero e 4,824 hash di timing breve coerenti. Totale R20-R32: 187 processi modello. I 6 controlli raw lunghi e le 2 catture lunghe profilate conservano l'errore baseline; non certificano la correttezza lunga.

[Manifest di validazione e provenienza](moe-pool-lazy-validation.json). Archivio locale `risultati/2026-10-07-pool-lazy/`, con raw, log, telemetria, source/build/helper/librerie e driver effettivamente caricati.

## Limiti e prossimo passo

La suite lunga normale verifica la correttezza del bypass lazy, non certifica una cache attiva a 10k token. I precedenti errori baseline R23/R25 rimangono archiviati e non sono dichiarati risolti. Corpus, pressione KV e concurrency1/2/4/8 restano aperti. S04/S09 placement mirato per layer e misure compute CPU/GPU sono il passo successivo per tentare un beneficio TG reale; partizione/warm-start S14 e fallback S07/S08 restano da implementare.

Riferimenti primari ricontrollati il 2026-10-07: [scheduler upstream](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp), [richiesta cache esperti #20757](https://github.com/ggml-org/llama.cpp/issues/20757), [RFC pool persistente #28248](https://github.com/ggml-org/llama.cpp/discussions/28248). Le misure riportate qui provengono dalla RX6800 locale; nessun risultato CUDA delle discussioni viene assunto per questa macchina.
