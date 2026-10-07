# R30: pool opzionale per un layer nello scheduler Vulkan

Data: 2026-10-07. Branch `experiment/moe-scheduler-pool`, [PR draft 11](https://github.com/Cartaz/llama.cpp-vulkan-moe/pull/11), sopra R29. Codice misurato locale `45bc2e846de4b311a1dafbc3e4cf7ed5a1850a72`, pubblicato come `00d131fe421ab31a3dc8ae380ae65ac823a99b91`: entrambi hanno tree `0e5b40c376172e7f7f49e18b91e629e6e6e100e3`. Il fix del parser host e' successivo, senza rebuild del motore. [Metadati](moe-scheduler-pool-validation.json). Archivio locale: `risultati/2026-10-07-scheduler-pool/`.

S04/S05 arrivano al percorso del modello per **un layer e uno stream**, con opzione OFF di default. Gate numerico esatto superato sul replay breve dopo warmup. Non e' una cache per tutti i layer, un nuovo placement o una riduzione dell'arena compute. T2 esteso e concurrency 2/4/8 restano aperti; M3 rimane parziale.

## Implementazione e costo atteso

Il collo bersaglio e' la ricopia di esperti CPU quando `MUL_MAT_ID` e' gia' assegnato a Vulkan. Conservarne la tripletta gate/up/down sul backend puo' evitare upload ripetuti su RDNA2. Il prezzo e' un buffer persistente aggiuntivo, lookup/remap sul thread host, caricamento dei miss e sincronizzazioni conservative. Il beneficio end-to-end richiede un A/B; la riduzione dei byte da sola non basta.

- Il core R29 viene spostato in un header interno GGML; la fixture conserva un alias. Nessun nuovo shader o API pubblica.
- `GGML_SCHED_EXPERT_POOL=layer:slots:budget_MiB` abilita un solo layer. Limiti: layer0..4096, slot1..256, budget1..2048MiB. Assente, vuoto o `0` disabilita; config invalida segnala e disabilita.
- Solo scheduler con una copia, senza callback di evaluation, stesso backend Vulkan e tre pesi host immutabili F32/Q4_K/Q6_K, con nomi canonici `blk.N.ffn_{gate,up,down}_exps.weight`.
- Il primo operatore compute di ogni segmento deve consumare il peso selezionato. Le viste iniziali sono riconosciute soltanto nel percorso opzionale; gli altri consumatori del peso impongono bypass.
- Riusa il readback degli ID esistente; non aggiunge un secondo readback. Solo le tre moltiplicazioni ricevono ID privati rimappati. Router, ID originali e altri consumatori mantengono i propri input.
- Admission dell'unione del batch, slot necessari pin, LRU deterministica, upload della tripletta soltanto sui miss. Il budget comprende buffer pesi e remap, con allocazione effettiva/allineamento del backend.
- Gli argomenti del grafo sono ripristinati dopo la sincronizzazione di ciascun segmento modificato, anche sulle uscite d'errore. La lease viene rilasciata dopo l'ultimo segmento selezionato.
- Capacita', budget, binding o modalita' non supportati usano il percorso normale. Il fallback runtime mantiene il backend originale: **non** implementa miss-CPU/hit-GPU. Il bypass per top-k duplicati non corregge il difetto sintetico baseline di R29; il routing normale usa top-k distinti.

Il pool aggiunge memoria persistente. L'allocator dello scheduler continua a riservare copie a forma completa: il buffer compute del modello resta 498.52MiB. Questo prototipo non promette risparmio VRAM. Non vengono indotti OOM del driver, pressione KV o abort del modello; i test di budget non equivalgono a queste prove.

## Parametri e due controlli distinti

Modello: Ornith-1.5-35B-Q4_K_M.gguf, 21,713,462,848 byte, SHA256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`, ricalcolato prima della campagna. Replay teacher-forced versionato `r22-244-128.csv`: 244token PP e128token TG, vocab248320. Non e' una misura semantica di qualita' o di un agente reale.

```text
-ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512
-fa on -ctk q8_0 -ctv q8_0 --fit off --verbosity 4 --load-mode mmap
MOE_REPLAY_CLEAR_DATA assente: default false
ON: GGML_SCHED_EXPERT_POOL=17:32:128
OFF: GGML_SCHED_EXPERT_POOL assente
```

1. **Normale**: soglia di offload non modificata, default32. Il PP usa piu' di32esperti nel layer17 e fa bypass; TG resta CPU per gli esperti host. Il pool non accelera quel decode.
2. **Offload minimo1**: `GGML_OP_OFFLOAD_MIN_BATCH=1` identico su B1, B2OFF e ON. La soglia globale offloada tutte le operazioni ammesse, non soltanto il layer17. Serve a verificare l'integrazione TG e a isolare il costo del pool. Le sue velocita' non vanno confrontate col profilo normale come guadagno della cache.

Release/native/shared, Vulkan ON, backend dynamic OFF, CPU compact/active OFF, OpenMP; tests/tools/server/app/MTMD OFF, target replay e scheduler-check. CMakeCache originali e shader/librerie congelati per entrambe le varianti. RX6800/RADV NAVI21, Mesa26.2.4-arch3.1, GCC16.2.1, CMake4.4.4/Ninja1.13.2, kernel7.2.9-1-cachyos. Ambiente pulito ed elenco/hash delle librerie effettivamente mappate per ogni processo; nessuna modifica a clock/tensione/power cap dell'utente.

B1 usa il motore originale `7fe450e19305b828c199d602c23a8337aaa1f03b` e il helper replay abbinato gia' congelato in R21: motore e helper sono provenienze distinte. B2OFF/ON usano esattamente gli stessi binari/librerie R30. Le directory frozen includono alias SONAME e sono verificate prima/dopo ogni processo. Guard RAM disponibile6GiB, VRAM iniziale<3GiB, cgroup24GiB/swap2GiB, timeout600s, telemetria2Hz e conferma della GPU prima del load.

## Gate prima del timing

| Controllo | Perimetro | Esito |
| --- | --- | --- |
| Scheduler fixture Release Vulkan | 22casi/132richieste,396vettori,28,154,880byte; due capture riuscite | Bit-identici OFF/ON, tutti finiti/nonzero |
| Oracolo indipendente F32 | Gate/up,1/33token, broadcast, ID strided | Esatto |
| Routing/lifetime | ID originali preservati, riordini all-hit, mixed, eviction, riuso grafo | PASS |
| Counters pool | 16casi ammissibili:20miss/8evictions ciascuno;3proiezioni su ogni ready | PASS; all-hit0byte pesi |
| Bypass | Pool4slot, budget1MiB/F32, config invalide, parallel/callback | Output identici al percorso OFF |
| ASAN/UBSAN/leak | CPU focused Q4,6richieste; Vulkan22casi/132richieste, anche F32/Q6/bypass | PASS, nessun errore |
| Modello normale | B1/B2OFF/ON,3processi,387vettori completi | Bit-identici, finiti/nonzero |
| Modello min-batch1 | B1/B2OFF/ON/ON-repeat,4processi,516vettori completi | Bit-identici, finiti/nonzero |
| Parser profilo | Nuovo evento admission e join PP/TG | 7testPASS CPU e7testPASS Vulkan, inclusa fixture runtime; profilo modello parse PASS |

SHA raw normale `b7bb28a86e623daf75f436f9bddb786cee6c43d8a28d1b85333f3f25d9cbb662`; min-batch1 `bcda1e4c633b1012e778ef1d0763fe8e8143e45a483ed4c3a71fe2d3aa74878b`. Il secondo differisce dal normale per la configurazione, ma coincide fra tutti i controlli abbinati. L'uguaglianza dei digest viene accettata solo dopo la verifica di ogni vettore finito/nonzero. Ogni ripetizione di timing mantiene anche il controllo hash per singola chiamata.

## Diagnostica, separata dal timing

`GGML_SCHED_EXPERT_POOL_LOG=PATH` abilita CSV append con ID pool/call, ragione, hit/miss/eviction, upload API, buffer e tempi host in us. Timestamp start/end permettono il join con intervalli PP/TG del replay. Per leggerli in ms dividere per1000. Il logger registra preparation, admission della prima proiezione e attese del ripristino; `expert_pool_admit` nel profiler comprende anche i lookup delle proiezioni successive. Non sono tempi device dei kernel; non sommare envelope/split/eventi annidati. Il logger e il profiler sono spenti nei campioni prestazionali.

Una capture per variante,1rep misurata dopo warmup, soglia1 e pool32slot:

- PP: bypass capacita',0proiezioni modificate. Pool allocato al primo warmup, initialization e buffer clear inclusi nei costi iniziali, non nascosti come lavoro gratuito.
- TG:128ready,384proiezioni,635hit/389miss su1024accessi (8esperti distinti per token) (hit62.01%),389evictions;8token completamente all-hit. Gli esperti ripetuti fra token sono ammessi.
- Payload API dei pesi layer17: OFF1,813,440,000byte, ON688,324,608byte, **-62.04%**. Include padding per gli upload originali; non e' traffico PCIe misurato o riduzione delle copie di tutti i layer.
- Admission con miss: p50 **0.356ms**, p95 **0.725ms**,120chiamate. Admission su tutti i token p50 **0.346ms**; lookup/readback e lavoro GPU hanno scope separati.
- Attese dei segmenti modificati: p50 **4.739ms**, p95 **4.912ms**, totale609.550ms. Comprendono completamento del compute, non sono tutte overhead aggiuntivo rispetto a OFF; non sottrarle ingenuamente al tempo totale.
- Pool pesi: **56,623,104byte =54MiB**; remap PP7808byte/TG32byte. Allocazione VRAM per-client totale aumenta di56,635,392byte fra i picchi dei due profili. VRAM residente resta circa11.622GiB, GTT residente aumenta da0.681 a0.734GiB: total e resident non sono sinonimi e il polling non attribuisce migrazioni a singoli buffer. Nessun risparmio VRAM dichiarato.

L'A/B dei tempi e' necessario anche perche' il profilo mostra costi residui e residenza variabile. Questi valori sono diagnostici sul breve, non ranking prestazionale o previsione per altri layer.

## A/B senza profiler

Protocollo registrato prima dei processi:4processi freschi per variante, una ripetizione warmup e3misurate; ordine OFF/ON/ON/OFF/OFF/ON/ON/OFF. Mediana delle3rep per processo, media delle4mediane per variante. Bootstrap indipendente per processo,20,000resample/seed30, CI95%; fascia di equivalenza PP/TG +/-3%, p95 +/-5%. Nessun profiler, logger del pool o dump raw nei campioni. I3,096hash per-call misurati coincidono col controllo raw min-batch1 validato.

| Metrica | OFF | ON | Variazione ON/OFF, CI95% | Verdetto |
| --- | --- | --- | --- | --- |
| PP token/s | 168.17 | 167.40 | -0.45%, [-2.31,+0.99]% | NESSUN_CAMBIAMENTO, fascia +/-3% |
| TG token/s | 9.425 | 9.527 | +1.08%, [+0.73,+1.44]% | NESSUN_CAMBIAMENTO, fascia +/-3% |
| TG p95 ms/token | 108.992 | 108.010 | -0.90%, [-1.79,-0.07]% | NESSUN_CAMBIAMENTO, fascia +/-5% |

La piccola differenza TG positiva e' rilevata in questo A/B ma resta entro la fascia pratica prestabilita: non viene promossa a un miglioramento rilevante o universale. PP e p95 risultano equivalenti entro le proprie fasce. La riduzione delle copie riguarda un layer di18host; altri layer, compute, readback e sincronizzazioni continuano a pesare. Le8esecuzioni aggiungono24rep misurate, senza aumentare artificialmente a24il numero di campioni indipendenti.

R30 esegue17processi modello:7raw gate +2profili +8timing. Totale cumulativo R20-R30:144processi; fixture e sanitizer sono conteggiati separatamente. Cold-start, TTFT applicativo e throughput di agenti reali non sono classificati da questo A/B.

## Tentativi conservati e limiti

Le prime capture sintetiche erano numericamente corrette ma non attivavano il pool: il segmento iniziava con VIEW e mancava il gate nella selezione della tripletta. Sono conservate ed escluse dalla validazione dell'integrazione; il fix riconosce il primo operatore compute soltanto quando l'opzione e' attiva. Le due capture finali usano veramente il pool e sono bit-identiche.

Il primo riepilogo del profilo ON rifiutava l'evento nuovo `expert_pool_admit`. Il modello aveva completato con successo: il parser e' stato esteso, i test esistenti aggiornati e lo stesso CSV riprocessato, senza trasformare l'errore di tooling in un campione di timing. Log di compilazione intermedia, errori e dati iniziali rimangono nell'archivio.

Il helper esegue una ripetizione di warmup prima del dump; i903vettori raw riguardano le ripetizioni misurate, non il cold-start completo del modello. Cold/all-hit/miss/eviction sono coperti dalle fixture e dai contatori; il gate raw cold del modello resta da estendere. Nessuna nuova affermazione sul contesto lungo instabile di B1, sulla qualita' semantica, su fusioni/kernel non coperti o su2/4/8stream. Il gate riguarda Ornith breve e le geometrie fixture dichiarate. Non e' dimostrato che tutte le architetture MoE siano compatibili; nomi fusi gate/up e layout diversi fanno bypass.

## Prossimo esperimento

R31: aggiungere la coppia breve/lungo richiesta dall'utente, circa10,000token PP +200token TG per il lungo. Sommare tutti i chunk PP, misurare TG/p95 separati e verificare prima ripetibilita' B1, dato il difetto lungo gia' documentato. Il replay244+128 rimane il controllo storico R30.

Passo successivo S14: admission lazy e bypass PP prima di allocare il pool. Nel profilo normale il prototipo riserva54MiB anche se il batch supera32slot e il decode resta CPU: il primo miglioramento piccolo e misurabile e' evitare questa allocazione inutile, preservando gli stessi logits e le stesse admission quando il pool serve. A/B R30/R31 con identici parametri, gate raw normale e soglia1, allocation/counters e memoria cold/warm; nessun guadagno di token/s senza nuova campagna di timing.

Poi controllo di placement mirato al singolo layer, senza soglia globale1: misurare il suo compute CPU/GPU e residenza total/resident, tenendo gli altri layer e i parametri uguali. Il pool da solo non sposta gli esperti CPU sulla GPU. Solo dopo questi controlli estendere layer, T2 held-out e concurrency2/4/8, o scegliere S07/S08 e nuove policy.

Il punto di inserimento e il comportamento di offload sono stati riletti nel [scheduler upstream fissato B4](https://github.com/ggml-org/llama.cpp/blob/f498f864fbc0472004ee1c3616c1188c68eb157f/ggml/src/ggml-backend.cpp) e nel [backend Vulkan dello stesso commit](https://github.com/ggml-org/llama.cpp/blob/f498f864fbc0472004ee1c3616c1188c68eb157f/ggml/src/ggml-vulkan/ggml-vulkan.cpp). Questa integrazione e i suoi risultati appartengono al fork: non sono una misura di una PR upstream o di hardware RDNA3/4.
