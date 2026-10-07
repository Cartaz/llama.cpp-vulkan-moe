# R31: test PP/TG breve e lungo, 200 token decode

Data: 2026-10-07. Branch `experiment/moe-short-long-bench`, sopra R30. Tooling `925d3ec0e1868280b0c2ba2dd76607de7bb45bc8`; motore B1 originale `7fe450e19305b828c199d602c23a8337aaa1f03b`, candidato OFF congelato R30 `45bc2e846de4b311a1dafbc3e4cf7ed5a1850a72`/remoto `00d131fe421ab31a3dc8ae380ae65ac823a99b91`, tree `0e5b40c376172e7f7f49e18b91e629e6e6e100e3`. Nessun rebuild o nuova modifica del motore.

L'utente ha chiesto due test complementari: breve per iterazioni rapide e lungo intorno a 10k token PP / 200 token TG. R31 aggiunge workload e un riepilogo che **somma tutti i chunk PP**; non usa il solo primo/ultimo chunk e non cambia retroattivamente il benchmark 244+128 di R30.

| Caso | Prompt | Decode | Contesto | Batch/ubatch | Chiamate PP per rep |
| --- | --- | --- | --- | --- | --- |
| Breve | 512 token | 200 token | 1024 | 512/512 | 1 |
| Lungo | 10,000 token | 200 token | 12,288 | 512/512 | 20: 19 da 512 + 272 |

Gli altri parametri sono identici: `-ngl 99 -ncmoe 18 -t 8 -tb 8 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off --load-mode mmap`. Soglia offload non modificata/default 32; CPU compact/active e pool OFF. La soglia globale 1 usata nel controllo cache R30 non e' il profilo di questi due test. PP, TG, p95 e memoria per caso sono distinti; non interpretare il rapporto breve/lungo come speedup di un'ottimizzazione, dato il cambio di prompt/contesto.

## Cosa viene misurato

Ogni chiamata cronometra `llama_decode` e `llama_synchronize` fino a completamento. PP token/s = token prompt / somma dei tempi di tutte le chiamate prefill. TG token/s = 200 / somma dei 200 passi a token singolo. p50/p95 TG sono le latenze delle chiamate. Caricamento, hash logits e scrittura raw fra chiamate sono esclusi: non e' TTFT applicativo o un benchmark end-to-end di un agente con tool.

`examples/moe-trace/replay-summary.py` valida ordine, rep, posizioni, dimensione dei chunk, continuazione, tempo positivo e hash. Rifiuta chunk mancanti/ripetuti, campioni incompleti e mismatch per-call. Il gate raw opzionale controlla dimensione esatta e ogni vettore finito/nonzero prima di accettare SHA256. NumPy serve soltanto per leggere i dump raw, il riepilogo CSV usa la libreria standard.

```bash
python3 examples/moe-trace/replay-summary.py replay.csv \
  --tokens examples/moe-trace/workloads/r31-long-10000-200.csv \
  --step 512 --json summary.json
```

Per il gate numerico aggiungere `--logits logits.bin --vocab 248320`; per confrontare tutti i passi, `--reference baseline-replay.csv`. L'esempio e' un comando riproducibile della repo, non il path dell'archivio locale.

## Dati e riproducibilita'

Due task sintetici di revisione C++, con stesso quesito e diversa lunghezza dell'estratto, delimitatori chat espliciti e prefix thinking. Il documento sorgente combina scheduler, buffer Vulkan e contesto del modello a snapshot congelato; i prefissi 512/10k token qui si fermano nel primo file dello scheduler; SHA e prompt completi sono versionati. Non e' una valutazione di qualita' semantica o un corpus di agenti reali.

Il tokenizer vocab-only usa header C API e librerie della baseline originale. La generazione greedy, senza callback, produce 200 token per ciascun prompt e verifica ogni vettore del cold run. `--no-escape` preserva i backslash del C++; i file non terminano con newline per coincidere col parsing `-f`. Gli ID prompt del generatore devono coincidere col tokenizer prima di congelare il CSV. Ogni replay impone gli stessi 200 token del proprio caso; non cambia la continuazione tra varianti.

RX 6800 / Ryzen 7 5700X3D / 32 GB, CachyOS; kernel `7.2.9-1-cachyos`, Mesa/RADV `26.2.4-arch3.1`, Vulkan API1.4.354. GCC16.2.1 20260810, CMake4.4.4, Ninja1.13.2. I CMakeCache BASE/POOL e le SHA di eseguibili, librerie effettivamente caricate e driver sono nell'archivio `risultati/2026-10-07-short-long/` e nel manifest di validazione. Build Release/native, Vulkan e OpenMP; CPU compact/active OFF. Le librerie B1 originali restano immutate; i helper di cattura/replay sono congelati separatamente.

Modello Ornith-1.5-35B Q4_K_M, 21,713,462,848 byte, SHA256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`, ricontrollata per R31. Dataset versionati al commit `1737545073a1e256b3f9fa19b3496f0dbf9e2483`. Manifest `examples/moe-trace/workloads/r31-benchmark-cases.json`; SHA CSV breve `711949862bf3674e00add770c2add1b3fac53d0f3695ee28769ca8a082c7052d`, lungo `b51f817559a54e37e34d64e9a5c37d0a7dc38101792ab3bdd9fbe01c44157889`.

Ambiente pulito e registrato per processo, nessun profiler/raw nelle misure temporali; `LD_LIBRARY_PATH` punta solo alla variante congelata. RX6800 verificata prima del load, MemAvailable >= 6 GiB, VRAM globale < 3 GiB prima della partenza, scope systemd MemoryMax24G/SwapMax2G e timeout600s. Clock utente invariati: limite core 2600 MHz, memoria 1075 MHz, offset -100 mV, cap 186 W; profilo e telemetria 2 Hz registrati. Nessuna esecuzione modello concorrente.

Il protocollo e' registrato prima delle inferenze e l'analisi prima dei timing: 4 processi freschi per caso, 3 ripetizioni dopo warmup. Mediana delle 3 rep per processo, media delle 4 mediane e CI95 bootstrap 20000 ricampionamenti a livello processo, seed 31. L'intervallo descrive questa campagna, con soli 4 processi. Nessun confronto di speedup fra i due workload.

## Gate e tempi

Entrambi i casi PASS: generatore greedy cold B1, due replay B1 freschi dopo warmup e replay del fork R30 con pool OFF hanno dump bitwise identici. Tutti i 1,684 vettori raw complessivi sono finiti/nonzero; ogni vettore ha 248,320 float32. Il controllo include ogni chunk PP e passo TG, senza ampliare tolleranze. Il dump contiene un vettore per chiamata: l'ultimo token di ciascun chunk PP, non i logits di tutti i token del prompt.

| Caso | Vettori per processo | Byte raw per processo | SHA256 comune ai 4 controlli |
| --- | --- | --- | --- |
| Breve | 201 | 199,649,280 | `70a46c02e881eccb8f109aeda82cbde51990391b077edd3515c206814f7c0424` |
| Lungo | 220 | 218,521,600 | `416e6948b1229aaf8cca9b272bd0af5d5c5da9750d28ca80d67e466fe6734494` |

Gli 11 test del parser/riepilogo e fixture runtime passano. Coprono aggregazione multi-chunk con coda, rep multiple, chunk mancanti, hash diversi e raw NaN/zero/dimensione errata. Il gate R31 lungo verifica questo nuovo workload/configurazione: non risolve o annulla le instabilita' precedentemente archiviate in R23/R25.

| Caso | PP token/s (CI95) | PP totale ms | TG token/s (CI95) | TG totale ms | TG p95 ms (CI95) |
| --- | --- | --- | --- | --- | --- |
| Breve | 259.04 [256.60, 261.40] | 1976.69 | 22.34 [22.25, 22.46] | 8952.53 | 51.45 [50.18, 52.35] |
| Lungo | 246.76 [246.30, 247.18] | 40525.22 | 16.52 [16.48, 16.56] | 12108.31 | 64.21 [63.80, 64.76] |

Tutti gli 8 processi di timing sono completi e i 5,052 hash per-call coincidono con il rispettivo riferimento validato. Ogni processo misura 3 rep dopo un warmup completo. La campagna R31 comprende 16 processi modello: 8 generatori/controlli raw e 8 timing. Totale R20-R31: 160. I tempi PP/TG nella tabella sono medie delle mediane per processo; i raw capture precedenti hanno solo valore diagnostico per la velocita'. Nessuna nuova ottimizzazione del motore viene classificata.

[Manifest completo di validazione e provenienza](moe-short-long-validation.json). Contiene analisi preregistrata, opzioni CMake, ambiente/argv, librerie e driver effettivamente caricati con SHA, dump validati, picchi memoria separati e hash dei file di campagna.

I log registrano buffer KV GPU di 10.62 MiB nel breve e 127.50 MiB nel lungo; compute arena GPU 498.52 MiB in entrambi. La memoria va letta separando total/resident VRAM e GTT dai campioni DRM del client. RSS comprende il modello mmap e non rappresenta RAM anonima; memory.peak dello scope e' riportata come misura distinta. Il manifest conserva questi valori distinti, senza attribuire migrazioni a un singolo buffer.

## Limiti e prossimo passo

Aggiornamento R32 (2026-10-07): una nuova esecuzione dell'identico motore B1, con gli stessi argomenti e SHA di tutte le librerie/driver, differisce dal raw lungo R31 in tutti i 220 vettori (max_abs3.71424, RMS0.108724, argmax invariati). R32 classifica ERRORE_BASELINE ed esclude il ranking lungo. I campioni R31 sono conservati e risultavano coerenti nella propria campagna; non costituiscono una garanzia di ripetibilita' fra campagne. [Diagnosi e controlli R32](moe-pool-lazy-rx6800.md).

Il lungo e' un gate nuovo: non eredita la validita' del breve. R23/R25 avevano gia' trovato instabilita' anche nel motore B1 a contesto lungo. Due processi B1 freschi devono produrre raw ripetibili alle condizioni registrate; se non coincidono, segnare ERRORE_BASELINE per quel caso e conservare tutto. Il breve puo' proseguire. Non allargare la tolleranza dopo aver visto i risultati e non pubblicare un vincitore sul lungo invalido.

Dopo gate valido, 4 processi nuovi/3 rep dopo warmup definiscono il riferimento temporale del singolo caso. Nel presente incremento non si confronta una nuova ottimizzazione del motore. Le future patch useranno la stessa coppia B1/B2OFF/ON, kernel/profiler separati dal timing, contesto/placement/KV fissi entro ciascun caso e verifica source/build/libs/token.

Per riprodurre il replay usare il helper `llama-moe-replay` costruito con le stesse opzioni del manifest e librerie della variante controllata:

```bash
MOE_REPLAY_IN=examples/moe-trace/workloads/r31-long-10000-200.csv \
MOE_REPLAY_REPS=3 ./build/bin/llama-moe-replay \
  -m /path/to/Ornith-1.5-35B-Q4_K_M.gguf \
  -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 12288 -b 512 -ub 512 \
  -fa on -ctk q8_0 -ctv q8_0 --fit off --load-mode mmap > replay.csv
```

Per il breve cambiare CSV in `r31-short-512-200.csv` e contesto in 1024. Il percorso `build/bin` e' illustrativo: nella campagna sono stati usati solo helper/librerie congelati. Il helper esegue un warmup completo e azzera le sequenze prima di ogni rep; il flag di clear dei dati KV non viene attivato, coerente col helper baseline originale. Cold generator e warm replay coincidono nei gate registrati.

Rimane da estendere il corpus, i turni cold/warm e la concurrency 1/2/4/8. S14 admission lazy e placement mirato rimangono passi successivi; i pool del profilo normale non accelerano il TG CPU e aggiungono memoria quando allocati inutilmente.
