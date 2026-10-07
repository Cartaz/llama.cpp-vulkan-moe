# R36: replay degli ingressi MoE reali del layer 17

Data: 2026-10-07. S09/T0, branch `experiment/moe-real-operator-replay`, [draft PR17](https://github.com/Cartaz/llama.cpp-vulkan-moe/pull/17) sopra PR16, non integrato. Codice locale `91fa63004721928eaacc0d5e6527db7eada40dbd`, remoto `ca4b1d7d280d41d31ca2ce968fe647dcaf223465`, tree comune `5114215426e00426c0001ac8fb03ac1044633f6b`.

R35 isolava la precisione delle attivazioni su dati sintetici. R36 ripete il confronto con pesi Q4_K autentici, ingressi F32 e routing catturati durante il replay Ornith. Il gate osservatore OFF/ON PASS e i risultati degli operatori confermano che la conversione degli ingressi spiega quasi tutta la differenza CPU/GPU nei nove casi selezionati. Non e' un fix dei logits finali, una certificazione semantica o un risultato di velocita'.

## Modifica minima e formato

`GGML_CPU_MOE_CAPTURE=PATH` abilita un osservatore in ingresso a `ggml_compute_forward_mul_mat_id`, soltanto sul worker0. Assente/vuoto e' disabilitato. Seleziona i nomi esatti `blk.17.ffn_{gate,up,down}_exps.weight`, Q4_K/256 esperti/top8, e soltanto gli ingressi a token singolo con le forme di Ornith. Legge ingressi F32 e ID I32 tramite gli stride; non scrive nei tensori, non cambia callback, grafo, scelta degli esperti o aritmetica, e non aggiunge barrier. Gli altri worker possono convertire lo stesso ingresso immutabile nel workspace. La critical section esistente serializza le scritture del singolo processo. Un errore I/O termina il processo; nessuna cattura parziale viene accettata.

E' uno strumento diagnostico ristretto a questo modello/layer e stream1, non un tracciatore generale. PATH deve essere nuovo per ciascun processo. Il file viene aperto in append e non contiene identificatori di sessione o posizioni token: il protocollo lega l'ordine dei record al replay congelato. Reps1 e warmup0 evitano record di altre esecuzioni. Non e' qualificato per processi concorrenti sullo stesso file. Anche OFF ha la verifica dell'ambiente sul worker0: nessuna assenza di overhead e' dichiarata.

Ogni record contiene8 uint64 nativi: magic `0x31454f4d55504347`, versione1, proiezione0/1/2, k, m, input_slots,8 ID,256 esperti; poi righe F32 packed nell'ordine degli slot e8 ID I32. La macchina misurata e' little-endian; il parser verifica header, dimensioni, EOF, ordine, finite/nonzero di ogni riga e range degli ID. Non vengono catturati gli output intermedi del modello.

`MOE_OPERATOR_INPUT_PREFIX=PREFIX` nel fixture esistente `--operator-numerics` carica pesi packed `PREFIX-{gate,up,down}-weights.bin` e, per ciascun pass0..2, `PREFIX-PROIEZIONE-PASS-original.bin` e `...-ids.bin`. Le dimensioni sono esatte, senza byte extra o mancanti; ingressi non finiti/nulli e ID fuori range vengono respinti prima del calcolo. Il modo sintetico R35 resta compatibile. `MOE_OPERATOR_INPUT_Q8_K=1` applica lo stesso roundtrip CPU Q8_K prima del calcolo; i due oracoli scalari con accumulo double restano quelli di R35.

Gate/up: pesi[2048,512,256], input[2048,1,1]; down: pesi[512,2048,256], input[512,8,1]. I tre tensori reali sono copiati byte per byte dal GGUF,144MiB ciascuno,432MiB totali. A differenza del fixture sintetico, tutti256 esperti hanno i loro pesi reali. Il replay seleziona gli ID naturali catturati, senza permutazioni artificiali.

## Protocollo e ripetibilita' del modello

Registrato prima dei raw in `protocol.json`: una richiesta,512PP+200TG, context1024, batch/ubatch512, tre passi di decode1/100/200 scelti prima delle catture. Prompt: revisione di un estratto C++, non generazione di un programma Python. I token della continuazione sono congelati nel CSV e alimentati uno alla volta, senza sampler nel replay; questo impedisce che una divergenza del token scelto cambi il contesto successivo. Ingressi deterministici non garantiscono da soli bit-repeat dei calcoli, che viene verificato.

Comando modello completo:

```text
replay -m /home/casa/Programmi/modelli/Ornith-1.5-35B-Q4_K_M.gguf -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off --verbosity 4 --load-mode mmap
```

Ambiente pulito: `LC_ALL=C`, `LD_LIBRARY_PATH=VARIANTE`, `MOE_REPLAY_IN=tokens.csv`, `MOE_REPLAY_REPS=1`, `MOE_REPLAY_WARMUP=0`, `MOE_REPLAY_LOGITS_OUT=RAW`; soltanto i due processi ON aggiungono `GGML_CPU_MOE_CAPTURE=PATH`. Non ci sono flag globali MMVQ sui processi modello, pool GPU attivo o callback di tracing. Scope modello24GiB/swap2GiB, riserva RAM6GiB, timeout600s e verifica RX6800/VRAM prima del caricamento.

Ordine dei cinque processi: B1 originale, fork R33 congelato, nuovo motore CPU con osservatore OFF, osservatore ON1, osservatore ON2. Helper replay R34 invariato e identico nelle quattro varianti. BASE e NEW sono le sei librerie modello congelate R34; CAPTURE riusa quelle NEW e sostituisce soltanto libggml-cpu con la build del codice R36. Snapshot dei sorgenti e SHA delle librerie realmente caricate sono conservati. Il primo rebuild Release precede il commit locale e CMake riporta `ab9502a-dirty`; il sorgente CPU congelato coincide byte per byte con il commit91fa630/tree dichiarato. La build ASAN viene configurata dopo il commit91fa630.

5/5 processi producono201 vettori di248320 logits,1005 vettori finiti/nonzero complessivi prima delle SHA. Tutti hanno SHA:

```text
70a46c02e881eccb8f109aeda82cbde51990391b077edd3515c206814f7c0424
```

E' anche la SHA del riferimento CPU R34. Le due catture hanno ciascuna600 record,200 per proiezione,2000 righe di ingresso finite/nonzero,6,611,200 byte. Ordine gate/up/down per ciascun passo; gli8 ID coincidono fra le tre proiezioni dello stesso token. Catture bit-identiche, SHA:

```text
551754a1fa05a983d80bd55fd500f8839da6500c3fc84ee682d87414b212d1dc
```

Gli ingressi selezionati sono ai passi1/100/200, posizioni replay512/611/711. Gate/up condividono l'ingresso: max_abs17.498/8.569/8.152, RMS1.069/0.797/1.008. Down ha max_abs2.102/1.177/1.571 e RMS0.124/0.097/0.126. ID e hash dei singoli file sono nel manifest; la selezione non e' stata adattata ai risultati.

## Replay isolato degli operatori

18 processi registrati: due per B1CPU, forkCPU, B1GPUauto, forkGPUauto, B1GPUF32, B1GPU-MMVQ e B1GPUF32 con inputCPUQ8_K; seconda serie in ordine inverso; quattro DEBUG corrispondenti ai modi GPU. Stesse librerie normali R35, helper R36 compilato contro header B1/fork con `-std=c++17 -O3 -DNDEBUG`. DEBUG sostituisce soltanto Vulkan con la libreria diagnostica R35 e usa il logger dispatch esistente. CPU8thread, grafo diretto con un solo MUL_MAT_ID; fallback GPU->CPU rifiutato.

```text
MOE_OPERATOR_INPUT_PREFIX=inputs/real MOE_OPERATOR_ORACLE_PREFIX=ORACLE MOE_OPERATOR_DATA_PREFIX=DUMP check --operator-numerics --output-bin ACTUAL.bin [--vulkan]
```

Auto: nessun flag. F32: `GGML_VK_DISABLE_MMVQ=1`. MMVQ: `GGML_VK_FORCE_MMVQ=1`. Input CPUQ8_K: `GGML_VK_DISABLE_MMVQ=1` e `MOE_OPERATOR_INPUT_Q8_K=1`. Questi flag globali isolano un operatore soltanto nel fixture; non sono un controllo di precisione mirato del modello completo. Scope8GiB/swap0, timeout180s, clock/telemetria/mappe/driver registrati, nessun ranking dei tempi.

18/18 processi accettati,1296 vettori attuali e2592 vettori oracolo validati. Sette repeat bit-identici, B1/fork CPU e GPUauto esatti, quattro DEBUG identici ai normali. Pesi packed, ingressi originali, ID e oracoli coincidono in tutte le configurazioni; solo gli input effettivi del controllo Q8_K vengono deliberatamente modificati.

Dispatch effettivi, con raw DEBUG esatto:

| Configurazione | Gate/up | Down |
| --- | --- | --- |
| Auto | quantize_q8_1_x4 + mul_mat_vec_id_q4_k_q8_1_f32 | mul_mat_vec_id_q4_k_f32 |
| F32 / F32 con inputCPUQ8_K | mul_mat_vec_id_q4_k_f32 | mul_mat_vec_id_q4_k_f32 |
| MMVQ forzato | quantize_q8_1_x4 + mul_mat_vec_id_q4_k_q8_1_f32 | quantize_q8_1_x4 + mul_mat_vec_id_q4_k_q8_1_f32 |

I tre gate scalari mantengono max_abs<=0.001 e relative_RMS<=0.0001 per ciascun pass: CPU/Q8_K max_abs7.784e-7, maxrelativeRMS2.318e-7; GPUF32/F32 max_abs5.349e-7, maxrelativeRMS2.353e-7; GPUF32 con inputQ8_K/Q8_K max_abs7.769e-7, maxrelativeRMS2.319e-7. Tutti PASS. RMS relativo = RMS errore / RMS riferimento. I limiti sui logits del modello restano argmax>=99%, max_abs<=0.5, maxRMS<=0.05 e maxKLsimmetrico<=0.01; non vengono sostituiti da quelli degli operatori.

RMS rispetto a CPU sui tre pass di ogni proiezione:

| Proiezione | GPUauto originale | GPUF32 originale | GPUF32 inputCPUQ8_K | Riduzione a pipeline F32 fissata |
| --- | ---: | ---: | ---: | ---: |
| gate | 0.0065260363 | 0.0065666585 | 1.43679747e-07 | 45703x |
| up | 0.00658796232 | 0.00660587828 | 1.38812609e-07 | 47588x |
| down | 0.000438748652 | 0.000438748652 | 8.41403715e-09 | 52145x |

Il confronto a pipeline fissata cambia soltanto la precisione degli ingressi e riduce l'errore45-52mila volte. Gate/up senza MMVQ non si avvicinano alla CPU finche' si mantiene l'ingresso F32 originale; la CPU usa la propria conversione Q8_K. Down rimane sulla stessa pipeline F32 anche in auto. I residui dopo l'intervento sono piccoli ma non bit-identici: max_absCPU/GPU gate9.537e-7, up6.407e-7, down5.215e-8. Forzare MMVQ non cambia gate/up e porta RMSdown0.0004462; non e' un fix comune.

## Validazioni e tentativo conservato

Due fixture aggiuntivi ASAN/UBSAN/leak PASS, CPU con osservatore ON e GPU con osservatore OFF:144 vettori attuali e288 oracoli. Risultati esatti rispetto ai normali CPU/GPUauto;9 record CPU equivalgono agli ingressi/ID dei file selezionati. Totale numerico20processi/1440vettori attuali/2880oracoli. Due controlli sintetici separati OFF/ON producono altri144 vettori raw identici a R35. Undici test legacy PASS. Sei input esterni malformati (mancante, troncato, extra byte, NaN, vettore nullo, ID256) rifiutati prima del calcolo con output vuoto. Nessun nuovo file tests/*.

Il primo fixture CPU termina correttamente ma il verificatore delle mappe pretende libllama/libcommon, presenti nel freeze modello e non collegate al fixture operatori. Il tentativo e il verificatore iniziale sono conservati in `attempt1/`, classificati HARNESS_ERROR,0processi accettati. Corretto il verificatore per richiedere le quattro libggml; rieseguiti tutti18 confronti secondo protocollo invariato. Non e' ERRORE_BASELINE. Una parte delle verifiche di compilazione/fixture sintetico/legacy e' stata eseguita mentre i primi controlli modello erano attivi: i tempi raccolti sono diagnostici e non consentono una stima dell'overhead dell'osservatore o un ranking.

Il modello rimane `Ornith-1.5-35B-Q4_K_M.gguf`,21,713,462,848 byte, SHA completa ricontrollata `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`. RX6800/5700X3D/32GB, kernel7.2.9-1-cachyos, Mesa/RADV26.2.4-arch3.1, GCC16.2.1 20260810, CMake4.4.4, Ninja1.13.2, Release/native/OpenMP/Vulkan, buildsharedON/backendDLoff; CPUcompact/activeOFF. Profilo utente2600/1075MHz,-100mV,186W invariato. Commit B1 `7fe450e19305b828c199d602c23a8337aaa1f03b`; fork librerieR33 `61da28a2efc1c1a4ece4d68cd99bea5d3ad08314`; DEBUG VulkanR35 `70dc36356ea4d47810872c45ea764c7acacb579b`. ASAN riusa ggml/base/Vulkan sanitized R35, sostituisce CPU R36 e compila helper con `-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer`, detect_leaks/halt_on_error attivi. Tutte le cache CMake, argv, sorgenti congelati, mappe, SHA e variabili sono nel manifest.

Upstream master ricontrollato al commit `50a6c5cf7c09ea5ea9c1937e4895ce88331c2869`; sorgenti CPU/Vulkan primari scaricati e hashati nel source-audit. Non viene sostituita la baseline ne' attribuito il comportamento congelato a master corrente senza misura. Fonti: [CPU upstream congelato](https://github.com/ggml-org/llama.cpp/blob/50a6c5cf7c09ea5ea9c1937e4895ce88331c2869/ggml/src/ggml-cpu/ggml-cpu.c), [Vulkan upstream congelato](https://github.com/ggml-org/llama.cpp/blob/50a6c5cf7c09ea5ea9c1937e4895ce88331c2869/ggml/src/ggml-vulkan/ggml-vulkan.cpp). Le conclusioni numeriche derivano dai raw locali su RX6800.

## Verdetto e prossimo passo

VALIDATO: osservatore breve esatto, catture ripetibili, replay di pesi/ID/attivazioni reali e precisione isolata nei nove casi. La quantizzazione degli ingressi e' la spiegazione dominante della differenza di questi operatori. Restano soltanto ingressi della traiettoria CPU, tre passi e un layer: non si misura direttamente l'output intermedio del modello o la traiettoria GPU, e il grafo isolato non contiene SiLU, moltiplicazione gate/up, merge, routing o accumulo fra layer.

S09/T0 successivo: controllo opt-in della precisione degli ingressi mirato al layer17 sul modello, prima gate/up/down separati e poi insieme, senza flag MMVQ globali. Il collo di bottiglia affrontato in questa fase e' la fedelta' numerica necessaria prima del placement ibrido, non la velocita'. Su RDNA2/Vulkan l'ipotesi e' far convergere i percorsi di conversione mantenendo separati copia, quantizzazione e dot; un roundtrip o quantizer aggiuntivo puo' costare memoria, dispatch e latenza. Implementare il minimo controllo selettivo e confrontare B1CPU, GPU originale, nuova opzione OFF/ON su identici token512PP+200TG, con repeat/cold/full logits e limiti invariati, poi misurare tempi senza dump soltanto dopo il gate. Non promuovere S07/S08 prima della verifica end-to-end.

Il caso lungo~10000PP+200TG resta ERRORE_BASELINE non ripetibile; R36 non lo ripete o risolve. Nessun nuovo risultato su TTFT applicativo, cache, throughput, concurrency1/2/4/8 o qualita' semantica. Per quella servira' anche una generazione libera greedy di piccoli programmi Python con verifica mediante esecuzione, separata dal replay e dopo il gate numerico. R20-R36:236processi modello cumulativi,5aggiunti qui.

Archivio: `risultati/2026-10-07-real-operator-replay/`. Evidenze portabili: [manifest](moe-real-operator-replay-validation.json). Raw, input packed, compilatori/helper/motori congelati, protocollo, catture, telemetria, mappe, source audit, oracoli e scripts sono conservati localmente; precedente archivio R35 e baseline immutati. Owner dirty files esclusi dai commit.
