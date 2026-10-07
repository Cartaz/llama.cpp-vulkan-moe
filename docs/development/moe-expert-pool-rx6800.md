# R29: pool esperti persistente, controlli isolati RX 6800

Data: 2026-10-07. Codice: `fa4afd46c0e6b5fc8fbf139210e75ef5c051bc44`,
tree `ec99cb4674d625000785a05da02a3db3f1ba4af8`, branch
`experiment/moe-expert-pool`, sopra R28. Metadati e protocollo:
[moe-expert-pool-validation.json](moe-expert-pool-validation.json).

S04/S05 avanzano da planner offline a **prototipo con buffer GPU reali**.
Il pool vive nel controllo operatori `llama-moe-scheduler-check --expert-pool`.
Non e' ancora collegato allo scheduler di inferenza: nessun flag del modello
lo attiva e questo incremento non esegue nuovi processi modello. I 127 processi
R20-R28 restano una campagna separata. M3 rimane parziale.

## Cosa implementa

- Una istanza per layer immutabile, un backend e una lease pendente.
- Slot compatti per la tripletta gate/up/down; tipi F32, Q4_K e Q6_K conservati.
- Budget in byte verificato anche per allineamento e allocazione del backend.
- Upload sincrono dei soli miss, mediante le API GGML esistenti; nessun nuovo shader.
- LRU deterministica: ID ordinati per admission, slot liberi prima dei residenti,
  eta' del batch e indice slot per risolvere i pareggi. Non espellere esperti
  necessari alla richiesta corrente.
- Remap degli ID nell'ordine originale top-k verso i kernel `MUL_MAT_ID` esistenti.
- Lease conservativa dell'intero pool: acquire durante lavoro pendente restituisce
  busy senza modifiche; release e distruzione attendono il backend.
- Richieste oltre capacita', pool non disponibile e routing non supportato
  restituiscono uno stato esplicito. La fixture esegue il fallback CPU con gli
  ID originali; input invalidi vengono rifiutati. Nessuna modifica al router.

Le sorgenti host e il backend devono vivere piu' del pool e i pesi devono restare
immutabili. La classe non e' thread safe, non permette piu' lease simultanee e non
implementa merge hit-GPU/miss-CPU. Il contatore upload misura payload API dei
pesi; non misura traffico PCIe, staging, trasferimenti di attivazioni o ID.

Il bersaglio e' la ricopia di pesi quando esperti host vengono eseguiti sulla GPU.
R20 misura zero upload di esperti nel decode ncmoe18: per accelerarlo bisogna
cambiare anche il placement. Il vantaggio su RDNA2 resta un'ipotesi: conservare
pesi caldi puo' evitare copie, ma consuma VRAM, carica i miss e qui sincronizza
ogni lease. La fixture tiene anche una copia completa GPU soltanto come controllo;
non rappresenta il budget finale della cache sul modello.

## Verifica

Release/native/Ninja, backend Vulkan/RADV, RX 6800 e Ryzen 7 5700X3D.
GCC16.2.1, Mesa26.2.4, kernel7.2.9-1-cachyos; valori completi, CMakeCache,
ambiente e SHA delle librerie effettivamente caricate nel manifest.
Release/ASAN hanno caricato le librerie dai path build /tmp tramite RUNPATH:
ogni SHA e' verificato uguale alla copia archiviata, che manca dei symlink SONAME.
I controlli B1/9375 usano le proprie directory frozen con alias.
Il controllo CPU default, troppo breve per il polling maps, e' ripetuto con
LD_DEBUG=libs: loader registrato e raw invariato; non e' un campione di timing. GPU fisica
verificata; nessun fallback involontario per i grafi residenti. Nessuna modifica
al profilo clock/tensione dell'utente.

| Controllo | Perimetro | Gate |
| --- | --- | --- |
| Pool CPU Release | 48 combinazioni,384 richieste | PASS |
| Pool Vulkan | Stesse48 combinazioni,384 richieste, due processi | PASS, raw ripetibili bit per bit |
| Gate/up/down | 256 grafi residenti +128 fallback per processo | 1152 vettori,78,807,040 byte, ogni vettore finito/nonzero |
| F32 gate/up | Oracolo scalare con valori diadici | Uguaglianza esatta |
| Q4_K e misti Q6_K | Pool contro pesi completi sullo stesso backend | Uguaglianza bit per bit, tolleranza non allargata |
| Fallback | Pesi CPU completi e ID originali | Uguaglianza bit per bit al controllo CPU |
| Management | Capacita'1/2/4/8, budget, geometria, LRU, layer separati, busy, cancel/retry, distruzione pendente | PASS |
| ASAN/UBSAN/leak | Management +3 casi finali/24 richieste: F32 token1, Q6-up token1, Q6-down token3 | PASS nel perimetro; suite Debug intermedia superseded, non dichiarata PASS |
| Modalita' scheduler precedente | CPU completo e Vulkan seriale, nuovo binario contro fixture9375 congelata | Raw bit-identici,4096/2048 byte |

Combinazioni: 256 esperti originali/top8,12 slot; token1/3/33,
broadcast on/off, ID contigui o view con stride16/offset4 byte. Gate/up256x256,
down256x8. Quattro triplette: F32/F32/F32, Q4/Q4/Q4, Q4/Q6/Q4, Q4/Q4/Q6.
Le dimensioni sono sintetiche, non quelle complete di Ornith.
Otto richieste per combinazione: cold, riordino hit, mixed, pressione capacita',
ID duplicati255, duplicati0/255, eviction248..255, riordino hit dopo eviction.

Ogni combinazione verifica20miss e8evictions. Gli hit unici sono28 con1token,
20 con3/33; le richieste duplicate entro top8 fanno bypass senza cambiare cache.
Gli esperti ripetuti fra token sono ammessi. Entrambi i riordini all-hit caricano
**zero byte di pesi**. I payload a12slot sono6,389,760/898,560/1,101,312/904,896 byte
per le quattro triplette; payload e allocazione risultano uguali su queste geometrie,
ma il budget non assume che questo sia sempre vero. I valori ms/token/s sono
**NON_MISURATI**: i tempi del runner sono diagnostici e non campioni prestazionali.

## Controllo fallito conservato

La prima fixture permetteva ID duplicati entro un token anche nei kernel GPU.
A33token, F32, otto copie di255 per token, pool e pesi completi GPU sono
bit-identici ma101,830 valori gate/up falliscono l'oracolo scalare. Il controllo
ridotto compilato contro header e librerie originali B1 `7fe450e19305b828c199d602c23a8337aaa1f03b`
riproduce la stessa discrepanza: **ERRORE_BASELINE sul solo input sintetico dichiarato**.
Primo esempio: gate token8/slot0/row0, GPU `-0.0153808594`, scalare `-0.0161132812`.
Non attribuire questo risultato al normale routing del modello, al driver o a B4.

Il confronto pool/full da solo non avrebbe trovato il difetto. Il nuovo contratto
rifiuta per il percorso residente gli ID duplicati dentro top-k e la fixture li
esegue sulla CPU. Non si cambiano gli ID, i pesi o la tolleranza. I tentativi
iniziali, errori di compilazione, dump e sorgenti diagnostiche restano conservati
in `risultati/2026-10-07-expert-pool/`; le prove fallite non diventano campioni validi.
Il controllo CPU dello stesso input duplicato supera l'oracolo scalare. Sono
verificati i rifiuti per budget e geometria; non sono stati indotti OOM del driver,
pressione KV o concorrenza. Il bypass unavailable e' verificato come stato,
mentre i grafi CPU di fallback coprono capacita' e duplicati.
Lo scheduler upstream al [controllo B4 fissato](https://github.com/ggml-org/llama.cpp/blob/f498f864fbc0472004ee1c3616c1188c68eb157f/ggml/src/ggml-backend.cpp)
e' stato riletto per il punto di integrazione. Non e' stato modificato o testato
con questo input duplicato: nessuna affermazione sul suo comportamento corrente.

## Passo successivo e A/B richiesto

Integrare il pool opzionale per un layer nel percorso scheduler, riusando buffer,
copie e kernel esistenti, senza callback di tracing che alteri i grafi. Garantire
ID originali sul fallback, lifetime fra tutte e tre le proiezioni, allocazione
fallita e ripresa. OFF deve mantenere il comportamento attuale; iniziare da
stream1. T0 sul modello e' un nuovo gate, non ereditato dalla fixture.

Prima del timing: stessi token, placement esplicito, commit/build/libs congelati,
raw finiti/nonzero e confronto B2OFF/ON con B1 separato; includere cold/warm,
all-hit, all-miss, eviction e bypass. Il controllo va fatto sul replay breve
ripetibile prima di estendere il contesto lungo con la sua instabilita' nota.
Poi T1/T3 A/B OFF/ON bilanciato, almeno4processi per variante e3ripetizioni dopo
warmup, cache16/32slot per il singolo layer solo entro budget misurato. Registrare
PP/TG separati, costo miss p50/p95 in ms, lookup/remap, upload API bytes e attese,
VRAM/GTT per-client, MemAvailable e costo init. Profiling diagnostico e timing
restano separati. Estendere layer e concurrency1/2/4/8 dopo la correttezza,
senza scegliere il vincitore da questo prototipo o dalla sola copertura logica.
