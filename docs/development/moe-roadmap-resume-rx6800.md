# Roadmap qualification resumed on RX 6800

Status: RUNNING, 2026-10-09. The owner requested autonomous continuation until roadmap completion or actual exhaustion of the account's five-hour usage window. M0–M3 extended acceptance remains governed by the frozen `2026-10-08-milestones-0-3/acceptance-protocol.json`. Passing a bounded diagnostic does not complete the milestone. Archives are relative to `risultati/`; large dumps and model files stay local.

Baseline B1 remains unmodified v0.5.0 `7fe450e19305b828c199d602c23a8337aaa1f03b`. Candidate controls use frozen R42 engine libraries; multilayer fixtures use the separately frozen current scheduler patch. Every run records executable/library hashes and actual loaded mappings. Model SHA is `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`. Hardware remains RX6800/RADV NAVI21, Ryzen5700X3D,32GB,16GiB BAR0; kernel7.2.9-1-cachyos/Mesa26.2.4/GCC16.2.1. User GPU settings2600/1075MHz,-100mV,186W are preserved. One GPU process at a time;24GiB cgroup memory,2GiB swap,6GiB available-RAM reserve,600s per-process deadline. Record resource failures without ranking them.

## R43b: extended fixed-token correctness

The prior campaign contains50 process results:49 full raw gates pass. One pool128 process at10000PP+200TG diverges in its third repetition, beginning at the13th physical prefill vector. All outputs remain finite/nonzero;208 of220 vectors differ thereafter. Other pool/control repeats do not erase this failure. `2026-10-09-roadmap-resume/long10000-prior-differences.json` preserves vector-level evidence. R47 registers new balanced resident/uncached/pool/pool+Q8-sync/logging controls to investigate this failure, without performance ranking or a causal claim.

## R44: six-family autonomous corpus

The frozen corpus has36 prompts,three TRAIN and three HELDOUT prompts per family,and seeds11/29/47:108 autonomous continuations. Sampling is temperature0.8/top-p0.95/top-k40,max128 tokens or EOG. Consecutive cases clear and synchronize model memory while scheduler pools persist. Baseline resident, candidate same-GPU transfer and pool128 compare full emitted logits and generated IDs. All108 autonomous continuations pass baseline resident/candidate transfer/pool byte parity:18fresh processes and40,086 emitted full raw vectors. All six legacy-callback batches fail autonomous trajectory parity and remain ineligible; see `2026-10-09-roadmap-resume/corpus-results.json`.

Two harness errors are preserved and corrected separately. C++ temperature0.8 is serialized as its actual float representation; the verifier now compares against float32(0.8). A final generated token can end with incomplete UTF-8, causing JSON serialization to abort after valid inference on both baseline and fork. The helper now writes original text bytes to a binary sidecar and uses replacement decoding only for JSON text. IDs/logits are unchanged, including all available prior raw prefixes. The original aborted runs remain archived.

## R45: multilayer scheduler fixtures

Independent pools and explicit GPU placement lists are implemented behind existing opt-in environment variables. Pool syntax extends to `layer:slots:budget_MiB[;...]`; GPU layer selection extends to comma-separated layers. Duplicate layers, malformed syntax,more than32 pools and aggregate budget over2048MiB disable the configuration. Legacy single-layer syntax and OFF remain supported.

Fresh CPU and RX6800 Vulkan fixtures pass for2/4 layers,F32 and Q4_K/Q6_K,changing inputs/strided IDs and reused scheduler plans with1→3→33→3→1 tokens. All12 Vulkan pools actually admit the three expert projections. Forced abort after admission restores graph source pointers; retry is byte-exact. CPU ASAN/UBSAN/leak check passes. The fixture's ID storage is explicitly retained as a graph output: reading an input after its last consumer otherwise tests allocator reuse rather than source immutability. The original failed assertion is preserved as a harness error. The fresh full suite passes25/25 gates:16 Release/ASAN CPU/Vulkan fixtures,three Khronos validation/synchronization fixtures,and six independent captured-CPU/transfer cost processes. All12 pools admit complete gate/up/down triplets in every multilayer Vulkan fixture. Current evidence does not qualify model multilayer output or performance. Shared backend wait durations are per-pool observations and must not be summed as disjoint time. The full transfer arena remains allocated.

## R49: routing callback after the up projection

The original callback requests the `ffn_moe_topk` view. It changes all emitted logits in the three fixed TRAIN trajectories tested on both unmodified B1 and candidate; it is ineligible for normal-path policy selection. The helper adds `MOE_CORPUS_CALLBACK_AFTER_UP=1` together with `MOE_CORPUS_CALLBACK=1`: it waits for the expert up projection and reads that operator's routing-ID source. Inference operators and engine libraries are unchanged. The legacy callback stays available for diagnosis.

Eight registered processes compare B1/candidate OFF,legacy callback,and repeated after-up callback on the same three fixed-token trajectories. Six OFF/after-up controls pass:2070 full raw vectors finite/nonzero and byte-exact. After-up routes are exact across baseline/candidate/repeats,and all naturally observed CPU-side layers0..17 agree with the callback-free candidate route logger. This is an observer-neutrality result on a bounded teacher-forced set,not a speedup or general quality result. An additional108-case autonomous cohort is registered before using these routes for policy analysis.

The initial verifier wrongly required41 normal layers. Model metadata has41 blocks and one next-token-prediction block; `llama_hparams::n_layer()` subtracts that block,so normal inference has40 layers0..39. The corrected audit requires all40 layers/top8 for every token and retains the original verifier failures. MTP is not enabled or qualified. `callback-after-up-audit.json` separates corrected geometry from actual legacy-callback raw failure. Preserving the router's full Vulkan fusion is a hypothesis for why the later boundary helps; this diagnostic alone does not establish causality.

## R50: exact offline analysis at corpus scale

Native route conversion can return all repetitions in one strict parse (`rep=None`),avoiding18 full rereads of each completed corpus profile. Single-repetition output stays compatible. A two-repetition fixture verifies exact output agreement and rejects incomplete observations. The scheduler/profile suite now has12 passing tests.

The complete-request planner partitions remaining request sets by cardinality. Equal-size subset membership is an exact lookup; smaller sets are either scanned or enumerated with combinations,whichever has fewer candidates. Integer gains,Fraction ranking,tie rules and resulting plans are unchanged. An independent subset oracle and full heterogeneous/minimum-reservation allocations agree; all25 profile tests pass. The original CPU-only TRAIN planning attempt was stopped before producing plans or opening HELDOUT because its pairwise subset scan scaled poorly. Frozen policy caps/minima remain unchanged. A fresh TRAIN run is running after isolated CPU/transfer cost fixtures; no inference-performance claim follows from this Python optimization.

An attempted priority pause checked the WORK model but omitted BASE; the baseline was still running. VRAM preflight rejected all25 subsequent fixture requests beforelaunch,with no overlapping model loaded. The baseline guard was resumed,all rejection artifacts retained,and a new archive now runs after checking every frozen model label and VRAM. A subsequent stale pause checkpoint was rejected before touching any process. These are scheduling harness errors,not test failures in the model or pool.

## Remaining acceptance work

R46 registers occupied contexts1/512/2048/8192/16128,ubatch512/1024/2048,Q8/F16 KV,three consecutive cycles and matched B1 resident/transfer/pool controls. State tests cover PP save/restore,rewind16,CPU abort/recovery,complete128-token decode and an independent no-state replay forOFF/1/2/4 GPU layers,including ASAN. R47 investigates the historical long-pool mismatch. Cost fixtures measure actual captured CPU projections and expert-triplet H2D/D2H copies. TRAIN-only policy allocation must be persisted before HELDOUT scoring. Independent balanced timing and CPU/GPU fidelity/quality gates remain separate. M4–M8 retain their roadmap dependencies; no default or integration is promoted by these partial results.

Reproducible local entry points,from the repository root:

```sh
python3 risultati/2026-10-09-roadmap-resume/callback-after-up-audit.py
python3 risultati/2026-10-09-roadmap-resume/corpus-after-up.py
python3 risultati/2026-10-09-roadmap-resume/state-campaign.py
python3 risultati/2026-10-09-roadmap-resume/state-controls.py
python3 risultati/2026-10-09-roadmap-resume/stress-campaign.py
```

Runners refuse to overwrite completed or partial runs. Consult the pipeline/checkpoint before launching any stage; parallel model processes violate this protocol.

## R52: separate warm and cold CPU weight cost

R45 uses repeated captured Q8_K input and selected expert weights; the selected working set can fit CPU caches. Its projection times therefore cannot establish cold expert-weight miss cost. A registered opt-in fixture uses `MOE_OPERATOR_COLD_WEIGHTS=1` to flush every64-byte cache line of selected host weight slices with x86 CLFLUSH/MFENCE before timing. Flush/synchronization before compute are excluded; input/output remain warm. It preserves numerical comparisons for every subset and repetition. Six balanced fresh warm/cold processes are queued against byte-identical archived FIXED engine libraries,with full output compared toR45. This excludes input quantization,routing/transfers/admission and end-to-end hybrid fallback.

CMake regenerated after a Git commit and rebuilt `ggml.c` with its updated `GGML_COMMIT`,changing library hashes. The R52 helper is instead compiled standalone directly against archived FIXED DSOs; rebuilt DSOs are not used as the qualified engine. Its exact command/source/artifact hashes are in `cold-cost-freeze.json`.

A600-second controller lease stops only owned queued experiments when the controller becomes unavailable. Its first180-second configuration expired during active controller work; that partial callback process and CPU planner attempt are preserved. Usage was66%,so this incident is not labeled quota exhaustion. Verified corpus checkpoints are resumed,and TRAIN allocation checkpoints are saved before HELDOUT scoring.

## Extended TRAIN/HELDOUT routing evidence

All21 TRAIN-only static quota allocations are frozen before HELDOUT scoring; two228.87MiB-cap allocations with minimum8 slots/layer are infeasible and remain so. TRAIN and HELDOUT each have54 complete callback-free CPU18 traces and162 per-case/phase analyses with windows1/8/32/128 and cold/prefill-warmed logical policies. HELDOUT scores every fixed allocation without retuning. Payload caps exclude allocator metadata,in-flight tensors,the retained transfer arena,KV and compute buffers.

The planner caches a layer's candidate ranking until that layer's selected experts change,and filters the remaining byte budget every iteration. Exact gains,Fraction ranking and ties stay unchanged. The26-test suite includes540 independently recomputed greedy allocation matrices and600 subset-gain comparisons. Already completed allocations remain preserved; three actual54-case TRAIN allocations spanning all three caps exactly match their earlier fully recomputed versions. No sample,policy,threshold or HELDOUT-dependent selection changes.

Pooled HELDOUT decode coverage for allocations without a minimum reservation:

| Cap MiB | Policy | Byte coverage | Complete per-layer request coverage | Layers with zero quota |
| --- | --- | --- | --- | --- |
| 228.87 | Uniform frequency | 13.32% | 0.00% | 0 |
| 228.87 | Global frequency/byte | 13.86% | 0.00% | 2 |
| 228.87 | Request bundles | 4.34% | 3.01% | 17 |
| 490.43 | Uniform frequency | 23.38% | 0.01% | 0 |
| 490.43 | Global frequency/byte | 23.80% | 0.03% | 0 |
| 490.43 | Request bundles | 6.55% | 6.01% | 16 |
| 1013.55 | Uniform frequency | 37.78% | 0.58% | 0 |
| 1013.55 | Global frequency/byte | 38.15% | 0.97% | 0 |
| 1013.55 | Request bundles | 13.83% | 11.78% | 15 |

These are logical static payload simulations,not runtime hit rates,transferred PCIe bytes or speed predictions. Complete-request greedy concentrates coverage in few layers and trades byte coverage for all-hit requests. No heldout token is fully covered across all18 selected layers by any feasible allocation. Low quotas can also fail an actual eight-expert pool admission; this simulation does not waive that runtime condition.

## R47 baseline instability and R55 async controls

The new R47 initial unmodified B1 resident process is itself not repeatable:rep0/1 agree;rep2 first diverges at physical prefill vector6/position3072.214/220 vectors differ,finite,first max absolute difference0.197236 and RMSE0.028863;argmax agrees throughout. This is a shared-reference failure,not evidence that the fork caused that particular divergence. It does not establish the same cause as the earlier R43b pool failure. A new Q8-FA synchronization-guard condition also diverges,so that existing opt-in barrier is not sufficient under the observed condition. All original gates remain unchanged and the cell is excluded from ranking.

The initial unstable B1 leaves the registered R47 online comparison without an eligible reference; subsequent empty assertion errors are retained as ineligible cross-reference checks,not automatically fork regressions. Offline within-process differences distinguish these from actual repeatability failures.

[Upstream issue25195](https://github.com/ggml-org/llama.cpp/issues/25195) reports an async transfer hazard on gfx1201/amdvlk and explicitly does not reproduce a native crash on RADV. It is a research lead,not evidence of this RX6800 failure. Current upstream de7fa0a and B1 source both expose `GGML_VK_DISABLE_ASYNC`; the source was inspected after an actual Gitfetch. R55 registers three balanced fresh processes each for B1resident/pool128,asyncON/OFF,plus B1ON/OFF model synchronization validation. Full raw repeatability and a predetermined chronological R43 B1reference are required. It uses existing runtime flags and unchanged archived engines,with no default promotion or performance ranking.

## R47 completed after quota recovery

The second continuation preserves the interrupted baseline files and verifies hashes for all 11 completed processes before skipping them. The final baseline and two logging controls complete the registered 14-process diagnostic. A separate complete raw audit uses the predetermined chronological R43 B1 reference:12/14 processes are fully repeatable and reference-exact. Initial B1 and the first Q8-sync condition remain actual within-process failures; respectively 214/220 and 210/220 vectors differ in rep 2, with no changed argmax in this fixed continuation. This does not establish semantic equivalence or a fix.

The original unstable B1 left the online comparator without a reference. Those later online failures stay archived; the descriptive audit identifies which processes were nevertheless fully repeatable and exact. The resumed three processes use unchanged first-repetition B1 bytes, and the independent audit verifies them against the separately frozen chronological R43 reference. No previously failed process becomes eligible for timing.

The first logging process records 880 layer 17 calls including warmup:all 20 prefill chunks per repetition exceed capacity and use the original transfer path with no remapped projections or pool uploads. Each 200-token decode uses all three projections. The first measured decode has 1462 logical hits, 138 misses/evictions and 244,187,136 uploaded bytes. These are actual pool counters in this logging control, not timing evidence or proof of the cause in a different unlogged failure. Pool payload/allocation metadata and the full transfer arena remain distinct.

R55 continues with the registered balanced async ON/OFF controls and full-model synchronization validation; cold costs, state/no-state, autonomous observer, timing and context/KV pressure are queued serially. Current upstream 8a1a9b5 was fetched and inspected; the delta from de7fa0a changes CUDA/SYCL/Hexagon and backend tests, with no Vulkan source change adopted. M0-M3 acceptance remains partial. Portable evidence: [R47 summary](data/moe-roadmap-resume-20261009/r47-completed-summary.json).

## R55 completed: async OFF does not contain the long failure

The twelve balanced normal processes are complete:all six B1 resident ON/OFF processes pass full raw repeatability and the frozen chronological R43 reference. Pool ON process1 and pool OFF process3 fail within-process byte parity in their first repetition; remaining repetitions and four other pool processes match. ON first differs at prefill vector4/position2559 (216/220 vectors; maxabs1.116323,RMSE0.233625). OFF first differs at vector16/position8703 (204/220 vectors; maxabs0.156506,RMSE0.0260143). Argmax is unchanged in these particular fixed trajectories; the byte correctness gate remains failed. Existing async disabling is not a demonstrated workaround. Six passing B1 processes do not erase the separate R47 B1 failure.

The two additional B1 synchronization-validation processes ON/OFF each complete one repetition without warmup and match the same frozen raw reference. Khronos layer mappings confirm that validation loaded. An independent audit scans both stdout and stderr:zero reported synchronization/validation errors. R45 validation fixtures were also audited across both streams, with zero errors in all three. The layer default duplicate-message limit remains10:message counts are lower bounds, and absence of reports does not prove absence of a race. R57 registers full pool ON/OFF validation with a dedicated validation file; it has no timing ranking. [Portable R55 evidence](data/moe-roadmap-resume-20261009/r55-completed-summary.json).

## R56: membership changes do not precede the first logit difference

Thirteen fresh processes capture all40x8 ordered routes over the original512PP+200TG teacher-forced workload, using the exact archived R39 engine DSOs and the existing after-up observer. Every observed full raw dump is byte-exact to its corresponding unobserved R40 intervention:2613 vectors,finite/nonzero; original/copy routes match and the all-projection repeat is exact. No kernel or engine was rebuilt for this attribution diagnostic.

For gate/up/all roundtrip interventions,the first logit difference is token512. At that token layer39 swaps ranks2/3 of the same experts154/110; expert membership first changes at token513. For down-only,first logit difference is token513,first ordered-route difference token514,first membership difference token515/layer36. Thus membership change is not necessary for the first observed logit amplification; even ordered routing is unchanged at the first down-only difference. This does not quantify the contribution of routing probabilities,intermediates or stored attention state,and does not explain the independent intermittent long-context failures. All CPU/GPU fidelity gates remain unchanged. [Portable attribution evidence](data/moe-roadmap-resume-20261009/r56-routing-attribution.json).

## R52: isolated cold CPU weight cost complete

All six standalone fixture processes (three warm,three CLFLUSH weight-cold) pass output byte parity to the archived R45 fixture. For eight selected experts,the median of independent process medians is down42-43us warm versus103.5-104us cold;gate39-43us versus111-113us;up37-58us versus110-112.5us across three retained passes. Flushing is outside the timer. These measure actual capturedQ8_K projection+sync work and exclude input quantization,routing,activation transfers,cache admission and end-to-end fallback. Warm variation is retained. No hybrid-speed claim follows. [Portable cost evidence](data/moe-roadmap-resume-20261009/r52-cold-cpu-summary.json).

## State helper amendment

The original R46 helper expected immediate restored cached logits. Both B1 and candidate state serialization code write model architecture and memory;they do not serialize output logits,despite the API header comment. Original return4 failures and partial files stay archived. A separately frozen helper now requires full byte consumption and byte-exact reserialization,then validates16 rewound logits,CPU abort/recovery and the full128-token continuation against an independent no-state trajectory. It links byte-identical archived B1/FIXED/ASAN engines. The two512/8192 prefixes,18 state variants and16 no-state controls remain required; numerical thresholds are unchanged. A missing new-label manifest entry caused a prelaunch harness failure and was corrected before state model execution;that attempt is preserved separately. State v2 qualification is running and is not yet accepted.

The initial successful state result is excluded:the previous controller was launched with a relative script path and survived an absolute-argument process match. It advanced to R57 while the replacement controller ran the first state case. VRAM guards rejected subsequent launches,but the first state/R57 pair overlapped. All collided state/control/observer/stress files and R57 partials are preserved under collision04 archives. The owner did not cause this incident. No benchmark timing is accepted. Controller05 holds an exclusive file lock;Python process ownership is resolved against /proc/PID/cwd,and all owned units were stopped before restarting the unchanged protocol serially. Source/library hashes and original failures remain unchanged.

Controller05 completes all nine512-prefix state cases,including four-layer ASAN/UBSAN:both restored state snapshots are byte-exact;16 rewound outputs and all128 primary decode outputs match same-placement B1. The8192-prefix matrix and16 independent no-state trajectories are pending,so the state milestone remains partial. Exact argv/environment arrays are in [command manifests](data/moe-roadmap-resume-20261009/continuation-command-manifests.json);[state protocol](data/moe-roadmap-resume-20261009/state-protocol-v2.json) and [standalone builds/engine hashes](data/moe-roadmap-resume-20261009/state-freeze-v2.json) retain reproducibility. No engine behavior or public API contract is changed by the helper correction.

## State v2 complete:18 lifecycle and16 independent controls

Controller05 completes all18 state variants and all16 independent no-state trajectories serially. Both512 and8192 prefixes use c16384,Q8KV,b/ub512,128 teacher-forced decode tokens and the original frozen placement choices OFF/1/2/4GPU layers;the four-layer sanitizer controls pass at both depths. All36 restore operations consume and reserialize the same bytes;all288 rewound outputs are exact. Primary state and independent no-state trajectories contribute4641 finite/nonzero full raw vectors and match byte-exact at each placement. Actual engine mappings/hashes and common/public headers match the archived B1/FIXED/ASAN sources. This closes the amended bounded state gate,not broadM3 acceptance or the separate10000-token intermittent failure. Original helper/collision failures stay preserved. [State evidence](data/moe-roadmap-resume-20261009/state-v2-completed-summary.json),[header identity](data/moe-roadmap-resume-20261009/state-v2-header-identity.json).

## R57 complete:divergence under active synchronization validation

Both pool ON/OFF validation processes complete warmup plus3repetitions at10000PP+200TG with unchanged frozen engines. Khronos layer loading is confirmed. The dedicated validation file is empty,and stdout/stderr report no validation or synchronization errors. Each stdout begins with one informational line announcing that log file,which makes the original strict CSV parser reject its header. Original online failures are retained. A separate derived CSV removes only that exact known prefix and strictly validates all660 inference rows;complete raw/mappings/validation streams are audited independently.

ON rep0 first differs at physical prefill vector11,last position6143:209/220 vectors differ thereafter;first maxabs0.229533,RMSE0.048247,and every vocabulary element differs in that first vector. Argmax agrees throughout this particular fixed continuation. ON reps1/2 match the frozen reference;OFF all3reps are fully repeatable/reference-exact. Thus a full raw divergence occurs with synchronization validation active and no reported error. This does not exclude a race or prove the cause;default duplicate suppression10 is retained. R55's separate OFF failure still prevents accepting asyncOFF as containment. No timing ranking. [Validation evidence](data/moe-roadmap-resume-20261009/r57-validation-completed-summary.json).

The full autonomous40-layer after-up observer cohort is running. A separately frozen R58 prototype adds final selected-expert coefficient bit capture at the same callback stage,without changing engine DSOs or the repository helper. It will run serially after the observer cohort and all CPU analysis exit,before registered timing. Any raw/route/geometry/identity/repeat failure stops attribution;no coefficient hypothesis is accepted before those gates.

## Autonomous after-up observer complete

All six fresh batches pass108autonomous TRAIN/HELDOUT continuations:13,362 full raw vectors and generated token IDs byte-exact to the corresponding B1resident trajectories. Ordered40x8 routes are complete,and all108 CPU18subtraces exactly match the independently archived callback-free native profiles. All324case/phase analyses complete and all worker subprocesses exit before subsequent model timing. The legacy callback's six autonomous failures remain archived. This closes the observer-neutrality/native40 profiling cohort,not extended turn/context,CPU/GPU fidelity or broaderT2/M2 acceptance. Logical cold/warm/static-prefill simulations remain separate from runtime transfers/cache measurements. [Observer evidence](data/moe-roadmap-resume-20261009/observer-completed-summary.json).

## R58:continuous router coefficients captured neutrally

A standalone helper prototype remembers final selected-expert weight tensor identities during existing ask callbacks,then reads F32coefficient bits at the already-qualified after-up callback stage. It adds no callback stop point or tensor writes;extra tensor readbacks stay inside that stage. The repository's active corpus helper and engine DSOs are unchanged. All13OFF/ON/repeat processes preserve2,613 complete raw vectors against their respective unobserved R40 interventions;all ordered routes match R56. Each of the seven observed processes captures227,840 finite final coefficient records with full40x8x712geometry. Original/copy coefficients and all-projection repeat coefficients are exact. [Protocol](data/moe-roadmap-resume-20261009/r58-protocol.json),[standalone build and engine hashes](data/moe-roadmap-resume-20261009/r58-freeze.json),[qualified source snapshot](data/moe-roadmap-resume-20261009/r58-corpus-observer.cpp).

First coefficient differences occur at layer18 in the same token as the first logit difference:gate/up/all token512,down token513. The first same-expert coefficient changes are2.98e-8,5.96e-8,1.49e-8 and1.49e-8 respectively. No earlier coefficient difference is observed. This localizes continuous propagation before expert membership changes;it does not explain the magnitude of later logit amplification or establish routing causality. Initial tiny differences and first violations of the existing R39fidelity envelope will be distinguished retrospectively after timing exits. Original numerical thresholds are unchanged. [Coefficient evidence](data/moe-roadmap-resume-20261009/r58-router-coefficient-summary.json).

## Independent timing input amendment

The original eighteen R51raw gate attempts abort before model initialization:CSVwriter's defaultCRLF header is rejected by the frozen replay helper's exactLFheader comparison. The resulting online library-mapping errors reflect processes too short to observe maps,not evidence of a loaded wrong engine. Original logs/manifests/results remain preserved and are ineligible. R51v2 changes only CSVline endings,verifies the exactLFheader and512/128token counts before launching,and retains all frozen engines,flags,orders,thresholds,raw gates and bootstrap rules. It starts at a verified interval between two complete stress processes;no live model is interrupted. The stress child/central parent remain paused until all timing and its CPUanalysis exit,then resume the registered stress order. [Timing amendment](data/moe-roadmap-resume-20261009/timing-protocol-v2.json).

The first pressure baseline process completes all15cases but fails cross-cycle raw parity. The uncached target control passes against the recorded first baseline trajectories;that PASS does not restore the unstable baseline cell's timing eligibility. Full pressure and independent timing qualification remain running.

The original state-output requirement is still unsatisfied on both B1 and candidate:the API header mentions cached logits/embeddings,but the inspected implementations serialize architecture and memory only. The amended18+16PASS gate is deliberately limited to serialized memory and subsequent teacher-forced continuation. It does not validate the first autonomous sample immediately after restore or restore sampler RNG/history. Original output-restoration failures remain a shared contract/documentation gap;they are not erased as mere parser noise,and the original acceptance protocol is not closed by the amended gate. A real client checkpoint must separately preserve valid output logits and sampler state or use a supported reevaluation policy before sampling. No such client workaround is promoted here.

## R51 v2: independent short multilayer timing completed

All54 registered processes pass:18 independent raw gates and36 independent timing runs across1/2/4 selected layers. The raw gates contain6966 finite,nonzero full-vocabulary vectors,byte-exact to same-placement B1. Timing uses four fresh processes per variant with one warmup and three measured repetitions,balanced resident/transfer/pool order,fixed512PP+128TG,c16384,Q8KV,b/ub512,FAon,mmap,fitOFF,t8/tb8. Pool logging,callbacks,raw writes,sanitizers and CPU analysis are absent from timing. Process rates aggregate token/time before taking independent-process medians. Bootstrap95% uses50000 draws,seed51 and the preregistered3% PP/TG and5% p95 margins. Resident drift stays below3.1%.

| Selected layers | TG resident / transfer / pool, tokens/s | Pool TG vs transfer, effect [95% CI] | Pool TG vs resident, effect [95% CI] | Pool PP vs resident | Median sampled peak VRAM saved vs resident |
| --- | --- | --- | --- | --- | --- |
|17|31.14 /27.28 /29.83|+9.35% [+7.65,+10.42]|-4.21% [-5.61,-3.11]|-4.08%|0.210GiB|
|16,17|32.16 /24.67 /29.73|+20.54% [+18.81,+22.12]|-7.54% [-9.79,-5.98]|-8.54%|0.455GiB|
|14,15,16,17|34.29 /20.45 /29.95|+46.44% [+45.69,+47.09]|-12.64% [-13.74,-11.43]|-15.56%|0.881GiB|

All three pool-vs-transfer TG intervals qualify as improvement. All three pool-vs-resident PP/TG intervals qualify as regression. Pool-vs-transfer PP is unchanged within the3% margin. TG p95 improves against transfer for2/4 layers; the1-layer interval crosses the5% practical boundary. Against resident,pool p95 is inconclusive for1/2 layers and regresses for4 layers. These are engine replay measurements,not application TTFT or agent throughput. The sampled VRAM values are total device peaks including desktop/driver allocations,not an allocator guarantee; actual pool payload is216MiB on layers14/15/17 and249MiB on layer16. Full transfer arenas remain allocated. The short gate does not erase R43/R47/R55/R57 long failures and cannot promote a default.

Original CRLF pre-model failures remain preserved. The v2 input amendment changes only line serialization; exact tokens,flags,engines,order,thresholds and all failures remain recorded. Timing occupied a verified idle boundary between complete stress processes; all CPU analysis exited before resuming the original stress queue. Portable protocol,freeze,command manifests and complete confidence intervals accompany this report.

## R59: magnitude of the roundtrip logit divergence

This retrospective analysis reuses all201 full-vector F64 metric rows for each R40 projection intervention and the original R39 limits:maximum absolute error0.5,RMS0.05,symmetric KL0.01 and whole-trajectory argmax agreement0.99. It introduces no new model runs,epsilon or acceptance thresholds.

| Intervention | First different logit token | First RMS >0.05 | First max abs >0.5 | First symmetric KL >0.01 | First membership change |
| --- | --- | --- | --- | --- | --- |
|gate|512|513|515|593|513|
|up|512|512|513|683|513|
|down|513|515|515|593|515|
|all|512|513|515|593|513|

The up intervention already fails the frozen RMS envelope at token512,one token before membership changes. Its first different output has maximum absolute error0.2377 and RMS0.0603 over248320 logits. The tiny first coefficient changes observed by R58 therefore do not imply tiny output error. All four interventions fail the original whole-trajectory fidelity screen. Temporal order alone does not identify the amplification mechanism:intermediate scales,weighted reductions and attention/recurrent state remain to be separated. Full numerical series are published; the optional plot was skipped because Matplotlib is unavailable in both available Python runtimes.

## Occupied-context pressure progress

The original36-process Q8/F16 KV and ubatch512/1024/2048 campaign remains running in its registered order. The first B1 resident process completes all15 cases but fails raw repeatability:only depth16128 cycle1 differs from its first occurrence,starting at physical prefill vector11 (zero-based),token6143. First-vector maximum error0.159593,RMS0.031174;argmax remains equal. Candidate transfer completes all15 cases byte-exact to the first B1 trajectories. That pass does not repair the shared baseline instability or make this pressure cell eligible for a performance ranking. All raw files and the independent full-vector audit remain local.

## R51 runtime counters and complete-request coverage

The six separate raw gate processes contain warmup plus three128-token TG repetitions for each selected layer. All TG calls admit full triplets; hits+misses equals8 perlayer/token. Warm repetitions achieve91.7-95.8% per-expert hits. Entire selected-layer requests are all-hit in79/128 tokens for one layer (61.72%),47/128 for two (36.72%),and25/26/27 of128 for four (19.53/20.31/21.09%). Both fresh raw processes agree exactly. Warmup complete-request coverage is lower:42.19%,18.75%,8.59%. These actual runtime counters differ from logical policy scores and are absent from measured timing runs. Their association with speed does not prove which miss/admission/transfer component dominates.

Each512-token prefill call falls back for capacity. Pool-upload bytes exclude ordinary fallback transfers and the full arena,so they are not total PCIe traffic. The warm layer17 pool uploads113.06MiB per128 TG tokens; heterogeneous layer16 uploads163.41-165.35MiB. No backend-wide wait duration is summed across pools.

## R60: autonomous client checkpoint registered

A separate prototype is linked to byte-identical frozen B1/FIXED engine libraries; no engine or active corpus helper changes. It compares six independent no-state processes and six client-checkpoint processes in the same balanced resident/transfer/pool/pool/transfer/resident order,each using exact R46 prefixes512/8192 and seeds11/29/47. The CPU common sampler retains its actual defaults:temperature0.8,top-k40,top-p0.95,min-p0.05,repeat penalty1.0;64 autonomous tokens or EOG. All four selected layers14-17 use the same GPU placement.

The client checkpoints before its first sample and after16 generated tokens. It saves engine memory,the last logit vector and a clone of common_sampler. On restore,it verifies consumed/reserialized memory bytes,copies the retained output into the exported mutable last-logit buffer,and clones the saved sampler. The first sampled ID and every later autonomous ID/full logit vector must match the original trajectory and an independent no-state B1 control. An early EOG that prevents the second checkpoint remains a failed coverage gate.

This explicitly tests a caller-side checkpoint. The original API cached-output restoration gate remains shared/unimplemented. Only a single last-output vector and CPU sampler without grammar/backend sampling are covered;disk sampler serialization,embeddings,multi-sequence/output metadata,active penalties and broader API repair are excluded. The campaign starts at a verified idle boundary after five complete stress processes and resumes the original stress order afterward. Status:RUNNING;no acceptance claim before all registered controls complete.

## Replay source provenance correction

The generic helper_source_sha256 field in the preserved R51 freeze/runner manifests was inherited from the scheduler fixture and identifies expert-pool-check.cpp,not the replay source. A separate audit now records the actual moe-replay.cpp SHA and proves it matches both the archived R41 guarded helper and commit d6844bd. The candidate scheduler build-source SHA matches committed snapshot5779fcd; unchanged B1 libraries remain identified separately. Original manifests are preserved,including this metadata error; executable/loaded-library hashes and numerical gates are unchanged. Source equality does not claim reproducible compilation or complete M0.

Current upstream8a1a9b5 was also inspected for checkpoint design. Its [state serializer](https://github.com/ggml-org/llama.cpp/blob/8a1a9b5126126e5228b95fa909d4b08fac65e8b3/src/llama-context.cpp#L3501) writes model architecture and memory. The [distribution sampler clone](https://github.com/ggml-org/llama.cpp/blob/8a1a9b5126126e5228b95fa909d4b08fac65e8b3/src/llama-sampler.cpp) copies RNG state,and [common_sampler_clone](https://github.com/ggml-org/llama.cpp/blob/8a1a9b5126126e5228b95fa909d4b08fac65e8b3/common/sampling.cpp) also copies accepted-token history. These code observations guide the bounded client prototype;no upstream fix or generic state compatibility is inferred.

R60 no-state controls complete6/6:four pass and two fail. Pool process1 differs only on prefix8192/seed11 in22/80 full vectors,first prefill vector4/token2559 (maximum error0.034292,RMS0.006309,248295 differing elements); all64 generated IDs remain identical. At that call all four pools report capacity fallback,zero remapped projections and zero pool-upload bytes;their payloads remain allocated from prior TG. Transfer process2 differs on prefix8192/seed29 from prefill vector13/token7167 (first maximum error0.191669,RMS0.032816);67/80 vectors differ and first generated ID changes at index28. Both resident controls are exact. Pool activity is not necessary for the failure in this cohort;the earlier long B1 failures remain separate evidence,not an excuse to pass these gates. Checkpoint processes continue as registered;first resident and transfer checkpoint controls pass,without repairing no-state qualification.

The client prototype is restricted to restoring into the same live context with one last output vector. It does not restore output bookkeeping in a new context or guarantee cross-context/model compatibility.

## R60 completed outcome

All12 registered model processes complete:10 PASS and two no-state controls FAIL. All six client-checkpoint processes pass36 autonomous cases with72 restored snapshots,including the first sampled ID after both checkpoints. Primary outputs contain5220 full raw vectors; restored continuations verify4032 full vectors. All72 memory snapshots are consumed and reserialized byte-exactly. In72 of72 restorations,the output buffer before the caller's copy differs from its checkpoint output. Both checkpoint pool processes admit triplets on all four layers.

The two no-state failures above remain explicit;the overall R60 gate is FAIL_INDEPENDENT_NO_STATE_CONTROLS. No failure replacement or default promotion. Caller-side resumption is validated only in this bounded same-context scope;the original engine cached-output API requirement remains shared/unimplemented. The original pressure campaign resumes after12 models,with five complete processes already retained.

## Resume06 and R61 preregistration

Resume06 reused five completed pressure processes only after verifying all archived files, original full-raw audit hashes, loaded engine libraries, fixed inputs, model hash and user dirty-file hashes. The quota-interrupted sixth folder remains unchanged. Its fresh suffix completed and failed raw parity too. At 2026-10-09T12:55:21.646476+00:00, 10/36 original pressure processes are complete (8 PASS, 2 FAIL); later exact controls do not repair failed baseline cells. The original registered order continues.

A separate source audit corrects the inherited stress helper metadata: the frozen7110f1 corpus binary has compile manifests matching source94a976 at2e08b57, not the earlier UTF8-only cfcda9/ed7a helper or expert-pool fixture76f4b5. Original metadata remains unchanged; the correction does not establish reproducible-bit compilation or close M0. Current upstream609290be has8 new commits since8a1a9b5, with no change in the selected Vulkan/scheduler/KV/state/sampling paths. No adoption.

R61 implements S14 as optional single-token pool admission: absent `GGML_SCHED_EXPERT_POOL_DECODE_ONLY` preserves existing behavior; presence, including value0, bypasses admission when the MoE IDs tensor has more than one token. It uses the existing copy fallback. Single-token prompt chunks can still admit, and batched decode also bypasses. This avoids possible pool uploads/remap/churn, but does not remove all prompt transfers and can increase cold TG misses. Resident payload remains allocated.

The engine delta is8 lines. Existing multilayer fixtures still verify all raw outputs, resizing and graph-source restoration; the injected cache abort must fire on eligible single-token shapes and must not fire on bypassed batches. Twelve CPU Release/ASAN controls pass full raw equality to the frozen prior candidate; these CPU controls do not exercise the Vulkan admission mechanism. Final fixture amendment only changes the Vulkan fault-hook expectations. GPU validation, model raw gates and timing are queued behind the completed original pressure campaign and its independent audit, under the same exclusive lock.

All7 R42 input files are registered before execution: short512+200 and all6 prior heldout families with64-token continuations, one prompt/family previously inspected.28 raw gates require same-GPU B1 parity and three repeat outputs. Only after all raw gates pass,112 separate timing processes provide4 fresh processes per variant and cell, balanced orders, warmup1+3rep, process bootstrap50k/seed61 and unchanged3% rate/duration,5% TGp95 margins. PP, TG and complete PP+TG decode duration are reported separately; this is not app TTFT, a cold-start speedup or general quality qualification. Pool counters use separate raw runs. No feature/default promotion or M0-M3 closure. Protocol/source/build identities and current progress are in `data/moe-roadmap-resume-20261009/`.

## Resume07: pressure recovery and prelaunch audits

The owner resumed work after an explicit pause at17/36 pressure processes (14PASS,3FAIL). All files of these17 runs, frozen engines, input/protocol, model and user dirty files were hash-verified. The interrupted Q8/ub2048/resident2 folder stays unchanged; a fresh resume07 suffix is used. The runner reconstructs the original first resident1 reference for every reused KV/ubatch group, including groups that were already partly complete. This corrects restart reference handling without changing engines, inputs, order or gates. The resumed resident2 run passes the original Q8/ub2048 reference. The first F16/ub512 resident also passes. Later progress is in the live checkpoint; the portable summary is explicitly a running snapshot.

The independent pressure audit will classify all36 registered attempts, including nonzero exits and unavailable first baseline references. It will separately report finite/nonzero vectors, bitwise differences and first divergence at the original output geometry. Missing references are never replaced by a later candidate. Its DESCRIPTIVE_COMPLETE status means all records were inspected, not that the pressure gate passed. The previous audit source and all original gates remain archived.

R61 source review confirms the shape bypass occurs before packed-ID construction and pool request/remap/acquisition. The scheduler's earlier route readback and used-ID scan, preparation and original copy fallback remain. No VRAM saving is claimed. GPU fixtures now inspect validation stdout as well as stderr and the validation file, and verify that the extracted validation layer was loaded. This runner amendment preceded all R61 GPU/model/timing measurements. R61 waits for the original pressure campaign and all36 audit records under the exclusive lock.

R62 is a separate B1 native-prefill diagnostic, registered and audited before launch. Eight fixed cases include two90-record tasks, four short controls and two long duplicates. Structured retrieval and code-trace oracles, duplicate identity, source, binary and B1 library hashes pass the prelaunch audit. It crosses CPU17/GPU17 weight placement with logical application batches versus legacy physical chunks, in two balanced fresh processes per configuration. Every emitted pre-sampling vector is retained and checked; invalid cases remain failures. The64-token cap does not qualify semantic answer quality. Differing autonomous histories are not a matched-input causal contrast. It starts only after R61 has joined, with one model process.

Upstream50e3e3e was checked from current code plus issues/PRs. The nine new commits include RMS workgroup overflow correction5e4878e/PR30145, static backend-sampler topology a518119/PR30223 and chat API refactoring8b54361/PR30210. The RMS change clamps dispatch dimensions and loops over excess channels/samples; no captured local shape establishes it as the cause of the archived divergences. Current helpers use CPU sampling, and all engines remain frozen. Any adoption needs a separate B4 gate and A/B on RX6800; no speedup or correctness fix is inferred here.

## R46 pressure completed: original gates retained

All36 registered processes finish with return0 and all540 fixed-token cases:32PASS and4FAIL. An independent audit checks72,612 complete248,320-element vectors against the original first-resident references. All vectors are finite/nonzero;569 differ and three later argmax positions change. All four failed processes diverge in prefill, and the first mismatching vector retains its argmax:

| Process | Case | First vector, zero based | Prefill token, zero based | Different vectors | Later argmax changes | First max abs / RMS |
| --- | --- | --- | --- | --- | --- | --- |
|Q8/ub512/resident1|depth16128/rep1|11|6143|149|1|0.159593/0.031174|
|Q8/ub512/resident2|depth16128/rep1|13|7167|147|0|1.612819/0.312954|
|Q8/ub2048/pool2|depth16128/rep1|5|12287|131|1|0.780067/0.134448|
|F16/ub512/resident2|depth8192/rep2|2|1535|142|1|0.765440/0.144926|

The other cases in these four processes are reference-exact. Every same-GPU uncached transfer process passes the original gate in this campaign. The two controls for each placement in Q8/ub1024,F16/ub1024 and F16/ub2048 also pass. These are bounded observations, not equivalent reliability or a causal proof. Three failed processes use unmodified B1 without a pool; the one pool failure remains separately ineligible. Exact later controls do not repair either kind of failure. Teacher-forced IDs are fixed by design, so later argmax changes do not represent generated-token parity.

Overall pressure qualification is FAIL_EXTENDED_PRESSURE. No timing ranking, new default or M0-M3 closure follows. Original output/protocol/order/partial folders remain; `pressure-completed-summary.json` and `stress-independent-audit-resume07.json` contain the full per-case hashes, first-difference geometry and separate finite checks. R61 GPU fixtures start after this completed audit under the same exclusive lock; its independent short gate and timing remain separate.

## R61 GPU mechanism gate

All12 fresh Vulkan fixture processes pass full raw parity to the previous candidate, including all four Khronos synchronization-validation conditions. An independent parser also verifies finite/nonzero float32 raw files, loaded validation libraries and both output streams. ON confirms48 batched skips in each scheduler fixture and288 in each multilayer fixture, with zero pool projection/hit/miss/eviction/upload counters on every skip. Single-token triplets are admitted in every fixture. OFF retains batched admission. Input resize,abort/retry and graph-source restoration remain checked by the existing helpers. Together with the12 prior CPU Release/ASAN controls, this closes the bounded mechanism gate, not model correctness or performance. All28 preregistered raw model gates are now running; timing remains gated on all seven replay cells.

## R61 model gate and R63 preregistration

All28 R61 model processes pass the original raw gate:7,092 complete vectors across seven archived inputs and resident/old/new-OFF/new-ON variants, with three measured repetitions per process. They are finite/nonzero and byte-exact to the same-GPU resident B1. Separate counters show batched ON calls bypass pool admission, while actual single-token triplets remain admitted. The112 independent warm timing processes are still running; full independent raw/CSV/CI audit waits for all R61/R62 GPU workers to join. No performance verdict is available at this checkpoint.

R63 preregisters an empty-pool first-replay comparison using the exact frozen R61 artifacts. Planning and structured inputs were selected from the earlier routing/churn observations; the512+200 replay is the negative control. Each of four variants runs four fresh processes per input. All48 complete raw gates must match the original first cold B1 and archived warm B1, and pass a separate full-vector audit, before48 independent timing processes are admitted. Common and replay warmup are OFF, with one replay per process. PP,TG,PP+TG decode duration and TGp95 use the same practical margins and process bootstrap as R61, with seed63. Driver and filesystem caches are uncontrolled and already warm; this tests empty expert pool/context, not model cold-start or application TTFT.

ON can remove prefill pool churn but may increase first-TG misses; zero pool upload is not zero total PCIe traffic, and persistent payload remains allocated. R63 waits for R61/R62 to join and for the root post-run audits, under the existing exclusive lock and memory/device/time guards. The protocol, runner sources and source hashes are frozen before any R63 inference. All cells, failures and original references remain reportable. No new default or M0-M3 acceptance follows.

## R61 warm replay completed and independently audited

All112 independent timing processes pass the original B1 fingerprint gate. A separate audit re-reads every original CSV, verifies isolated no-raw/no-counter environments and actual loaded DSOs, and recalculates PP/TG sums, p95, drift, medians and the50,000 process-level bootstrap draws. All seven cells remain eligible under the preregistered20% resident drift limit. Full raw audit independently passes28 model processes and7,092 finite/nonzero complete vectors.

ON versus new OFF, percent effect and95% process-bootstrap interval:

| Fixed input | PP tokens/s | TG tokens/s | PP+TG decode duration | TGp95 |
| --- | --- | --- | --- | --- |
|heldout-arithmetic|-0.21% [-0.71,+0.10] NESSUN_CAMBIAMENTO|-2.16% [-3.21,+0.71] INCONCLUDENTE|+1.43% [-0.41,+2.24] NESSUN_CAMBIAMENTO|+1.37% [+0.25,+2.74] NESSUN_CAMBIAMENTO|
|heldout-code|+0.33% [+0.05,+0.62] NESSUN_CAMBIAMENTO|+0.90% [-1.14,+2.43] NESSUN_CAMBIAMENTO|-0.70% [-1.68,+0.70] NESSUN_CAMBIAMENTO|-0.73% [-2.31,+2.97] NESSUN_CAMBIAMENTO|
|heldout-italian|-0.28% [-0.83,+0.25] NESSUN_CAMBIAMENTO|-0.23% [-1.98,+1.58] NESSUN_CAMBIAMENTO|+0.42% [-0.85,+1.34] NESSUN_CAMBIAMENTO|-0.20% [-2.12,+2.27] NESSUN_CAMBIAMENTO|
|heldout-planning|-2.56% [-3.58,-2.11] INCONCLUDENTE|+0.57% [-0.41,+2.71] NESSUN_CAMBIAMENTO|+0.68% [-0.82,+1.38] NESSUN_CAMBIAMENTO|+0.27% [-3.39,+2.09] NESSUN_CAMBIAMENTO|
|heldout-retrieval|-0.40% [-0.76,+0.06] NESSUN_CAMBIAMENTO|+0.02% [-1.61,+1.45] NESSUN_CAMBIAMENTO|+0.19% [-0.82,+1.31] NESSUN_CAMBIAMENTO|+0.79% [-1.11,+3.20] NESSUN_CAMBIAMENTO|
|heldout-structured|-2.36% [-2.72,-2.16] NESSUN_CAMBIAMENTO|+1.72% [+0.16,+4.10] INCONCLUDENTE|-0.15% [-1.70,+0.86] NESSUN_CAMBIAMENTO|-2.44% [-4.11,+0.45] NESSUN_CAMBIAMENTO|
|short|-0.14% [-0.72,+0.54] NESSUN_CAMBIAMENTO|+1.73% [-0.60,+2.85] NESSUN_CAMBIAMENTO|-1.32% [-2.13,+0.56] NESSUN_CAMBIAMENTO|-2.36% [-3.76,+0.39] NESSUN_CAMBIAMENTO|

Every PP+TG duration comparison is NESSUN_CAMBIAMENTO under the3% practical margin. TG for arithmetic/structured and PP for planning remain INCONCLUDENTE; every other ON/OFF metric is within its preregistered practical band. Old versus new OFF is NESSUN_CAMBIAMENTO for all four metrics in every input. No useful warm latency gain is established. ON versus resident TG points are negative in all seven inputs, but all intervals cross the practical regression boundary and remain INCONCLUDENTE. No global winner or new default follows.

Counters explain the intended mechanism, not a measured speedup: ON has no batched pool admission, and planning/structured warm TG reaches512/512 expert hits without misses/evictions/pool uploads. In the first block, ON instead raises TG misses to99/112 versus26/50 under old/OFF. Original fallback copies and persistent payload remain. R61 timings have a warmup and cannot measure this first-block cost; R63 uses separate fresh empty-pool raw/timing processes. Model loading, sampling and application TTFT are excluded.

## R62 native-prefill diagnostic completed

Eight processes report64 cases and3,484 complete emitted vectors. All are finite/nonzero. Six processes pass internal/fresh repeats; GPU17/logical1 fails the code-trace duplicate, and GPU17/logical2 fails comparison to that anomalous first-process duplicate. Full independent audit finds64 differing vectors beginning at vector0 after the3174-token prefill, with identical prompt IDs, identical generated64-token histories and unchanged first argmax. First max absolute difference is0.1988800764 and RMS0.02911014985. The second-process internal duplicate and other first-process cases remain exact. The original anomaly is never substituted. Overall native repetition gate remains FAIL_NUMERICAL_OR_REPEAT.

CPU17 logical versus legacy physical output is byte-exact in every case. GPU17 logical versus legacy is exact except the original anomalous duplicate. All four CPU processes and both GPU legacy processes pass in this bounded cohort; this does not establish equivalent reliability or a geometry fix. CPU17/GPU17 first differences normally occur at vector1, after the same first sampled token; later autonomous histories can differ, including two short controls. Neither semantic equivalence nor a placement fix is established. The64-token cap and raw-I/O helper remain diagnostic.

R62 audit was amended only to report first-divergence geometry for every internal/fresh repeat after the original failure appeared. Original auditor SHA/source is retained; gate, input, binary and measurement protocol are unchanged. All R61/R62 model workers joined before full audits. Their complete audit marker admits R63 with its own full-raw gate, while preserving both long-pressure and native-repeat failures.

## R64 context lifecycle diagnostic preregistered

R62 localizes an anomalous baseline duplicate at the first output after prefill, with identical prompt and generated histories. R64 tests the same complete eight-case sequence on the unmodified B1 DSOs, crossing logical/physical prefill with reuse versus fresh experimental context. Four fresh processes per cell give16 processes and128 cases. The helper uses the same public context/threadpool factory in both modes. The original common context remains unused and retained identically, preserving model sampling metadata; at most one experimental context is alive. Fresh synchronizes and destroys it before recreation. Destructor order releases the context before its threadpools and before the model. This includes scheduler/buffer/threadpool lifecycle and cannot identify an isolated memory-clear cause.

Full emitted raw vectors, prompt/generated IDs, within-process duplicates and fresh repeats are checked. Original first references are never replaced. First-seven-case compatibility against the original benign R62GPUlegacy1 archive is separate from the final anomalous duplicate; a compatibility failure prevents interpretation. The protocol, prototype, runners and54 public-header hashes are frozen before compilation/inference. Standalone helper compilation and model runs wait for R63 timing and the root independent post-run audit. No engine/kernel/driver change, performance ranking, reliability certification, semantic equivalence or new default is implied.

## R63 empty-pool first replay completed and audited

All48 fresh raw processes pass, with5,296 complete finite/nonzero vectors exact to the original first chronological empty-pool B1 and archived warm B1. A separate auditor verifies full raw files, first-call zero hits, actual admitted TG triplets, ON batched bypass and loaded DSOs. All48 separate timing processes pass original fingerprints, and their CSV/rate/p95/median/drift/bootstrap intervals independently recalculate exactly. All three inputs stay eligible under the original20% resident drift limit. Each process has common/replay warmup OFF and one replay; shader/file caches are already warm. No model cold-start or application TTFT is measured.

ON versus new OFF, percent effect and95% process-bootstrap interval:

| Fixed input | PP tokens/s | TG tokens/s | PP+TG decode duration | TGp95 |
| --- | --- | --- | --- | --- |
|heldout-planning|+2.20% [+0.61,+3.26] INCONCLUDENTE|-2.38% [-4.24,-0.10] INCONCLUDENTE|+1.05% [-0.88,+1.87] NESSUN_CAMBIAMENTO|+4.89% [+2.21,+8.70] INCONCLUDENTE|
|heldout-structured|+1.09% [+0.74,+2.29] NESSUN_CAMBIAMENTO|-1.20% [-5.28,+0.77] INCONCLUDENTE|+0.14% [-1.26,+3.01] INCONCLUDENTE|+2.18% [-2.28,+8.86] INCONCLUDENTE|
|short|+1.29% [-2.84,+3.72] INCONCLUDENTE|-0.97% [-2.35,+5.00] INCONCLUDENTE|+0.29% [-3.83,+1.14] INCONCLUDENTE|-1.17% [-5.75,+2.38] INCONCLUDENTE|

Planning PP+TG duration is NESSUN_CAMBIAMENTO within the3% margin; structured and short durations remain INCONCLUDENTE. All TG contrasts and planning/short PP and all p95 contrasts are INCONCLUDENTE. Structured PP stays within the3% practical band. No metric establishes a practical improvement or regression under the original thresholds. Old/new-OFF cold PP is within3% in all inputs; TG/p95 and short duration remain inconclusive. Four fresh processes per variant are retained without post-hoc selection or extension.

Warm and empty-pool campaigns together do not demonstrate a practical latency benefit of the bypass. This is a bounded negative/inconclusive result, not proof of equivalence on all workloads. ON remains opt-in and defaultOFF. The mechanism avoids batched pool admission but leaves original fallback copies and persistent payload. Prior long-pressure, native-repeat, API/state and CPU/GPU failures remain open. The96-process execution index records actual commands, environment, loaded engine/driver identities, clocks,power,affinity and original CSV/raw hashes with shared configurations stored once.

After all R63 workers and audits joined, R64 standalone helper compiled successfully with GCC16.2.1 and the exact original B1 DSOs. The frozen source/header/protocol hashes and compile command are recorded. R64 lifecycle model diagnostics are now running; source compilation does not qualify their outputs or establish a fix.

## R64 completed: baseline lifecycle remains unqualified

All16 processes complete128 cases and7,088 full finite/nonzero vectors. Twelve processes pass the registered repeat gates; four fresh/legacy processes fail. Full independent audit identifies one original anomalous retrieval duplicate in fresh-legacy1, after the3294-token prompt. All34 emitted vectors differ, beginning at vector0, with identical prompt IDs, identical34 generated IDs and unchanged first argmax. First max absolute difference is0.1889941692 and RMS0.03670543680. Subsequent fresh-legacy2/3/4 internal duplicates are exact but disagree with the frozen anomalous first reference; they are not three additional independent anomalies. Original references and all four process failures remain.

First-seven-case compatibility is111/112 exact, with that same first retrieval duplicate failing. Per the preregistered rule this closes causal interpretation. The helper visibly creates fresh context generation7 for this case, but context/threadpool/scheduler/buffer lifecycle and the identically retained unused common context are a bundle; no isolated memory-clear cause or repair is established. No performance ranking or reliability certification follows. R62,pressure,API/state and CPU/GPU gates remain open.

## Current upstream79e reviewed; R65 preregistered

A fresh upstream fetch reaches79e2e74eb11022c1ba2e438df7f0ca2d4c10f8b6, version0.6.0-dev. Four new commits since50e include embedding GET_ROWS ordering8e2d31e/PR30160, llama-bench fit-context correctionf39148a/PR28331, UI changes and CUDA-only roundf79e/PR30229. Current code and the merged PRs were read. GET_ROWS now goes at the front of the graph to avoid interleaving embedding PAD/MUL/SCALE and unwanted splits. Issue30033 is on SYCL/Intel; none of its throughput figures are AMD evidence. The roundf edit changes only ggml-cuda/unary.cu and is excluded from this Vulkan/GCC build. No RDNA2 fix or gain is inferred. See the published source review with primary commit/PR links; the exact selected patch remains in the local archive with its recorded SHA.

R65 builds the full unmodified79e source in a separate detached checkout, with Release/shared/native/OpenMP/Vulkan,CUDAOFF,backendDLOFF and Ninja2jobs. The original B1 engine, helper and model remain unchanged. Candidate helper compilation uses byte-identical canonical2e08 replay/phase sources with the candidate0.6ABI headers. Full model SHA is reverified before building; clean source,commands,cache,compiler and artifacts are recorded. No current source is adopted into the fork.

Four fixed inputs are registered before build/inference: planning119+64 andshort512+200 may qualify short timing; code3174+64 andretrieval3294+34 are diagnostic only. Long fixed IDs come from the original benign R62GPUlegacy1 and can include a forced terminal EOG; they are not native answer quality. AllCSVheaders,LFformat,IDbounds and PP/TG counts were checked. All32 fresh raw processes (four/variant/input,three measured repetitions andone replay warmup) must be finite/nonzero,self-repeat exact and byte-exact to the first chronological B1. Missing/failed B1 references are never replaced. Every attempted outcome is independently classified.

At most16 independent no-raw/no-counter short timing processes follow their own cell's full raw audit. Long cells never enter timing even if raw passes. Both short cells remain reportable, including ineligible cells. Rates,PP+TG decode duration andp95 retain3%/5% practical margins,50,000 process-bootstrap draws withseed65 and20% B1 drift exclusion. The current engine bundles multiple upstream changes, so any result cannot be attributed toPR30160 without a separate isolated A/B. No default,integration,semantic,TTFT,concurrency or M0-M3 closure follows. R65 waits for all R64 workers and root post-run audit under the exclusive lock.

## R65 build complete, numerical campaign in progress

The first local clone attempt could not read the requested upstream tree and stopped before compilation/inference. Its source/log/status remain archived. The preregistered V2 preparation explicitly fetched the same exact public79e SHA into a new isolated checkout, with unchanged inputIDs and acceptance criteria. Build succeeds in241s,including the canonical helper; independent source/cache/ELF/artifact audit passes14 frozen files. Candidate source stays clean; B1 files are byte-identical to R61-BASE. Exact commands/cache/source/library/model identities are preserved.

Planning completes8 raw processes:4B1PASS and4B4FAIL against chronological B1. Short and long cases remain in progress in the attached snapshot. The B4 difference does not alone establish a bug or semantic regression. Original gates remain unchanged; the planning timing cell is excluded. A separate posthoc within-version fresh-repeat description was selected after observing this difference and will run only after the complete original raw audit, without replacing the B1 reference or changing timing eligibility. No final performance,quality or acceptance conclusion is available.
