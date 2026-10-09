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
