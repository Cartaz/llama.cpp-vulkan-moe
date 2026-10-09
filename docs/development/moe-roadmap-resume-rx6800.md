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
