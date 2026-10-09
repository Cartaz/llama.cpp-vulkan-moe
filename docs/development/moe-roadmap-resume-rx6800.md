# Roadmap qualification resumed on RX 6800

Status: RUNNING, 2026-10-09. The owner requested autonomous continuation until roadmap completion or actual exhaustion of the account's five-hour usage window. M0–M3 extended acceptance remains governed by the frozen `2026-10-08-milestones-0-3/acceptance-protocol.json`. Passing a bounded diagnostic does not complete the milestone. Archives are relative to `risultati/`; large dumps and model files stay local.

Baseline B1 remains unmodified v0.5.0 `7fe450e19305b828c199d602c23a8337aaa1f03b`. Candidate controls use frozen R42 engine libraries; multilayer fixtures use the separately frozen current scheduler patch. Every run records executable/library hashes and actual loaded mappings. Model SHA is `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`. Hardware remains RX6800/RADV NAVI21, Ryzen5700X3D,32GB,16GiB BAR0; kernel7.2.9-1-cachyos/Mesa26.2.4/GCC16.2.1. User GPU settings2600/1075MHz,-100mV,186W are preserved. One GPU process at a time;24GiB cgroup memory,2GiB swap,6GiB available-RAM reserve,600s per-process deadline. Record resource failures without ranking them.

## R43b: extended fixed-token correctness

The prior campaign contains50 process results:49 full raw gates pass. One pool128 process at10000PP+200TG diverges in its third repetition, beginning at the13th physical prefill vector. All outputs remain finite/nonzero;208 of220 vectors differ thereafter. Other pool/control repeats do not erase this failure. `2026-10-09-roadmap-resume/long10000-prior-differences.json` preserves vector-level evidence. R47 registers new balanced resident/uncached/pool/pool+Q8-sync/logging controls to investigate this failure, without performance ranking or a causal claim.

## R44: six-family autonomous corpus

The frozen corpus has36 prompts,three TRAIN and three HELDOUT prompts per family,and seeds11/29/47:108 autonomous continuations. Sampling is temperature0.8/top-p0.95/top-k40,max128 tokens or EOG. Consecutive cases clear and synchronize model memory while scheduler pools persist. Baseline resident, candidate same-GPU transfer and pool128 compare full emitted logits and generated IDs. The campaign is running; see `2026-10-09-roadmap-resume/corpus-results.json`.

Two harness errors are preserved and corrected separately. C++ temperature0.8 is serialized as its actual float representation; the verifier now compares against float32(0.8). A final generated token can end with incomplete UTF-8, causing JSON serialization to abort after valid inference on both baseline and fork. The helper now writes original text bytes to a binary sidecar and uses replacement decoding only for JSON text. IDs/logits are unchanged, including all available prior raw prefixes. The original aborted runs remain archived.

## R45: multilayer scheduler fixtures

Independent pools and explicit GPU placement lists are implemented behind existing opt-in environment variables. Pool syntax extends to `layer:slots:budget_MiB[;...]`; GPU layer selection extends to comma-separated layers. Duplicate layers, malformed syntax,more than32 pools and aggregate budget over2048MiB disable the configuration. Legacy single-layer syntax and OFF remain supported.

Fresh CPU and RX6800 Vulkan fixtures pass for2/4 layers,F32 and Q4_K/Q6_K,changing inputs/strided IDs and reused scheduler plans with1→3→33→3→1 tokens. All12 Vulkan pools actually admit the three expert projections. Forced abort after admission restores graph source pointers; retry is byte-exact. CPU ASAN/UBSAN/leak check passes. The fixture's ID storage is explicitly retained as a graph output: reading an input after its last consumer otherwise tests allocator reuse rather than source immutability. The original failed assertion is preserved as a harness error. Fresh full sanitizer/Vulkan validation/cost suites are queued; current evidence does not qualify model multilayer output or performance. Shared backend wait durations are per-pool observations and must not be summed as disjoint time. The full transfer arena remains allocated.

## R49: routing callback after the up projection

The original callback requests the `ffn_moe_topk` view. It changes all emitted logits in the three fixed TRAIN trajectories tested on both unmodified B1 and candidate; it is ineligible for normal-path policy selection. The helper adds `MOE_CORPUS_CALLBACK_AFTER_UP=1` together with `MOE_CORPUS_CALLBACK=1`: it waits for the expert up projection and reads that operator's routing-ID source. Inference operators and engine libraries are unchanged. The legacy callback stays available for diagnosis.

Eight registered processes compare B1/candidate OFF,legacy callback,and repeated after-up callback on the same three fixed-token trajectories. Six OFF/after-up controls pass:2070 full raw vectors finite/nonzero and byte-exact. After-up routes are exact across baseline/candidate/repeats,and all naturally observed CPU-side layers0..17 agree with the callback-free candidate route logger. This is an observer-neutrality result on a bounded teacher-forced set,not a speedup or general quality result. An additional108-case autonomous cohort is registered before using these routes for policy analysis.

The initial verifier wrongly required41 normal layers. Model metadata has41 blocks and one next-token-prediction block; `llama_hparams::n_layer()` subtracts that block,so normal inference has40 layers0..39. The corrected audit requires all40 layers/top8 for every token and retains the original verifier failures. MTP is not enabled or qualified. `callback-after-up-audit.json` separates corrected geometry from actual legacy-callback raw failure. Preserving the router's full Vulkan fusion is a hypothesis for why the later boundary helps; this diagnostic alone does not establish causality.

## R50: exact offline analysis at corpus scale

Native route conversion can return all repetitions in one strict parse (`rep=None`),avoiding18 full rereads of each completed corpus profile. Single-repetition output stays compatible. A two-repetition fixture verifies exact output agreement and rejects incomplete observations. The scheduler/profile suite now has12 passing tests.

The complete-request planner partitions remaining request sets by cardinality. Equal-size subset membership is an exact lookup; smaller sets are either scanned or enumerated with combinations,whichever has fewer candidates. Integer gains,Fraction ranking,tie rules and resulting plans are unchanged. An independent subset oracle and full heterogeneous/minimum-reservation allocations agree; all25 profile tests pass. The original CPU-only TRAIN planning attempt was stopped before producing plans or opening HELDOUT because its pairwise subset scan scaled poorly. Frozen policy caps/minima remain unchanged. A fresh TRAIN run is queued after isolated CPU/transfer cost fixtures; no inference-performance claim follows from this Python optimization.

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
