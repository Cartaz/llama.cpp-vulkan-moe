# Layer reservations and expanded normal-route pilot

R28, 2026-10-07, advances S02/S06. The R27 complete-request greedy planner concentrated its payload in a few layers. This increment adds optional minimum 4/8 slots ranked by TRAIN frequency per layer before residual global-frequency/byte or complete-request-bundle allocation. It also expands the normal-route corpus from the previous four-case development pilot to 12 new prompts, six TRAIN / six HELDOUT in paired code, dependency planning, evidence retrieval, arithmetic, structured/tool planning and Italian explanation families. Runtime cache is not implemented by this Python-only change.

The bottleneck targeted is uneven cache capacity: leaving a layer without any slots guarantees fallback for that layer. On RX6800/RDNA2, a reservation could spread a bounded GPU pool across layers while preserving original routing and weights; however it spends memory on experts with lower observed reuse and can reduce complete-request coverage. This is a logical membership experiment, not measured GPU compute/fallback latency or PCIe savings. Current ncmoe18 has no TG expert-weight uploads to remove; cache operation will change placement and requires explicit fallback, remap and slot-lifetime controls.

## Protocol and provenance

Policies, new messages/splits and caps were fixed before model captures or HELDOUT scoring. [Messages](../../examples/moe-trace/workloads/r28-routing-corpus.json) are versioned; exact rendered prompts and fixed-token workloads are also saved for replication. The two TRAIN/two HELDOUT cases previously inspected in R21/R27 remain development data and are excluded from this new comparison. Each new family supplies one TRAIN and one different HELDOUT prompt. This is still below T2's full three-prompt/three-continuation-per-family target: no stochastic seed sweep, long conversation, semantic quality or real-agent task success is claimed.

Reuse exact frozen original B1 engine `7fe450e19305b828c199d602c23a8337aaa1f03b` / replay helper and normal collector `9375d46b7f7e1099bdee4d92cc6fc091057a0194`, tree `b6464350c9251c43bc38f5ca1d61566b54667840`, from R21. No engine rebuild or B4 data is mixed into this comparison. Model 21,713,462,848 bytes, SHA256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f` is freshly rehashed before loading; actual GGUF template SHA `f55f52930aa8bf44ab5cb85f99370fcc3c56e9a85640b812086d5330bce5d86b`. Jinja2 rendering with thinking OFF. Native/Release/Ninja/Vulkan/shared build flags, compiler/cache/binary/library hashes, system versions and every actual command/environment are recorded in validation metadata. Frozen source/library identities are separate from the updated Python planner source hashes.

Same RX 6800 / 5700X3D / 32 GB, RADV NAVI21 / Mesa 26.2.4-arch3.1, kernel 7.2.9-1-cachyos. GPU settings unchanged: 2600/1075 MHz, -100 mV, 186 W. Args ngl99/ncmoe18/t8/tb8/c1024/b512/ub512/FAon/Q8KV/fitOFF/mmap. Greedy 64-token cap. One model process at a time, clean environment and explicit frozen libs, 24 G scope / 2 G swap, 6 GiB availability floor, 600 s timeout and 2 Hz memory/DRM telemetry.

Each case requires normal callback-free generator capture, B1 fixed-token replay with warmup+one raw repetition, and collector replay with warmup+one raw repetition and host routing/phase profile. Every full-vocabulary endpoint must be finite/individually nonzero and byte-identical across all three. Replay hashes must agree and CPU18/top8 route joins, including warmup, must be complete. Raw/profile runs are correctness/routing captures, never PP/TG performance samples.

## Reservation implementation

`plan(..., minimum_slots=0)` retains previous defaults exactly. New `minimum_slots` manifest list requests optional variants. Reserve most-frequent observed TRAIN IDs per layer with stable expert-ID ties, then apply the same residual greedy policy at the shared byte cap. Uniform policy remains the baseline. Allocated payload and slack are reported; no byte padding or invented unobserved expert IDs. An insufficient cap or insufficient observed TRAIN support produces explicit `INFEASIBLE`, without silently lowering the minimum.

Caps 239,984,640 / 514,252,800 / 1,062,789,120 bytes correspond to uniform 7/15/31 slots across CPU18 (228.867/490.430/1013.555MiB). Min 4 reserves 130.781 MiB; min 8 reserves 261.563 MiB and therefore cannot fit the first cap. Expert triplet costs remain 1,769,472 or 2,039,808 bytes according to the actual layer layout. Allocator metadata, in-flight slots and KV/compute/desktop reserves remain outside this offline cap.

`--plans-json` saves all TRAIN allocations before parsing HELDOUT routing contents or attaching scores. Hash/model tags are validated; this tooling trusts paired trace/model provenance, not a GGUF loader or runtime remap format. All variants are prespecified; no policy promotion follows from favorable held-out scores. Layer quotas are capacity reservations, not guarantees that a held-out token's entire top8 fits or hits.

Validation: 20 offline tests pass, including heterogeneous exact-fit/infeasible minima, no unseen padding, residual bundle behavior, whole-request counts, deterministic/default compatibility and a persisted TRAIN plan even when subsequent HELDOUT parsing rejects malformed top-k data. Full default R27 report/allocations remain identical. The first new residual-choice fixture had an incorrect expected ratio; costs were corrected before model runs, with the initial failure retained in the archive.

## Outcome

All 12 cases pass the prescribed gates: 36 successful model processes, 2,340 complete finite/nonzero vocabulary distributions (780 distinct endpoints captured three ways), exact per-case generator/B1/collector parity and complete CPU18 routing. New prompts contain 113..151 PP tokens and each supplies 64 generated continuation tokens. All source/library paths are verified. Captures remain in `risultati/2026-10-07-layer-reservations/`; [validation metadata](moe-layer-reservations-validation.json) includes every process, input/source/model/build/library/system hash, commands, protocol and per-case evidence. [Full policy analysis](moe-layer-reservations-analysis.json) preserves allocations and all TRAIN/HELDOUT scores.

At the shared 514,252,800-byte cap (490.430 MiB), pooled HELDOUT results cover 6,912 token/layer requests and 55,296 expert activations:

| Policy | Reserved slots/layer | Payload MiB | Layers without slots | Activation coverage | Complete top-8 requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uniform frequency | 0 | 490.430 | 0 | 23.607% | 0.014% |
| Global frequency/byte | 0 | 489.984 | 0 | 23.617% | 0.029% |
| Global frequency/byte | 4 | 489.586 | 0 | 23.683% | 0.014% |
| Global frequency/byte | 8 | 488.930 | 0 | 23.729% | 0% |
| Complete-request bundles | 0 | 489.727 | 16 | 6.507% | 4.456% |
| Complete-request bundles | 4 | 490.219 | 0 | 14.413% | 4.181% |
| Complete-request bundles | 8 | 489.984 | 0 | 18.725% | 2.459% |

Min 4 spreads bundle capacity across every layer: complete requests number 289, compared with 308 without reservation. Activation coverage rises, complete-request coverage falls slightly. Min 8 spreads more payload but complete-request coverage falls further. A layer left at four slots cannot contain an entire top-8 request; its partial hits require mixed execution or bypass. Min 8 removes that capacity obstacle, but does not guarantee matching expert IDs. This does not select a latency winner: partial hits still need mixed execution or a whole-request fallback, and their costs are not measured here. Uniform/global frequency optimize a different coverage objective. These are descriptive counts on six HELDOUT prompts, not independent per-request samples for statistical significance.

The 228.867 MiB cap yields two explicit INFEASIBLE min-8 variants, as preregistered. All other 19 of 21 allocations fit their caps and retain their requested minima. At 1,013.555 MiB, bundle min 0 / 4 / 8 complete-request rates are 12.066% / 11.646% / 8.058%, with 15 / 0 / 0 zero-quota layers. Full per-case, byte-weighted and per-layer accounting is preserved in the archive; the versioned analysis contains every allocation and case score. **No proposal covers a complete token across all 18 observed layers** on any of the six HELDOUT cases at any cap. The other 22 active layers are outside this trace scope.

All TRAIN plans were persisted before HELDOUT routing parsing and remain identical after attaching scores. No policy, prompt or threshold was changed after observing these outcomes. R27 figures use a different training/evaluation corpus and are not pooled or treated as a causal before/after A/B. Default minima remain zero. There is no runtime GPU allocation, inferred PCIe traffic reduction, PP/TG speed result, semantic score or baseline/default promotion.

Decision: reservations solve the zero-quota capacity problem in the simulation, with an explicit coverage tradeoff. S06 now has a tested minimum-reservation comparison; M2 remains partial because full corpus/continuation coverage, miss cost and runtime remap/lifetime are missing. The next runtime block is S04/S05: a bounded pool with original expert identity, safe slots, defined admission and CPU fallback. Its exact A/B must hold model/workload/payload cap fixed, validate all-hit/all-miss/mixed/eviction correctness independently, then measure TG, transfers, fallback cost and p95 for uniform versus reserved/global policies. Logical coverage alone cannot promote one of these allocations.

## Reproduce

Generation uses the existing normal helper with `MOE_TRACE_NO_CALLBACK=1` (presence disables legacy callback), actual rendered prompt and 64-token cap. Replay uses saved token IDs and fresh output files:

```sh
env -i PATH=/usr/bin:/bin HOME="$HOME" XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" LC_ALL=C LD_LIBRARY_PATH=/absolute/path/frozen/COLLECTOR MOE_REPLAY_IN=/absolute/path/r28-heldout-code.csv MOE_REPLAY_REPS=1 MOE_REPLAY_LOGITS_OUT=/absolute/path/fresh-logits.bin GGML_SCHED_ROUTE_PROFILE=/absolute/path/fresh-routes-host.csv GGML_SCHED_PROFILE=/absolute/path/fresh-scheduler.csv MOE_REPLAY_PROFILE=/absolute/path/fresh-phases.csv /absolute/path/frozen/COLLECTOR/replay -m /absolute/path/Ornith-1.5-35B-Q4_K_M.gguf -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off --verbosity 4 --load-mode mmap > fresh.csv 2> fresh.log
python3 examples/moe-trace/route-summary.py fresh-routes-host.csv --phases fresh-phases.csv --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 --output fresh-routing-normal.csv --json fresh-route-join.json
python3 examples/moe-trace/test-profile.py
python3 examples/moe-trace/layer-budget.py /absolute/path/verified-manifest.json --plans-json fresh-train-plans.json --json fresh-policy-analysis.json
```

B1 replay uses original libraries and no collector/profile variables. Check complete raw equality before using routes in a manifest; supply all case/model/layout hashes, strict geometry and declared TRAIN/HELDOUT splits. Manifest and captured commands will be included in the final validation data. S04/S05 exact pool/remap/lifetime and admission/fallback implementation remains the next runtime block; numeric/quality and TG/H2D/p95 tests must be independent of logical coverage.
