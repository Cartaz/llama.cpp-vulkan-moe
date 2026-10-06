# S02/S06 pilot: normal routing and complete expert requests

Date: 2026-10-07. R21 implements an opt-in route collector and request-level cache statistics, and validates a short train/held-out corpus on Ornith. Status: VALIDATO for the declared collection geometry and pilot; M2 remains TEST_PARZIALI. A runtime GPU expert cache is not implemented. [Validation metadata](moe-cache-plan-validation.json) records source, builds, model, commands, libraries, tests and archive hashes. Local archive: `risultati/2026-10-07-cache-plan/`.

## Correctness before policy selection

The initial legacy callback capture fails the normal-output gate: all65 logits vectors differ from an untraced original replay on the same recorded tokens, max absolute difference3.512509, RMS0.298977, two different argmax values. Both outputs are finite/nonzero. The untraced fork and original replay match exactly. Compiling the trace helper against original baseline headers and linking only original baseline libraries reproduces the callback output exactly. This is ERRORE_BASELINE in the shared diagnostic path; its cause is not established. The baseline without a callback matches its own untraced replay. These failed captures and controls remain archived; process exit0 does not mean the acceptance gate passed.

The new collector observes IDs that the scheduler already has: the existing selective expert-upload readback, or host IDs after a synchronous native CPU graph has completed. It adds no GPU graph cuts, readbacks or synchronization. CPU callback mini-graphs are supported for fixtures, but the new corpus generator uses `MOE_TRACE_NO_CALLBACK=1`. The legacy callback remains the default for compatibility. The trace helper rejects nonfinite/all-zero logits and can write a fresh raw dump through `MOE_TRACE_LOGITS_OUT`.

`GGML_SCHED_ROUTE_PROFILE` writes a separate opt-in CSV with scheduler/call identity, tensor geometry, local token/rank/expert and host timestamp. It shares the existing log mutex and emits a completion footer. `route-summary.py` joins it with explicit replay phases and exports canonical routes. It checks dimensions, ID ranges, completeness, duplicate triplet agreement, scheduler completion and interval containment. It requires one complete scheduler call per evaluation phase; it refuses to guess offsets for multiple scheduler calls/ubatches. Fully GPU-resident expert routes are not observable through this mechanism. CPU layers0..17 are the validated model scope, not all40 active expert layers. Empty environment paths are OFF; errors warn/disable logging and incomplete output fails the converter.

On the same old244+128 token input, normal collection matches the B1 raw logits exactly, SHA256 `b7bb28a86e623daf75f436f9bddb786cee6c43d8a28d1b85333f3f25d9cbb662`. Comparing CPU18 routes to the archived legacy callback gives5355 different ranks among53568 observations and715 different expert sets among6696 token-layer requests (10.68%). R05 remains historical diagnostic-path simulation; its old hit rates cannot select a normal inference cache until revalidated. This does not revoke the separate byte-exact R03/R20 replay checks.

## Corpus and frozen build

Engine candidate: published `9375d46b7f7e1099bdee4d92cc6fc091057a0194`, source tree `b6464350c9251c43bc38f5ca1d61566b54667840`. Baseline engine: untouched v0.5.0 `7fe450e19305b828c199d602c23a8337aaa1f03b`. Earlier legacy controls used helper `a26c9a34be3eab4662a6b94f79b8104fa3afea6f`; original-linked helper provenance is distinguished from engine provenance in metadata. Build: Release/Ninja/native, Vulkan ON, CPU_MOE_COMPACT/ACTIVE OFF; kernel7.2.9-1-cachyos, Mesa26.2.4-arch3.1/RADV NAVI21, GCC16.2.1 20260810, CMake4.4.4, Ninja1.13.2. Model SHA256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`, 21,713,462,848 bytes; RX6800/5700X3D/32GB, unchanged2600/1075MHz,-100mV,186W.

Four curated synthetic agent-like prompts were fixed before generation: training code132 tokens and planning125; held-out retrieval175 and reasoning141. Each has64 greedy continuation tokens. The model's actual GGUF chat template was rendered with jinja2 3.1.6, thinkingOFF, template SHA256 `f55f52930aa8bf44ab5cb85f99370fcc3c56e9a85640b812086d5330bce5d86b`. Prompt/messages/token hashes and split are recorded. This is one short continuation per prompt, not the full T2 family/seed matrix, live agents, or a semantic quality score.

For each prompt, the no-callback generator, original replay, and new collector replay produce identical65-vector/64,563,200-byte dumps. Every vector is finite/nonzero. All18 CPU layers/top8/256 experts are complete in warmup and measured replay matrices. Six earlier diagnostic/control processes, twelve new corpus processes and one fixed-workload check total19 successful processes; the initial legacy acceptance failure is retained separately. Every process uses frozen libraries, explicit environment, one stream, context1024, batch/ubatch512, ncmoe18, eight generation/batch threads, FAon, Q8 KV and fitOFF. Memory guards are24G scope/2G swap/6GiB available floor, timeout600s, telemetry2Hz. No speed ranking uses dump or collection runs.

CPU Release, ASan/UBSan and Vulkan suites with GPU logging OFF/ON pass7 tests each, including independently specified route IDs, strided/repeated IDs, complete footers, invalid/empty/error paths and byte-exact outputs. The32 Vulkan operator evaluations and32 diagnostic phase joins pass. The offline profile suite passes14 tests, including a cache that hits individual activations but cannot hold a complete request. Collector overhead has not been certified separately.

## Complete-request availability at equal byte budgets

The analyzer now accepts explicit `--layers` and reports `all_hit`, `all_miss`, `mixed`, `exceeds_capacity`, and total token-layer requests, alongside activation hits. The first three partition requests; exceeds_capacity is an overlapping property. Membership is checked before admission. LRU eagerly admits misses; a CPU-bypass cache that does not upload misses has different state. These are logical policy estimates, not measured saved transfers, runtime residency or TG gains.

Gate/up/down payload per expert is1,769,472 or2,039,808 bytes depending on layer. Uniform CPU18 quotas7/8/15/31/62 consume228.867/261.563/490.430/1013.555/2027.109MiB. A uniform256MiB budget cannot hold top8 in every layer. The actual allocator needs alignment, inflight buffers and context/desktop reserve in addition to these payloads.

| Normal decode profile | Cold LRU activation hits,15slots | Cold complete requests,15slots | Warm complete requests,15slots | Cold complete requests,31slots |
| --- | ---: | ---: | ---: | ---: |
| Training code | 37.88% | 0.95% | 1.30% | 3.65% |
| Training planning | 41.91% | 0.52% | 0.87% | 5.99% |
| Held-out retrieval | 36.13% | 0.61% | 0.95% | 4.08% |
| Held-out reasoning | 41.95% | 0.95% | 1.39% | 5.82% |

Static sets trained on only the two training decode streams have held-out activation coverage19.04%/14.27% at15slots, but zero complete requests. At31slots coverage33.62%/26.67% yields only one complete request out of1152 in each held-out case (0.09%). Prefill-trained static sets and all quotas are retained in metadata. There is no held-out leakage in the training static profile.

This pilot gives little support for an all-hit-only GPU dispatch at about490MiB. S04/S05/S07 must define mixed-hit execution, miss admission, CPU bypass, reduction order, remap and pinned slot lifetime. Eager logical LRU cannot forecast CPU-bypass speed. The data justify measuring these alternatives, not a speculative performance claim. M2 still needs a broader corpus, layer-aware allocation at equal total bytes, actual miss costs and runtime correctness.

## Reproduce the validated collection path

Use fresh output paths and the recorded frozen library directory. Generate with the callback disabled:

```sh
MOE_TRACE_NO_CALLBACK=1 MOE_TRACE_OUT=/absolute/path/unused-routing.csv MOE_TRACE_TOKENS_OUT=/absolute/path/tokens.csv MOE_TRACE_LOGITS_OUT=/absolute/path/generation-logits.bin llama-moe-trace -m /absolute/path/Ornith-1.5-35B-Q4_K_M.gguf -f /absolute/path/prompt.txt -n 64 -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off
GGML_SCHED_ROUTE_PROFILE=/absolute/path/routes.csv GGML_SCHED_PROFILE=/absolute/path/scheduler.csv MOE_REPLAY_PROFILE=/absolute/path/phases.csv MOE_REPLAY_IN=/absolute/path/tokens.csv MOE_REPLAY_REPS=1 MOE_REPLAY_LOGITS_OUT=/absolute/path/collector-logits.bin llama-moe-replay -m /absolute/path/Ornith-1.5-35B-Q4_K_M.gguf -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off > replay.csv
python3 examples/moe-trace/route-summary.py /absolute/path/routes.csv --phases /absolute/path/phases.csv --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 --output /absolute/path/routing-normal.csv --json route-join.json
python3 examples/moe-trace/analyze.py /absolute/path/routing-normal.csv --phase decode --layers 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 --slots 0,7,8,15,31,62 --expert-bytes /absolute/path/expert-bytes.json --json cache-summary.json
```

Compare raw outputs to a finite/nonzero original replay before trusting routes. The generator's routing file is header-only in no-callback mode. `analyze-normal.py`, train sets, prompt rendering and all original artifacts are retained in the archive. No B3 promotion or integration follows from a logical hit-rate result.
