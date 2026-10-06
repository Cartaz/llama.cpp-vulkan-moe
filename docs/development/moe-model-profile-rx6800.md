# S03 model validation, logger cost and Vulkan diagnostics

Date: 2026-10-07. R20 follows the user's explicit request to resume model benchmarks. Status: VALIDATO for the stated single-stream replay; S03 remains TEST_PARZIALI for a complete calibrated timeline and larger-context memory budget. Exact sources, frozen artifacts, commands, loaded libraries, per-process observations and archive hashes are in [validation metadata](moe-model-profile-validation.json). Local archive: `risultati/2026-10-07-model-profile/`. Earlier interrupted campaigns remain preserved.

## Workload and controls

Ornith-1.5-35B-Q4_K_M, SHA-256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`, 21,713,462,848 bytes. Fixed recorded 244-token prefill plus 128 continuation tokens; token-file SHA-256 `85935136b367ed46e47e967b05495774bc290d2d32dd92776c0c242d744cf29c`. This is teacher-forced replay, separate from synthetic llama-bench and live agent tasks.

Baseline B1: untouched v0.5.0 `7fe450e19305b828c199d602c23a8337aaa1f03b`. Initial candidate: `0ae4bbd609e93ce8c6a7a9cda5775b0cee5f557b`. Timestamp fix tested locally at `5dcc3e6416a175986e451ff1cdc244e256c25a4b`; published code commit `af0195af1cfe57dd74a19f4919712c89ed9e761c` has the identical source tree `8f647dcf5b6d8a8ccab4aa7bee59ebc2d4d9f897`. Commit metadata differs; the frozen build records its actual local SHA. The three preexisting quality/README edits were excluded from commits and the replay build targets.

Both variants use `-ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off`, default mmap. Candidate build: Release/Ninja/native, Vulkan ON, CPU_MOE_COMPACT/CPU_MOE_ACTIVE OFF. A clean full Vulkan build replaced the earlier mixed-library smoke setup. GPU: RX 6800, RADV NAVI21, Mesa 26.2.4-arch3.1; kernel 7.2.9-1-cachyos; GCC 16.2.1 20260810, CMake 4.4.4, Ninja 1.13.2. Existing 2600/1075 MHz, -100 mV, 186 W configuration was unchanged. Frozen libraries were selected explicitly through LD_LIBRARY_PATH; common loaded-library hashes remained stable.

One model process at a time, each in a separate systemd user service: MemoryMax24G, MemorySwapMax2G, OOMPolicy stop; pre-load/during-run MemAvailable floor6GiB, device identity check, 600s timeout. All 30 processes completed successfully without triggering a guard. Env was rebuilt with `env -i`; no inherited GGML/RADV/VK experiment flags. Telemetry and DRM fdinfo were sampled at2Hz. Hashing occurred outside inference timers. Warmup was excluded, recorded separately in paired profiles. No clock/voltage experiment was performed.

## Correctness and measured cost

Six raw checks cover B1, initial candidate OFF/ON, original GPU logger and fixed logger OFF/ON. All 129 logits vectors in each check are finite and nonzero. Each complete 128,133,120-byte dump is byte-identical, SHA-256 `b7bb28a86e623daf75f436f9bddb786cee6c43d8a28d1b85333f3f25d9cbb662`, also matching R03. Every measured repeated call matches the validated baseline hash. Raw-dump runs are correctness diagnostics, excluded from performance ranking.

T1: OFF/ON/ON/OFF repeated twice, followed by the one preregistered extension because the p95 interval crossed its margin. Eight independent processes per condition, each warmup plus three measured repetitions. Bootstrap50,000 resamples of process means, seed20261007; repetitions within one process were not treated as independent samples. Practical margins:3% PP/TG,5% decode p95. No further extension was made.

| Metric | Host profiler OFF | Host profiler ON | Change; bootstrap95% interval | Verdict |
| --- | ---: | ---: | --- | --- |
| PP tokens/s | 167.936 | 168.159 | +0.13%; [-0.82%, +1.07%] | NESSUN_CAMBIAMENTO within3% |
| TG tokens/s | 29.052 | 27.289 | -6.07%; [-6.68%, -5.49%] | REGRESSIONE from instrumentation |
| Mean per-process decode p95, ms | 40.336 | 42.670 | +5.79%; [+4.70%, +6.81%] | INCONCLUDENTE against5% margin |

T1b independently compares B1 to the fixed fork with both profilers OFF: BASE/OFF/OFF/BASE repeated twice, four processes per variant, three measured repetitions each, seed20261008. PP170.583 ->169.371 (-0.71%, CI[-1.69%, +0.47%]); TG29.039 ->28.490 (-1.89%, CI[-2.88%, -0.97%]); p9540.299 ->41.211ms (+2.26%, CI[+0.50%, +3.60%]). All three intervals fit the preset practical margins. This does not establish zero overhead or a speed improvement. Results apply to this replay and configuration; bootstrap intervals from four/eight processes do not certify other workloads.

## Host components by phase

Every paired model profile associates all compute_splits calls with explicit phases, with zero unattributed compute calls. The following are means over24 measured ON repetitions, excluding warmup. Envelopes and their children are separate; do not add rows to derive elapsed time or overlap.

| Scope | Prefill244, ms | Decode128, ms |
| --- | ---: | ---: |
| Complete replay evaluation intervals | 1451.083 | 4690.722 |
| compute_splits, inclusive | 1347.917 | 4276.179 |
| Selective expert_upload host calls | 1128.249 | 0 |
| Vulkan compute_call host API | 109.113 | 1951.829 |
| CPU compute_call | 0.176 | 988.192 |
| Vulkan-to-CPU copy_wait | 0 | 738.044 |
| Vulkan-to-CPU synchronous tensor_copy | 0 | 312.048 |
| Final Vulkan scheduler_wait | 98.893 | 399.569 |
| Vulkan input_wait | 92.408 | 82.088 |
| Router readback | 2.162 | 0 |

Prefill selective-upload API payload is5,404,684,800 bytes per repetition, including padding. Decode has18,948,096 bytes of synchronous Vulkan-to-CPU copies,35,781,120 bytes of synchronous CPU-to-Vulkan copies and152,044,032 bytes of accepted asynchronous CPU-to-Vulkan payload. These are requested API bytes, not physical PCIe measurements. Vulkan host API durations include submission behavior and waits and are not GPU kernel time.

At ncmoe18, the first18 expert layers stay on CPU during single-token decode. They move to Vulkan for sufficiently large prefill batches through the existing offload policy (minimum batch32 by default in this fork). The zero decode expert-upload count is meaningful: a decode cache would change execution placement, not merely remove current weight copies. CPU computation and activation transfers/waits are visible TG targets. Prefill upload remains a separate measured target. No speedup is inferred by subtracting these scopes.

## GPU diagnostic correction and correlation

The original logger passes this model replay but aborts in the full synthetic parallel/view case at `ctx->compute_ctx.expired()`, independently of host profiling. Copies and event waits can already open a compute context before graph execution. The fix removes the invalid empty-context requirement and reuses the existing context through the existing helper. It adds no normal-mode synchronization or changes to routing/copy regions.

With GGML_VK_PERF_LOGGER_FREQUENCY=1, each graph prints a host-clock interval, backend, query count and concurrent-mode flag before its timing block. `profile-summary.py --vulkan-log` joins each marker by containment to exactly one Vulkan compute_call and an explicit replay phase; it rejects missing/duplicate blocks, mismatched counts/totals, unsupported concurrent logging, crossing intervals and unpaired calls. These markers preserve the existing logger and queries rather than creating a new query subsystem. Earlier log files without markers cannot be correlated by this parser.

Fixed model diagnostic:55 Vulkan graph calls in each prefill and2432 in each decode128, including separate warmup. Measured query intervals total309.924ms for PP and3719.045ms for decode. PP's largest grouped operation is Q4_K MUL_MAT_ID512x2048,168.575ms; decode's largest grouped operation is Q4_K MUL_MAT_ID_VEC512x2048,370.890ms. Full operation totals are in the archived schema-3 JSON. The raw outputs remain identical. The logger inserts per-operation barriers and waits; these are serialized diagnostic intervals, not normal-mode occupancy or a calibrated CPU/GPU critical-path timeline. They must not be added to the host table or used to rank shader changes without an OFF benchmark. Vulkan timestamp units and support are described by the [Khronos specification](https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#queries-timestamps).

CPU Release and ASan/UBSan parser/runtime suites:6 tests each. Full clean RX6800 Vulkan fixture:6 tests with GPU logging OFF and6 with GPU logging ON, including32 exact operator evaluations and32 GPU call/phase joins, strided/repeated IDs, callbacks and parallel scheduling. The independent scalar/finite/nonzero checks and byte-exact OFF/ON outputs pass. Correlation tests reject truncated, miscounted, outside-call and concurrent logs.

## Memory and next decision

Identical requested allocator sizes for B1 and candidates: CPU_Mapped model9049.31MiB; Vulkan model11532.90MiB; Vulkan KV10.62MiB; recurrent state62.81MiB; Vulkan compute498.52MiB; VulkanHost compute9.29MiB; VulkanHost output0.95MiB. These categories are not all physical device residency, and CPU-mapped model bytes are not private RAM.

During T1, sampled device-wide VRAM peaks range13,857,189,888..13,888,954,368 bytes (about12.90..12.93GiB), including desktop. Per-client driver peaks are reported separately in metadata: requested VRAM about11.826GiB and resident VRAM about11.683GiB. Driver resident GTT about0.620GiB is distinct from global GTT peaks0.731..0.776GiB. RSS/HWM includes about20.37GiB of mappings. Cgroup charges vary with ownership of shared page cache: the first baseline reached17.8GiB while warm-run services charged much less. This is not evidence of a model RAM saving. DRM requested/resident counters and deprecated aliases follow [kernel documentation](https://docs.kernel.org/gpu/drm-usage-stats.html); do not sum aliases. Peaks sampled at2Hz can miss shorter spikes and do not certify a larger-context cache budget.

S04/S05 must now account for CPU/GPU placement, triplet bytes, pinned lifetime, mixed-hit requests and CPU fallback cost. S06 needs a budget after weights/KV/recurrent/compute plus desktop margin, with context-dependent checks. Follow-up M2 work selects CPU layers explicitly and measures complete top-k availability rather than confusing activation hit rate with executable GPU requests. Corpus/held-out, runtime cache, actual miss costs, longer prompts/contexts, sampled generation quality and1/2/4/8-stream measurements remain separate gates.

## Reproduce

The local archive preserves the full runner, preregistration, all results and frozen libraries. Builds and model processes use the exact flags/hashes in metadata. A standard paired diagnostic invocation is:

```sh
MOE_REPLAY_IN=/absolute/path/tokens.csv MOE_REPLAY_REPS=1 GGML_SCHED_PROFILE=/absolute/path/fresh-scheduler.csv MOE_REPLAY_PROFILE=/absolute/path/fresh-phases.csv GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_FREQUENCY=1 llama-moe-replay -m /absolute/path/Ornith-1.5-35B-Q4_K_M.gguf -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off > replay.csv 2> vulkan.log
python3 examples/moe-trace/profile-summary.py /absolute/path/fresh-scheduler.csv --phases /absolute/path/fresh-phases.csv --vulkan-log vulkan.log --json diagnostic.json
```

Use new paths and the same process provenance for all three files, unset GGML_VK_PERF_LOGGER_CONCURRENT, and use the recorded frozen library path. Run speed comparisons with both loggers absent and without raw dumps. No B3 promotion or integration is justified by this measurement increment.
