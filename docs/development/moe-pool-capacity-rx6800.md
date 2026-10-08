# R41: single-layer expert-pool capacity on RX 6800

2026-10-08, branch `experiment/moe-pool-capacity`, stacked on R40. On the declared same-GPU replay, 64/128 slots improve decode over targeted GPU execution without a pool. The 128-slot pool reaches 30.18 token/s versus 27.29 without the pool, but remains slower than a resident layer at 31.81. It reserves 216 MiB less expert-weight VRAM than the resident layer. This is a conditional short-workload result, not a new default: CPU/GPU fidelity, held-out workloads and an unexplained invalid process remain open.

## Fixed implementation and protocol

All variants use the same frozen R39 WORK engine: CPU capture implementation `f3a5150eb0c6eb57e60effe6498b7f6319e73136`, inherited R37 diagnostic graph and scheduler pool. B1 remains unmodified v0.5.0 `7fe450e19305b828c199d602c23a8337aaa1f03b`; it is not the performance denominator here. The only new C++ change is the standalone replay guard at `25e605e1c807e988f696395ef481dec412a94752`: scan every complete warmup/measured logits vector for finite values and at least one nonzero element before hashing or writing it. Scan time is outside the decode+synchronize timer. No engine, shader, routing, quantization or cache-policy change.

WORK is a mixed, explicitly frozen build: only its CPU library was rebuilt with Release/native/OpenMP, Vulkan OFF and input diagnostics OFF; other libraries retain the R37 Release/native/Vulkan ON/input diagnostics ON build. Both builds have compact/active CPU features OFF, shared libraries ON, backend dynamic loading OFF, Ninja, GCC16.2.1 and release flags `-O3 -DNDEBUG`. The guard helper uses C++17 `-O2 -DNDEBUG` and the unchanged libraries. Per-library hashes and build configuration are in [portable validation](moe-pool-capacity-validation.json); do not interpret the WORK CMake cache as the configuration of every loaded library.

Hardware: RX6800/RADV NAVI21 16 GiB with ReBAR16GiB, Ryzen5700X3D, 32GiB RAM, CachyOS; kernel7.2.9-1-cachyos, Mesa26.2.4-arch3.1, CMake4.4.4, Ninja1.13.2. User GPU settings2600/1075MHz, offset-100mV and186W retained. Model Ornith1.5 35B Q4_K_M,21713462848bytes, SHA256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`.

```text
-ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512
-fa on -ctk q8_0 -ctv q8_0 --fit off --load-mode mmap --verbosity 4
```

One stream, frozen512PP+200TG tokens,201complete248320-logit vectors/repetition, no sampling. Target sets `GGML_SCHED_EXPERT_GPU_LAYER=17`. Pool variants additionally set `GGML_SCHED_EXPERT_POOL=17:SLOTS:256`, SLOTS8/16/32/64/128. Resident removes both variables and prepends `-ot ^blk[.]17[.]ffn_(gate|up|down)_exps[.]weight$=Vulkan0` before `-ncmoe18`. Effective layer17 placement is checked in logs. Environment is cleared and reconstructed: `LC_ALL=C`, frozen `LD_LIBRARY_PATH`, `MOE_REPLAY_IN`, reps and warmup; exact full argv/env/loaded-library hashes are retained for each process. No affinity pinning was applied; inherited per-process affinity was not captured. No background GPU inference. Each process has MemoryMax24G/SwapMax2G,6GiB MemAvailable reserve, RX6800 preflight and600s timeout.

The initial registered campaign ran14cold raw gates and stopped after its third timing process failed fingerprints. Nine separately registered diagnostics followed. A new guard-only helper then ran an entire balanced campaign:7unlogged cold raw gates and28timing processes,4fresh processes/variant,3measured repetitions after one complete warmup. Orders, failure policy and margins were registered before execution. Logging/capture is OFF during ranked timing. Six later logged profiles measure routing/counters only and are not ranked. All initial data and outliers are retained; no outcome-selected replacement.

## Correctness and the retained failure

The21cold gates provide4221byte-exact full raw vectors across target, pools and resident. The9diagnostic processes provide4623finite/nonzero exact vectors, including warm/cold16-slot controls, logging OFF/ON and target/resident repeats. Guarded timing validates all16884measured fingerprints against the original target and scans5628warmup vectors directly. Six profiles validate3618measured fingerprints and1206warmup vectors; normalized layer17 route sequences agree exactly in repetition0, including complete PP and TG coverage.

Initial `r41-timing-s16-1` has603incorrect fingerprints. Its PP/TG signatures equal canonical all-positive/all-negative qNaN vector hashes; without its raw dump this is evidence of suspected NaNs, not raw-certified NaNs. The initial timing campaign is not ranked. The nine raw diagnostics did not reproduce the incident. The new guard rejects such output; it does not fix or explain its cause. Overall promotion stays blocked by this unresolved reliability observation and the separate CPU/GPU fidelity screen. Related upstream [issue25195](https://github.com/ggml-org/llama.cpp/issues/25195) concerns a different GPU/driver/long-context situation and establishes no attribution here.

The first warm-profile target process completed, then an omitted harness reference caused a postprocessing KeyError. Its original manifest/logs were retained and all603fingerprints independently validated; the harness was corrected and the five remaining cases continued. No inference was rerun or selectively substituted.

## Performance and memory tradeoff

Values are means of four independent process medians, each median covering three repetitions. Independent-process bootstrap20000/seed41 gives95% intervals; practical margins PP/TG3%, p95 latency5%. These are replay decode throughput and per-token decode p95, not TTFT or real-agent latency.

| Variant | Weight pool MiB | PP token/s | TG token/s | TG p95 ms | Global peak VRAM GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Target, no pool | 0 | 277.73 | 27.29 | 41.53 | 13.376 |
| 8 slots | 13.5 | 278.58 | 27.54 | 41.34 | 13.388 |
| 16 slots | 27 | 278.55 | 28.24 | 40.84 | 13.404 |
| 32 slots | 54 | 278.02 | 28.61 | 40.38 | 13.427 |
| 64 slots | 108 | 278.34 | 29.73 | 38.85 | 13.480 |
| 128 slots | 216 | 278.06 | 30.18 | 38.20 | 13.585 |
| Layer resident | 432 | 291.20 | 31.81 | 36.53 | 13.808 |

Against target,64slots TG+8.92% CI[7.83,10.20] and p95-6.45%[-7.66,-5.21]: MIGLIORAMENTO;128slots TG+10.58%[9.62,11.65], p95-8.03%[-9.12,-6.90]: MIGLIORAMENTO. PP is equivalent within3%.16/32slots TG+3.46%[2.12,4.83]/+4.82%[2.99,6.51]: INCONCLUDENTE at the practical3% boundary, despite positive point estimates. Target first-to-last drift PP-0.68%,TG-0.30%,p95+0.99% meets the registered20% guard.

One ~1.032s CPU harness compilation overlaps `r41b-timing-s8-2`; its exact PP/TG phase was not captured. An addendum registered before aggregation marks every8-slot contrast INCONCLUDENTE. All four samples remain; no replacement or claim of equivalence for8slots.

Against resident,128slots TG-5.11%[-6.31,-3.62] and PP-4.51%[-4.84,-4.23]: REGRESSIONE; p95+4.57%[3.39,5.65]: INCONCLUDENTE at5%.64slots TG-6.53%[-7.83,-4.96], p95+6.37%[5.04,7.63]: REGRESSIONE. The resident tradeoff remains meaningful:128slots save216MiB requested expert-weight allocation but lose decode speed. The existing full transfer arena is retained; pool allocations increase memory relative to target. This is not proof of eliminating that arena or reducing system RAM.

VRAM/GTT are global0.5s samples including desktop use, not exact private peaks. PeakGTT is0.322-0.338GiB across variants. RSS is about20.37GiB with the shared mmap model; child cgroup charge is about0.16GiB and swap0, not total model RAM. MemAvailable minima are about25GiB and include reclaimable page cache. PSS/private residency is NON_MISURATO. Full per-process measurements are retained; do not subtract these noisy global peaks as exact allocation savings.

## Routing, hits, transfers and remaining cost

Layer17 touches212distinct experts in512PP and165in200TG, top8 per decode token. Decode working-set payload is278.4375MiB; the128-slot pool216MiB cannot hold it all. Each slot contains gate/up/down Q4_K payload1769472bytes (1.6875MiB), plus32bytes total remap. Prefill exceeds the capacity path and does not populate the pool in this protocol. Cold decode begins with8misses. Warmup and repetitions retain cache state across context clears; cold and warm figures remain separate.

| Slots | Cold hit % | Warm rep0/1/2 hit % | Cold upload MiB | Warm rep0/1/2 upload MiB |
| --- | ---: | --- | ---: | --- |
| 8 | 31.500 | 31.500 /31.500 /31.500 | 1849.500 | 1849.500 /1849.500 /1849.500 |
| 16 | 47.750 | 47.563 /47.563 /47.563 | 1410.750 | 1415.813 /1415.813 /1415.813 |
| 32 | 62.125 | 62.438 /62.500 /62.500 | 1022.625 | 1014.188 /1012.500 /1012.500 |
| 64 | 78.250 | 79.250 /79.250 /79.250 | 587.250 | 560.250 /560.250 /560.250 |
| 128 | 88.438 | 93.375 /93.313 /93.313 | 312.188 | 178.875 /180.563 /180.563 |

These count logical API weight upload payload, not physical PCIe traffic. Uncached target requests2700MiB over200TG; resident has no decode expert-weight upload, with432MiB loaded beforehand. At128slots warm rep0, admission p50/p95 is3/638us but wait p50/p95 remains2008/2197.55us. Larger caches reduce miss uploads; remaining synchronization/execution/other layers still limit end-to-end speed. Logged timings do not establish a physical transfer bottleneck or predict multi-layer scaling.

Minimal next A/B: keep the engine/guard fixed; register held-out prompts and routing-family splits before cache selection, same capacities and resident/target controls, at least4fresh processes each, full finite/nonzero raw gates first, then unlogged timing and separate counter profiles. Extend context/KV and layers only after the same-GPU fidelity gates pass; compare aggregate and per-stream results separately at1/2/4/8streams. S09 still needs the downstream amplification/actual CPU-GPU fidelity explanation. No changes to default placement, pool policy or B3.

## Validation and artifacts

Profile/parser24PASS; scheduler11PASS including CPU/Vulkan profiler parity with no skip. The actual frozen `evaluate` function passes12negative and2positive harness controls: NaN, infinity and all-zero vectors rejected in both warmup and measured phases before raw writes; valid vectors accepted. R39 ASAN/UBSAN/leak raw parity is preserved; the R41 engine itself was not rebuilt with sanitizers. `git diff --check` passes.

R41 executes67model processes:17initial,9diagnostics,35guarded,6profiles; historical R20-R41 total377. Local archive `risultati/2026-10-08-pool-capacity/` retains all protocols/addenda, frozen binaries/source/configuration, full raw gates, CSV fingerprints, telemetry, route/phase/pool logs and analyzers. [Portable validation](moe-pool-capacity-validation.json) publishes process metadata, hashes, aggregate intervals, all profile counters and failure history without large binary/log artifacts. S04 capacity is VALIDATO only within this conditional short same-GPU perimeter; T2/M2/M3, semantic quality, long-context repeatability and overall reliability remain TEST_PARZIALI.
