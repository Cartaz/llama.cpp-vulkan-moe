# Host load-mode replication on the measured prefill bottleneck

Date: 2026-10-07. R22 repeats the existing `mmap`/`none` comparison on the clean current engine after R20 identified prefill host upload costs. Status: VALIDATO on fixed244+128 replay; broader T4/T6/T7 gates remain open. This is a configuration replication, not a new engine optimization. [Validation metadata](moe-host-transfer-recheck-validation.json) includes exact commands, frozen artifacts, system versions, every process and statistics. Local archive: `risultati/2026-10-07-host-transfer-recheck/`.

## One-variable A/B

Both modes use frozen engine `9375d46b7f7e1099bdee4d92cc6fc091057a0194`, tree `b6464350c9251c43bc38f5ca1d61566b54667840`, and the same six engine libraries. Release/native/Ninja, VulkanON, CPU_MOE_COMPACT/ACTIVE OFF; GCC16.2.1 20260810, CMake4.4.4, Ninja1.13.2. Model is Ornith-1.5-35B-Q4_K_M, 21,713,462,848 bytes, freshly rechecked SHA256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`. Fixed token-file SHA256 `85935136b367ed46e47e967b05495774bc290d2d32dd92776c0c242d744cf29c`.

RX6800/RADV NAVI21, Mesa26.2.4-arch3.1, kernel7.2.9-1-cachyos, Ryzen5700X3D/32GB. The existing2600/1075MHz,-100mV,186W profile was untouched. Every command uses ncmoe18, ngl99, eight generation/batch threads, context1024, batch/ubatch512, FAon, Q8 KV and fitOFF. Only `--load-mode mmap` versus `--load-mode none` changes. Environment is rebuilt with env-i, explicit frozen LD_LIBRARY_PATH and no inherited experiment flags. A single process runs at a time, scope24G/swap2G,6GiB MemAvailable floor and600s timeout; no guards triggered.

Two correctness dump processes, eight uninstrumented speed processes and one separate NONE host diagnostic completed. Fresh mmap/none raw dumps are finite/nonzero at every129-vector call and byte-identical to untouched B1, SHA256 `b7bb28a86e623daf75f436f9bddb786cee6c43d8a28d1b85333f3f25d9cbb662`,128,133,120bytes. Every repeated measured call matches that reference hash. Dump/profile runs are excluded from ranking. No sampled semantic quality score, application TTFT or concurrency is inferred from replay.

## Performance and costs

Preregistered order mmap/none/none/mmap repeated twice; four independent processes per mode, one warmup and three measured repetitions each. Means are per-process token/elapsed ratios. Bootstrap50000 resamples at process level, seed20261009; margins3%PP/TG,5%decode p95. No adaptive extension or discarded samples. First/last mmap changes are-2.47%PP,+1.59%TG,-0.61%p95, below the gross20% drift alarm; practical verdicts use the smaller preset margins.

| Metric | mmap | none | Change, bootstrap95% interval | Verdict |
| --- | ---: | ---: | --- | --- |
| PP tokens/s | 168.824 | 225.892 | +33.80%; [+32.26%, +35.17%] | MIGLIORAMENTO |
| TG tokens/s | 28.853 | 28.837 | -0.057%; [-1.22%, +1.25%] | NESSUN_CAMBIAMENTO within3% |
| Mean process decode p95, ms | 40.754 | 40.602 | -0.374%; [-1.49%, +0.64%] | NESSUN_CAMBIAMENTO within5% |
| Mean warm-file-cache initialization to replay-ready log, s | 5.017 | 9.893 | +4.875s | Descriptive startup cost |

Initialization is relative to common logging initialization and precedes replay warmup; it is not complete application cold-start TTFT or a cold-disk measurement. File-cache ownership and prior processes affect loading. Only inference rates/p95 receive the preset statistical verdict.

| Memory observation, sampled at2Hz | mmap | none |
| --- | ---: | ---: |
| Per-process driver resident GTT peak | 0.620GiB | 9.571..9.593GiB |
| Per-process driver requested GTT peak | 0.477GiB | 9.429..9.450GiB |
| Global GTT maximum, including desktop | 0.752GiB | 9.706GiB |
| Per-process driver resident VRAM peak | 11.683GiB | 11.689..11.694GiB |
| Global VRAM maximum | 12.901GiB | 12.907GiB |
| Minimum system MemAvailable | 24.805GiB | 16.941GiB |
| Maximum sampled cgroup memory peak | 5.772GiB | 16.407GiB |

Requested allocator buffers: mmap CPU_Mapped model9049.31MiB; none Vulkan_Host model8642.81MiB. Both retain Vulkan model11532.90MiB, KV10.62MiB, recurrent62.81MiB, Vulkan compute498.52MiB, Host compute9.29MiB and output0.95MiB. Buffer classes differ; these logs do not enumerate a complete private-RAM footprint. RSS/HWM peaks20.37GiB mmap versus0.47GiB none are mapping/accounting differences, not a model RAM reduction. Driver-backed host allocation is visible in GTT and system availability. Cgroup shared file-cache charges depend on ownership/order and cannot by themselves estimate the mode's marginal RAM cost. Requested/resident counters are distinct; deprecated DRM aliases are not added. See [kernel accounting definitions](https://docs.kernel.org/gpu/drm-usage-stats.html).

The NONE cgroup sampled swap maximum is114688bytes (mmap0). System SwapFree decreases elsewhere during the campaign; global swap cannot be attributed to the model. Sampled GPU hotspot peaks48C in both modes, effective maximum core2604MHz and memory1074MHz, power peaks185W mmap/175W none. These ranges include load/idle phases and do not measure energy per token or utilization equivalence. Sampled peaks can miss brief spikes. This short context does not certify a larger cache/context or eight-stream budget.

## Host timing interpretation

The separate NONE host-profile run requests the same5,404,684,800 bytes of selective expert uploads during prefill,3168 API calls including padding. Its prefill interval1144.219ms includes995.154ms compute_splits. Upload host calls themselves total1.725ms, but input_wait totals863.794ms and final scheduler_wait144.718ms. R20's separately collected mmap profile had1128.249ms upload host calls and92.408ms input_wait. The load mode changes the copy/submission path and where host waiting occurs. The1.7ms upload scope does not mean the payload transfer vanished, and subtracting scopes across campaigns does not establish physical bandwidth or overlap. Host/GPU envelopes must not be summed.

NONE decode remains CPU placement for the first18 expert layers and performs no expert weight uploads. This explains why the PP gain is not evidence of a decode cache benefit. S10/S11 still need exact region/queue/lifetime measurements and a bounded staging experiment at the same load mode before claiming overlap. The allocation cost of keeping about8.4GiB in Vulkan_Host must be compared with bounded staging and context memory, not hidden behind RSS.

Decision: retain the existing explicit load-mode choice for further experiments. Do not change defaults or promote B3 based on the brief replay. R23 separately checks a3107-token prompt and finds a failed repeatability gate even on the original baseline; [long-prompt diagnostic](moe-long-prefill-recheck-rx6800.md) documents that limit. The short A/B remains reproducible on its declared workload.

## Reproduce

The exact token input is versioned at [r22-244-128.csv](../../examples/moe-trace/workloads/r22-244-128.csv).

Use the frozen engine/library hashes in metadata, the exact token file, fresh output paths and no other experiment flags. Each speed invocation has one internal warmup and three measured repetitions:

```sh
env -i PATH=/usr/bin:/bin HOME="$HOME" XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" LC_ALL=C LD_LIBRARY_PATH=/absolute/path/frozen/COLLECTOR MOE_REPLAY_IN=/absolute/path/tokens.csv MOE_REPLAY_REPS=3 /absolute/path/frozen/COLLECTOR/replay -m /absolute/path/Ornith-1.5-35B-Q4_K_M.gguf -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off --verbosity 4 --load-mode mmap > mmap.csv 2> mmap.log
env -i PATH=/usr/bin:/bin HOME="$HOME" XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" LC_ALL=C LD_LIBRARY_PATH=/absolute/path/frozen/COLLECTOR MOE_REPLAY_IN=/absolute/path/tokens.csv MOE_REPLAY_REPS=3 /absolute/path/frozen/COLLECTOR/replay -m /absolute/path/Ornith-1.5-35B-Q4_K_M.gguf -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off --verbosity 4 --load-mode none > none.csv 2> none.log
```

The archive runner adds the recorded resource scope, telemetry, Vulkan identity/preload checks, library verification and per-call reference checks. Run four fresh processes per condition in the declared balanced order. `analyze.py` in the archive reproduces all bootstrap and memory summaries. Hashes/dumps remain outside inference timers.
