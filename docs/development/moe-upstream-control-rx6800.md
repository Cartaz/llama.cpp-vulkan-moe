# Clean recent upstream control and host-loading replication

R26, 2026-10-07. A separate unmodified upstream B4 checkout at `f498f864fbc0472004ee1c3616c1188c68eb157f`, tree `672c77b5702e28f11f8d93113c82eab26d160753`, passes the prescribed pairwise repeatability screen on short244+128 and long3107+128 teacher-forced replays. This does not establish general determinism or fix the old baseline. [Validation metadata](moe-upstream-control-validation.json) records the independent source/build/library/model freeze, every command/capture, protocols, system and effective initialization. Archive: `risultati/2026-10-07-upstream-control/`.

B1 `7fe450e19305b828c199d602c23a8337aaa1f03b` and prior experiments remain unchanged. The [upstream revision](https://github.com/ggml-org/llama.cpp/commit/f498f864fbc0472004ee1c3616c1188c68eb157f) is a control, not a cherry-pick into this fork. B4 is GGML0.26.0/llama0.6.0, compared with the historical0.25.1/0.5.0. Compile the identical79c7778 replay helper outside the upstream checkout against B4 headers and six B4 libraries; no helper API adaptation was needed. Upstream checkout stayed clean.

Build: GCC16.2.1 20260810, Release/Ninja/native/Vulkan/shared libraries; GGML_BACKEND_DL OFF, common ON, examples/tools/tests/MTMD OFF. Current common build has OpenSSL enabled; deprecated LLAMA_CURL=OFF is recorded as passed, not claimed to override all newer options. CMake4.4.4/Ninja1.13.2; full cache and binary hashes in the archive. Model21713462848bytes SHA256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f` was freshly rehashed before the first process. Same RX6800/5700X3D/32GB, RADV NAVI21/Mesa26.2.4-arch3.1, kernel7.2.9-1-cachyos; unchanged2600/1075MHz,-100mV,186W.

## Correctness controls

Arguments: ngl99/ncmoe18/t8/tb8/b512/ub512/FAon/Q8KV/fitOFF/mmap, c1024 short or c8192 long. Same recorded [short](../../examples/moe-trace/workloads/r22-244-128.csv) and [long](../../examples/moe-trace/workloads/r23-code-review-3107-128.csv) inputs. Four fresh B4 processes in order short/long/long/short, one internal warmup plus one raw repetition. Metadata-only memory reset; no callback/profiler or inherited GGML/RADV/VK experimental variables. Every distribution is finite and individually nonzero, vocabulary248320.

| Workload | Full vectors | Pairwise different vectors | Raw SHA256 |
| --- | ---: | ---: | --- |
| 244+128 | 129 | 0 | `1c80dca21c7768d9fa61f49d72e452e5c1277137386b08107cdfb54404aa991e` |
| 3107+128 | 135 | 0 | `3d5b21e1e60f779a9c247f810e13891bdca7c66625b0443cdd12b43921fbf7ef` |

The short B4 canonical differs from the original B1 canonical in29/129vectors: max absolute0.06257987, overall RMS0.001059668, zero argmax differences. This is not bit-equivalence or a semantic quality result. New math/defaults can legitimately change logits; the cause is not established here. No correctness-equivalent B1-versus-B4 speed ranking is made. The old long baseline remains variable as recorded in R23/R25; it is not overwritten by a new reference or compared against an arbitrarily chosen old dump.

Effective placement is the same requested ncmoe18/ngl99, and default op-offload minimum remains32. Short B4 allocations: GPU model11532.90MiB, CPU_Mapped model9049.31MiB, KV10.62MiB, recurrent62.81MiB, GPU compute501.00MiB, host compute17.30MiB and host output0.95MiB. Long KV85.00MiB, host compute24.30MiB. Historical fork short compute498.52/9.29MiB differs. Identical requested flags alone do not establish identical graphs or memory needs across versions.

Source inspection finds the examined Gated Delta Net and flash_attn_cm2 shaders identical to B1, while scheduler and recurrent-state code changed. This does not identify an upstream fix or a root cause. Reports such as [upstream issue27237](https://github.com/ggml-org/llama.cpp/issues/27237), on other hardware/drivers and software, are not RX6800 causal evidence. Additional state/history/shape and quality controls remain necessary.

## Same-version load-mode A/B

After the initial screen, a second protocol was fixed before any NONE control or timing run. Two fresh B4 NONE short captures must match the B4 mmap canonical completely; both pass129-vector raw equality. Only then run four fresh processes/mode, order mmap/none/none/mmap/mmap/none/none/mmap, one warmup plus three measured repetitions, no raw dump/profiler/callback. Validate all387calls/process against own finite/nonzero canonical. Shared24G scope/2G swap/6GiB availability floor/600s timeout/2Hz telemetry and one GPU model process at a time.

All eight timing processes finish with every387-call hash matching the B4 canonical:3,096validated measured calls. Fourteen successful B4 model processes total (four initial raw, two NONE raw, eight T1); no speed sample comes from the raw captures. Process metrics aggregate token/elapsed across three repetitions; reported comparison is the ratio of process medians. Process bootstrap50,000draws/seed20261010, practical margins3% PP/TG and5%p95 were fixed before execution.

| Metric | mmap median | none median | Change / bootstrap95% | Verdict |
| --- | ---: | ---: | --- | --- |
| PP tokens/s | 179.537 | 246.548 | +37.324% [36.922,37.383] | MIGLIORAMENTO |
| TG tokens/s | 28.240 | 28.443 | +0.717% [-0.298,1.366] | NESSUN_CAMBIAMENTO within3% |
| TG call p95 ms | 41.179 | 40.887 | -0.709% [-1.759,1.284] | NESSUN_CAMBIAMENTO within5% |

First/last mmap control drift is-0.010% PP/+0.606% TG. This is replication of an existing loading option inside B4; it introduces no backend optimization or runtime cache. R22 is historical replication context, not pooled data or proof of a B4 speedup over B1.

The tradeoff remains material. Sampled client resident GTT0.6272GiB becomes9.5656..9.6455GiB; client requested GTT0.4847GiB becomes9.4231..9.5030GiB. Resident VRAM11.6853..11.6861GiB becomes11.6977..11.7007GiB. Global VRAM peaks12.963GiB mmap and12.936GiB none include desktop activity and are not a causal memory saving. GPU model/KV/recurrent/compute requested allocations are unchanged across the short load-mode comparison; NONE replaces CPU_Mapped9049.31MiB by Vulkan_Host8642.81MiB model payload.

Sampled MemAvailable minimum26.171GiB mmap versus17.121GiB none, warm-file-cache replay-ready initialization5.528s versus8.729s (process medians). This is not cold-disk loading or application TTFT. Mmap RSS peaks20.372GiB, NONE0.469GiB: driver host allocations are omitted from ordinary RSS, so the lower RSS is **not RAM savings**. Cgroup peak4.258GiB/8.473GiB depends on file-cache ownership and process order; it must not replace whole-system memory accounting. Sampled per-scope swap is zero, while global SwapFree varies across both modes and cannot be attributed to this model alone. Peaks are2Hz sampled bounds, requested and resident allocations differ; [DRM accounting definitions](https://docs.kernel.org/gpu/drm-usage-stats.html) explain these fields.

Decision: preserve B4 as a separate pinned control and `none` as a short-workload PP/memory/startup tradeoff. No default/B3 promotion, engine integration, general quality guarantee or long NONE speed claim. Extend normal generation/quality and long repeatability before choosing a new runtime baseline. S10/S11 still require bounded host staging plus source/slot lifetime controls; no overlapping transfer timeline was measured here.

## Reproduce

Separate unmodified upstream build:

```sh
cmake -S /absolute/path/llama-b4 -B /absolute/path/b4-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DGGML_VULKAN=ON -DGGML_NATIVE=ON -DGGML_BACKEND_DL=OFF -DLLAMA_BUILD_COMMON=ON -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_MTMD=OFF -DLLAMA_CURL=OFF
cmake --build /absolute/path/b4-build --target llama-common -j 8
```

Compile this fork's identical replay helper against B4 headers and explicit B4 library directory, with `-std=c++17 -O3 -DNDEBUG`, `-lllama-common -lllama -lggml -lggml-base -pthread -ldl` and origin rpath; complete exact compiler argv is in metadata. Use fresh outputs:

```sh
env -i PATH=/usr/bin:/bin HOME="$HOME" XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" LC_ALL=C LD_LIBRARY_PATH=/absolute/path/frozen/B4 MOE_REPLAY_IN=/absolute/path/r22-244-128.csv MOE_REPLAY_REPS=1 MOE_REPLAY_LOGITS_OUT=/absolute/path/fresh-logits.bin /absolute/path/frozen/B4/replay -m /absolute/path/Ornith-1.5-35B-Q4_K_M.gguf -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off --verbosity 4 --load-mode mmap > fresh.csv 2> fresh.log
```

For the long gate use c8192 and3107+128token input. For timed short runs remove raw output, use REPS=3, validate every hash against the B4 canonical and vary only load-mode. Keep PP, TG, p95, initialization, RAM/GTT and semantic quality separate.
