# S03 initial scheduler profiling on RX 6800

Date: 2026-10-06. Status: IMPLEMENTATO / TEST_PARZIALI. Branch: `experiment/moe-scheduler-profile`, based on `fad18a1af87c9567a97fdc3645057858156b419e` (S01/S02 foundations). Exact tested source and binary hashes are in [validation metadata](moe-scheduler-profile-validation.json). No model benchmark or answer-quality campaign was resumed. PP, TG, TTFT, profiler overhead and performance gains are NON_MISURATO.

## Purpose and scope

S03 prepares measurements before choosing a persistent expert cache or a kernel optimization. On the RX 6800, host routing, expert upload and synchronization can contribute to latency along with GPU compute. This is a hypothesis to measure on the model, not a conclusion from the small operator fixture.

`GGML_SCHED_PROFILE=/absolute/path/fresh.csv` enables host scopes in the existing GGML scheduler. Absent or empty means OFF. The patch adds no GPU synchronization and does not change expert selection, copy regions or graph placement. OFF avoids clocks, file writes and profiling locks, but retains scope objects and branches; zero overhead is not claimed. ON serializes CSV writes and allocates quoted names, so its overhead must be measured separately.

The profile covers existing allocation, scheduler splits, router readback, routing scan, expert uploads, generic tensor copies, backend compute calls and waits. Existing Vulkan timestamp logging is reused for a separate diagnostic run; this patch does not add a second Vulkan query system.

## CSV interpretation

The schema is `scheduler,call,split,event,source,destination,tensor,start_us,duration_us,bytes,padding_bytes,buffer_bytes,status`. Times use the existing host monotonic microsecond clock. Divide by 1000 to express milliseconds. Scheduler IDs are process-local; call IDs count graph-compute invocations. Explicit allocation before the first compute has call 0. `scheduler_end` is a completion marker when the scheduler is freed, not a measurement of backend destruction.

| Event or counter | Meaning and limit |
| --- | --- |
| `compute_splits`, `split` | Inclusive host envelopes. Nested events and their logging contribute to these times. Do not add envelopes to child scopes. |
| `compute_call` | Duration of the backend graph-compute API. CPU may compute synchronously; Vulkan can enqueue work. This is not GPU active time or a per-node duration. The tensor name identifies the first node or callback boundary. |
| `input_wait`, `source_wait`, `split_wait`, `copy_wait`, `callback_wait`, `scheduler_wait` | Host time inside preexisting synchronization calls. A wait can cover earlier queued work; it is not an additional independent compute cost. |
| `event_wait_enqueue` | Time to request an existing queue/event dependency; not its eventual GPU stall duration. |
| `router_readback` | API read plus its existing synchronization. Inspect source backend: IDs can originate on CPU. Byte span can include stride gaps. |
| `expert_upload` | One requested selective-copy region; bytes include padding, with padding also reported separately. |
| `tensor_copy_async`, `tensor_copy` | Requested tensor payload. A rejected async attempt records zero bytes; its synchronous fallback records the payload. |
| `buffer_bytes` on `split` | Scheduler-reserved buffer size. Excludes model weights, KV/recurrent state and other backend allocations. Maxima are not physical VRAM residency. |

Copy payloads are not measured PCIe traffic. Overlapping scopes must not be summed as elapsed time. Host timestamps and existing GPU query durations do not establish a shared-clock timeline or overlap percentage.

Use one fresh regular output file per process; do not share it across processes. Within one process, schedulers use a common lock and unbuffered streams so headers and complete records are visible between scheduler writes. Open/write failure logs a warning and disables profiling for that scheduler while inference continues. Profile paths can fail independently of inference correctness.

```sh
python3 examples/moe-trace/profile-summary.py /absolute/path/fresh.csv --json /absolute/path/summary.json
```

The streaming parser validates the schema, counters, operation status, increasing call IDs and completion markers. It rejects failed or unfinished profiles and reused completed scheduler IDs. The summary groups event/source/destination/tensor and reports inclusive compute time separately. It does not assign prompt/decode phases or calculate tokens/s.

## Operator correctness and exact byte checks

`llama-moe-scheduler-check` builds F32 MUL_MAT_ID graphs with 32 input columns, 8 output rows, 8 experts, top-k 2 and 1 or 3 tokens. It tests contiguous and strided IDs, callbacks OFF/ON, scheduler parallel mode OFF/ON, and two executions of the same allocated graph. The second execution changes input values and repeats expert 7. A CPU preprocessing split exercises strided router readback; a separate VIEW-first case retains the existing full-copy fallback.

Every output element is checked against an independent scalar dot product. Power-of-two inputs permit exact equality. All outputs must be finite and each execution must have nonzero output. The Vulkan option requires an RX 6800 GPU and checks actual scheduler placement; it refuses CPU fallback.

Each runtime-suite invocation compares profiler OFF/ON raw output and covers 16 scheduler instances / 32 operator evaluations. It also verifies an empty environment value, an unopenable path and, on Linux, a failed write to `/dev/full`. The failure cases produce the same raw output.

For CPU-source weights and Vulkan compute, one expert is 1024 bytes:

| Execution | Expected selective API payload | Padding | Regions |
| --- | --- | --- | --- |
| First, 1 token: experts 0,1 | 2560 bytes | 512 bytes | 1 |
| First, 3 tokens: experts 0,1,4,7 | 5120 bytes | 1024 bytes | 3 |
| Second, repeated expert 7 | 1024 bytes | 0 bytes | 1 |
| VIEW-first full-copy fallback, either pass | 8192 bytes generic copy | Selective counter absent | No selective regions |

Router-read spans are 8 bytes for 1 token, 24 for contiguous 3-token IDs, and 136 for the CPU strided view. The suites verify these exact counters, not just the presence of events. All CPU Release, CPU ASan/UBSan and RX 6800 suites passed (3 tests each); S01/S02 offline tests passed (13). Full raw output is byte-identical across profiler OFF/ON and the prebuilt control executable: SHA-256 `b8ccf66c028651dc191d799d009ffd6fc97887c28fbf4aa2a7781e48f1c7f5a2`.

## Reproduce the checks

Commands run from the repository root. These are correctness probes, not performance benchmarks.

```sh
cmake -S . -B /tmp/moe-s03-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=OFF -DGGML_CPU_MOE_COMPACT=OFF -DGGML_CPU_MOE_ACTIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_CURL=OFF
cmake --build /tmp/moe-s03-build --target llama-moe-scheduler-check -j2
python3 examples/moe-trace/test-scheduler-profile.py --check /tmp/moe-s03-build/bin/llama-moe-scheduler-check -v

cmake -S . -B /tmp/moe-s03-asan -G Ninja -DCMAKE_BUILD_TYPE=Debug -DGGML_VULKAN=OFF -DGGML_CPU_MOE_COMPACT=OFF -DGGML_CPU_MOE_ACTIVE=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_CURL=OFF '-DCMAKE_C_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer' '-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer'
cmake --build /tmp/moe-s03-asan --target llama-moe-scheduler-check -j2
python3 examples/moe-trace/test-scheduler-profile.py --check /tmp/moe-s03-asan/bin/llama-moe-scheduler-check -v
python3 examples/moe-trace/test-profile.py -v
```

Local Vulkan checks reuse the existing, unchanged `build-control-off/bin/libggml-vulkan.so`, linked to the newly built scheduler/base/CPU. This avoids replacing benchmark libraries. It is not a clean full Vulkan rebuild. Backend interfaces were unchanged; `ldd` and binary hashes are archived. The source revision of the older prebuilt control binaries is not independently certified by this check.

```sh
c++ -std=c++17 -DGGML_SCHED_CHECK_VULKAN examples/moe-trace/scheduler-check.cpp -Iggml/include -L/tmp/moe-s03-build/bin -Lbuild-control-off/bin -Wl,-rpath,/tmp/moe-s03-build/bin -Wl,-rpath,/home/casa/Programmi/Ottimizzazione_llama_cpp_moe/llama-moe-ab/build-control-off/bin -lggml -lggml-base -lggml-cpu -lggml-vulkan -o /tmp/moe-s03-check-vulkan
python3 examples/moe-trace/test-scheduler-profile.py --check /tmp/moe-s03-check-vulkan --vulkan -v
c++ -std=c++17 -DGGML_SCHED_CHECK_VULKAN examples/moe-trace/scheduler-check.cpp -Iggml/include -Lbuild-control-off/bin -Wl,-rpath,/home/casa/Programmi/Ottimizzazione_llama_cpp_moe/llama-moe-ab/build-control-off/bin -lggml -lggml-base -lggml-cpu -lggml-vulkan -o /tmp/moe-s03-check-control
```

A clean build with `GGML_VULKAN=ON` also exposes `--vulkan` through the CMake target, but that complete rebuild was not run here. Environment at validation had no GGML/RADV/VK/MESA overrides or LD_LIBRARY_PATH. System: kernel `7.2.9-1-cachyos`, GCC `16.2.1 20260810`, CMake `4.4.4`, Ninja `1.13.2`, Mesa/RADV `26.2.4-arch3.1`, Vulkan API `1.4.354`, RX 6800 / RADV NAVI21. CPU build: native ON, dynamic backend loading OFF, compact/active MoE experiments OFF.

An additional existing `test-backend-ops` executable was run against the new CPU/base libraries in correctness mode: four Q4_K/Q6_K MUL_MAT_ID cases with and without broadcast passed. This is a narrow operator regression check; the independent fixture above provides the CPU/Vulkan scheduler checks. The full backend suite and CI were not run.

```sh
LD_LIBRARY_PATH=/tmp/moe-s03-build/bin:/home/casa/Programmi/Ottimizzazione_llama_cpp_moe/llama-moe-ab/build-control-off/bin build-B/bin/test-backend-ops test -b CPU -o MUL_MAT_ID -p 'n_mats=256,n_used=8.*n=1,'
```

## Existing GPU timestamp logger limitation

With `GGML_VK_PERF_LOGGER=1`, both the prebuilt control and candidate abort at `GGML_ASSERT(ctx->compute_ctx.expired())` in the parallel + strided + no-callback case, after the first ten fixture cases pass. This reproduces without `GGML_SCHED_PROFILE`; the assertion alone does not establish its root cause. It is a shared diagnostic-path limitation, not a validated upstream-v0.5.0 defect or a performance result.

Without the GPU timestamp logger, all parallel and serial fixture cases pass. Restricting the existing logger to serial cases gives 16 scalar-checked evaluations and identical candidate/control raw output:

```sh
GGML_VK_PERF_LOGGER=1 /tmp/moe-s03-check-control --vulkan --output-bin /tmp/control-all.bin
# Reproduces the assertion; the resulting run is invalid.
GGML_VK_PERF_LOGGER=1 /tmp/moe-s03-check-control --vulkan --serial --output-bin /tmp/control-serial.bin
GGML_VK_PERF_LOGGER=1 GGML_SCHED_PROFILE=/tmp/fresh-serial.csv /tmp/moe-s03-check-vulkan --vulkan --serial --output-bin /tmp/candidate-serial.bin
cmp /tmp/control-serial.bin /tmp/candidate-serial.bin
```

The existing logger can alter synchronization and is for diagnostic runs. Its successful serial probe does not validate arbitrary graphs, concurrent requests or PP/TG attribution. Relevant implementation references: [upstream scheduler](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp) and [upstream Vulkan backend](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp), checked on 2026-10-06. The actual tested fork artifacts are identified by the validation metadata.

## Remaining gates and next implementation

S03 remains partial: add explicit replay phase/call correlation, validate GPU timestamp logging on the intended graph, and account for weights/KV/recurrent/compute/staging plus resident/requested VRAM and GTT. Then assemble the exposed critical path instead of adding overlapping durations. A persistent cache design must retain exact routing and in-flight slot ownership; the current patch provides no cache.

On authorized benchmark resumption, use frozen model/token inputs and identical placement, ubatch, KV and environment. First compare profiler OFF against host-profile ON in balanced fresh-process order, measuring PP/TG/TTFT and profiler overhead. Run GPU timestamp diagnostics separately and reject assertion/NaN/fallback cases. Validate all-hit/all-miss/cold/warm copy accounting before S04/S05; compare end-to-end elapsed time after optimizing the largest exposed cost. Keep model benchmarks separate from these synthetic correctness checks.

Local raw archive: `risultati/2026-10-06-scheduler-profile/`, containing logs, CSV/JSON, outputs, compiler caches, `ldd`, Vulkan inventory, source snapshots, library hashes and artifact checksums. It does not overwrite earlier campaign results. The small validation manifest is versioned with this report.
