# MoE profiling foundations: S01/S02

Date: 2026-10-06. Base fork: `45b5cf1df08d8dfe52190c13c160d3262dd9b3ec`, on top of `c679dc22012ee9281db57336ff8fd17a93bc54b2`. Original baseline: v0.5.0 `7fe450e19305b828c199d602c23a8337aaa1f03b`. Implementation branch: `experiment/moe-profile-foundations`.

This implements the first offline tools in S01/S02. It does not implement an expert GPU cache, resume the paused benchmark campaign, or establish a PP/TG benefit. M1/M2 gates remain incomplete. The backend, routing, weights and application defaults are unchanged.

## Bottleneck and scope

Selecting a cache budget from activation counts alone can favor small experts, and one short routing trace does not describe reuse or a domain change. S02 now reports entropy, reuse and multiple working-set windows. These measurements can inform S04-S06 on RX 6800, where cache capacity competes with dense weights, KV, recurrent state and compute buffers in 16 GiB. Whether a cache improves decode still requires measured miss cost and an end-to-end A/B.

S01 adds a live-process manifest. Dynamic backend libraries can differ from the executable's build or the libraries returned by `ldd`; hashing paths found in the running process helps detect that confounder. Hashing consumes I/O and must run outside inference timers. The tools use the Python standard library and do not load a model or launch a benchmark.

## Routing analysis

Existing CLI options and per-layer JSON fields remain available. JSON schema version 2 adds `evaluated_tokens`, `entropy_bits`, `entropy_observed_normalized`, `cold_activations`, two reuse histograms, `working_set_windows` and `cache_summary`.

```sh
python3 examples/moe-trace/analyze.py record-routing.csv \
  --phase decode --slots 0,16,32,64,96,128 --prefill-policies \
  --windows 1,8,32,128 --json decode-profile.json
python3 examples/moe-trace/analyze.py record-routing.csv \
  --phase prefill --slots 0,16,32,64,96,128 \
  --windows 1,8,32,128 --json prefill-profile.json
```

- Entropy is Shannon entropy in bits, over distinct expert activations per token. The normalized value divides by `log2(observed_unique_experts)`, not the model's total expert count. A single observed expert has value zero. This is concentration in the observed trace, not proof that unseen experts are inactive.
- `reuse_token_gap_histogram` counts evaluated token groups strictly between two uses of an expert in one layer. Consecutive uses have gap zero. Missing absolute token positions do not add evaluated groups.
- `reuse_distinct_intervening_histogram` counts distinct experts in those intervening groups. Experts co-selected in the previous or current top-k group do not create a distance. This atomic group statistic is not a rank-ordered LRU stack distance; use the separate cache simulation to estimate LRU coverage.
- Cold activations have no prior use in the selected analysis. They are reported separately from finite reuse distances.
- Window statistics include complete windows only and remain null when too few groups exist. New histograms retain counts, not the entire token history; memory scales with window lengths, observed experts and histogram bins. Exact distinct reuse computation scans the observed expert set per activation, so analysis time can grow with large expert counts.
- `--phase all` joins prefill/decode history. Phase-specific analyses start empty; `--prefill-policies` separately warms caches from prefill while reuse/entropy statistics still cover decode only.

Groups must be contiguous, ranks must increase from zero, and token positions must increase independently within each phase/layer. Layer-major batches from the trace callback are supported. Duplicate expert IDs in a top-k group retain the existing distinct-activation semantics. Analyze separate requests separately; do not concatenate traces that reset token positions. No multi-sequence cache simulation is claimed.

### Byte weighting

Supply a JSON object mapping every evaluated layer to the storage bytes for one complete routed expert. Include all applicable gate/up/down tensors; for fused gate/up, include the fused tensor once. Shared experts and dense weights are excluded. Derive sizes from the actual model's tensor types and layout; no default or model size is guessed.

```sh
python3 examples/moe-trace/analyze.py record-routing.csv \
  --phase decode --slots 0,16,32,64 --prefill-policies \
  --expert-bytes expert-bytes.json --json byte-profile.json
```

The input shape is `{"0": positive_integer_bytes, "1": positive_integer_bytes, ...}`. All layers must be present; missing entries, booleans, non-integral values and zero/negative sizes are rejected. One size is assumed for all experts in a layer. Models with heterogeneous expert sizes within a layer need a future format extension.

Per-layer cache entries add capacity, requested, hit and miss bytes. Global summaries include `byte_hit_rate = sum(hit_bytes) / sum(requested_bytes)` and the total modeled cache capacity. Cold LRU, warm LRU and static prefill use the same slots and byte budget. Byte totals count one complete expert request per evaluated token and distinct expert, exclude initial warm/static priming, and can count the same expert repeatedly. They are logical estimates, not scheduler H2D traffic: real prefill copies can share weights across many tokens, copy padding or gaps, and include other tensors. Metadata, alignment, staging and in-flight allocation costs are also excluded. S03/S06 must measure those costs before setting a GPU budget.

## Runtime manifest

Capture an already started process after backend loading, with benchmark timing paused. This command does not start it:

```sh
python3 examples/moe-trace/manifest.py \
  --pid "$BENCH_PID" --repo . --cmake-cache build-vulkan/CMakeCache.txt \
  --workload agent-tokens.csv --model /path/to/Ornith-1.5-35B-Q4_K_M.gguf \
  --log initialization.log --require-vulkan --json candidate-freeze.json
```

The manifest records the commit and tracked diff, CMake cache hash and build settings, executable hash, mapped `.so` file hashes, command, CPU affinity, relevant target-process environment, absent known flags, workload/model hashes, kernel, CPU, installed build-tool/Mesa package versions, initialization log, process/machine memory snapshots and DRM counters. Preserve untracked sources separately. The installed compiler version is the compiler at the CMake path now; it cannot prove which version compiled an older binary. Retain build logs too.

Model hashing is optional and can read tens of gigabytes. For a complete campaign manifest, provide `--model` during untimed preparation. With no model, the field is absent, not verified. Environment variables are read from the target's initial `/proc/PID/environ`; later in-process changes may not appear. Retain effective configuration logs, not just requested arguments. Relevant environment capture excludes unrelated variables and credentials.

The tool refuses unreadable/deleted/replaced mapped libraries, files that change during hashing, and a process whose start-time identity changes. When `maps` and `stat` device IDs differ, it records both IDs, sets `library_device_identity_complete=false`, and reports the limitation in `optional_capture_errors`. `--strict-map-device` instead fails. Inode agreement alone is not complete identity verification. File hashes identify the current file on disk; they do not hash relocated process memory or prove that an in-place modified library still matches mapped code. Freeze artifacts before process launch and preserve them throughout the run.

`--require-vulkan` requires a mapped `libggml-vulkan` library. This is a guard against an unloaded backend, not proof that work executes on RX 6800. Keep the runner's pre-load device validation, memory reserve, cgroup limits, GPU contention checks and effective placement inspection. GPU/RAM values are snapshots, not peaks or PCIe transfer counters. Optional tool-version failures are recorded explicitly.

To enforce the same frozen artifacts/configuration on another process of the **same variant**:

```sh
python3 examples/moe-trace/manifest.py \
  --pid "$BENCH_PID" --repo . --cmake-cache build-vulkan/CMakeCache.txt \
  --workload agent-tokens.csv --model /path/to/Ornith-1.5-35B-Q4_K_M.gguf \
  --require-vulkan --expect candidate-freeze.json --json candidate-repeat.json
```

The check rejects changed source SHA/diff, binary, CMake/build settings, environment, arguments, affinity, workload/model identity or mapped library paths/hashes/set. Take one reference per B1/B2 variant; their inference libraries are intentionally different. Repeat capture at the same backend-loading stage. This does not validate output, latency or numerical correctness. Failed capture/verification exits with an error without writing a new report; use unique output paths so an older file cannot be mistaken for a successful new capture.

## Validation and next A/B

```sh
python3 examples/moe-trace/test-profile.py -v
git diff --check
```

The suite checks legacy phase/LRU/prefill behavior, atomic reuse against an independent history-union oracle, short windows, entropy, unequal per-layer byte weighting, byte-budget equality across policies, invalid inputs, artifact hashes, mapped-library identity checks, frozen-manifest drift and live capture of the Python test process without loading a model. The historical stride-corrected `record-routing.csv` from `risultati/2026-10-04-replay/` also reproduced every existing per-layer field for prefill, decode and all (40 layers each). This is offline compatibility evidence, not a new inference benchmark.

All code changes run outside the inference path. No inference rebuild is required. PP, TG, TTFT, semantic quality and runtime memory benefit are **NON_MISURATO** for this change. Existing benchmark pauses remain in force.

After a requested campaign resume, first freeze separate B1/B2 controls with the same model, fixed token workload, effective placement, batch/ubatch and runtime environment. Reproduce T0 logits and finite/nonzero checks, then T1 process-level controls before tuning. Collect S02 traces separately on Italian, code, tools and long chat, with held-out requests never used to choose static experts. Compare cold/warm/static at the reported equal byte capacity, per phase and domain. Use S03 timestamps/CPU markers and measured upload/readback bytes to distinguish CPU compute, transfer and synchronization; offline hit rates cannot substitute for that A/B. S03 timing instrumentation, corpus collection, byte-budget quota planning and S04/S05 remain next work.

## Source review

Read target and baseline `ggml/src/ggml-backend.cpp`; those files are byte-identical in the checked local snapshots. Also consulted the [upstream scheduler](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-backend.cpp), [routing trace PR #28544](https://github.com/ggml-org/llama.cpp/pull/28544), [pool RFC #28248](https://github.com/ggml-org/llama.cpp/discussions/28248) and [prefill transfer issue #25859](https://github.com/ggml-org/llama.cpp/issues/25859) on 2026-10-06. These references motivate measurement and lifetime checks; their reported performance is not a result for this RX 6800 implementation.


## ReBAR identity guard (R38, 2026-10-08)

Manifest schema2 records PCI device address, driver, vendor/device/subsystem IDs, total and CPU-visible VRAM, PCI resource sizes/flags and maximum link capabilities. Current link speed/width and used memory remain snapshots. Hardware verification compares stable identity, kernel, CPU and architecture; it excludes PCI base addresses, DRM card numbering, busy counters, used memory and current link state, which can change with power management.

`--expect` now rejects a different BAR aperture or visible VRAM, including the256MiB to16GiB transition. Regenerate schema1 freeze manifests before using the schema2 hardware guard: missing identity cannot verify the old hardware. The manifest records measurements; it does not change firmware/ReBAR or assign a throughput gain to a transfer mechanism. Linux documents `mem_info_vis_vram_total` as CPU-visible VRAM [here](https://docs.kernel.org/gpu/amdgpu/driver-misc.html).

Target environment capture also includes `LLAMA_MOE_`, all `MOE_`, `MESA_` and `OMP_` variables so input-precision, fixture and CPU-thread controls participate in freeze checks. None of these fields run in the inference path. The existing offline suite tests small/large BAR transitions, unused resources, invalid ranges, dynamic busy counters and a live Linux snapshot;21tests pass. R38 real-model results and exact protocol are in the separate ReBAR report.
