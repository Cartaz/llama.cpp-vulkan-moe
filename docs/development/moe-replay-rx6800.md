# RX 6800 fixed-token replay and active-expert experiment, 2026-10-04

The fixed-token replay preserves complete model logits across upstream, compact OFF and compact ON. The new active-expert CPU path also preserves those logits, but does not demonstrate a throughput improvement. Both CPU experiments remain OFF by default. Prefill-informed cache analysis is an offline simulation; no GPU expert cache is implemented.

## Reproduction

Hardware: AMD Radeon RX 6800 16 GB (RADV NAVI21), Ryzen 7 5700X3D, 32 GB RAM. System: CachyOS kernel `7.2.9-1-cachyos`, Mesa/vulkan-radeon `26.2.4-1`, GCC `16.2.1+r23+gd564253eb6c8-1`, CMake `4.4.4-1.1`, Ninja `1.13.2-3.1`. The CPU governor is `performance`; GPU power profile is `BOOTUP_DEFAULT`. No clocks, power profiles or governors were changed.

Model: `Ornith-1.5-35B-Q4_K_M.gguf`, 21,713,462,848 bytes, SHA-256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`. Unmodified upstream inference libraries: `7fe450e19305b828c199d602c23a8337aaa1f03b` (v0.5.0). The replay source at local commit `624e9b7346a962e89683aa34b7bfbdf54d0effc7` has SHA-256 `4e804289e18e127fe5f0e2429010cff56f18a0eebcb701c0312e5f6cddb7b245`; the corresponding published source tree is commit `a3371be58ad2f5eae1ad71a29fbfa5f092fd82b2`.

All model runs use:

```sh
-m /home/casa/Programmi/modelli/Ornith-1.5-35B-Q4_K_M.gguf \
  -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512 \
  -fa on -ctk q8_0 -ctv q8_0
```

The workload is one recorded C++ review prompt, 244 prefill tokens and 128 evaluated greedy decode tokens. `agent-tokens.csv` has SHA-256 `85935136b367ed46e47e967b05495774bc290d2d32dd92776c0c242d744cf29c`. `MOE_REPLAY_IN` points to this CSV; `MOE_REPLAY_REPS` controls repetitions. Each process evaluates the complete workload once before measuring repetitions. Sequence memory is cleared before each workload. No routing callback or raw-logit output is enabled during performance runs. Hashing and CSV output are outside each timed inference call. The model uses 40 routed layers, 256 experts, top-8 selection; `-ncmoe 18` places layers 0..17 experts on CPU.

Local archives `risultati/2026-10-04-replay/` and `risultati/2026-10-04-active/` contain individual calls, commands, relevant environment variables, CMake caches, executable/library hashes, model/driver/kernel metadata, one-second RAM/VRAM/temperature/power/clock telemetry and scripts. `summary.json` retains each repetition and process, including decode call p50/p95. Performance means below first aggregate tokens/time within each process, then average independent process results. Repetitions within one process are not treated as independent process samples.

## Replay validation and compact workspace comparison

Seven malformed workload files are rejected. Upstream BASE, fork compact OFF and fork compact ON produce byte-identical complete logits after each prefill chunk and decode call: 128,133,120 bytes, SHA-256 `b7bb28a86e623daf75f436f9bddb786cee6c43d8a28d1b85333f3f25d9cbb662`. All per-call logits hashes also match across measured repetitions. The same replay source was compiled against BASE/OFF libraries without editing the upstream checkout; common/llama headers were verified identical before using that ABI.

Six processes, three measured repetitions each, in BASE/OFF/ON/ON/OFF/BASE order:

| Process | Build | Prompt token/s | Decode token/s |
| --- | --- | ---: | ---: |
| 1 | BASE | 171.447 | 30.048 |
| 2 | compact OFF | 173.341 | 29.832 |
| 3 | compact ON | 172.304 | 30.054 |
| 4 | compact ON | 172.983 | 30.121 |
| 5 | compact OFF | 172.480 | 29.848 |
| 6 | BASE | 172.364 | 29.358 |

Compact ON vs OFF process means: prompt -0.15%, decode +0.83%. With only two processes per variant and observed baseline variability, this does not establish a throughput improvement. The independently verified workspace reduction remains the reason to retain compact routing as an opt-in experiment. These real token calls cannot be directly compared with random-token llama-bench, tracing latency, total application latency, TTFT or concurrent serving.

## Active-expert CPU path

`GGML_CPU_MOE_ACTIVE=ON` builds an ascending active-expert list for up to 8 tokens and resets only active chunk counters. Workers traverse that list; larger batches retain the original traversal. This targets CPU dispatch overhead, not expert weight bandwidth or GPU work. Both main builds use `GGML_CPU_MOE_COMPACT=ON`; the only inference build option difference is `GGML_CPU_MOE_ACTIVE`. Builds are Release/native CPU/Vulkan with Ninja and tests enabled.

Validated CPU source corresponds to local `6d2ea1decc797dbca57d1914cef20b2b4be31613`, published as `10b0cade93078154852809eb8006b0e9c5d6a66b` with an identical tree. The builds were made before that commit, on `624e9b7` plus the archived `source.patch`; generated version strings therefore predate the source commit. `benchmark-metadata.json` identifies the measured source SHA-256s, commit and exact binary/library hashes. No rebuild or other benchmark ran during the eight model performance processes.

Validation:

- All 361 deterministic guarded cases pass in active OFF/ON, with byte-identical raw outputs. These include scalar F32 checks, Q4_K/Q6_K, strided and broadcast views, repeated IDs, changed inputs/routing on reused plans, multiple worker counts and the 8/9-token boundary.
- Both builds pass all 955 existing CPU `MUL_MAT_ID` cases sequentially. This suite shares compiled routing with its reference; the independent scalar/cross-build tests are necessary as well.
- Compact OFF / active ON also passes all 361 cases with identical outputs.
- Both active variants match the upstream full-model logits file byte for byte, retaining the SHA-256 above.
- The 2048-token Q4_K planned CPU workspace increases from 750,176 to 751,212 bytes with active traversal. No model weight or VRAM saving is claimed.

Eight processes, five measured repetitions each:

| Process | Active | Prompt token/s | Decode token/s | Peak total GPU VRAM, GiB |
| --- | --- | ---: | ---: | ---: |
| 1 | OFF | 168.702 | 22.479 | 13.246 |
| 2 | ON | 168.420 | 22.664 | 13.245 |
| 3 | ON | 167.946 | 22.419 | 13.258 |
| 4 | OFF | 132.801 | 29.236 | 13.318 |
| 5 | ON | 168.528 | 22.151 | 13.332 |
| 6 | OFF | 168.128 | 22.520 | 13.243 |
| 7 | OFF | 167.956 | 22.230 | 13.265 |
| 8 | ON | 133.550 | 29.235 | 13.298 |

Mean prompt: OFF 159.397, ON 159.611 token/s (+0.13%). Mean decode: OFF 24.116, ON 24.117 token/s (+0.004%). Process standard deviations are about 17.4-17.7 prompt token/s and 3.42 decode token/s. The process-local regimes dominate the difference between variants. Report all eight results; do not silently discard the faster-decode/slower-prefill regime. Peak process RSS is about 20.37 GiB; device-wide VRAM includes desktop applications and is not isolated model allocation.

A diagnostic BASE/OFF/OFF/BASE sequence using the untouched upstream libraries, three repetitions and verbose initialization, gives prompt/decode token/s of 168.263/22.603, 167.131/22.181, 132.701/28.850 and 167.343/22.604. Thus the upstream build also exhibits the lower decode regime; the new CPU path is not sufficient to explain it. Verbose logs show unchanged fitted parameters, 9049.31 MiB CPU-mapped model weights, 11532.90 MiB Vulkan weights, 498.52 MiB Vulkan compute buffer, 10.62 MiB KV and 62.81 MiB recurrent state. The root cause of the process variation is still unresolved.

The existing synthetic CPU operator benchmark was run OFF/ON/ON/OFF on Q4_K, 128 experts, top-8, m=768, k=2048. Mean microseconds per call:

| Tokens | OFF | ON | ON time change |
| --- | ---: | ---: | ---: |
| 1 | 53.240 | 53.860 | +1.16% |
| 4 | 202.580 | 200.920 | -0.82% |
| 8 | 395.675 | 394.105 | -0.40% |
| 32 (original traversal) | 1628.205 | 1615.415 | -0.79% |

Two processes per variant cannot resolve such small differences, especially with a similar change in the 32-token control. This is a CPU synthetic workload with 128 experts, not the 256-expert model workload or Vulkan throughput. No active-path speedup is established. The first operator CSV-format attempt omitted timing fields in upstream's printer; console reruns provide the usable measurements and both attempts are retained. Command timestamps confirm sequential execution.

## Prefill-informed policies

The analyzer adds `--phase decode --prefill-policies`. A warmed LRU carries prefill cache state into decode. A static cache freezes the most frequent prefill experts, with ascending expert ID as the tie-breaker. Decode frequencies never choose static slots. Hand-calculated fixtures verify cold/warm/static hits, layer isolation, no future-frequency leakage, phase-order rejection and rejection of a missing prefill. Existing JSON for prefill/decode/all on the real trace is unchanged without this flag.

For the same 244+128 trace, logical coverage on CPU layers 0..17 (18,432 decode activations):

| Slots per layer | Cold LRU | Warm LRU | Static prefill |
| --- | ---: | ---: | ---: |
| 16 | 40.99% | 41.54% | 32.30% |
| 32 | 55.96% | 56.91% | 47.42% |
| 48 | 66.39% | 67.97% | 57.19% |
| 64 | 72.18% | 74.85% | 65.54% |
| 96 | 79.41% | 84.20% | 76.64% |
| 128 | 82.57% | 90.17% | 83.52% |

These exclude priming cost, transfers, dense/KV capacity, asynchronous eviction and CPU fallback costs. They do not predict token/s. At small slot counts, freezing the prompt's hottest experts gives lower coverage than adaptive LRU on this one continuation. Validate on multiple held-out prompts and 1/2/4/8 streams before choosing a cache policy. This measurement is single-stream only.
