# RX 6800 host transfers and large-prefill copy experiment, 2026-10-04

Loading CPU expert weights in Vulkan host buffers improves single-stream prompt processing in two fixed-token workloads. This is a configuration improvement available in the existing backend, not a measured gain from the CPU routing patches. No decode throughput improvement is established. An isolated scheduler experiment preserved logits but made the 512-token workload 13.2% slower; it was rejected and removed from the branch.

## Video and source review

The requested [Codacus video](https://www.youtube.com/watch?v=VytSYCDhWQ0) and [Cloud Codes video](https://www.youtube.com/watch?v=_Jdjq6pgIRg) concern the same reported experiment. The second presentation is not an independent replication. Direct playback/transcripts were unavailable in this session; the review uses indexed descriptions/chapters and the author's public code.

The author's [original branch README](https://github.com/thecodacus/llama.cpp/tree/fable5/prefetch-experts) reports 1143 -> 1880 token/s prefill (+64.5%) for Qwen3.6-35B-A3B on RTX 3060 12 GB, `-ngl 99 -ncmoe 26 -p 2048 -n 0 -r 5 -b 2048 -ub 2048`. This is an author-reported CUDA result, not an RX 6800 replication. Token-identical output does not by itself prove byte-identical complete logits.

Relevant commits read from reference branch head `5e7f6271c06b9104862ab799278a1b7f1323a449`:

- [20f5994](https://github.com/thecodacus/llama.cpp/commit/20f5994bfeb91d24da328077c4b6095998cc9888) restores loader hooks for registering mmap CPU weights with CUDA, including page alignment and unregister-before-unmap lifetime. Reported pinning-only prefill: 1144 -> 1385 token/s.
- [1163cb3](https://github.com/thecodacus/llama.cpp/commit/1163cb34939fe4a9cb07aec034c5954144497ae9) uploads complete expert tensors on another CUDA backend/stream with event-ordered temporary buffers when `top_k * tokens >= 2 * experts`. It avoids routing-ID readback for those copies. It does not predict routing, change selected experts or provide persistent expert caching.
- [5f83fbb](https://github.com/thecodacus/llama.cpp/commit/5f83fbbe7c668c59912a1fe09e86a0ef580406c4) changes to three slots and fixes a fallback use-after-free: restore graph tensor pointers after launch, allocate replacements before freeing, size slots from graph maxima. A port must preserve these lifetime requirements.

These mechanisms primarily address batched prefill with CPU-resident experts. Their benefits do not establish faster one-token generation. Extra full-tensor traffic and VRAM slots can make small or sparse batches slower. A separate Vulkan backend object must not be assumed to provide an independent CUDA-like stream without checking queue/event semantics.

## Reproduction

Hardware, model and baseline are those in [the replay report](moe-replay-rx6800.md): RX 6800 16 GB/RADV NAVI21, Ryzen 7 5700X3D, 32 GB RAM; kernel `7.2.9-1-cachyos`; Mesa/vulkan-radeon `26.2.4-1`; GCC `16.2.1+r23+gd564253eb6c8-1`; CMake `4.4.4-1.1`; Ninja `1.13.2-3.1`. CPU governor `performance`, GPU profile `BOOTUP_DEFAULT`, unchanged throughout. Model SHA-256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`.

Runtime screen source tree: `6eab0aae2e67d69c616a88b28adeb6a804b727a6`, clean. Inference libraries in `build-B` are Release/native/Ninja/Vulkan, compact CPU ON, active CPU OFF. Replay binary SHA-256 `339f68d37a2923d0d213cbc0d707319a0377b95c073329520770d862a6ee8e64`; replay source SHA-256 `4e804289e18e127fe5f0e2429010cff56f18a0eebcb701c0312e5f6cddb7b245`. Upstream baseline remains unmodified at `7fe450e19305b828c199d602c23a8337aaa1f03b`. Archives retain exact binary/library hashes and CMake caches; generated version strings predate source publication as explained in the prior report.

Short workload: same 244 prompt + 128 evaluated continuation tokens as the replay report. Flags:

```sh
-ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512 \
  -fa on -ctk q8_0 -ctv q8_0 --fit off --verbosity 4 --load-mode mmap
```

The 512-token workload is a prefix of a recorded C++ review prompt, with no decode continuation. SHA-256 `0900b116875d6ff3f8cd9f97cfd95d72433489fc502e0acd3d9dcec7b30c0501`. It uses `-c 8192 -b 2048 -ub 512` and the remaining flags above. The replay calls at most `min(batch, ubatch)` tokens at once. This is a real-code token replay, distinct from both synthetic llama-bench and a live agent/server workload.

Each process warms the complete workload once, then clears sequence memory before every repetition. Only decode calls plus synchronization are timed; sequence clearing, hashing, raw-logit output and model initialization are outside that timer. No routing callback or raw output is used for performance. Results aggregate tokens/time per process and then average process rates; repetitions are not independent process samples. Only one stream is tested; 2/4/8-stream behavior remains unvalidated.

## Runtime screen

Nine processes, three measured repetitions each. Only the indicated configuration changes from control:

| Configuration | PP token/s | TG token/s |
| --- | ---: | ---: |
| control | 165.59 | 22.07 |
| load-mode none | 222.58 | 22.25 |
| decode threads 4, batch threads 8 | 164.63 | 23.34 |
| decode threads 6, batch threads 8 | 165.16 | 23.06 |
| physical-core affinity ff, strict, both pools 8 | 167.45 | 23.29 |
| GGML_VK_DISABLE_ASYNC=1 | 168.09 | 22.29 |
| GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 | 181.41 | 28.87 |
| RADV_PERFTEST=nogttspill | 79.15 | 4.87 |
| control repeated | 178.99 | 22.89 |

All measured call hashes match the original upstream short replay. Single-process screens identify candidates, not demonstrated gains. The previous report's process-local 22/29 token/s regimes still complicate TG interpretation. `nogttspill` strongly regresses this configuration and is not part of the candidate preset. Vulkan timestamp profiling was separate; gaps and synchronization can contribute to diagnostic durations, so grouped timestamps are not proof of pure kernel bottlenecks.

## Confirmed host-buffer PP improvement

`ggml_backend_vk_set_tensor_2d_async` directly enqueues a GPU transfer when `ggml_vk_buffer_write_2d_async` recognizes a Vulkan host allocation. An ordinary mmap pointer falls back to a CPU staging copy plus `ggml_vk_synchronize`. Loading with `--load-mode none` actually produces `Vulkan_Host` expert allocations in these runs. This makes a relevant part of the pinning idea testable without a CUDA patch.

Short workload, five repetitions/process, mmap/none/none/mmap:

| Process | Load | PP token/s | TG token/s |
| --- | --- | ---: | ---: |
| 1 | mmap | 169.732 | 29.006 |
| 2 | none | 239.708 | 22.838 |
| 3 | none | 239.877 | 22.851 |
| 4 | mmap | 178.105 | 22.591 |

PP process means: 173.919 -> 239.793 token/s (+37.88%). TG means: 25.799 -> 22.844 (-11.45%); the distinct process regimes confound attribution, but this possible cost must not be hidden. The standalone replay linked to unmodified upstream libraries with load-mode none produces the original complete raw logits byte for byte (128,133,120 bytes, SHA-256 `b7bb28a86e623daf75f436f9bddb786cee6c43d8a28d1b85333f3f25d9cbb662`). Performance call hashes match as well.

512-token prefix, three repetitions/process, mmap/none/none/mmap: PP 286.85/365.82/364.83/286.96 token/s. Means: 286.91 -> 365.32 (+27.3%). All call hashes match between load modes and repetitions. This is a configuration gain, not a new source-code gain.

Costs: load initialization approximately 12 s vs 4 s, CPU expert weights `Vulkan_Host` 8642.81 MiB vs `CPU_Mapped` 9049.31 MiB, peak total GTT approximately 9.6-9.8 vs 0.9 GiB, minimum system MemAvailable approximately 17.4 vs 23.3 GiB in the short comparison. GPU-resident weights remain 11532.90 MiB; device-wide VRAM also includes desktop allocations. Driver-owned host allocations are not fully represented by process RSS. The much lower RSS in none mode does not demonstrate lower physical RAM use. This preset needs memory-budget and startup-latency checks for the intended workload.

## Full-copy hypothesis and routing coverage

Routing recorded on the first code-prompt chunk, CPU layers 0..17, 256 experts, top-8:

| Prefix tokens | Mean unique experts/layer | Min/max | Ideal full/selected expert payload ratio |
| --- | ---: | --- | ---: |
| 64 | 108.17 | 71/171 | 2.37 |
| 128 | 148.61 | 105/204 | 1.72 |
| 256 | 182.50 | 139/235 | 1.40 |
| 512 | 209.00 | 170/251 | 1.22 |

The video threshold would fire at 64 tokens, but nearly all experts are not used in this prompt. These are logical unions from one tracing run, not measured PCIe bytes or a general distribution. The ratios exclude grouping/padding overhead and assume equal payload per expert. They motivate a conservative experiment threshold and a separate measurement of copy costs.

## Isolated scheduler experiment

The archived prototype `GGML_SCHED_MOE_FULL_COPY=1` opted into a full upload of CPU-resident MoE weight tensors when the routed batch has at least 512 tokens and no evaluation callback. Default OFF. It keeps the existing destination event wait and source backend synchronization, uses the same destination allocation, then skips only the routing-ID download and selected-expert grouping. It changes no arithmetic, tensor lifetime, routing decisions or decode path. It does not add a second backend, prefetch slots or an expert cache.

This is a 12-line experiment targeting readback/copy-call overhead. It trades additional expert payload for fewer host interactions and deliberately separates that hypothesis from the video's multi-stream overlap. An isolated Release/native/Vulkan `build-moe-transfer` uses the same CMake settings as `build-B`; the original builds remain untouched. Its archive records the exact patch against `6eab0aa`, compiler/build commands, cache and binary/library hashes.

On the 512-token prefix in load-mode none, three-repetition raw logits are byte-identical between experiment OFF/ON and match the original build's call hash. A separate debug run confirms 54 full weight uploads per prefill pass, 108 including warmup and the measured replay; debug logging is disabled for the performance comparison.

Four processes, five repetitions each, OFF/ON/ON/OFF, load-mode none, same binary and flags:

| Process | Full copy | PP token/s |
| --- | --- | ---: |
| 1 | OFF | 366.011 |
| 2 | ON | 317.579 |
| 3 | ON | 317.798 |
| 4 | OFF | 366.013 |

Means: OFF 366.012, ON 317.689 token/s (-13.20%). Every measured hash matches the original build. Raw-logit validation spans 2,979,840 bytes, SHA-256 `c2f4c7cf2977d3008c885c268cd6f6dc9e3a8c7c73719e4f8b381fdeb899452f`. The experiment is rejected for this workload; its scheduler changes were removed before publication. The patch and isolated binary remain archived for reproducibility. No environment variable from this prototype is supported by the resulting branch.

This rejects full copying without additional overlap at 512 tokens on this configuration. It does not measure the author's second-stream prefetch, larger batches or other prompt distributions. A later overlap experiment must isolate transfer queue concurrency, event dependencies, buffer lifetimes, allocation fallback and additional VRAM before testing 512/2048/4096 prompts and generation independently. The long-prompt correctness issue below must also be resolved for those larger workloads.

## Long-prompt correctness investigation

The first 4096-token prefill attempt (`-c 8192 -b 2048 -ub 512`, mmap, three repetitions) failed the identical-final-hash check. The first chunk is stable, later chunks vary. The original files are retained; the script failure is not discarded as timing noise. Workload SHA-256 `d12d7e429c2e6b75498d912c4e779aae88e867cefd3dd80e8395aee34a1ed8c5`.

Three-repetition raw-logit diagnostics against unmodified upstream also vary: maximum absolute difference 4.5844, maximum per-call RMS 0.6812. The fork varies too (max 1.4382, RMS 0.2587). All captured values are finite and argmax matches at all sampled chunk endpoints, including BASE-vs-fork, but that does not certify identical long-prompt inference. This is not a measured regression introduced by the new full-copy experiment, which was not applied at that point.

Separate checks with full memory zeroing, `LLAMA_GRAPH_REUSE_DISABLE=1` and `GGML_VK_DISABLE_ASYNC=1` do not eliminate variation at 4096 tokens. A 1024-token control was stable in one process; other diagnostic processes varied, so shorter length alone does not establish a universal boundary. Root cause remains unresolved. Do not promote the 4096-token or larger-ubatch performance results until correctness is understood. The validated short replay and first 512-token chunk are reported separately.

Archives: `risultati/2026-10-04-runtime/` and `risultati/2026-10-04-full-copy/` retain full commands, environment variables, telemetry (including system MemAvailable), source patch, CMake caches, binary hashes, raw logits and diagnostics. Large outputs remain local and are not checked into Git.
