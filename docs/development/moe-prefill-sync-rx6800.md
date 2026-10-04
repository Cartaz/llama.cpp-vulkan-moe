# RX 6800 long-prefill synchronization, 2026-10-04

An optional GPU barrier before Vulkan Flash Attention with Q8_0 K and V makes the tested 4096-token replay deterministic and byte-identical to a layer-fenced reference. This is a conservative experimental workaround, not an identified root-cause fix or a demonstrated throughput optimization. It remains disabled by default.

```sh
GGML_VK_FA_Q8_SYNC=1 ./build-moe-transfer/bin/llama-moe-replay ...
```

The variable is checked for presence once per process. Unset it to disable the option; setting it to `0` still enables it. The change adds five lines to `ggml_vk_flash_attn` and calls the existing `ggml_vk_sync_buffers` only when both attention K and V have type Q8_0. Native F16 KV and other cache types do not get this extra barrier. It inserts a GPU memory/execution dependency without a CPU fence wait, queue change, arithmetic change, new allocation or persistent expert cache. It can reduce GPU overlap; its cost must be measured on each workload.

## Reproduction

Parent source: `2f6f6c35e58a41d051d086a3db25f9fce2d9535d`, tree `ace8825914da956e23ab15c0acb0e0e29ea671c7`, plus the archived five-line patch. Release/native/Ninja/Vulkan, compact CPU ON, active CPU OFF. Original `build-B` and unmodified upstream baseline `7fe450e19305b828c199d602c23a8337aaa1f03b` remain unchanged. Final Vulkan source blob `06f3564cbbc9761c5076cb2435c113a9a356927c`; replay binary SHA-256 `314cb6fbdb2729cb9fae3c534c437209436da87d19363e41aa340b020cad948d`; Vulkan library SHA-256 `d909e7f5918fd1e06a53da2a9648eaa76d6c46aecec967f29fc61e9766bc2e18`. The isolated build directory was reused after the rejected full-copy experiment; old experiment hashes describe earlier binaries, not the current directory contents.

RX 6800 16 GB / RADV NAVI21, Ryzen 7 5700X3D, 32 GB RAM; kernel `7.2.9-1-cachyos`; Mesa/vulkan-radeon `26.2.4-1`; GCC `16.2.1+r23+gd564253eb6c8-1`; CMake `4.4.4-1.1`; Ninja `1.13.2-3.1`. CPU governor `performance` and GPU `BOOTUP_DEFAULT`, unchanged. Model: Ornith-1.5-35B-Q4_K_M.gguf, SHA-256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`.

Long workload is the first 4096 recorded tokens of a C++ review prompt, no decode continuation, SHA-256 `d12d7e429c2e6b75498d912c4e779aae88e867cefd3dd80e8395aee34a1ed8c5`. Flags:

```sh
-ngl 99 -ncmoe 18 -t 8 -tb 8 -c 8192 -b 2048 -ub 512 \
  -fa on -ctk q8_0 -ctv q8_0 --fit off --verbosity 4 --load-mode mmap
```

The replay evaluates one ubatch per call and captures the last-token logits of each call, not logits for every prompt token. Each process warms the full workload once and clears sequence memory before each measured repetition. The short workload and baseline raw-logit reference are described in [the replay report](moe-replay-rx6800.md). Only one stream is tested here; concurrency and live server behavior remain unvalidated.

## Isolation results

The previous [host-transfer report](moe-host-transfer-rx6800.md) records large differences between repeated long-prefill logits in both the fork and unmodified baseline. Full memory zeroing, graph-reuse disabling and async disabling did not eliminate them.

Additional diagnostic runs are not valid performance measurements:

| Diagnostic | Result |
| --- | --- |
| load-mode none | Later chunks still vary |
| GGML_VK_SERIALIZE_SUBMISSIONS=1 | Identical hashes, but every captured logit is zero; rejected |
| GGML_VK_DISABLE_FUSION=1 | Millions of nonfinite values in this configuration; rejected |
| GGML_VK_DISABLE_GRAPH_OPTIMIZE=1 | Later chunks still vary |
| Readback callback on first four layers | Traced logical tensor values stable, later logits vary |
| Readback callback on every layer output | All eight endpoint logits stable |
| GPU barrier before every operation/fusion | Three repetitions exactly match the layer-fenced endpoint hashes |
| Barriers before copies / matmul / recurrent ops separately | Each group leaves some later chunks variable |
| Barriers before Flash Attention only | Three repetitions match the all-barrier raw reference |
| Barriers before RoPE only | Three repetitions also match the all-barrier raw reference |

The node callback hashes logical tensor elements using all strides, excluding padding. Callbacks alter graph splitting/fusion and synchronization; they do not identify the first failing node by themselves. The barrier-group prototypes scan every member of a fused operation. They are archived diagnostics and are not supported environment modes on the branch.

These results implicate synchronization but do not uniquely identify a missing tensor dependency, an auxiliary scratch hazard or a driver issue. Both attention-only and RoPE-only barriers work, so attributing the fault specifically to Flash Attention would overstate the evidence. An earlier five-line prototype inserted the barrier for every Flash Attention call. It passed the Q8_0 replay but produced 3,228,160 nonfinite captured values in the F16 control. It was rejected. A narrower trial gated on `use_dequant_kv` still leaves a variable later endpoint, so that predicate is insufficient. The final option is restricted by the actual K/V types, both Q8_0, and does not claim to identify the failing internal pass. The F16 layer-fenced reference also varies (max absolute difference 1.499, RMS up to 0.301), so F16 is not promoted as an alternative configuration. A final-option F16 control is finite and nonzero, but remains variable; the type guard excludes it from the extra barrier. The Q8_0-only workaround supports further experiments while the exact dependency is investigated.

## Candidate validation

The final five-line option inside Flash Attention passes:

- 4096-token replay, ubatch 512, three repetitions each in mmap and none mode: all eight endpoints stable, finite and nonzero, complete captured logits byte-identical to the all-barrier reference.
- Short 244 prompt + 128 fixed continuation token replay: all 129 captured logit vectors byte-identical to the original unmodified-upstream reference.
- 362 selected existing `test-backend-ops` Flash Attention cases against CPU, both option OFF and ON, covering f16/q8_0 KV, views, masks, GQA and prefill layouts. Command: `test-backend-ops -b Vulkan0 -o FLASH_ATTN_EXT -p 'hsk=(64|128|256|576),.*nb=(64|75),.*type_K=(f16|q8_0),type_V=(f16|q8_0),'`.

Long raw output: 24 x 248320 float32 values, 23,838,720 bytes, SHA-256 `79b4b83d577c2a18cf4582082f16a13455f22c5c2c615140cc88cf8237673373`. Short raw output: 128,133,120 bytes, SHA-256 `b7bb28a86e623daf75f436f9bddb786cee6c43d8a28d1b85333f3f25d9cbb662`. Equal hashes alone are insufficient: checks also verify expected sizes, finite values and nonzero vectors, specifically rejecting the serialized-mode false positive.

## Short performance cost

Four independent processes, five repetitions each, OFF/ON/ON/OFF, final scoped-option binary, load-mode none, original short flags. No routing callback, raw-logit dump or debug logger. Timing covers decode calls plus synchronization; initialization, sequence clearing and hashing are outside the timer. PP and TG aggregate tokens / elapsed time per process.

| Process | Barrier | PP token/s | TG token/s |
| --- | --- | ---: | ---: |
| 1 | OFF | 229.575 | 29.155 |
| 2 | ON | 224.736 | 28.546 |
| 3 | ON | 226.290 | 29.247 |
| 4 | OFF | 224.112 | 28.726 |

Means: PP 226.844 OFF vs 225.513 ON (-0.59%); TG 28.941 OFF vs 28.896 ON (-0.15%). This sample shows a small cost, not a speed improvement from the barrier. All call hashes match the original short reference.

## Confirmed long-prefill host-load gain

Final Q8_0-only option ON throughout, ubatch 512, four consecutive processes in mmap/none/none/mmap order, three repetitions each. No routing callback, raw-logit dump or debug logger. All 96 captured endpoint hashes match the finite/nonzero raw reference. This measures a configuration gain available in the existing loader, with the optional synchronization workaround held constant, not faster arithmetic from the barrier.

| Process | Load | PP token/s |
| --- | --- | ---: |
| 1 | mmap | 256.973 |
| 2 | none | 323.220 |
| 3 | none | 324.455 |
| 4 | mmap | 257.875 |

Means: 257.424 -> 323.838 token/s (+25.80%). This is prompt processing at a loaded model, not cold-start latency or generation throughput. The host-load costs in the previous report still apply: slower initialization and substantially more pinned/driver-owned host memory. No TG improvement is established.

mmap: device-wide peak VRAM 13.40 GiB, peak GTT 1.05 GiB, minimum system MemAvailable 25.66 GiB. none: device-wide peak VRAM 13.43 GiB, peak GTT 9.71 GiB, minimum system MemAvailable 17.28 GiB. Desktop allocations are included; process RSS does not capture all Vulkan host memory.

## Larger batches

The earlier broader prototype at ubatch 1024 and 2048 passes two-repetition full-logit checks against a separate layer-boundary callback reference using the same batch size. All values are finite and nonzero. This establishes repeatability and agreement with the fenced computation for that batch, not equivalence between batch sizes.

Cross-batch comparison at token 4095 against ubatch 512 is unexpectedly large: ubatch 1024 max absolute difference 7.800, RMS 1.334, normalized MSE 0.04653; ubatch 2048 max absolute difference 13.704, RMS 2.704, normalized MSE 0.19117. Argmax matches at the captured common endpoints. Different batching can change arithmetic and routing, but these differences are not dismissed as harmless rounding. Larger-batch throughput screens are exploratory and are not promoted as verified optimizations. A trusted numerical comparison is still required.

Subsequent autonomous-generation checks with the final Q8-only option compare against unmodified upstream at the same batch size. For seeds 42 and 12345, candidate ubatch 2048 matches upstream ubatch 2048 exactly over 128 generated tokens and all pre-sampling logits fingerprints, while both differ from ubatch 512 starting at generated token 5. The final-option PP benchmark confirms 766.919 token/s at ubatch 2048 versus 324.403 at ubatch 512. These findings retain the large-batch configuration as a candidate and show that the sampled output change also occurs in upstream. See [large-batch generation validation](moe-large-batch-generation-rx6800.md) for controls and limits; semantic quality across batch sizes remains unmeasured.

## Quality scope

Changing load mode does not change the model, selected precision or quantization. In the tested replays the full captured vocabulary distributions match byte for byte, a stronger check than matching only the chosen token. The short reference comes from unmodified upstream. The long reference is a computation with explicit synchronization because unmodified upstream long-prefill runs are themselves variable. These are sampled call endpoints and fixed-token paths, not a semantic quality suite over arbitrary prompts or a proof that every hidden state matches. The follow-up generation validation is limited to sampled same-batch generation equivalence; no universal semantic-quality claim is made for larger batches or F16.

## Upstream review and archives

Reviewed current [upstream Vulkan code](https://github.com/ggml-org/llama.cpp/blob/7f2dd88b0ac393357ae6a9e1992185a48c20b13b/ggml/src/ggml-vulkan/ggml-vulkan.cpp), blob `a4c7bcc8c3ccf16b120f7ab795b7abd5926d782f`, on 2026-10-04. The graph tensor hazard lists and Flash Attention auxiliary synchronization structure remain present. [Issue 28056](https://github.com/ggml-org/llama.cpp/issues/28056) describes a HIP integrated-GPU input overwrite race; this is a different hardware/backend scenario and does not establish the cause here. The CUDA-only race fix in PR 28475 is also not a Vulkan fix.

`risultati/2026-10-04-correctness/` retains commands, environment, source patches, CMake cache, exact binary/library hashes, raw logits, logical-node traces and telemetry including system MemAvailable. Do not interpret RSS alone as total RAM use of Vulkan host allocations. The option has no validated behavior on other GPUs or concurrent streams. F16 remains an unresolved control rather than a supported workaround target. Keep it separable from expert-copy, cache and prefetch experiments.
