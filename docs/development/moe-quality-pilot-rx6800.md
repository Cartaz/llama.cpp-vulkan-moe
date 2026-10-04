# RX 6800 answer-quality benchmark pilot, 2026-10-04

The branch now has a separate llama-moe-quality utility and a deterministic dataset generator/scorer. It compares correctness of complete responses between ubatch 512 and 2048. Matching response text is not required for a correct solution. See [the protocol](moe-quality-ab-rx6800.md) for tasks, controls, run commands and the planned held-out evaluation.

## Implemented tests

Four synthetic task families with independent answer oracles: exact retrieval from document records, filtered ledger arithmetic, multi-hop pointer following, and Python code tracing for stable expert deduplication or compact-routing prefix offsets. These are original local tests, not official RULER or HumanEval scores. Code tracing does not replace a code-generation execution benchmark.

Complete native-template chat prompts, identical token IDs checked between A/B, actual long prompts 3170-3668 tokens in the executed pilots. Short controls have 76-94 tokens. One stream, batch 2048 and context 8192 throughout; only ubatch changes in the primary pairs. GPU layers 99, CPU MoE layers 18, threads 8/8, Flash Attention on, Q8_0 K/V, fit off, load-mode none, final optional Q8 barrier ON in both variants. No routing callback, forced continuation, grammar or initialization warmup. Clear sequence/KV/recurrent memory before each task. Reinitialize the sampler with the same seed for each case. Duplicate-case token comparisons check actual repeatability rather than assuming that temperature zero guarantees it.

The grader compares typed values, preserves leading zeros and array order, rejects duplicate JSON keys, tracks strict JSON formatting separately from correctness, and separately records truncation. It parses final content with llama.cpp's native reasoning parser; numbers appearing in intermediate reasoning do not pass a failed final answer. Oracle and grading self-checks include known wrong values, wrong types/order, duplicate keys and malformed responses. Only benchmark-authored code runs to validate the code-tracing oracle; model-generated code is not executed.

## Results and calibration

| Profile | Unique long tasks | Correct at 512 | Correct at 2048 | Extra solution errors at 2048 | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Greedy, thinking OFF, 128-token budget | 8 | 3 | 3 | 0 | Low floor on arithmetic and logic; insufficient acceptance test |
| Greedy, thinking ON, 512-token budget | 3 | 2 | 2 | 0 | Code task interrupted before final answer in both variants |
| Author sampling, thinking ON, 2048-token budget | 2 | 2 | 2 | 0 | Both complete arithmetic and code-tracing tasks correctly |

Short controls and duplicate cases are excluded from the primary score. All executed duplicate cases have identical generated token IDs within each variant/profile. None of these small samples establishes statistical non-inferiority; equal scores and McNemar p=1 do not prove general quality equivalence.

The no-thinking arithmetic control produces 57 instead of the correct 69 in both batch variants. An untouched upstream 512 control has exactly the same input and generated token IDs for all four short tasks, including this error. With thinking enabled the arithmetic result is 69. This suggests that profile/budget calibration is necessary; it does not uniquely attribute initial errors to the model, quantization, template or numerical backend.

The code-tracing task in the greedy-thinking pilot needs more than 512 generated tokens to finish. The author-profile calibration finishes with 1212 generated tokens at ubatch 512 and 1167 at ubatch 2048, including EOG. Both return the correct prefix offsets [0, 2, 6, 6, 7]. The 2048 response uses a Markdown JSON fence, violating the requested raw-JSON format; the 512 response is raw JSON. The scorer reports this format difference separately instead of silently treating it as either a wrong arithmetic result or complete equivalence.

An untouched upstream run at ubatch 2048, native reasoning and the same sampling has identical prompt IDs, all 1167 generated token IDs, and the same fenced final answer as the fork. This observed format change also occurs without the fork modifications. It remains a workflow-relevant cross-batch difference, even though the numerical solution is correct.

The author profile is temperature 0.6, top-p 0.95, top-k 20, seed 42; min-p=0 to avoid an extra filter. The [official Ornith model card](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B) recommends the first three parameters for general tasks and describes native reasoning. This calibration is not a reproduction of the author's published benchmark protocol.

## Timing scope

The two unique calibrated long tasks take 87.552 seconds in total at 512 and 75.741 seconds at 2048, excluding initialization and sequence clearing. These are instrumented loaded-task timings, including sampling and finite-logit checks; different reasoning lengths also contribute. They do not establish a pure token-generation throughput improvement. Separate per-item timing, completion status and PP are recorded by the scorer. The earlier controlled PP benchmark remains the relevant performance result, approximately 767 versus 324 prompt token/s on its fixed 4096-token workload.

## Prepared larger evaluation

Two generated datasets are ready, each with 64 unique long tasks (16 per family), four short controls and two duplicates: 70 calls per variant and seed. One uses 90 document records, the other 140. The longer corpus is generated but has not yet been inference-tested; its actual token lengths/context headroom must pass the harness checks. Generator seed 20261004. Independent held-out seeds can be chosen via --generator-seed and their dataset hashes frozen before tuning.

These prepared 64-task corpora are not reported as completed benchmarks. Before accepting the faster configuration, calibrate task difficulty and answer budget in the native reasoning profile, run the larger paired evaluation at predeclared sampling seeds, and inspect per-family losses/wins, truncation and format failures. Follow with execution-based coding tasks and representative agent requests. The current prompts are English; Italian task variants should be part of the real-user stage. Concurrency 1/2/4/8 and other cache types remain separate tests.

## Exact artifacts

Unmodified upstream baseline: 7fe450e19305b828c199d602c23a8337aaa1f03b, build-A untouched. Inference source for the fork pilots has the final Q8-only workaround published in ce3ba740ee565300b58a87601cf4c36d5a26489c; parent at integration was f42a6d363d479e97975a210f291aa87b60e14802. Numerical backend source remains unchanged by this benchmark utility. Test-time Vulkan library SHA-256 d909e7f5918fd1e06a53da2a9648eaa76d6c46aecec967f29fc61e9766bc2e18.

RX 6800 16 GB/RADV, Ryzen 7 5700X3D, 32 GB RAM; kernel 7.2.9-1-cachyos, Mesa/vulkan-radeon 26.2.4-1, GCC 16.2.1+r23+gd564253eb6c8-1, CMake 4.4.4-1.1, Ninja 1.13.2-3.1; Release/native/Ninja/Vulkan, compact CPU ON, active CPU OFF. CPU governor performance and GPU BOOTUP_DEFAULT unchanged. Model Ornith-1.5-35B-Q4_K_M.gguf SHA-256 ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f.

risultati/2026-10-04-quality/ contains datasets, exact harness variants, compile commands, source/binary/library hashes, old and final CMake caches, rendered prompts, native chat template, input/output token IDs, raw response and parsed final text, completion state, command/environment metadata, grader reports and GPU/system-memory telemetry. The exact test-time libraries are preserved in tested-libraries with SONAME aliases. Compiling the final CMake target updates only build metadata in libggml-base and libllama-common; the saved old libraries retain the exact earlier hashes.

The CMake target builds successfully. A native-generation smoke probe matches the calibrated control's prompt and output token IDs, and rejects an invalid output ID before writing its path. The final tool also rejects a nonpositive answer budget before model/GPU initialization. The published Python utility passes its deterministic generation/oracle/grading self-checks. Experimental prototype source hashes, final tool hashes and these validations are archived separately; do not substitute the final binary hash for the earlier prototype runs.

Dataset SHA-256:

- pilot.jsonl: afe24e6f7049eaaf5ff5a5b56b649c7cf17406cf5d3ec239ec85a56dcb968e09

- full.jsonl: 8f2e15862a6d0c3888fc2ca4831e348a6510394b877d83c99ea3bc624b720952

- full-longer.jsonl: 1d10c4e878f1b2a7a64a46f7ee3d7c9e7dadd145b2e8dcf645f837765af39e79
