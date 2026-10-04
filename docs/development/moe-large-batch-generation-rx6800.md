# RX 6800 large-batch generation validation, 2026-10-04

The ubatch 2048 configuration is retained as a performance candidate. With identical prompt, seed and sampling, it generates the same tested 128-token continuation as unmodified upstream using ubatch 2048. Changing ubatch from 512 to 2048 changes the continuation in both upstream and this branch. Earlier cross-batch logit differences did not establish a quality regression caused by the branch.

## Autonomous generation

4096 identical recorded prefill tokens, seeds 42 and 12345, temperature 0.8, top-k 40, top-p 0.95, min-p 0.05, other common sampler defaults. Context 8192, batch 2048, GPU layers 99, CPU MoE layers 18, CPU/batch threads 8, Flash Attention on, Q8_0 K/V, fit off. No forced continuation, routing callback or initialization warmup. Both implementations sample and feed back their own generated tokens. The prompt is a recorded code-review prefix; these are 128-token continuations, not scored complete answers.

Each seed has five fresh-process runs: upstream ubatch 512, candidate ubatch 2048 twice, upstream ubatch 2048, and candidate ubatch 512. Ten runs total.

| Comparison | Seed 42 | Seed 12345 |
| --- | --- | --- |
| Candidate 2048 vs upstream 2048 | Identical 128 token IDs and text | Identical 128 token IDs and text |
| Candidate 2048 repeated | Identical | Identical |
| Candidate 512 vs upstream 512 | Identical | Identical |
| 2048 vs 512, both implementations | First difference at generated token 5 | First difference at generated token 5 |

All 128 pre-sampling logits fingerprints match in each same-batch comparison. The harness checks every captured distribution for finite, nonzero values. Fingerprints are not full raw-logit dumps. The prior broader-barrier experiment retained full logits matching a separately layer-fenced computation at the same batch size; this new experiment tests the final Q8-only guard against untouched upstream directly.

Baseline: upstream `7fe450e19305b828c199d602c23a8337aaa1f03b`, unchanged build-A libraries, load-mode mmap. Candidate: tested source at `ce3ba740ee565300b58a87601cf4c36d5a26489c`, unchanged build-moe-transfer libraries, compact CPU ON, active CPU OFF, load-mode none and GGML_VK_FA_Q8_SYNC=1. Both use the identical standalone harness and common sampler. No inference libraries were rebuilt. Candidate Vulkan library SHA-256 `d909e7f5918fd1e06a53da2a9648eaa76d6c46aecec967f29fc61e9766bc2e18`.

## Final-option prompt-processing measurement

Same final candidate build, Q8-only option ON, load-mode none for both ubatches. Same 4096-token workload and other flags above. Order 512/2048/2048/512, three timed repetitions per process after one untimed warmup. Timing covers decode plus synchronization; loading, hashing and sequence clearing are outside the timer. No routing callback, raw-logit dump or debug logger. This measures prompt processing of an already loaded model, not generation throughput or startup latency.

| Process | Ubatch | PP token/s |
| ---: | ---: | ---: |
| 1 | 512 | 324.567 |
| 2 | 2048 | 765.524 |
| 3 | 2048 | 768.315 |
| 4 | 512 | 324.238 |

Means: 324.403 -> 766.919 token/s (+136.41%). All per-call fingerprints repeat identically within each process. All 12 final-prefill fingerprints match the finite/nonzero distributions captured during autonomous generation at their respective batch size. This confirms the earlier approximately 760 PP token/s screen with the final Q8 guard.

Device-wide sampled peak VRAM: ubatch 512 13.41 GiB, ubatch 2048 15.48 GiB. Peak GTT: 9.76 / 9.77 GiB. Sampling is once per second and includes desktop allocations. Larger batches leave less VRAM headroom. The pinned-host load-mode startup and RAM costs from the earlier report still apply.

## Interpretation and limits

A different continuation across batch sizes is not evidence by itself that semantic quality is worse. The same-batch upstream control shows that the observed continuation change also occurs without the fork modifications. Exact original-output fidelity requires specifying the batch as well as the prompt, seed and sampler. This does not identify the numerical cause of the cross-batch differences or dismiss their large magnitude as harmless rounding.

Keep ubatch 2048 available as a measured experimental configuration. Its generated-output equivalence is supported for the two sampled seeds at this prompt and hardware. Semantic quality across complete representative prompts, other cache types, longer sequences and concurrent streams remains unmeasured. No universal equivalence claim or token-generation speed gain is made.

RX 6800 16 GB / RADV NAVI21, Ryzen 7 5700X3D, 32 GB RAM; kernel `7.2.9-1-cachyos`, Mesa/vulkan-radeon `26.2.4-1`, GCC `16.2.1+r23+gd564253eb6c8-1`, CMake `4.4.4-1.1`, Ninja `1.13.2-3.1`; Release/native/Ninja/Vulkan. CPU governor performance and GPU BOOTUP_DEFAULT unchanged. Model Ornith-1.5-35B-Q4_K_M.gguf SHA-256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`.

Archives: `risultati/2026-10-04-generation/` holds the original identical harness, exact compile commands and hashes, and short-prompt generation tests. `risultati/2026-10-04-generation-large/` holds the large-batch runner, comparison matrices, token IDs/fingerprints, output text, command/environment metadata, benchmark results and GPU/system-memory telemetry. The final-option libraries and source hashes are unchanged from the previous synchronization report.
