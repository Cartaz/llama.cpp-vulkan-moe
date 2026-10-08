# R37: selective model input precision on RX 6800

Measurements: 2026-10-07, before ReBAR activation. Report finalized: 2026-10-08. Branch: `experiment/moe-model-input-precision`, draft PR18. Local source commit `8caa0bb9d8a28666c15641f778a90527655902ca`, published equivalent `f0965aa6460b1cbd9047eeda2e4a9ca76b22e23a`; both have tree `73550921af312cfbb312ddcd11b62019cad6b25b`. Original B1 is v0.5.0 `7fe450e19305b828c199d602c23a8337aaa1f03b`.

The diagnostic works and its OFF path preserves B1, but Q8_K input rounding does not restore full-model CPU/GPU fidelity. All interventions fail the unchanged numerical screen. Keep it opt-in for diagnosis; reject it as a fidelity fix and make no throughput or semantic-quality claim. ReBAR results belong to a separate campaign, R38.

## Implementation and hypothesis

R36 isolated identical real weights, expert IDs and inputs and found CPU Q8_K activation conversion dominated the operator-level difference. R37 tests that hypothesis on the actual model trajectory. `LLAMA_MOE_INPUT_DIAGNOSTICS` is a compile-time option, OFF by default, requiring built-in CPU and Vulkan backends. At layer17 single-token projections, `LLAMA_MOE_INPUT_CONTROL=f32` selects Vulkan F32; `copy` inserts CPU identity custom nodes and the same F32 path; `gate`, `up`, `down` or `all` instead perform CPU-trait Q8_K conversion followed by F32 dequantization on the selected inputs. Absent, empty and `0` keep the original graph. Invalid modes abort; custom input nodes reject LoRA. CPU remains an explicit control.

Scheduler copies prefix input names with the backend, so Vulkan recognizes diagnostic markers after that prefix. This correction was registered before measurements. No router ID, coefficient or weight is edited. The controls add transfers/custom operations and may change rounding and graph splits. Identity-copy controls isolate the latter effects. This is a numerical prerequisite for S09 hybrid placement on RDNA2, with additional CPU/copy costs; it is not a speed optimization.

## Frozen protocol

34 preregistered model processes plus three separately registered CPU follow-ups: each512PP+200TG,201 complete248320-logit vectors,7437 vectors total. One sequence, context1024, batch/ubatch512,8threads for PP/TG, Q8_0 K/V, FAon, mmap, fitOFF, no warmup, one repetition. A recorded teacher-forced token CSV fixes the continuation; it does not sample free-running answers. All vectors are finite/nonzero before SHA validation.

```text
replay -m /home/casa/Programmi/modelli/Ornith-1.5-35B-Q4_K_M.gguf -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off --verbosity 4 --load-mode mmap
```

The full-placement cases add `-ot '^blk[.]17[.]ffn_(gate|up|down)_exps[.]weight$=Vulkan0'` before `-ncmoe`. Single-projection cases replace that group with gate, up or down while keeping other layer17 projections on CPU. Environment: `LC_ALL=C`, frozen variant `LD_LIBRARY_PATH`, `MOE_REPLAY_IN=tokens.csv`, `MOE_REPLAY_REPS=1`, `MOE_REPLAY_WARMUP=0`, `MOE_REPLAY_LOGITS_OUT=RAW`; each intervention adds its named `LLAMA_MOE_INPUT_CONTROL`. The profile/debug/sanitizer cases add only the separately recorded instrumentation. All exact argv/environment/library hashes are in the validation manifest.

Screen registered before raw measurements: argmax agreement>=99%, max absolute logit error<=0.5, RMS of every vector<=0.05 and symmetric KL of every vector<=0.01. A failure is retained without relaxing limits. Model processes run alone in systemd24GiB/swap2GiB scopes, RAM reserve6GiB, timeout600s, GPU preflight and telemetry. Timings are diagnostic, not ranked; compilation and fixture activity overlapped parts of the campaign.

## Results against original CPU placement

Prefill logits match in every case; differences begin at the first decode step. Each row uses all201vectors, not only the first step.

| Intervention | First TG RMS | Maximum vector RMS | Maximum absolute error | Maximum symmetric KL | Argmax differences /201 | Screen |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Original GPU | 0.067523 | 0.611467 | 2.231972 | 0.026661 | 0 | FAIL |
| GPU F32 | 0.101110 | 0.480285 | 2.441733 | 0.036540 | 2 | FAIL |
| Identity copy + GPU F32 | 0.101110 | 0.480285 | 2.441733 | 0.036540 | 2 | FAIL |
| Q8_K gate | 0.069472 | 0.653638 | 1.938758 | 0.022684 | 1 | FAIL |
| Q8_K up | 0.069049 | 0.516192 | 2.612192 | 0.039058 | 1 | FAIL |
| Q8_K down | 0.062283 | 0.683858 | 2.488664 | 0.051896 | 4 | FAIL |
| Q8_K all / GPU F32 | 0.045377 | 0.542968 | 2.154535 | 0.019718 | 1 | FAIL |
| Q8_K all / CPU | 0.045377 | 0.701015 | 2.458070 | 0.033247 | 2 | FAIL |
| Q8_K gate only GPU | 0.048220 | 0.402112 | 2.221948 | 0.026001 | 1 | FAIL |
| Q8_K up only GPU | 0.042704 | 0.488671 | 2.300020 | 0.035017 | 1 | FAIL |
| Q8_K down only GPU | 0.045377 | 0.417402 | 2.334493 | 0.017278 | 1 | FAIL |

Original CPU and GPU have all201argmax equal yet fail the numerical screen. Q8_K all reduces the maximum RMS from0.611467 to0.542968; that remains over10times the limit. Single projections and GPUF32 also fail. Comparing transformed CPU to transformed GPU starts very close (first TG RMS6.265e-7, max_abs3.338e-6) but later reaches RMS0.463829, max_abs2.652600 and one changed argmax. Agreement on the first operator or first token cannot certify the trajectory.

The CPU Q8_K roundtrip itself differs from the original CPU path: first TG RMS0.045377, later maximum RMS0.701015 and two changed argmax. The real isolated CPU fixture exposes non-idempotent numerical behavior too:9480/73728 values differ, max_abs4.768e-7 and RMS4.469e-8. This establishes a nonzero perturbation in the small control; it does not identify a unique cause of the much larger model difference.

## Structural and numerical validation

All12registered repeat pairs are bit-identical. Absent/empty/zero diagnostic modes and OFF CPU/GPU are exact to their respective B1 placements. GPU identity-copy equals GPUF32 byte for byte. Supplemental CPU identity-copy repeats equal original CPU, and CPU f32 is a no-op. Therefore a CPU identity copy alone is insufficient to explain the roundtrip difference.

Three DEBUG variants, scheduler-profile all and both full-model ASAN/UBSAN/leak cases match their normal counterparts. DEBUG confirms200gate/up/down F32 dispatches per projection for f32/copy/all; scheduler profile records200CPU custom computes and200copies per projection. The profile events are host events, not GPU timestamps. Four real-operator follow-ups (72actual vectors each,144oracle vectors each) and11existing legacy tests pass. The selected diagnostic F32 fixture matches the previously frozen globally disabled-MMVQ isolated fixture; global MMVQ flags are absent from model runs. Default-OFF build and invalid configuration checks are preserved in the archive. No new tests/* file was added.

RX6800/5700X3D/32GiB, kernel7.2.9-1-cachyos, Mesa/RADV26.2.4-arch3.1, GCC16.2.1, CMake4.4.4, Ninja1.13.2. Release/native/OpenMP/Vulkan, sharedON/backendDLoff, CPUcompact/activeOFF. Existing user GPU2600/1075MHz,-100mV,186W settings were not changed. Model21713462848bytes, SHA256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`. CMake caches and selected compile commands identify diagnostic/default/debug/sanitized builds; frozen helpers remain R36.

## Decision and next experiment

VALIDATO diagnostic controls and repeatability; SCARTATO as a fidelity fix. PP/TG gain, TTFT, semantic quality, cache hit rates and concurrency are NON_MISURATO. Long~10000PP+200TG remains the separately documented ERRORE_BASELINE. R20-R37 contain273model processes,37added here.

Next: rebaseline frozen controls after the user's ReBAR change, then locate the first intermediate divergence with an observer that passes OFF/ON raw parity before attributing differences. Capture the actual CPU quantized data in addition to F32 inputs; a dequantize/requantize roundtrip cannot stand in for the original CPU conversion. Compare projection output, activation, merge and layer output on the first decode step before extending to later tokens. Keep unchanged logits gates and use a separate performance run without capture only after the relevant fidelity gate. No hybrid placement promotion follows from R37.

Archive: `risultati/2026-10-07-model-input-precision/`. [Portable validation](moe-model-input-precision-validation.json) includes frozen identity, protocols, commands, loaded libraries, raw checks, comparisons and dispatch evidence. Owner dirty files are excluded. Primary source review made before R37 at upstream `b86d2f07542b29ab099aed34fd6b6d1b2fd4b81c`: [graph](https://github.com/ggml-org/llama.cpp/blob/b86d2f07542b29ab099aed34fd6b6d1b2fd4b81c/src/llama-graph.cpp), [Vulkan](https://github.com/ggml-org/llama.cpp/blob/b86d2f07542b29ab099aed34fd6b6d1b2fd4b81c/ggml/src/ggml-vulkan/ggml-vulkan.cpp), [CPU](https://github.com/ggml-org/llama.cpp/blob/b86d2f07542b29ab099aed34fd6b6d1b2fd4b81c/ggml/src/ggml-cpu/ggml-cpu.c). These observations concern frozen local builds, not a claim about current master performance.
