# R38: ReBAR baseline and memory allocation control on RX 6800

Date: 2026-10-08. Branch: `experiment/moe-rebar-controls`, based on R37 documentation commit `c663e8059d4b9d803ce47a56b4aecfc13876b2f4`. Inference engines are unchanged frozen B1 v0.5.0 `7fe450e19305b828c199d602c23a8337aaa1f03b` and R37 `8caa0bb9d8a28666c15641f778a90527655902ca` (published code-equivalent `f0965aa6460b1cbd9047eeda2e4a9ca76b22e23a`). No recompile, firmware setting, clock or voltage change was made.

ReBAR is now active: PCI resource0 is16GiB, resource2 is256MiB, and amdgpu reports17163091968bytes both total and CPU-visible VRAM. All six fresh raw controls are valid and repeat/variant/allocation checks are exact. The five controls with historical counterparts also match before ReBAR byte for byte. The separate synthetic before/after campaign reports TG12815.666->33.812token/s (+115.82%,2.158x), with PP512-0.16% and PP4096+0.17%. The new real-token replay establishes a current baseline at31.902token/s after512PP, with eight fresh timing processes.

## Development change and bottleneck

Small-BAR tuning and new ReBAR measurements must not silently share a hardware baseline. Manifest schema2 adds PCI device/driver identity, resource sizes/flags, total and CPU-visible VRAM and maximum link capabilities to the frozen hardware signature, together with kernel, CPU and architecture. `--expect` rejects a change in that signature. Used memory, busy counters, card numbering, PCI base addresses and current link speed/width remain dynamic snapshots and do not invalidate equivalent hardware. Schema1 references must be recaptured before schema2 verification; missing hardware identity is not proof of equivalence. Environment capture now includes `LLAMA_MOE_`, `MOE_`, `MESA_` and `OMP_` controls.

The implementation is an offline extension of existing manifest infrastructure. It does no work in inference and needs no inference rebuild. The existing21-test suite passes, including small/large aperture and visible-VRAM changes, moving PCI addresses, dynamic counters/link state, invalid resource ranges and a live Linux process snapshot.

The performance question is whether the historical host-visible allocation workaround still helps under ReBAR. It targets buffer allocation/CPU access and possible staging cost on discrete RADV. It can also add transfers by excluding directly mapped VRAM. R38 changes only the existing `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM` flag at fixed weights, placement and tokens; it does not change the allocator or claim a specific transfer mechanism was measured.

## Raw controls and correctness

Order: B1 CPU1, B1 layer17GPU1, fork OFF layer17GPU, B1 layer17GPU with disable flag, B1 CPU2, B1 layer17GPU2. Each512PP+200TG, no warmup, one repetition, context1024, batch/ubatch512, threads8, mmap, FAon, Q8_0 K/V, fitOFF. Six processes yield1206raw248320-logit vectors, all finite/nonzero. CPU repeats have SHA `70a46c02e881eccb8f109aeda82cbde51990391b077edd3515c206814f7c0424`; GPU repeats, forkOFF and changed allocation have SHA `311b49e5aac3ca8604e8e1fe4c8a36127cba23d9bcb1dc4b5abb86574dd4cf95`. These equal their R37 placement references.

This confirms unchanged output for each placement on the short workload. CPU and GPU placement still differ from one another as recorded in R37 and fail its numerical screen. ReBAR does not resolve that known difference. Long-context repeatability, semantic quality and concurrent requests are not validated here.

## Controlled allocation A/B

Four fresh processes per variant, two balanced blocks: default/disabled/disabled/default, then repeated. One untimed full replay warmup and three timed repeats per process. No raw dump, observer, scheduler profile, concurrent build or other model process during timing. All4824timed calls match the appropriate raw-reference fingerprints after their timers. Warmup, model load, hashing, output I/O and initialization are excluded; this measures decode+synchronize, not application TTFT or free-running generation.

| Metric | Default | Disable host-visible VRAM | Disable change | Process bootstrap95% | Verdict |
| --- | ---: | ---: | ---: | --- | --- |
| PP512 token/s | 290.489 | 288.904 | -0.55% | [-0.72%, -0.35%] | NESSUN_CAMBIAMENTO |
| TG200 token/s | 31.902 | 30.780 | -3.52% | [-5.31%, -1.78%] | INCONCLUDENTE |
| Mean process p95 ms | 36.516 | 37.995 | +4.05% | [+2.52%, +5.61%] | INCONCLUDENTE |

PP/TG use total tokens divided by total elapsed time. p95 is the mean of each process's pooled600decode-call p95. Bootstrap resamples four independent processes per variant,20000draws,seed6800; it does not treat repeated calls as independent processes. Practical margins3% throughput and5% p95 were registered before measurements. The estimator script was written while timing was underway; thresholds, order and sample count were unchanged. End/start drift is recorded for both variants and never exceeds the20% alarm threshold.

| Process | PP token/s | TG token/s | p95 ms |
| --- | ---: | ---: | ---: |
| default-1 | 290.888 | 32.197 | 36.300 |
| disabled-1 | 289.434 | 31.112 | 37.399 |
| disabled-2 | 288.736 | 30.185 | 38.672 |
| default-2 | 289.911 | 32.450 | 36.153 |
| default-3 | 290.107 | 31.675 | 36.658 |
| disabled-3 | 288.735 | 30.695 | 38.197 |
| disabled-4 | 288.713 | 31.146 | 37.711 |
| default-4 | 291.055 | 31.309 | 36.952 |

Completely fixed replay command for both timing variants:

```text
replay -m /home/casa/Programmi/modelli/Ornith-1.5-35B-Q4_K_M.gguf -ngl 99 -ot '^blk[.]17[.]ffn_(gate|up|down)_exps[.]weight$=Vulkan0' -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off --verbosity 4 --load-mode mmap
```

The override places all three layer17 projections on GPU; other CPU MoE layers remain as configured. Default environment: `LC_ALL=C`, `LD_LIBRARY_PATH=R37/frozen/BASE`, `MOE_REPLAY_IN=R38/short/tokens.csv`, `MOE_REPLAY_REPS=3`, `MOE_REPLAY_WARMUP=1`; disabled adds only `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`. The flag is presence-based; setting it to0 does not disable the workaround. CPU raw controls omit the override; NEW raw uses R37/frozen/NEW. Exact paths, launch argv and loaded library hashes are in the portable manifest.

Scripts in `risultati/2026-10-08-rebar-runtime/`: `run-controls.py`, `run-performance.py`, `summarize.py`, `report.py`, and the adapted frozen runner. They refuse existing measurement logs. Do not rerun into this directory; prepare a fresh campaign with the same frozen engines and token hashes. Initial preparation had two import/path errors before any model launch; logs are retained,0model measurements lost, order/criteria unchanged.

## System and scope

RX6800/5700X3D/32GiB, kernel7.2.9-1-cachyos, Mesa/RADV26.2.4-arch3.1, GCC16.2.1, CMake4.4.4, Ninja1.13.2. Engines Release/native/OpenMP/Vulkan, sharedON/backendDLoff, CPUcompact/activeOFF; diagnostic optionON only in the frozen NEW libraries, runtimeOFF. Existing user2600/1075MHz,-100mV,186W settings are preserved. Model21713462848bytes, full SHA256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`, rechecked before R38. PCIe link16.0GT/s x16 observed. BAR and GPU settings are checked between processes. RAM reserve6GiB, separate systemd24GiB/swap2GiB scopes, timeout600s, telemetry0.5s. No guard triggered.

Minimum MemAvailable: default24.71GiB, disabled24.64GiB. Peak total-device VRAM: default13.51GiB, disabled13.55GiB; peaks include desktop and initialization. These are sampled values, not exact allocation high-watermarks. VRAM/GTT and cgroup samples are retained; they do not measure transfer bytes or cache hits.

The synthetic reference uses ncmoe17, batch2048/ubatch512 and TG128 from empty context. R38 uses ncmoe18 plus layer17GPU, batch512 and TG200 after512real prompt tokens. Their absolute rates are separate baselines. The synthetic OFF/ON uses identical binary/driver/settings but only a before/after transition, without an OFF return. R37 timing was diagnostic with overlapping activity and cannot supply a clean historical real-replay speedup.

## Research decision

Keep the ReBAR default as the current control. Do not carry forward the old small-BAR provisional tuning or infer a general placement optimum from this one prompt. The disable-flag verdicts above apply only to this workload. R37 remains a rejected fidelity fix; follow-up should capture actual CPU quantized inputs and the first divergent intermediate with an observer parity gate.

Current upstream reviewed at `9c2e0e491a822adae1f0b1c831adb4160057d24f`: [Vulkan allocation](https://github.com/ggml-org/llama.cpp/blob/9c2e0e491a822adae1f0b1c831adb4160057d24f/ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp), [driver sysfs documentation](https://docs.kernel.org/gpu/amdgpu/driver-misc.html), and [small-BAR issue27097](https://github.com/ggml-org/llama.cpp/issues/27097), whose RX7900XTX numbers are not RX6800 evidence. [PR29887](https://github.com/ggml-org/llama.cpp/pull/29887) was merged2026-10-07 at `d6cf9acb25e6c657d6c6a4645ab7b69fce119995`; its generic GPU MoE cache uses layout-grouped LRU banks, uploads misses and bypasses admission for large batches. Code review confirms a backend synchronization/readback before slot updates, so benefit and overlap on RADV need measurement. Its published RTX4090/5090 speedups are not predictions for RDNA2. Prioritize a separate frozen upstream cacheOFF/ON control, GPU-resident parity, budget and small-batch checks before duplicating this integration or cherry-picking it into the preserved fork.

Archive: `risultati/2026-10-08-rebar-runtime/`. [Portable validation](moe-rebar-runtime-validation.json) includes protocols, model/build identity, actual commands/libraries, raw checks, all process samples, telemetry references, source audit and synthetic reference. R38 adds14real-model processes, totalR20-R38=287. No TTFT, semantic score, long-context or1/2/4/8stream claim. No experimental feature integrated into the release baseline.
