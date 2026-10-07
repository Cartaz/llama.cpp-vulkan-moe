# Explicit replay memory-data reset diagnostic

Date: 2026-10-07. R25 adds a helper-only control after [R23](moe-long-prefill-recheck-rx6800.md) failed long-prompt repeatability and [R24](moe-chunk-repeatability-rx6800.md) did not reproduce it with shorter prefixes. Status: IMPLEMENTATO and validated for option/default compatibility; the true-clear long gate FAILS even on the original engine. This is a negative correctness diagnostic, not an inference fix or performance optimization. [Validation metadata](moe-state-reset-validation.json) contains source, compiler/build/cache/library hashes, commands, protocols and every capture. Archive: `risultati/2026-10-07-state-reset/`.

## Small separable change

Published code `79c777827a3bdb42e9c2684fdb2aba65c98168b1`, tree `b38c33825b7dd4babab2d9bcde4f35e48eca4e5a`. `MOE_REPLAY_CLEAR_DATA=1` makes the replay helper call the existing `llama_memory_clear(..., true)` before its warmup and each repetition. Absent,0 and empty retain the previous false value. Other values reject before model loading. A setup log records `clear_data` so reset conditions are visible in provenance. The helper's default behavior and inference source under ggml/src, src, include and common are unchanged relative to9375; freshly rebuilt library hashes, including build metadata, are recorded separately.

The hypothesis targets residual data across replay repetitions, not a presumed RDNA2 kernel bottleneck. Metadata-only reset normally keeps buffer contents; true clearing also clears the memory buffers through the existing backend API. It can cost memory traffic and synchronization. Clearing stays outside existing per-evaluation PP/TG intervals, as it did before this patch. No free reset, TTFT improvement, zero overhead or normal application behavior is inferred. The option changes only the replay helper; llama-server/CLI defaults are not changed.

For the original engine, the identical modified helper was separately compiled with original baseline headers and linked only to the six frozen original libraries at `7fe450e19305b828c199d602c23a8337aaa1f03b`. The baseline checkout/libraries were not modified. Helper source and engine source are distinct in the manifest. Candidate Release/native/Ninja/Vulkan build uses GCC16.2.1 20260810; CPU_MOE_COMPACT/ACTIVE remain OFF.

## Preregistered long gate and outcome

Same versioned [3107+128 tokens](../../examples/moe-trace/workloads/r23-code-review-3107-128.csv) as R23, SHA256 `487fee2134b6f69f2ab1e1cef26e64757f1b5ec240a71e9bb014e6e916ceac4f`. Ornith Q4_K_M model SHA256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`,21,713,462,848bytes; model rechecked at R22 freeze and unchanged, size checked per process. RX6800/5700X3D/32GB, RADV NAVI21/Mesa26.2.4-arch3.1, kernel7.2.9-1-cachyos, unchanged2600/1075MHz,-100mV,186W.

Flags ncmoe18/ngl99/t8/tb8/c8192/b512/ub512/FAon/Q8KV/fitOFF/mmap. Two fresh processes per engine, order original/candidate/candidate/original, one internal warmup and one raw repetition, clear_data=true for both. Clean environment, explicit frozen libraries, one stream,24G scope/2G swap/6GiB availability floor/600s timeout and2Hz telemetry. No evaluation callback or profiler. All four processes completed, with every135-vector/vocabulary248320 distribution finite/nonzero.

| Comparison with true clearing | Different vectors | First difference | Max absolute / RMS over all values | Different argmax endpoints |
| --- | ---: | --- | --- | ---: |
| Original run1 vs original run4 | 134/135 | Prefill position512, second chunk | 6.900958 / 0.132741 | 2 |
| Candidate run2 vs candidate run3 | 0 | None | 0 / 0 | 0 |
| Candidate run2 vs original run4 | 0 | None | 0 / 0 | 0 |

The original pair's different argmax endpoints are decode positions3117 and3132. Candidate pair and second original are byte-identical, SHA256 `a5a40821a05d567051db5b6129ac7f2518ccd2dc02be1481a5295b1acde40b1b`. The first original has a different raw hash. The required all-four bit-equality gate fails. Two matching candidate runs cannot override the failed original pair or certify general determinism.

True clearing is therefore insufficient to contain the long failure. The changed divergence endpoint compared with R23 does not prove that clearing makes inference worse, that stale memory is the cause, or that a particular shader/driver is faulty. Old false-clear captures and new true-clear captures have different helper/build provenance and are not a causal performance A/B. A conditional matched-helper false-clear extension was recorded before reading the first result; it was not triggered because the true-clear gate already failed. No speed campaign or tolerance relaxation followed.

## Default compatibility and option checks

Three separate short244+128 correctness processes use the new candidate helper with absent,0 and empty option values, each with one internal warmup and one raw repetition. Context1024; remaining flags unchanged. All129-vector outputs are finite/nonzero and byte-identical to original B1, SHA256 `b7bb28a86e623daf75f436f9bddb786cee6c43d8a28d1b85333f3f25d9cbb662`. Every per-call hash matches the recorded canonical reference and every setup marker reports clear_data=0. These are compatibility checks, not speed samples.

Six additional non-model launches test both helpers: invalid2 rejects before model loading;0/empty pass option parsing and fail at an intentionally absent model path. Their complete logs/commands are recorded. Total: seven successful model processes and six expected non-model option launches. Existing offline/runtime suites were already passed at R21; this helper change is covered by the new CLI and full-output model checks rather than redundant operator tests.

Decision: keep the control opt-in for diagnosis, preserve false as default, and record the negative hypothesis result. Long-context quality/profile promotion remains conditional on a reproducible normal baseline. A recent upstream B4 control and isolation of tensor-shape/history dependencies remain separate work; no engine patch or upstream fix is claimed.

## Reproduce

Use the same new helper for both conditions, original libraries for the original-engine control, fresh outputs and explicit clean environments. The true-clear original capture is:

```sh
env -i PATH=/usr/bin:/bin HOME="$HOME" XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" LC_ALL=C LD_LIBRARY_PATH=/absolute/path/frozen/BASE_CLEAR MOE_REPLAY_IN=/absolute/path/r23-code-review-3107-128.csv MOE_REPLAY_REPS=1 MOE_REPLAY_CLEAR_DATA=1 MOE_REPLAY_LOGITS_OUT=/absolute/path/fresh-logits.bin /absolute/path/frozen/BASE_CLEAR/replay -m /absolute/path/Ornith-1.5-35B-Q4_K_M.gguf -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 8192 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off --verbosity 4 --load-mode mmap > fresh.csv 2> fresh.log
```

Repeat in fresh processes and compare all135 finite/nonzero distributions, not just argmax values. Set0 or omit the variable for a matched-helper false-clear control. Clearing costs need an explicit reset/application timer before any latency ranking.
