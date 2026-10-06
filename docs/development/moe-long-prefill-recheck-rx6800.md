# Long code-review prompt: original-engine repeatability gate

Date: 2026-10-07. R23 advances T4 after the short [R22 load-mode replication](moe-host-transfer-recheck-rx6800.md). Status: ERRORE_BASELINE for original-engine repeatability on this declared workload; speed gate FAILED. No long-prompt speed comparison or B3 promotion is valid. [Validation metadata](moe-long-prefill-recheck-validation.json) records frozen artifacts, prompt/template/token hashes, commands, all captures and comparisons. Local archive: `risultati/2026-10-07-long-prefill-recheck/`.

## Workload and protocol

One curated synthetic agent-like code-review prompt supplies20 queue handlers with retry/error-handling code. Actual GGUF chat template, thinkingOFF, jinja2 3.1.6. It tokenizes to3107 prompt tokens, followed by128 greedy continuation tokens generated without an evaluation callback. This is a new held-out prompt from the R21 four-prompt routing pilot, not a semantic quality evaluation or live agent task. Token-file SHA256 `487fee2134b6f69f2ab1e1cef26e64757f1b5ec240a71e9bb014e6e916ceac4f` was frozen after generation and before replay.

Original engine B1: untouched v0.5.0 `7fe450e19305b828c199d602c23a8337aaa1f03b`. Candidate: frozen `9375d46b7f7e1099bdee4d92cc6fc091057a0194`, tree `b6464350c9251c43bc38f5ca1d61566b54667840`, same libraries/build as R22. Ornith-1.5-35B-Q4_K_M,21,713,462,848bytes, SHA256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`, freshly verified at R22 freeze; file size and frozen library hashes checked per process. Model was unchanged between campaigns. RX6800/RADV NAVI21, Mesa26.2.4-arch3.1, kernel7.2.9-1-cachyos, Ryzen5700X3D/32GB; unchanged2600/1075MHz,-100mV,186W.

Context8192, batch/ubatch512, ngl99/ncmoe18/t8/tb8, FAon/Q8 KV/fitOFF. Prompt evaluates as six512-token chunks and one35-token chunk; each emits one complete vocabulary distribution, followed by128 decode distributions. All replay processes have one internal warmup and one measured correctness repetition. No host/route/GPU logger is enabled. Environment is rebuilt with explicit frozen libraries; scope24G/swap2G/floor6GiB,600s timeout and2Hz telemetry. Processes run sequentially. Requested placement and buffer geometry are retained in metadata.

Before the first process, the protocol required finite/nonzero complete logits and bit equality between no-callback generation, original replay, candidate mmap and candidate none before any timing campaign. Candidate mmap failed the per-call reference check, so the balanced speed campaign was not started. The candidate process itself exited0; the wrapper rejected its output before writing a normal result manifest. Its planned manifest, raw/CSV/log/maps/telemetry remain intact. A separately named post-validation manifest records the observed process success and failed comparison, without inventing missing wall times.

After failure, a separate diagnostic extension was declared: repeat original mmap, repeat candidate mmap, then capture candidate none. These are raw correctness diagnostics only; they do not waive the original acceptance gate. All six model processes completed, and zero speed processes ran.

## Observations

All six complete captures have135 vectors/vocabulary248320,134,092,800bytes each. Every vector is finite and nonzero. Each comparison to the first original capture agrees for the first two prompt chunks and diverges at position1024, the third512-token chunk. The remaining133 vectors differ.

| Capture compared with first original | Max absolute logits difference | RMS over all captured values | Different argmax endpoints |
| --- | ---: | ---: | ---: |
| No-callback generator | 3.431840 | 0.110396 | 4 |
| Candidate mmap | 5.172958 | 0.148936 | 2 |
| Repeated original mmap | 3.431840 | 0.115102 | 2 |
| Repeated candidate mmap | 3.456722 | 0.104021 | 3 |
| Candidate none diagnostic | 3.431840 | 0.115102 | 2 |

The repeated original's two different argmax endpoints occur at prompt positions2048 and2560; all its decode argmax values happen to match the first original. Full decode distributions still differ. The candidate none diagnostic matches the repeated original capture byte-for-byte; this one match does not establish determinism or a load-mode fix. All raw hashes and exact endpoint lists are retained.

This establishes failure of the original engine's strict repeatability gate at these settings. It does not establish a regression caused by R21 or R22, a semantic quality score, GPU race, driver defect or particular faulty shader. The first two matching chunks help bound a reproducer, but do not prove the problem begins inside a specific layer or operation. R21's callback-specific divergence and this no-callback long-prompt failure are separate diagnostic findings.

Earlier [R06/R07 long-prompt diagnostics](moe-host-transfer-rx6800.md) already recorded finite output differences even in the unmodified original. The present campaign has its own prompt, frozen engine and exact controls; it is not substituted for those archives. Current upstream reports [#21888](https://github.com/ggml-org/llama.cpp/issues/21888) and [#24812](https://github.com/ggml-org/llama.cpp/issues/24812) were checked as related Qwen/Vulkan investigations on other GPUs. They are not evidence of the local cause or a validated RX6800 workaround. Current upstream GDN shader code was also inspected; no upstream fix was adopted or benchmarked in this increment.

## Decision and next diagnostic

Preserve the failed gate and keep short-workload R22 conclusions limited to244+128. Do not rank long-prompt load modes or promote a general profile from these captures. The next correctness work should reduce the original-engine reproducer around the third chunk, record repeatability per chunk, and test one predeclared change at a time. The [R24 prefix screen](moe-chunk-repeatability-rx6800.md) subsequently finds two matching runs at1024 and1536, so the third-chunk ordinal alone is insufficient. A recent upstream B4 build is a separate control with its own source/build/runtime freeze, not a replacement for the historical baseline. CPU reference or synchronization changes need distinct controls; an instrumented output that becomes stable alone does not prove normal-mode correctness.

M2 and bounded transfer design can continue on their validated short geometry while long-prompt promotion remains conditional. Runtime cache, calibrated normal GPU timeline, broad quality and1/2/4/8-stream validation remain incomplete.

## Reproduce the failed gate

The exact [3107+128 token input](../../examples/moe-trace/workloads/r23-code-review-3107-128.csv), [messages](../../examples/moe-trace/workloads/r23-code-review-messages.json), rendered prompt and hashes are versioned under `examples/moe-trace/workloads/`.

Use the exact archived token file and frozen B1/COLLECTOR library hashes, clean environment and new paths. Both commands use identical flags and chunking:

```sh
env -i PATH=/usr/bin:/bin HOME="$HOME" XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" LC_ALL=C LD_LIBRARY_PATH=/absolute/path/frozen/BASE MOE_REPLAY_IN=/absolute/path/tokens.csv MOE_REPLAY_REPS=1 MOE_REPLAY_LOGITS_OUT=/absolute/path/original-logits.bin /absolute/path/frozen/BASE/replay -m /absolute/path/Ornith-1.5-35B-Q4_K_M.gguf -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 8192 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off --verbosity 4 --load-mode mmap > original.csv 2> original.log
env -i PATH=/usr/bin:/bin HOME="$HOME" XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" LC_ALL=C LD_LIBRARY_PATH=/absolute/path/frozen/BASE MOE_REPLAY_IN=/absolute/path/tokens.csv MOE_REPLAY_REPS=1 MOE_REPLAY_LOGITS_OUT=/absolute/path/repeated-original-logits.bin /absolute/path/frozen/BASE/replay -m /absolute/path/Ornith-1.5-35B-Q4_K_M.gguf -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 8192 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off --verbosity 4 --load-mode mmap > repeated-original.csv 2> repeated-original.log
```

Check all135 finite/nonzero vectors before comparing raw bytes and per-call hashes. The archive preserves generation, original/candidate captures, the preregistered conditional speed protocol and the later diagnostic protocol. Dump-run timings are not performance evidence.
