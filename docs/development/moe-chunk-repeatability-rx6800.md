# Reduced original-engine chunk repeatability screen

Date: 2026-10-07. R24 reduces the [R23 long-prompt failure](moe-long-prefill-recheck-rx6800.md) without changing the original engine. Status: TEST_PARZIALI, two matching processes per reduced case; the full failure remains unresolved. [Validation metadata](moe-chunk-repeatability-validation.json) contains commands, frozen libraries, token hashes, raw checks and all launch records. Archive: `risultati/2026-10-07-chunk-repeatability/`.

Original v0.5.0 `7fe450e19305b828c199d602c23a8337aaa1f03b` only. Use the first1024 or1536 prompt tokens from R23 and reuse its first32 continuation IDs, with positions reindexed. This is synthetic teacher forcing to test repeatability, not a realistic continuation or semantic quality score. Versioned inputs: [1024+32](../../examples/moe-trace/workloads/r24-prefix-1024-32.csv), [1536+32](../../examples/moe-trace/workloads/r24-prefix-1536-32.csv); hashes in [workload manifest](../../examples/moe-trace/workloads/manifest.json).

Same frozen B1 libraries and model as R23: Ornith Q4_K_M SHA256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`, RX6800/RADV NAVI21, Mesa26.2.4-arch3.1, kernel7.2.9-1-cachyos, Ryzen5700X3D/32GB. Existing2600/1075MHz,-100mV,186W profile unchanged. Flags: ngl99/ncmoe18/t8/tb8/c8192/b512/ub512/FAon/Q8KV/fitOFF/mmap. Each process has one internal warmup and one raw correctness repetition. Clean environment, explicit libraries, scope24G/swap2G,6GiB availability floor,600s timeout and2Hz telemetry; no profiler and no simultaneous model process.

Preregistered order1024/1536/1536/1024; two fresh processes per prefix, no adaptive extension. All four model processes completed. The initial launch rejected the CRLF input header before model loading, exit1 after40ms service runtime. This was a harness formatting error. Original files/freeze/log remain preserved; new separately frozen LF files retain the same token IDs. This rejected launch is excluded from model-process count.

| Prefix | Captured vectors per run | Raw bytes per run | Pairwise result |
| --- | ---: | ---: | --- |
| 1024+32 | 34 | 33,771,520 | Bit-identical; finite/nonzero at every vector |
| 1536+32 | 35 | 34,764,800 | Bit-identical; finite/nonzero at every vector |

All compared values, per-call hashes and argmax endpoints match within each pair. No speed or answer-quality ranking is reported, and two matches are not a general determinism certificate. The1536 case does not reproduce the full3107 failure. Therefore the ordinal third chunk alone is insufficient as a reproducer. Later warmup/continuation history, total prompt length, changing tensor shapes or allocator/state dependencies remain hypotheses; this screen does not identify their cause. In particular, it does not prove that the third chunk is correct on all longer prompts.

The replay helper clears memory metadata with `llama_memory_clear(..., false)` before every repetition. Testing data clearing is a separate possible diagnostic variable; this observation does not show that false clearing is faulty. Any such probe must use the same original engine with a separately frozen helper and preserve the default helper as control.

To reproduce, run each versioned token file twice in fresh processes using the original frozen engine and the same configuration, with fresh dump paths:

```sh
env -i PATH=/usr/bin:/bin HOME="$HOME" XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" LC_ALL=C LD_LIBRARY_PATH=/absolute/path/frozen/BASE MOE_REPLAY_IN=/absolute/path/r24-prefix-1536-32.csv MOE_REPLAY_REPS=1 MOE_REPLAY_LOGITS_OUT=/absolute/path/fresh-logits.bin /absolute/path/frozen/BASE/replay -m /absolute/path/Ornith-1.5-35B-Q4_K_M.gguf -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 8192 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off --verbosity 4 --load-mode mmap > fresh.csv 2> fresh.log
```

The original R23 failing workload and short R22 result remain separate records. Reducing a failed input to a passing one does not establish a fix or authorize general profile promotion.
