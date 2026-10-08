# R39: actual CPU quantized work and upstream MoE cache pilot

2026-10-08, RX 6800/RADV with ReBAR 16 GiB. Implementation commits `f3a5150eb0c6eb57e60effe6498b7f6319e73136` and `3a423b9804ffec74d600f003374592874232bbe0`, branch `experiment/moe-quantized-work-replay`. Original B1 remains v0.5.0 `7fe450e19305b828c199d602c23a8337aaa1f03b`. WORK uses frozen R37 libraries with only the CPU library replaced; clean upstream B4 is `9c2e0e491a822adae1f0b1c831adb4160057d24f`. The helper is the same no-callback source, recompiled for the upstream ABI.

The new observer reproduces real CPU projection outputs exactly. It captures the Q8_K work buffer that the CPU dot product actually consumed, resolving the missing input evidence from R37. This validates capture and isolated replay; it does not fix full-model CPU/GPU fidelity. The separate upstream cache pilot is repeatable but fails the unchanged numerical screen against upstream OFF. No speed ranking or cache promotion follows.

## Minimal diagnostic change and gates

`GGML_CPU_MOE_CAPTURE_WORK=PATH` captures completed single-token layer17 gate/up/down operations only. An extra CPU thread barrier completes the output before thread0 appends a version2 record: F32 input, selected IDs, actual Q8_K work bytes and F32 output. Existing `GGML_CPU_MOE_CAPTURE` version1 remains compatible. Absent/empty WORK adds no barrier. This opt-in instrumentation does not modify graph tensors, weights or routing; capture I/O and barriers invalidate performance comparisons.

`cpu-capture.py` validates shapes, IDs, Q8_K scale/block sums, finite/nonzero input/output, projection order and gate/up input identity. Extraction refuses overwrite. Existing `expert-pool-check` accepts captured Q8_K directly and replays identical real weights/IDs without substituting a dequantize/requantize roundtrip. The existing parser tests cover truncated/malformed captures, incompatible layouts and safe extraction.

The preregistered model replay is 512PP+200TG,201 complete248320-logit vectors, one stream, context1024, no warmup, one repetition, mmap and Q8_0 K/V. Every model process uses:

```text
replay -m /home/casa/Programmi/modelli/Ornith-1.5-35B-Q4_K_M.gguf -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off --verbosity 4 --load-mode mmap
```

Environment includes frozen `LD_LIBRARY_PATH`, `LC_ALL=C`, `MOE_REPLAY_IN=tokens.csv`, `MOE_REPLAY_REPS=1`, `MOE_REPLAY_WARMUP=0`, `MOE_REPLAY_LOGITS_OUT=RAW`. Observer cases add their capture paths. ASAN/UBSAN/leak enables halt-on-error. Exact commands, environment, helper/library hashes and raw checks are in [portable validation](moe-quantized-work-validation.json). Model processes run alone in systemd MemoryMax24G/SwapMax2G scopes with a6GiB available-RAM reserve, RX6800 preflight and600s timeout. GPU clocks2600/1075MHz, offset-100mV and cap186W remain the user's settings. Model21713462848bytes, SHA256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`; kernel7.2.9-1-cachyos, Mesa/RADV26.2.4-arch3.1, GCC16.2.1, CMake4.4.4, Ninja1.13.2, Release/native/Vulkan/OpenMP. CMake caches are frozen locally with hashes in validation.

## Completed observer and operator validation

Five release model controls (old CPU, WORK OFF, two WORK ON, simultaneous legacy/work capture) are bit-exact across all1005logit vectors. The full-model ASAN/UBSAN/leak run is clean and matches all201release vectors. Two work captures match byte for byte:600records each,2000F32input vectors and4800projection output vectors. Legacy F32 capture matches the previous R36 capture.

At decode steps1/100/200, direct CPU replay matches all nine real projection outputs byte for byte. Re-quantizing the captured original F32 input also reproduces all nine actual CPU Q8_K byte arrays. Thus the isolated original CPU quantizer and dot product are verified against what the model used, rather than merely compared with another fixture.

Eight operator processes validate576actual vectors and1152F64oracle vectors, with repeat-exact CPU original, CPU captured and GPU captured-F32. CPU captured and GPU captured-F32 both pass the registered per-case oracle limits max_abs<=0.001 and relative_RMS<=0.0001. Their actual errors are far smaller:

| GPU captured F32 vs real CPU | Maximum absolute error | RMS over three steps | Relative RMS |
| --- | ---: | ---: | ---: |
| gate | 9.5367e-7 | 1.4368e-7 | 2.5871e-7 |
| up | 6.4075e-7 | 1.3881e-7 | 2.8236e-7 |
| down | 5.2154e-8 | 8.4140e-9 | 3.0766e-7 |

GPU original-F32 instead has RMS0.00657/0.00661/0.000439 for gate/up/down against the same CPU outputs. This establishes activation quantization as the dominant isolated operator difference on these inputs. It does not explain the entire trajectory or turn R37 into a passing fidelity experiment. R40 must inspect intervention intermediates on the model.

## Separate current-upstream cache pilot

[Upstream PR29887](https://github.com/ggml-org/llama.cpp/pull/29887), merged2026-10-07, runs small host-expert MUL_MAT_ID operations on GPU with an LRU expert cache. Four valid controls use identical frozen B4 libraries and the same workload, OFF/512MiB/512MiB/OFF. A first complete OFF process was rejected by the harness for demanding an unused libllama-bench-impl.so; its raw/log/maps are preserved. The preregistered addendum removes only that unrelated library requirement; all six engine libraries remain mandatory. No inference or threshold changes.

Each mode repeats byte for byte. Upstream OFF versus fork CPU passes the numerical screen: maximum RMS0.01997, max_abs0.08896, no argmax changes; the difference begins in prefill and is a version comparison, not cache effect. Cache ON preserves prefill exactly but differs from OFF at decode step1. Across201vectors it reaches RMS1.14125, max_abs4.09976, symmetric KL0.04355 and one changed argmax. It fails the unchanged limits RMS<=0.05, max_abs<=0.5, KL<=0.01 and argmax>=99%.

Both cache runs report10870hits,17930misses,37.74%hit rate and32761.27MiB uploaded over200decode steps. This is a compatibility and numerical pilot; timings are not ranked, GPU-resident comparison and capacity sweep are absent, and semantic quality is unmeasured. Cache executes CPU-placed experts on GPU, so this screen does not isolate a cache-remap bug from ordinary CPU/GPU numerical differences. A same-GPU reference is required before attribution.

## Decision and next work

VALIDATO observer/actual CPU replay on the declared short workload; TEST_PARZIALI S09. Cache numerical screen FAIL, without a semantic-quality verdict. Retain opt-in diagnostics; do not merge a hybrid default or promote cache capacity based on the pilot. R39 adds11model processes including sanitizer and the rejected harness control;10valid processes provide2010complete vectors. R20-R39 executed298model processes under the historical accounting convention. No long-context, concurrency, TTFT or semantic-quality gate is closed here.

Archive `risultati/2026-10-08-quantized-work-replay/` preserves all captures, raw logits, inputs, repeated controls, source, frozen libraries, CMake caches, telemetry and protocols. Next: locate the first projection/intermediate difference under the CPU Q8_K roundtrip with raw-neutral observation, and qualify same-GPU cache behavior separately before performance work.
