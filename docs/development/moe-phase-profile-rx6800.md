# S03 replay phase correlation on RX 6800

Date: 2026-10-07. Status: IMPLEMENTATO / TEST_PARZIALI. Branch: `experiment/moe-phase-profile`, based on `0ffed2a10e4201ace20cb589900050ba8cde58df` (R18). Exact tested source and binary hashes are in [validation metadata](moe-phase-profile-validation.json). The model benchmark and answer-quality campaigns remain paused. PP/TG gains, logger overhead and quality are NON_MISURATO.

## Implementation

`MOE_REPLAY_PROFILE=/absolute/path/fresh-phases.csv` enables explicit phase intervals in `llama-moe-replay`. Absent or empty is OFF. The helper refuses to overwrite an existing phase file and reports failure if the file cannot be created or completed. This is a separate file from `GGML_SCHED_PROFILE`.

The sidecar schema is `rep,phase,position,n_tokens,start_us,end_us,status`. Each interval starts immediately before the existing `llama_decode` call and ends after the existing `llama_synchronize`. Both use `ggml_time_us`, the same host clock as R18. Writing the phase record, reading/hashing logits, dumping raw output and printing the replay summary remain outside that interval. No new synchronization is added. The existing replay stdout schema remains unchanged.

Warmup records have `rep=-1`; measured repetitions start at 0. Phase labels come from the replay workload: `prefill` is prompt processing and `decode` is the fixed continuation, one token per replay call. A one-token prompt tail remains prefill. Token count is not used to infer the phase. This replay is teacher-forced; it does not measure sampling or autonomous generation latency.

The successful run closes with `profile_end`. Failure or interruption must not produce a valid completed report. The shared `phase-profile.h` writer is also exercised by the small operator fixture. No scheduler/Vulkan backend source or public API is changed in this increment.

## Correlation and report

```sh
python3 examples/moe-trace/profile-summary.py /absolute/path/scheduler.csv --phases /absolute/path/phases.csv --json /absolute/path/summary.json
```

Use distinct fresh files produced by the same process and run. Timestamp containment cannot authenticate file identity; record file hashes and process provenance with the existing manifest tools. The join is for sequential, synchronized replay evaluations. Concurrent or overlapping evaluation windows are rejected; this is not yet a multi-request profiler.

The parser validates successful status, completion, ordered non-overlapping intervals, repetition IDs, phase order and continuous token positions. A scheduler scope must fit completely inside one interval to receive a phase. Scopes crossing a boundary are rejected instead of being split or assigned by their start time. Ambiguous zero-duration scopes on a shared boundary are also rejected. Every phase evaluation must contain at least one complete `compute_splits` call; one evaluation may contain several calls or schedulers.

With `--phases`, JSON schema version 2 adds:

| Field | Meaning |
| --- | --- |
| `phase_evaluations` | Recorded evaluations, elapsed milliseconds and exact scheduler/call pairs for contained `compute_splits` scopes. |
| `phase_summary` | Totals per repetition and declared phase: evaluation count, tokens, full interval time, scheduler compute calls and event counters/times in milliseconds. Warmup stays in its own repetition. |
| `unattributed_events`, `unattributed_compute_calls` | Work outside evaluation intervals, including initialization/allocation or other host activity. It is retained, not assigned to decode. |

Without `--phases`, the R18 schema-1 summary remains available and accepts its earlier scheduler CSVs. All R18 interpretation limits still apply: inclusive envelopes must not be added to their child events; `compute_call` on Vulkan is host API time, not GPU active time; copy bytes are API payloads, not measured PCIe traffic. The phase duration includes graph preparation and completion waits, but the event breakdown is still partial. Do not treat sums of nested events as elapsed PP/TG or calculate overlap by subtraction.

## Checks performed without a model

The operator fixture uses explicit prefill/decode labels on its first/second execution, with one repetition per graph case. Its three-token second execution is a synthetic correlation case, not an actual single-stream TG sample. It shares the writer with the replay but does not call llama_decode.

- CPU Release, RX 6800 Vulkan and CPU ASan/UBSan: 5 tests passed in each suite.
- Each runtime suite invocation checks the 16 graph cases / 32 evaluations from R18, including reused allocations, changed inputs, repeated and strided IDs, callbacks and scheduler parallel mode.
- All 32 phase evaluations match the expected scheduler/call pairs exactly. The independently checked selective-upload totals are 46080 bytes for the first executions and 12288 for the second executions; full-copy cases remain separate generic-copy counters.
- Raw output is byte-identical with logging OFF, paired logs ON and the earlier fixture executable: SHA-256 `b8ccf66c028651dc191d799d009ffd6fc97887c28fbf4aa2a7781e48f1c7f5a2`. Every value also passes the existing finite/nonzero/scalar-reference checks.
- Offline cases verify warmup separation, a one-token prefill interval, microsecond/millisecond conversion, nested-scope handling, out-of-window retention, boundary rejection and incorrect/missing phase records.
- Existing phase output is not overwritten. A Linux child-process file-size limit forces a final write failure; the program fails and the incomplete file is rejected.
- S01/S02 offline suite: 13 tests passed. The replay executable and its CPU dependencies compiled successfully; no model was loaded. The full CI and a clean full Vulkan build were not run.

## Reproduce

Commands run from the repository root, using the same CPU Release/ASan builds and unchanged prebuilt Vulkan backend described in [R18](moe-scheduler-profile-rx6800.md).

```sh
cmake --build /tmp/moe-s03-build --target llama-moe-scheduler-check llama-moe-replay -j2
python3 examples/moe-trace/test-scheduler-profile.py --check /tmp/moe-s03-build/bin/llama-moe-scheduler-check -v
cmake --build /tmp/moe-s03-asan --target llama-moe-scheduler-check -j2
python3 examples/moe-trace/test-scheduler-profile.py --check /tmp/moe-s03-asan/bin/llama-moe-scheduler-check -v
c++ -std=c++17 -DGGML_SCHED_CHECK_VULKAN examples/moe-trace/scheduler-check.cpp -Iggml/include -L/tmp/moe-s03-build/bin -Lbuild-control-off/bin -Wl,-rpath,/tmp/moe-s03-build/bin -Wl,-rpath,/home/casa/Programmi/Ottimizzazione_llama_cpp_moe/llama-moe-ab/build-control-off/bin -lggml -lggml-base -lggml-cpu -lggml-vulkan -o /tmp/moe-phase-check-vulkan
python3 examples/moe-trace/test-scheduler-profile.py --check /tmp/moe-phase-check-vulkan --vulkan -v
python3 examples/moe-trace/test-profile.py -v
```

To inspect a paired synthetic report, choose paths that do not already exist:

```sh
MOE_REPLAY_PROFILE=/tmp/fresh-phases.csv GGML_SCHED_PROFILE=/tmp/fresh-scheduler.csv /tmp/moe-phase-check-vulkan --vulkan --output-bin /tmp/phase-output.bin
python3 examples/moe-trace/profile-summary.py /tmp/fresh-scheduler.csv --phases /tmp/fresh-phases.csv --json /tmp/phase-summary.json
```

System: RX 6800 / RADV NAVI21, kernel `7.2.9-1-cachyos`, Mesa `26.2.4-arch3.1`, GCC `16.2.1 20260810`. No inherited GGML/RADV/VK/MESA overrides or LD_LIBRARY_PATH were set for the checks. Release remains native ON, compact/active MoE features OFF and dynamic backend loading OFF. The paired diagnostic run sets only the two logging paths. R18's prebuilt Vulkan library hash is unchanged. The CPU/base libraries and new replay binary are identified separately in the validation manifest.

Local archive: `risultati/2026-10-07-phase-profile/`, containing paired CSV/JSON, raw output, logs, source snapshots, build cache, Vulkan inventory, `ldd` and checksums. Earlier archives are preserved.

## Remaining gates

The actual model replay join still needs runtime validation when campaigns resume: identical frozen tokens, model, placement, batch/ubatch, KV and environment; compare OFF/ON output, measure logging overhead and keep the uninstrumented end-to-end reference. Synthetic labels validate the correlation mechanism, not model phase correctness or a speedup.

GPU active durations/shared-clock correlation, the R18 parallel/view timestamp-logger assertion and full weights/KV/recurrent/compute/staging/resident-memory accounting remain unresolved. These are the next S03 gates before ranking optimizations or claiming an exposed critical path.

Current upstream [context code](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp) and [simple example](https://github.com/ggml-org/llama.cpp/blob/master/examples/simple/simple.cpp) were checked on 2026-10-07. This increment uses the APIs in the recorded fork commit; it does not assume that the fork's decode API matches current upstream.
