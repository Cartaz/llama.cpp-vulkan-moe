# R42: bounded reliability and held-out same-GPU validation

2026-10-08, branch `experiment/moe-heldout-validation`, stacked on R41. This experiment extends correctness checks to independent processes and six existing held-out prompt families. It does not change inference code, rank performance or explain the original R41 invalid process. Pool capacity, routing and placement remain opt-in; no new default or B3.

## Frozen identity and registered scope

Helper `25e605e1c807e988f696395ef481dec412a94752`, SHA256 `d71a81d1616852d7a0511fc83014814d6900d0188212ac0e9e1fe88ded2d8e00`, and the exact R41 GUARD engine libraries. Original B1 remains unmodified v0.5.0 `7fe450e19305b828c199d602c23a8337aaa1f03b`; B1 CPU logits are not the denominator for same-GPU qualification. R39 WORK combines a CPU-only rebuilt library (Release/native/OpenMP; Vulkan/input diagnostics OFF) with the remaining R37 Release/native/Vulkan/input diagnostics ON libraries. Compact/active CPU features OFF, shared libraries ON, backend DL OFF, release flags `-O3 -DNDEBUG`; helper C++17 `-O2 -DNDEBUG`. Exact loaded-library hashes are retained per process.

RX6800/RADV NAVI21/ReBAR16GiB, Ryzen5700X3D/32GiB/CachyOS, kernel7.2.9-1-cachyos, Mesa26.2.4-arch3.1, GCC16.2.1/CMake4.4.4/Ninja1.13.2. GPU2600/1075MHz,-100mV,186W retained. Ornith1.5 35B Q4_K_M,21713462848bytes, SHA256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`.

```text
-ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512
-fa on -ctk q8_0 -ctv q8_0 --fit off --load-mode mmap --verbosity 4
```

Target sets `GGML_SCHED_EXPERT_GPU_LAYER=17`; pool sets `GGML_SCHED_EXPERT_POOL=17:SLOTS:256` additionally. Resident removes these variables and prepends `-ot ^blk[.]17[.]ffn_(gate|up|down)_exps[.]weight$=Vulkan0` before `-ncmoe18`. Cleared environment reconstructed with `LC_ALL=C`, frozen `LD_LIBRARY_PATH`, tokens path/repetitions/warmup and raw output path. No silent CPU fallback or global batch-threshold change. Each process runs alone in MemoryMax24G/SwapMax2G with RX6800 preflight,6GiB MemAvailable reserve and600s timeout.

Protocol registered before execution, source parent `ca5892f7b7f4af4799264ce3440476edf5551ba1` (same tree as published R41 `492ab9fc5d84824e9a5c427389572615eea87047`). Three orders cover target/16slots/128slots/resident,12fresh processes, each512PP+200TG with one complete warmup and three measured repetitions. All603measured raw vectors/process must match the201-vector original R41 target repeated three times, not merely its fingerprints. Every warmup vector is scanned directly by the unchanged guard; raw is measured-only. These extra controls are a bounded audit, not a reliability-rate estimate.

Then the existing R28 held-out code/planning/retrieval/arithmetic/structured/Italian cases, each64fixed continuation tokens, are replayed through target/64slots/128slots/resident:24cold raw gates. Finally target/128slots run with route/phase/pool logging after one full warmup:12profiles with measured raw parity to their own cold controls. Capacity was selected on the distinct R41 replay; no capacity tuning uses these results. R28 prompts were previously inspected for offline policy research, so this is not a newly blind corpus. One prompt/family and one fixed continuation do not meet the full T2 requirement of three prompts/family and three continuations.

No performance ranking: raw I/O, PSS sampling and route logging are diagnostic workloads. Teacher-forced sequences are fixed; no semantic score, autonomous generation, long context, concurrency or CPU/GPU fidelity verdict follows from these gates. Resource/device failures stop execution; any correctness discrepancy is retained without selective replacement and prevents dependent qualification.

## Results

48model processes completed:12bounded reliability,24cold held-out gates,12logged profiles. All9576complete measured raw vectors are finite/nonzero and byte-exact to their declared same-GPU references, including all three repetitions in each reliability process.3192warmup vectors are scanned directly (not raw-dumped). Historical R20-R42 total425. No new invalid process; original R41 incident remains unexplained.

| Family | PP tokens | TG distinct experts | Cold-start TG hit % | Repeat TG hit % | Cold-start TG upload MiB | Repeat TG upload MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code | 113 | 102 | 80.078 | 100.000 | 172.125 | 0.000 |
| planning | 119 | 99 | 94.922 | 94.531 | 43.875 | 47.250 |
| retrieval | 139 | 112 | 78.125 | 100.000 | 189.000 | 0.000 |
| arithmetic | 133 | 96 | 81.250 | 100.000 | 162.000 | 0.000 |
| structured | 126 | 112 | 90.234 | 92.188 | 84.375 | 67.500 |
| italian | 149 | 92 | 82.031 | 100.000 | 155.250 | 0.000 |

Every measured profile has complete layer17 PP/TG routes, agreeing exactly between target and128slots. Gate/up/down duplicate observations agree; all130warmup/measured phase calls are covered per profile. Uploads count logical API expert-weight payload (1.6875MiB/expert), not physical PCIe traffic. Cache state persists across context clear; these warm results repeat the same fixed continuation, not a new turn/domain. Prefill bypass reasons and admission/wait distributions remain in validation. Profiles are never ranked.

The two admission cases expose a prefill/decode cache interaction. Planning and structured both use125distinct PP experts, fitting128slots; PP+TG unions are144/163experts. In the repeated prompt, PP admits16/35experts and evicts16/35slots, uploading27.000/59.0625MiB before TG. Decode then incurs28/40misses (47.250/67.500MiB), despite the standalone decode sets99/112fitting the pool. Four other PP sets exceed128 and bypass; repeated decode has zero weight uploads there. These are measured policy effects, not a proof of application speedup.

The next minimal S14 A/B is an opt-in prefill bypass or batch cap, preserving default admission, weights, routing, retained slots and lifetime. Target: avoid replacing reusable decode experts during the next prefill. Tradeoff: losing useful PP admission can increase PP transfers and cold-start misses. Register both planning/structured and existing bypass-family controls; compare normal128slots vs decode-only128slots at the same216MiB, cold and repeated PP+TG, full raw/logged-neutrality gates first, then at least4fresh processes/variant with unlogged timing. Report total PP+TG elapsed time, PP, TG, per-token p95, PP/TG upload payload and cache misses separately; application TTFT needs its own measurement. A larger hit rate alone is insufficient. This change is proposed, not implemented by R42.

Cold and warm profiles are separately paired with their unlogged raw controls. This broadens the declared same-GPU correctness perimeter to six prompt families; it does not validate CPU/GPU numerical equivalence or semantic quality.

Full measurements are recorded in [portable validation](moe-heldout-validation.json). The original R41 invalid process and its suspected NaN signatures remain preserved in the R41 archive; absence of a new incident does not establish a fix.

## Memory and observer limits

The archived runner preserves R39 launch/guard/provenance behavior and adds read-only `/proc/PID/smaps_rollup` PSS/private-clean/private-dirty telemetry plus live `Cpus_allowed_list`. Affinity is inherited, not pinned; one live observation is recorded per process. PSS and private residency are sampled every0.5s and are not exact allocator accounting. RSS, cgroup charge, MemAvailable and global VRAM/GTT remain distinct. Mapped shared weights can raise RSS without the same increase in private RAM; global GPU memory includes desktop use. Short audit sampled peak RSS20.370GiB and PSS20.348GiB; all48processes record inherited CPU0-15, cgroup swap0 and minimum MemAvailable25.026GiB. PSS/private-clean can include file-backed mmap pages; these totals do not establish anonymous/locked RAM use or a RAM saving. Sampling may add overhead, which is why no timing result is promoted here.

## Current upstream audit

Upstream checked at `71ad0590f4808b6202f9213d166913858c73b1bc`, separately from frozen R39 B4 `9c2e0e491a822adae1f0b1c831adb4160057d24f`. Source diff shows device-specific cache banks/budget selection in [PR30112](https://github.com/ggml-org/llama.cpp/pull/30112) and sparse coopmat2 attention changes in [PR30003](https://github.com/ggml-org/llama.cpp/pull/30003). R42 does not adopt these changes or infer a speedup on RX6800. The previously reviewed [issue25195](https://github.com/ggml-org/llama.cpp/issues/25195) describes gfx1201/AMDVLK/long-context behavior; it provides no cause attribution for this RX6800/RADV short incident. Source files, diff and hashes are archived.

## Tests and reproduction

R42 reruns24profile/parser tests and14actual-evaluate guard controls (12negative,2positive). The first guard harness launch lacked its shared-library path and failed in the loader before any test; its log is preserved. Correcting `LD_LIBRARY_PATH` runs the same frozen binary. R41 scheduler11PASS including Vulkan and R39 ASAN/UBSAN/leak parity are inherited for the unchanged engine; neither is represented as a fresh R42 run. Python runner/analysis syntax checks pass.

Local archive `risultati/2026-10-08-heldout-validation/` preserves the registered protocol, input hashes and copied tokens, all raw logits, CSV fingerprints, phase/route/pool logs, process/library/affinity metadata, PSS/DRM/global telemetry and analysis scripts. Portable validation includes complete argv/environment with deduplicated shared-library/profile registries, individual raw hashes and route/counter aggregates. No engine binaries, model weights or large raw dumps are committed.

Next qualification requires new held-out prompts/continuations and context/KV-pressure controls with raw parity before timing. S09 downstream amplification and full CPU/GPU fidelity remain open; multi-layer and1/2/4/8stream cache promotion need their own gates. A future B4 rerun must freeze the newer upstream revision independently and cannot retroactively replace R39 results.
