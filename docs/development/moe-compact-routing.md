# Compact CPU MoE routing experiment

This experiment starts from fork commit `bd6833de1659904b47117ff7cf7dd6ef24bbe740`, based on llama.cpp v0.5.0 (`7fe450e19305b828c199d602c23a8337aaa1f03b`). Enable it with `-DGGML_CPU_MOE_COMPACT=ON`. It is OFF by default and reports the CPU backend feature `MOE_COMPACT=1` when enabled.

## Change

CPU `MUL_MAT_ID` normally reserves a row mapping for every expert and every selected expert/token pair. The experimental path counts pairs per expert, computes offsets, and packs the existing mappings into one contiguous array. It preserves the order within each expert, the selected IDs, the vector-dot kernels, the IQ panel dispatch, and the worker scheduling.

For 256 experts, top-8 routing, and 2048 tokens, the row table shrinks from 32 MiB to 128 KiB, with an additional 2 KiB offset array. This saves temporary CPU workspace, not model weight memory or VRAM. The extra counting pass can cost time. A reduction in workspace does not establish a throughput improvement.

## Compare builds

Use the same source revision, compiler, and options except for the experimental build option. In fish:

```fish
for mode in OFF ON
    cmake -S . -B build-moe-$mode -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_CPU_MOE_COMPACT=$mode -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=ON
    cmake --build build-moe-$mode --target llama-bench test-backend-ops llama-moe-routing-check -j 8
    ./build-moe-$mode/bin/test-backend-ops test -b CPU -o MUL_MAT_ID > test-moe-$mode.log
    ./build-moe-$mode/bin/llama-moe-routing-check --output-bin routing-$mode.bin > routing-$mode.log 2> workspace-$mode.log
end
cmp routing-OFF.bin routing-ON.bin
diff -u routing-OFF.log routing-ON.log
```

`llama-moe-routing-check` requires a CPU backend linked at build time (`GGML_BACKEND_DL=OFF`, the default). It covers F32, Q4_K, and Q6_K weights with 256 experts and top-8 selection; 1, 7, and 65 tokens; 1, 4, and 8 threads; broadcast inputs; repeated IDs; and strided input and ID views. A final Q4_K case uses 2048 tokens. Each case reuses its plan for a second pass with changed IDs and inputs, checks workspace guards, and emits deterministic output hashes. F32 outputs are checked against independent scalar dot products. Raw output files allow exact comparison of the quantized paths between builds.

The CPU reference in `test-backend-ops` shares the compiled routing implementation, so that suite alone cannot independently verify this layout change. Use the cross-build comparison and scalar checker as well.

For Ornith on RX 6800, freeze the v0.5.0 runtime flags from the sweep, including `ncmoe`, threads, context, batch, ubatch, flash attention, and KV types. Pass those exact runtime flags to both binaries. Record each commit, all build options, the model hash, and the benchmark results. Keep a separately tuned comparison separate from the fixed-flag A/B comparison.

## Validation limits

Local checks use a Linux x86-64 CPU build with `GGML_NATIVE=OFF` and `GGML_VULKAN=OFF`. RX 6800 Vulkan execution and Ornith end-to-end throughput remain to be measured on the target machine. AddressSanitizer and UndefinedBehaviorSanitizer checks use `detect_leaks=0` because leak detection is unavailable in the execution environment.

All 217 deterministic cases pass with the option OFF and ON, with byte-identical raw outputs (SHA-256 `3d87bacdf6d90e8aad55b747679628a33736e7f490b9e988264181045eaea5dc`). The compact build also passes the 217 cases with AddressSanitizer and UndefinedBehaviorSanitizer; its output hashes match the OFF build. Total planned workspace for the 2048-token Q4_K case decreases from 34,171,480 to 750,176 bytes, including activation conversion and worker scratch.

Both builds pass all 955 CPU `MUL_MAT_ID` operator cases, including the 12 new 256-expert cases. Run the full OFF and ON suites sequentially: running both together exceeded the local resource limit (exit 137), while both sequential runs completed successfully. Validation used GCC/G++ 13.3.0, CMake 4.4.3, and Ninja 1.13.2.

## Next experiment

A GPU expert cache needs persistent device slots, expert ID remapping, staging buffers, buffer lifetime and synchronization rules, and a CPU miss path. This change does not implement that cache. Relevant upstream proposals: [hybrid expert cache](https://github.com/ggml-org/llama.cpp/discussions/24528) and [persistent expert pool](https://github.com/ggml-org/llama.cpp/discussions/28248).
