# MoE routing trace

This example records routed expert IDs for a prompt and a greedy continuation. It is adapted from [upstream PR #28544](https://github.com/ggml-org/llama.cpp/pull/28544) at commit `879f4d67a107c799ec571ee4e42398b971226d73`. This fork adds token and phase indices and an offline cache simulator.

```sh
MOE_TRACE_OUT=ornith-routing.csv ./build-vulkan/bin/llama-moe-trace \
  -m /path/to/ornith.gguf -f /path/to/prompt.txt -n 1024 -ngl 99 -fa on
python3 examples/moe-trace/analyze.py ornith-routing.csv --phase decode --slots 16,32,48,64,96,128 --window 32 --json ornith-routing-summary.json
```

The CSV columns are `phase,token,layer,rank,expert`. `token` is the zero-based position in this single sequence. `rank` is the expert's position in the top-k result. Prefill and decode rows are distinct. The callback reads the I32 view span, including stride gaps, after each layer; it forces scheduler synchronization and changes measured latency. Run throughput tests separately, without the trace callback.

The analyzer simulates an independent LRU cache per layer. It checks all experts selected for a token against the cache **before** inserting any of them. The printed hit ratio is an upper bound on the fraction of expert activations that could be served by an ideal cache of that size: it excludes transfer time, capacity occupied by dense weights and KV, asynchronous update delays, CPU work, and graph overhead. A `--slots 0` run gives the cold baseline. Each phase starts with an empty simulated cache; use `--phase all` only to examine a sequential prefill-to-decode transition.

Record the GGUF name and hash, `git rev-parse HEAD`, build flags, compiler, Mesa/RADV, kernel, GPU memory use, and the prompt when sharing results. Compare PP and TG throughput against the unmodified `v0.5.0` tag using the same settings and workload. Then test 1, 2, 4 and 8 streams independently; this single-sequence trace does not predict concurrent cache contention.

The JSON report contains expert frequencies, top-16 activation share, per-layer LRU hits and misses, and unique experts in a rolling window of evaluated tokens. Window statistics use only complete windows; they are null when the trace is shorter than `--window`. These counts describe logical expert activations, not bytes transferred or measured GPU cache hits.

Prefill traces from the original example at `0a5bc4d` read the top-k view as contiguous and can contain incorrect expert IDs after the first token of each batch. Regenerate those traces with the stride-aware reader before using them for placement or cache decisions. Decode traces with one token per batch are unaffected by the row-stride error.
