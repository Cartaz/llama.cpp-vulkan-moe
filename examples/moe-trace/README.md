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

## Fixed-token replay

Record the token IDs of the prompt and greedy continuation as well as the routing:

```sh
MOE_TRACE_OUT=agent-routing.csv MOE_TRACE_TOKENS_OUT=agent-tokens.csv \
  ./build-vulkan/bin/llama-moe-trace -m model.gguf -f prompt.txt -n 128 \
  -ngl 99 -ncmoe 18 -t 8 -c 1024 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0
MOE_REPLAY_IN=agent-tokens.csv MOE_REPLAY_REPS=3 \
  ./build-vulkan/bin/llama-moe-replay -m model.gguf \
  -ngl 99 -ncmoe 18 -t 8 -c 1024 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 > replay.csv
```

The workload CSV is `phase,token,id`: a nonempty prefill followed by optional decode tokens, with consecutive zero-based positions. Replay uses these IDs directly, ignores sampling parameters and `-n`, and does not install a routing callback. Keep the model/tokenizer, workload file and batch/ubatch sizes fixed across builds. Context size must fit the full workload.

Replay clears sequence memory, evaluates the complete workload once as a warmup, then runs the requested repetitions from empty sequence memory. Its CSV columns are `rep,phase,position,n_tokens,elapsed_us,logits_hash`. Each time is a synchronized decode call; the prefill sum and decode sum are separate. Hashing, CSV output and optional raw-logit writing occur outside each measured call. These are inference-call timings, not request TTFT, sampling time, total application latency or concurrent serving performance. The FNV-1a hash covers the last token's full float logits for each call; it is not a cryptographic checksum.

Set `MOE_REPLAY_LOGITS_OUT=logits.bin` for a byte comparison between builds using the same workload and batching. The file concatenates native float logits after each prefill chunk and each decode call, in repetition order; its vocabulary size comes from the model. Retain binary SHA-256s and an exact `cmp` for correctness evidence. Measure throughput separately with raw-logit output disabled.

The same replay source can be compiled against an unmodified upstream build's libraries without modifying that checkout. Check that the headers and library ABI match, and record the replay source hash separately from the inference library commit. A fixed input sequence controls token-induced routing changes; it does not guarantee identical routing if logits or backend arithmetic change.

## Prefill-informed cache policies

Use `--phase decode --prefill-policies` to compare the existing cold LRU with two additional per-layer policies:

```sh
python3 examples/moe-trace/analyze.py agent-routing.csv --phase decode \
  --prefill-policies --slots 0,16,32,48,64,96,128 --json prefill-policies.json
```

`warm_lru` processes the prefill into an LRU cache, then counts hits and performs LRU updates during decode. `static_prefill` selects the most frequent prefill experts once and holds those slots fixed throughout decode; equal frequencies use ascending expert ID. It never uses decode frequencies to select experts. Each request checks its distinct top-k experts before any cache update. Layers have independent caches. Layers absent from prefill start empty; the trace must contain some prefill and must place all prefill rows before decode.

Only decode activations enter the hit-rate denominator. Per-layer JSON includes training activation counts and `prefill_cache` hits, misses and rates. Existing output is unchanged when the option is omitted. These policies assume the initial slots are populated before decode and exclude priming cost, transfers, eviction synchronization and cache capacity shared with dense weights/KV. They measure logical coverage, not a working GPU cache or an end-to-end speedup. Validate prompt-informed placement on multiple held-out prompts before choosing a policy.

## Answer-quality A/B

The separate `llama-moe-quality` target and `quality-bench.py` compare answer correctness at ubatch 512 and 2048 using fixed synthetic tasks and automatic oracles. See the [protocol](../../docs/development/moe-quality-ab-rx6800.md) and [pilot results](../../docs/development/moe-quality-pilot-rx6800.md). The prepared larger datasets are not reported as completed evaluations.
