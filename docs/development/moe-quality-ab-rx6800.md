# Deterministic quality A/B protocol: ubatch 512 vs 2048

This suite tests task correctness, not identical prose. It is a local synthetic regression suite, not an official RULER, HumanEval, MMLU or GSM8K score.

## Primary comparison

Use the same tested fork executable/library set, model file, quantization, attention, cache types, context, load mode and CPU/GPU placement. Enable GGML_VK_FA_Q8_SYNC=1 in both variants. Change only ubatch: 512 or 2048. Keep batch=2048 and context=8192. One sequence, one inference stream. The standard upstream build can be used as an additional same-batch control for any detected regression; do not mix a load-mode or synchronization change into the primary comparison.

Native GGUF chat template, complete system/user messages, fixed template time. Compare the same thinking setting in A/B; use enable_thinking=true for the final Ornith quality profile, and keep the initial no-thinking screen separate. No JSON grammar or answer-constraining logit bias. The final quality profile keeps native reasoning enabled and uses the author-recommended temperature=0.6, top-p=0.95 and top-k=20, with min-p=0 (no extra filter), and predeclared seeds [42, 12345, 20261004]. Start with a 2048-token answer budget at the same context in A/B; retain truncation as a separate failure metric and calibrate budgets before freezing a held-out test set. Fixed seeded sampling is repeatable only when the numerical backend is stable; duplicate-case checks are required. Greedy temperature=0, seed=42 remains a separate diagnostic screen. Greedy removes sampling randomness but does not by itself guarantee backend determinism. Include duplicate cases and compare their generated token IDs. If duplicates differ, suspend statistical interpretation and diagnose state/scheduling first.

Save exact source commit, CMake cache, compiler/driver/kernel versions, model and binary/library hashes, environment, rendered prompts, input token IDs, generated token IDs, response text and completion status. Match input token IDs exactly between A/B. Clear sequence/KV/recurrent memory before each independent task. No context shortening, chat-history reuse, prompt caching or truncation.

## Tasks and independent answer oracles

| Family | Complete question | Deterministic grading |
| --- | --- | --- |
| Retrieval | Locate an exact six-digit code among 90 records; vary position near start, middle and end | Exact string, including leading zeros |
| Arithmetic | Filter ledger rows, multiply quantity by price, sum and apply a one-time rebate | Exact integer calculated by Python |
| Logic | Follow exactly N links in a shuffled directed chain among distracting records | Exact final node, from graph traversal |
| Code tracing | Read Python stable-dedup or compact-routing prefix-offset code and compute its result on the listed input | Exact ordered integer array; oracle independently executes only benchmark-authored code |

The model is not given the expected answer or grader metadata. The generator seed is fixed, datasets are serialized and hashed, and each instance includes oracle inputs for independent verification. The scorer does not ask an LLM to judge responses. It distinguishes correct values, invalid format, truncation and repeat instability. Ordered arrays are not treated as sets; booleans are not accepted as integers; a matching number elsewhere in an explanation is not accepted.

A short control for each family must fit <=512 actual model tokens. Long cases must exceed 2048 actual model tokens, and prompt + answer budget must fit the same context. Abort if these conditions fail. Many ordinary short MMLU/GSM8K questions alone will not exercise the larger prefill path; arbitrary filler is not a substitute for testing retrieval and reasoning on relevant long inputs.

Pilot: two instances per family (8 primary cases), four short controls and two duplicate cases = 14 tasks per variant. The pilot validates the evaluation pipeline and screens for large failures; it cannot certify non-inferiority.

Prepared extended screen: 16 instances per family (64 primary cases), four short controls and two duplicates = 70 tasks per variant. A second dataset with 140 instead of 90 document records tests longer prompts; verify actual token lengths and context headroom before running. The current greedy pilot shows a low score floor on some families; successful calibration in the native-reasoning profile is required before treating the extended set as an acceptance test. Expansion should add difficulty/positions/lengths, not just identical filler. Generate a separate held-out set with a distinct predeclared --generator-seed and freeze its hash before tuning. A stronger follow-up should include hundreds of independently generated cases and complete real task prompts, with a held-out fixed test set. Do not tune changes against the held-out cases.

## Paired results and decision

Report correct/total for each family and overall. Report A-correct/B-wrong (losses), A-wrong/B-correct (wins), both wrong, and both correct. Do not hide one family's regression in a global average. Short controls and duplicate cases are excluded from the primary score.

The current scorer reports an exact McNemar test on discordant cases. For extended held-out runs, add paired item bootstrap intervals for score differences, stratified by family and clustered by question across seeds. Small pilot p-values are only descriptive. Equal scores or a nonsignificant test do not prove equivalent quality. Any extra large-batch errors should be inspected and reproduced, including an unmodified upstream control with the same batch. An acceptable non-inferiority margin must be chosen before the final held-out run; it is a product decision, not a value to choose after seeing the scores.

Keep deterministic repeated-case failures, nonfinite/zero logits, crashes, prompt overflow, truncation and malformed outputs as explicit failure metrics. If both variants fail the same tasks, the benchmark needs easier controls or better coverage before it can distinguish a batch regression.

Measure PP separately from generation and latency. Loaded-model first-token latency, time to complete a correct answer, total correct tasks per wall-clock time, peak VRAM/GTT and system MemAvailable matter. The pilot generation timer includes sampler and finite-value checks and is not interchangeable with the earlier pure decode TG benchmark. Repeat throughput measurements in ABBA order after correctness is established; do not infer a TG improvement from PP alone.

## Additional established benchmarks

- Ornith sampling and reasoning profile: https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B . These recommendations are from the model author and do not reproduce its full published benchmark protocol.
- RULER's retrieval, multi-hop tracing and aggregation tasks are useful templates for a broader long-context suite: https://github.com/NVIDIA/RULER . The local tasks above are inspired by these categories and are not its full evaluation protocol.
- HumanEval / stronger execution-based code tests measure functional correctness rather than matching source text: https://github.com/openai/human-eval . A later code-generation stage needs isolated execution, fixed tests and a sufficient answer budget. Code tracing in the current pilot does not replace it.
- llama-perplexity offers perplexity / KL-distribution comparisons: https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md . Use it as a supplementary numerical warning signal with identical corpus/context/tokenization and verified recurrent-state support, not as the only answer-quality measure. Prefer answer-token teacher-forced likelihood for the local tasks when adding a diagnostic.

After the native-reasoning quality profile, repeat selected fixed instances with the previously used temperature 0.8/top-k 40/min-p 0.05 and the same predeclared seed list for both batches, to check that user profile separately. Keep it separate from the greedy diagnostics and author-recommended profile. Cluster seed repetitions by question for confidence intervals; they are not independent new questions. Then test representative agent workloads and concurrency 1, 2, 4 and 8, preserving the same correctness graders; the current quality pilot is single-stream only.

## Reproduction

`quality-bench.py selftest` checks generator repeatability, independent oracles and positive/negative grading controls.

`quality-bench.py generate --per-family 2 --out pilot.jsonl` creates the pilot; `--per-family 16` creates the extended screen. Add `--records 140` for the longer document profile. The current generator seed is 20261004.

Build the llama-moe-quality target after configuring the existing fork. Set QUALITY_CASES to the JSONL file and QUALITY_OUT to an existing output directory. Set QUALITY_THINKING=1 for native reasoning; unset it for the no-thinking diagnostic (presence enables it even if its value is 0). Run with the same model and flags for both ubatches:

```sh
GGML_VK_FA_Q8_SYNC=1 QUALITY_THINKING=1 QUALITY_CASES=/absolute/pilot.jsonl QUALITY_OUT=/absolute/A ./build/bin/llama-moe-quality -m /absolute/model.gguf -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 8192 -b 2048 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off --load-mode none --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 --seed 42 -n 2048
```

For B, change only `-ub 2048` and the output directory. Use the scorer with the same dataset:

```sh
python3 quality-bench.py score --cases pilot.jsonl --a A/responses.jsonl --b B/responses.jsonl --out score.json
```

## Executed pilot

See [the pilot report](moe-quality-pilot-rx6800.md) for the no-thinking screen, native reasoning/budget calibration, original-upstream controls and observed format difference. The two prepared 64-task corpora are not completed quality evaluations.
