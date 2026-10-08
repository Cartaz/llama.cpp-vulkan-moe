# R40: first CPU roundtrip intermediates on RX 6800

2026-10-08, ReBAR16GiB, branch `experiment/moe-roundtrip-intermediates`. R39 frozen WORK engine `f3a5150eb0c6eb57e60effe6498b7f6319e73136`, inherited R37 input diagnostic graph and no-callback replay helper. No inference code changes in R40. The exact first arithmetic difference of the CPU intervention is now measured: Q8_K dequantize/requantize changes a block scale, while its integer quants remain identical. This localizes the roundtrip perturbation; the much larger downstream logits amplification remains unexplained. Keep the intervention rejected as a fidelity fix.

## Protocol and observation gate

Twelve new model processes: fresh original CPU control; copy/gate/up/down/all each with WORK observation OFF and ON; another all-ON repeat. Same512PP+200TG/201complete248320-logit vectors, c1024,b/ub512,Q8_0K/V,mmap,FAon,t/tb8,fitOFF,one stream,no warmup as R39. Each mode adds `LLAMA_MOE_INPUT_CONTROL=MODE`; observed processes add `GGML_CPU_MOE_CAPTURE_WORK=CAPTURE`. The observer OFF/ON gate is exact full raw logits with finite/nonzero checks before hashes. All repeat requires byte-exact full capture and logits. Identity copy is the structural control; original must match R39. No timing ranking or threshold relaxation.

The exact base command is:

```text
replay -m /home/casa/Programmi/modelli/Ornith-1.5-35B-Q4_K_M.gguf -ngl 99 -ncmoe 18 -t 8 -tb 8 -c 1024 -b 512 -ub 512 -fa on -ctk q8_0 -ctv q8_0 --fit off --verbosity 4 --load-mode mmap
```

`LD_LIBRARY_PATH` points to R39 frozen WORK; `MOE_REPLAY_IN` is the frozen R39 token file, `MOE_REPLAY_REPS=1`, `MOE_REPLAY_WARMUP=0`, `MOE_REPLAY_LOGITS_OUT=RAW`, `LC_ALL=C`. The registered protocol and per-process full argv/environment/loaded engine hashes are in [validation](moe-roundtrip-intermediates-validation.json); R39 validation contains all build/model/hardware hashes. Model SHA256 `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`,21713462848bytes; RX6800/5700X3D/32GiB, kernel7.2.9-1-cachyos, Mesa/RADV26.2.4-arch3.1,GCC16.2.1,CMake4.4.4,Ninja1.13.2,Release/native/Vulkan/OpenMP. User2600/1075MHz,-100mV,186W settings remain unchanged. Single process, MemoryMax24G/swap2G,6GiB available-RAM reserve and RX6800 preflight.

R40 adds an explicit parser `--allow-different-inputs` option for deliberate single-projection interventions. Default parsing remains strict; enabling the option only permits gate/up input differences. Layout, routing, triplets, finite/nonzero and Q8_K block checks remain required. Tests verify that ordinary parsing rejects the intervention and optional parsing still rejects invalid data. This does not weaken a logits fidelity gate.

## Results

All five observer OFF/ON pairs match byte for byte; all repeat capture/logits match. Identity-copy input/Q8_K/output captures equal the original across all600records, and all201logit vectors equal the original. The12processes validate2412full logit vectors; six observed processes provide3600records and28800projection output vectors.

At decode step1, the selected gate/up roundtrip changes one block scale by1.862645149230957e-9. No integer quant or block sum changes. Recomputed projection output differs by <=4.7684e-7. The unselected up in gate-only, and unselected gate in up-only, remain exact. The layer17 down input changes only at small floating-point scale after activation; down quantized scales change, with no integer quant or sum changes at that step. Full routing IDs at layer17 are identical on the first step in every mode.

| CPU intervention | First changed projection output | First changed full logits | First changed logits RMS | Maximum logits RMS | Changed argmax /201 | Screen |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| copy | none | none | 0 | 0 | 0 | PASS, exact |
| gate | gate, step1 | 1 | 0.0453775 | 0.516267 | 3 | FAIL |
| up | up, step1 | 1 | 0.0603153 | 0.651788 | 1 | FAIL |
| down | down, step2 | 2 | 0.0126037 | 0.423207 | 2 | FAIL |
| all | gate, step1 | 1 | 0.0453774 | 0.701015 | 2 | FAIL |

All prefill vectors are exact. Down-only first-step F32 input is rounded but its actual Q8_K bytes and projection outputs remain exact; its first logits vector remains exact too. The first down-only projection/logits difference occurs at step2. The screen preserves R37/R39 max_abs<=0.5,RMS<=0.05,KL<=0.01,argmax>=99%. Gate/up/down/all all fail at least one condition. Paired observed runs remain exact to unobserved runs even where those runs fail the comparison to original.

## Interpretation and next work

Established: the roundtrip is not byte-idempotent on real model inputs. The first gate/up output perturbation is caused by the measured scale change under an otherwise exact CPU path. R39 already proved direct original-F32 CPU conversion matches the real work bytes and direct replay matches actual model output; it must remain the original reference. Identity copy and raw-neutral observer controls exclude those mechanisms as necessary causes here.

Unresolved: the downstream amplification from <=4.7684e-7 in layer17 projection output to~0.22/0.24maximum first-step logit error. No claim is made that one measured scale explains the original CPU/GPU path difference, or that all amplification is a recurrent-state bug. Subsequent layer/router/normalization/KV/recurrent intermediates need their own observer parity before attribution. These are teacher-forced numerical diagnostics, not free-generation semantic evaluations.

Keep roundtrip as opt-in diagnosis, SCARTATO as a fidelity fix. PP/TG,TTFT,semantic quality,long context and concurrency are NON_MISURATO. R39+R40 do not authorize a new default or integration. The independently valid same-GPU pool perimeters can proceed to S04 capacity benchmarking as R41 while S09 stays open.

Archive `risultati/2026-10-08-roundtrip-intermediates/`: frozen protocol,exact commands,600-record captures,complete raw logits,per-vector CSV,per-record intermediate comparisons and telemetry. R20-R40 executed310model processes under historical accounting,12new here. Existing profile tests24PASS; scheduler offline10PASS with compiled runtime validated separately when the model GPU is idle.
