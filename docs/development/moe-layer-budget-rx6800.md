# Static layer-aware payload planner pilot

R27, 2026-10-07. S06 now has a separable [offline planner](../../examples/moe-trace/layer-budget.py), not a runtime GPU cache. It targets wasted payload from uniform layer quotas. Frequency differs across layers and expert triplets cost either1,769,472 or2,039,808bytes on this model. Allocating an existing bounded pool according to TRAIN reuse could improve coverage on RDNA2 without changing routing or inference math; benefit in latency still requires a real exact cache and CPU fallback. [Validation data](moe-layer-budget-validation.json) includes every allocation, input/model/layout/template SHA, TRAIN/HELDOUT scores and test evidence. Archive: `risultati/2026-10-07-layer-budget/`.

The two TRAIN and two HELDOUT normal-path traces from [R21](moe-cache-plan-rx6800.md) passed complete-logit parity against original B1. This pilot uses only their64-token decode segments, layers0..17,256experts/top8. Prompt routes and the old callback corpus are excluded. Model SHA `ca6ea26329c88b78ffd90a85163be2e746c2fafd1024f56db47e499f117f9a7f`; template SHA `f55f52930aa8bf44ab5cb85f99370fcc3c56e9a85640b812086d5330bce5d86b`. No new model process or output-quality score is involved. The planner trusts declared paired trace/model provenance, verifies file/layout hashes and strict route geometry, and persists layer-local expert IDs plus byte quotas. It is not a GGUF loader or a runtime remap format.

Three policies were fixed before HELDOUT scoring: uniform TRAIN-frequency quotas; global TRAIN activations per byte; greedy bundles that maximize newly completed weighted TRAIN top-k requests per added byte. Bundles admit missing experts atomically and rescore after every admission. This is a heuristic, not optimal knapsack. Deterministic ties make results reproducible. No HELDOUT data changes a plan. A shared cap is used, with exact allocated payload and slack reported; global policies need not consume precisely the same payload. Allocator overhead, reservations, in-flight slots and source lifetimes are outside this simulation.

At the main cap514,252,800bytes (490.430MiB, uniform15 slots/layer):

| Policy | Payload MiB | Layers with zero slots | HELDOUT retrieval activation / complete request | HELDOUT reasoning activation / complete request |
| --- | ---: | ---: | ---: | ---: |
| Uniform frequency | 490.430 | 0 | 19.043% / 0% | 14.269% / 0% |
| Global frequency/byte | 489.469 | 0 | 20.085% / 0% | 13.845% / 0% |
| Complete-request bundles | 490.266 | 15 | 9.766% / 3.819% | 8.247% / 2.344% |

Global frequency redistributes slots to1..23per layer; improving activation coverage on retrieval does not establish a general improvement. Bundle allocation concentrates on layers1/5/8 with10/163/116slots. It increases complete requests but sacrifices activation coverage and leaves15layers entirely on fallback. Complete means one token/layer's top8. **No policy serves a whole decode token across all18selected layers from cache** on either HELDOUT case. The other22model layers are outside these coverage numbers.

The fixed228.867MiB and1,013.555MiB caps are also recorded. At the latter, bundle complete-request coverage is9.028%/3.819%, with13zero-quota layers; uniform is0.087%/0.087%. These are logical observations on two HELDOUT prompts, not saved PCIe bytes, a performance confidence interval or a selected winner. Current ncmoe18 performs no TG expert-weight uploads, so a future cache changes placement and compute/fallback costs; multiplying logical hits by bytes does not measure existing transfers removed.

Validation: all18offline tests pass (14existing plus4planner cases). Fixtures cover heterogeneous tight budgets, indivisible bundles, shared-request gains, counts and whole-token partition, deterministic ties, malformed geometry, hash/model identity rejection, and unchanged plans when HELDOUT data changes. No engine/operator rerun is needed for this Python-only planner.

Reproduce from a fresh manifest with verified trace hashes, separate TRAIN/HELDOUT cases and expert-triplet byte layout:

```sh
python3 examples/moe-trace/test-profile.py
python3 examples/moe-trace/layer-budget.py risultati/2026-10-07-layer-budget/manifest.json --json fresh-layer-budget.json
```

The versioned validation JSON preserves this manifest. Paths to the raw R21 traces refer to the local archive. Minimum layer reservations, a broader untouched corpus, miss-cost weighting and runtime remap/lifetime remain open. The exact next A/B is uniform versus minimum-reservation versus global policy at the same available payload cap on new held-out traces, followed by T0 exact runtime all-hit/all-miss/mixed/eviction and T3 measuring TG, H2D/D2H, fallback cost and p95. Keep the feature OFF by default until those gates pass.
