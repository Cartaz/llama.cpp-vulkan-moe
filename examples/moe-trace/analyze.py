#!/usr/bin/env python3
"""Estimate per-layer LRU hit rates from llama-moe-trace CSV output."""

import argparse
import csv
import json
from collections import Counter, OrderedDict, defaultdict, deque


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="CSV produced by llama-moe-trace")
    parser.add_argument("--phase", choices=("prefill", "decode", "all"), default="decode")
    parser.add_argument("--slots", default="0,16,32,48,64,96,128", help="slots per layer")
    parser.add_argument("--json", dest="json_path", help="write per-layer statistics as JSON")
    parser.add_argument("--window", type=int, default=32, help="working-set window in evaluated tokens per layer")
    parser.add_argument("--prefill-policies", action="store_true", help="compare decode with prefill-warmed LRU and fixed prefill-frequency slots")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    if args.prefill_policies and args.phase != "decode":
        parser.error("--prefill-policies requires --phase decode")
    slots = sorted(set(int(s) for s in args.slots.split(",")))
    if not slots or min(slots) < 0:
        parser.error("--slots must contain non-negative integers")

    caches = defaultdict(lambda: {size: OrderedDict() for size in slots})
    hits = Counter()
    layer_hits = Counter()
    windows = defaultdict(deque)
    working_set = defaultdict(Counter)
    window_sizes = defaultdict(list)
    activations = Counter()
    frequency = defaultdict(Counter)
    phases_seen = set()
    prefill_frequency = defaultdict(Counter)
    warm_caches = defaultdict(lambda: {size: OrderedDict() for size in slots})
    static_slots = {}
    policy_hits = Counter()
    decoding_seen = False
    group = None
    selected = []

    def access(cache, size, distinct):
        count = sum(expert in cache for expert in distinct)
        for expert in distinct:
            if expert in cache:
                cache.move_to_end(expert)
            elif size:
                if len(cache) == size:
                    cache.popitem(last=False)
                cache[expert] = None
        return count

    def process():
        if group is None:
            return
        phase, token, layer = group
        # The whole top-k set is requested at once; later ranks cannot hit
        # experts first selected by an earlier rank of the same token.
        distinct = list(dict.fromkeys(selected))
        if args.prefill_policies:
            if phase == "prefill":
                prefill_frequency[layer].update(distinct)
                for size in slots:
                    access(warm_caches[layer][size], size, distinct)
                return
            if layer not in static_slots:
                ranked = sorted(prefill_frequency[layer], key=lambda expert: (-prefill_frequency[layer][expert], expert))
                static_slots[layer] = {size: set(ranked[:size]) for size in slots}
            for size in slots:
                policy_hits[(layer, size, "warm_lru")] += access(warm_caches[layer][size], size, distinct)
                policy_hits[(layer, size, "static_prefill")] += sum(expert in static_slots[layer][size] for expert in distinct)
        if args.phase == "all":
            phase = "all"
        activations[phase] += len(distinct)
        frequency[(phase, layer)].update(distinct)
        key = (phase, layer)
        windows[key].append(distinct)
        working_set[key].update(distinct)
        if len(windows[key]) > args.window:
            for expert in windows[key].popleft():
                working_set[key][expert] -= 1
                if working_set[key][expert] == 0:
                    del working_set[key][expert]
        if len(windows[key]) == args.window:
            window_sizes[key].append(len(working_set[key]))
        for size in slots:
            cache = caches[(phase, layer)][size]
            count = access(cache, size, distinct)
            hits[(phase, size)] += count
            layer_hits[(phase, layer, size)] += count

    with open(args.trace, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["phase", "token", "layer", "rank", "expert"]:
            parser.error("expected phase,token,layer,rank,expert CSV header")
        for row in reader:
            phase = row["phase"]
            if args.prefill_policies:
                if phase not in ("prefill", "decode") or (decoding_seen and phase == "prefill"):
                    parser.error("prefill policies require prefill rows before decode rows")
                decoding_seen = decoding_seen or phase == "decode"
            if args.phase != "all" and phase != args.phase:
                if not (args.prefill_policies and phase == "prefill"):
                    continue
            if phase == args.phase or args.phase == "all":
                phases_seen.add(phase)
            key = (phase, int(row["token"]), int(row["layer"]))
            if group != key:
                process()
                group = key
                selected = []
            selected.append(int(row["expert"]))
    process()

    if not activations:
        parser.error("trace has no matching expert activations")
    if args.prefill_policies and not any(prefill_frequency.values()):
        parser.error("prefill policies need prefill expert activations")
    print(f"phases={','.join(sorted(phases_seen))} layers={len({key[1] for key in frequency})} activations={sum(activations.values())}")
    print("slots/layer  hits  activations  hit_rate")
    for size in slots:
        count = sum(hits[(phase, size)] for phase in activations)
        total = sum(activations.values())
        print(f"{size:11d}  {count:8d}  {total:11d}  {count/total:8.2%}")
    layers = []
    for (phase, layer), counts in sorted(frequency.items()):
        total = counts.total()
        sizes = window_sizes[(phase, layer)]
        layers.append({"phase": phase, "layer": layer, "activations": total,
                       "unique_experts": len(counts), "expert_frequency": dict(sorted(counts.items())),
                       "top16_fraction": sum(count for _, count in counts.most_common(16)) / total,
                       "window_tokens": args.window, "full_windows": len(sizes),
                       "window_unique_mean": sum(sizes) / len(sizes) if sizes else None,
                       "window_unique_max": max(sizes) if sizes else None,
                       "cache": [{"slots": size, "hits": layer_hits[(phase, layer, size)],
                                  "misses": total - layer_hits[(phase, layer, size)],
                                  "hit_rate": layer_hits[(phase, layer, size)] / total} for size in slots]})
        print(f"layer {layer:3d}: unique={len(counts):4d} hottest={counts.most_common(1)[0][1]/total:.2%}")
        if args.prefill_policies:
            layers[-1]["prefill_training_activations"] = prefill_frequency[layer].total()
            layers[-1]["prefill_cache"] = [
                {"policy": policy, "slots": size, "hits": policy_hits[(layer, size, policy)],
                 "misses": total - policy_hits[(layer, size, policy)],
                 "hit_rate": policy_hits[(layer, size, policy)] / total}
                for policy in ("warm_lru", "static_prefill") for size in slots]
    if args.prefill_policies:
        print("prefill policy   slots/layer  decode_hits  activations  hit_rate")
        total = sum(activations.values())
        for policy in ("warm_lru", "static_prefill"):
            for size in slots:
                count = sum(policy_hits[(layer["layer"], size, policy)] for layer in layers)
                print(f"{policy:15s} {size:11d} {count:12d} {total:12d} {count/total:8.2%}")
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as stream:
            json.dump({"trace": args.trace, "phase": args.phase, "layers": layers}, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
