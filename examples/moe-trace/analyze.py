#!/usr/bin/env python3
"""Estimate per-layer LRU hit rates from llama-moe-trace CSV output."""

import argparse
import csv
import json
import math
from collections import Counter, OrderedDict, defaultdict, deque


class RoutingProfile:
    def __init__(self, window_lengths):
        self.tokens = 0
        self.last_seen = {}
        self.cold = 0
        self.token_gaps = Counter()
        self.distinct_gaps = Counter()
        self.windows = {length: (deque(), Counter(), Counter()) for length in window_lengths}

    def access(self, experts):
        # Count reuse before updating any expert in this top-k set.
        for expert in experts:
            if expert not in self.last_seen:
                self.cold += 1
                continue
            previous = self.last_seen[expert]
            self.token_gaps[self.tokens - previous - 1] += 1
            self.distinct_gaps[sum(position > previous for position in self.last_seen.values())] += 1
        for expert in experts:
            self.last_seen[expert] = self.tokens
        self.tokens += 1
        for length, (queue, counts, histogram) in self.windows.items():
            queue.append(experts)
            counts.update(experts)
            if len(queue) > length:
                for expert in queue.popleft():
                    counts[expert] -= 1
                    if counts[expert] == 0:
                        del counts[expert]
            if len(queue) == length:
                histogram[len(counts)] += 1

    def report(self, frequencies):
        total = frequencies.total()
        entropy = -sum((count / total) * math.log2(count / total) for count in frequencies.values())
        return {
            "evaluated_tokens": self.tokens,
            "entropy_bits": entropy,
            "entropy_observed_normalized": entropy / math.log2(len(frequencies)) if len(frequencies) > 1 else 0.0,
            "cold_activations": self.cold,
            "reuse_token_gap_histogram": dict(sorted(self.token_gaps.items())),
            "reuse_distinct_intervening_histogram": dict(sorted(self.distinct_gaps.items())),
            "working_set_windows": [
                {"tokens": length, "full_windows": histogram.total(),
                 "unique_mean": sum(size * count for size, count in histogram.items()) / histogram.total() if histogram else None,
                 "unique_max": max(histogram) if histogram else None,
                 "unique_histogram": dict(sorted(histogram.items()))}
                for length, (_, _, histogram) in sorted(self.windows.items())],
        }


def parse_sizes(value):
    sizes = sorted(set(int(size) for size in value.split(",")))
    if not sizes or min(sizes) < 0:
        raise ValueError("expected non-negative integers")
    return sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="CSV produced by llama-moe-trace")
    parser.add_argument("--phase", choices=("prefill", "decode", "all"), default="decode")
    parser.add_argument("--slots", default="0,16,32,48,64,96,128", help="slots per layer")
    parser.add_argument("--json", dest="json_path", help="write per-layer statistics as JSON")
    parser.add_argument("--window", type=int, default=32, help="working-set window in evaluated tokens per layer")
    parser.add_argument("--windows", default="1,8,32,128", help="additional working-set windows in evaluated tokens per layer")
    parser.add_argument("--expert-bytes", help="JSON object mapping layer IDs to bytes for one complete gate/up/down expert")
    parser.add_argument("--layers", help="comma-separated layer IDs; omitted selects all layers")
    parser.add_argument("--prefill-policies", action="store_true", help="compare decode with prefill-warmed LRU and fixed prefill-frequency slots")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    if args.prefill_policies and args.phase != "decode":
        parser.error("--prefill-policies requires --phase decode")
    try:
        slots = parse_sizes(args.slots)
        selected_layers = set(parse_sizes(args.layers)) if args.layers is not None else None
        window_lengths = parse_sizes(args.windows)
        if min(window_lengths) < 1:
            raise ValueError("--windows must be positive")
        expert_bytes = None
        if args.expert_bytes:
            with open(args.expert_bytes, encoding="utf-8") as stream:
                raw_sizes = json.load(stream)
            if not isinstance(raw_sizes, dict):
                raise ValueError("--expert-bytes must be a JSON object")
            expert_bytes = {}
            for layer, size in raw_sizes.items():
                if str(int(layer)) != layer or int(layer) < 0 or type(size) is not int or size < 1:
                    raise ValueError("--expert-bytes needs non-negative layer IDs and positive integer byte sizes")
                expert_bytes[int(layer)] = size
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))

    caches = defaultdict(lambda: {size: OrderedDict() for size in slots})
    hits = Counter()
    layer_hits = Counter()
    windows = defaultdict(deque)
    working_set = defaultdict(Counter)
    window_sizes = defaultdict(list)
    activations = Counter()
    frequency = defaultdict(Counter)
    profiles = defaultdict(lambda: RoutingProfile(window_lengths))
    phases_seen = set()
    prefill_frequency = defaultdict(Counter)
    warm_caches = defaultdict(lambda: {size: OrderedDict() for size in slots})
    static_slots = {}
    policy_hits = Counter()
    requests = defaultdict(Counter)
    decoding_seen = False
    group = None
    selected = []
    last_tokens = {}

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

    def record_request(phase, layer, size, policy, count, distinct):
        stats = requests[(phase, layer, size, policy)]
        stats["requests"] += 1
        stats["all_hit"] += count == len(distinct)
        stats["all_miss"] += count == 0
        stats["mixed"] += 0 < count < len(distinct)
        stats["exceeds_capacity"] += len(distinct) > size

    def request_report(stats):
        return {key: stats[key] for key in ("requests", "all_hit", "all_miss", "mixed", "exceeds_capacity")} | {
            "all_hit_rate": stats["all_hit"] / stats["requests"] if stats["requests"] else None}

    def process():
        if group is None:
            return
        phase, token, layer = group
        previous = last_tokens.get((phase, layer), -1)
        if token <= previous:
            parser.error("token groups must increase within each phase/layer; concatenate separate traces with separate analyses")
        last_tokens[(phase, layer)] = token
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
                for policy, count in (("warm_lru", access(warm_caches[layer][size], size, distinct)),
                                      ("static_prefill", sum(expert in static_slots[layer][size] for expert in distinct))):
                    policy_hits[(layer, size, policy)] += count
                    record_request(phase, layer, size, policy, count, distinct)
        if args.phase == "all":
            phase = "all"
        activations[phase] += len(distinct)
        frequency[(phase, layer)].update(distinct)
        profiles[(phase, layer)].access(distinct)
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
            record_request(phase, layer, size, "cold_lru", count, distinct)

    with open(args.trace, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["phase", "token", "layer", "rank", "expert"]:
            parser.error("expected phase,token,layer,rank,expert CSV header")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                parser.error(f"CSV line {reader.line_num}: expected five columns")
            phase = row["phase"]
            try:
                if phase not in ("prefill", "decode"):
                    raise ValueError("unknown phase")
                token, layer, rank, expert = (int(row[field]) for field in ("token", "layer", "rank", "expert"))
                if min(token, layer, rank, expert) < 0:
                    raise ValueError("negative token, layer, rank or expert")
            except (ValueError, TypeError) as error:
                parser.error(f"CSV line {reader.line_num}: {error}")
            if selected_layers is not None and layer not in selected_layers:
                continue
            if args.prefill_policies:
                if phase not in ("prefill", "decode") or (decoding_seen and phase == "prefill"):
                    parser.error("prefill policies require prefill rows before decode rows")
                decoding_seen = decoding_seen or phase == "decode"
            if args.phase != "all" and phase != args.phase:
                if not (args.prefill_policies and phase == "prefill"):
                    continue
            if phase == args.phase or args.phase == "all":
                phases_seen.add(phase)
            key = (phase, token, layer)
            if group != key:
                process()
                group = key
                selected = []
            if rank != len(selected):
                parser.error(f"CSV line {reader.line_num}: ranks must be consecutive from zero within each token/layer")
            selected.append(expert)
    process()

    if not activations:
        parser.error("trace has no matching expert activations")
    if selected_layers is not None and selected_layers != {layer for _, layer in frequency}:
        parser.error("--layers includes layers without matching activations")
    if args.prefill_policies and not any(prefill_frequency.values()):
        parser.error("prefill policies need prefill expert activations")
    if expert_bytes is not None:
        missing = {layer for _, layer in frequency} - expert_bytes.keys()
        if missing:
            parser.error(f"--expert-bytes missing evaluated layers: {sorted(missing)}")
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
                                  "hit_rate": layer_hits[(phase, layer, size)] / total,
                                  "whole_request": request_report(requests[(phase, layer, size, "cold_lru")])} for size in slots]})
        print(f"layer {layer:3d}: unique={len(counts):4d} hottest={counts.most_common(1)[0][1]/total:.2%}")
        layers[-1].update(profiles[(phase, layer)].report(counts))
        if args.prefill_policies:
            layers[-1]["prefill_training_activations"] = prefill_frequency[layer].total()
            layers[-1]["prefill_cache"] = [
                {"policy": policy, "slots": size, "hits": policy_hits[(layer, size, policy)],
                 "misses": total - policy_hits[(layer, size, policy)],
                 "hit_rate": policy_hits[(layer, size, policy)] / total,
                 "whole_request": request_report(requests[(phase, layer, size, policy)])}
                for policy in ("warm_lru", "static_prefill") for size in slots]
        if expert_bytes is not None:
            size_bytes = expert_bytes[layer]
            layers[-1]["expert_bytes"] = size_bytes
            for cache in layers[-1]["cache"] + layers[-1].get("prefill_cache", []):
                cache.update({"capacity_bytes": cache["slots"] * size_bytes,
                              "requested_bytes": total * size_bytes,
                              "hit_bytes": cache["hits"] * size_bytes,
                              "miss_bytes": cache["misses"] * size_bytes})
    if args.prefill_policies:
        print("prefill policy   slots/layer  decode_hits  activations  hit_rate")
        total = sum(activations.values())
        for policy in ("warm_lru", "static_prefill"):
            for size in slots:
                count = sum(policy_hits[(layer["layer"], size, policy)] for layer in layers)
                print(f"{policy:15s} {size:11d} {count:12d} {total:12d} {count/total:8.2%}")
    if args.json_path:
        summary = []
        policies = ("cold_lru", "warm_lru", "static_prefill") if args.prefill_policies else ("cold_lru",)
        for policy in policies:
            for size in slots:
                def policy_count(layer):
                    if policy == "cold_lru":
                        return layer_hits[(layer["phase"], layer["layer"], size)]
                    return policy_hits[(layer["layer"], size, policy)]
                entry = {"policy": policy, "slots_per_layer": size, "activations": sum(activations.values()),
                         "hits": sum(policy_count(layer) for layer in layers)}
                combined = Counter()
                for layer in layers:
                    combined.update(requests[(layer["phase"], layer["layer"], size, policy)])
                entry["whole_request"] = request_report(combined)
                if expert_bytes is not None:
                    requested = sum(layer["activations"] * layer["expert_bytes"] for layer in layers)
                    hit_bytes = sum(policy_count(layer) * layer["expert_bytes"] for layer in layers)
                    entry.update({"capacity_bytes": sum(expert_bytes[layer] * size for layer in {item["layer"] for item in layers}),
                                  "requested_bytes": requested, "hit_bytes": hit_bytes,
                                  "miss_bytes": requested - hit_bytes, "byte_hit_rate": hit_bytes / requested})
                summary.append(entry)
        with open(args.json_path, "w", encoding="utf-8") as stream:
            json.dump({"schema_version": 3, "trace": args.trace, "phase": args.phase,
                       "selected_layers": sorted(selected_layers) if selected_layers is not None else None,
                       "whole_request_metrics": "pre-request membership in logical caches; all_hit/all_miss/mixed partition token-layer requests; exceeds_capacity overlaps these counts; no runtime cache or transfer saving is implied",
                       "byte_metrics": "logical estimates, not measured transfers" if expert_bytes is not None else "unavailable",
                       "cache_summary": summary, "layers": layers}, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
