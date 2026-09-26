#!/usr/bin/env python3
"""Estimate per-layer LRU hit rates from llama-moe-trace CSV output."""

import argparse
import csv
from collections import Counter, OrderedDict, defaultdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="CSV produced by llama-moe-trace")
    parser.add_argument("--phase", choices=("prefill", "decode", "all"), default="decode")
    parser.add_argument("--slots", default="0,16,32,48,64,96,128", help="slots per layer")
    args = parser.parse_args()
    slots = sorted(set(int(s) for s in args.slots.split(",")))
    if not slots or min(slots) < 0:
        parser.error("--slots must contain non-negative integers")

    caches = defaultdict(lambda: {size: OrderedDict() for size in slots})
    hits = Counter()
    activations = Counter()
    frequency = defaultdict(Counter)
    phases_seen = set()
    group = None
    selected = []

    def process():
        if group is None:
            return
        phase, token, layer = group
        # The whole top-k set is requested at once; later ranks cannot hit
        # experts first selected by an earlier rank of the same token.
        distinct = list(dict.fromkeys(selected))
        activations[phase] += len(distinct)
        frequency[(phase, layer)].update(distinct)
        for size in slots:
            cache = caches[(phase, layer)][size]
            hits[(phase, size)] += sum(expert in cache for expert in distinct)
            for expert in distinct:
                if expert in cache:
                    cache.move_to_end(expert)
                elif size:
                    if len(cache) == size:
                        cache.popitem(last=False)
                    cache[expert] = None

    with open(args.trace, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["phase", "token", "layer", "rank", "expert"]:
            parser.error("expected phase,token,layer,rank,expert CSV header")
        for row in reader:
            phase = row["phase"]
            if args.phase != "all" and phase != args.phase:
                continue
            phases_seen.add(phase)
            key = ("all" if args.phase == "all" else phase, int(row["token"]), int(row["layer"]))
            if group != key:
                process()
                group = key
                selected = []
            selected.append(int(row["expert"]))
    process()

    if not activations:
        parser.error("trace has no matching expert activations")
    print(f"phases={','.join(sorted(phases_seen))} layers={len({key[1] for key in frequency})} activations={sum(activations.values())}")
    print("slots/layer  hits  activations  hit_rate")
    for size in slots:
        count = sum(hits[(phase, size)] for phase in activations)
        total = sum(activations.values())
        print(f"{size:11d}  {count:8d}  {total:11d}  {count/total:8.2%}")
    for (_, layer), counts in sorted(frequency.items()):
        total = counts.total()
        print(f"layer {layer:3d}: unique={len(counts):4d} hottest={counts.most_common(1)[0][1]/total:.2%}")


if __name__ == "__main__":
    main()
