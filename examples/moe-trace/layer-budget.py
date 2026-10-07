#!/usr/bin/env python3
"""Plan static expert payloads from TRAIN routes and score fixed HELDOUT routes."""

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_routes(path, layers, n_layers, top_k, experts, prompt, decode):
    groups = defaultdict(list)
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["phase", "token", "layer", "rank", "expert"]:
            raise ValueError("unexpected route header")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("invalid route columns")
            token, layer, rank, expert = (int(row[key]) for key in ("token", "layer", "rank", "expert"))
            phase = "prefill" if token < prompt else "decode"
            if row["phase"] != phase or not 0 <= token < prompt + decode or not 0 <= layer < n_layers or not 0 <= expert < experts:
                raise ValueError("invalid phase, token, layer or expert")
            chosen = groups[token, layer]
            if rank != len(chosen) or rank >= top_k or expert in chosen:
                raise ValueError("duplicate or incomplete top-k")
            chosen.append(expert)
    if len(groups) != (prompt + decode) * n_layers or any(len(ids) != top_k for ids in groups.values()):
        raise ValueError("missing tokens, layers or ranks")
    return {(token, layer): frozenset(ids) for (token, layer), ids in groups.items() if token >= prompt and layer in layers}


def frequencies(traces, sizes):
    result = {layer: Counter() for layer in sizes}
    for groups in traces:
        for (_, layer), ids in groups.items():
            result[layer].update(ids)
    return result


def plan(traces, sizes, budget, policy):
    if type(budget) is not int or budget < 0 or not sizes or any(type(size) is not int or size < 1 for size in sizes.values()):
        raise ValueError("invalid payload budget or expert sizes")
    counts = frequencies(traces, sizes)
    selected = {layer: set() for layer in sizes}
    used = 0
    if policy == "uniform_frequency":
        slots = budget // sum(sizes.values())
        for layer in sorted(sizes):
            selected[layer].update(sorted(counts[layer], key=lambda expert: (-counts[layer][expert], expert))[:slots])
        used = sum(len(ids) * sizes[layer] for layer, ids in selected.items())
    elif policy == "global_frequency_per_byte":
        ranked = sorted(((layer, expert, count) for layer, freq in counts.items() for expert, count in freq.items()),
                        key=lambda item: (-Fraction(item[2], sizes[item[0]]), item[0], item[1]))
        for layer, expert, _ in ranked:
            if used + sizes[layer] <= budget:
                selected[layer].add(expert)
                used += sizes[layer]
    elif policy == "request_bundle":
        requests = {layer: Counter() for layer in sizes}
        for groups in traces:
            for (_, layer), ids in groups.items():
                requests[layer][ids] += 1
        while True:
            best = None
            for layer in sorted(sizes):
                missing = Counter()
                for ids, count in requests[layer].items():
                    rest = ids - selected[layer]
                    if rest:
                        missing[rest] += count
                for rest in missing:
                    cost = len(rest) * sizes[layer]
                    if used + cost > budget:
                        continue
                    gain = sum(count for other, count in missing.items() if other <= rest)
                    key = (-Fraction(gain, cost), cost, layer, tuple(sorted(rest)))
                    if best is None or key < best[0]:
                        best = key, layer, rest, cost
            if best is None:
                break
            _, layer, rest, cost = best
            selected[layer].update(rest)
            used += cost
    else:
        raise ValueError("unknown policy")
    assert used == sum(len(ids) * sizes[layer] for layer, ids in selected.items()) and used <= budget
    return {"policy": policy, "budget_bytes": budget, "payload_bytes": used, "slack_bytes": budget - used,
            "zero_quota_layers": [layer for layer in sorted(sizes) if not selected[layer]],
            "layers": {str(layer): {"slots": len(ids), "expert_bytes": sizes[layer], "payload_bytes": len(ids) * sizes[layer],
                                     "experts": sorted(ids)} for layer, ids in sorted(selected.items())}}


def score(groups, allocation, sizes):
    selected = {int(layer): set(data["experts"]) for layer, data in allocation["layers"].items()}
    stats = Counter()
    per_token = defaultdict(list)
    for (token, layer), ids in groups.items():
        hits = len(ids & selected[layer])
        stats.update(requests=1, activations=len(ids), hits=hits, requested_bytes=len(ids) * sizes[layer], hit_bytes=hits * sizes[layer],
                     all_hit=int(hits == len(ids)), all_miss=int(hits == 0), mixed=int(0 < hits < len(ids)))
        per_token[token].append(hits == len(ids))
    assert stats["requests"] == stats["all_hit"] + stats["all_miss"] + stats["mixed"]
    if not groups or any(len(complete) != len(sizes) for complete in per_token.values()):
        raise ValueError("score requires every selected layer per token")
    return dict(stats, activation_hit_rate=stats["hits"] / stats["activations"], byte_hit_rate=stats["hit_bytes"] / stats["requested_bytes"],
                all_hit_rate=stats["all_hit"] / stats["requests"], tokens=len(per_token), all_layer_hit_tokens=sum(all(value) for value in per_token.values()))


def run(manifest_path):
    root = Path(manifest_path).resolve().parent
    manifest = json.loads(Path(manifest_path).read_text())
    model_sha = manifest["model_sha256"]
    if not isinstance(model_sha, str) or len(model_sha) != 64 or any(char not in "0123456789abcdef" for char in model_sha):
        raise ValueError("invalid model SHA256")
    geometry = manifest["geometry"]
    layers = manifest["selected_layers"]
    if not layers or sorted(set(layers)) != layers or any(type(layer) is not int or not 0 <= layer < geometry["layers"] for layer in layers):
        raise ValueError("invalid selected layers")
    if not 0 < geometry["top_k"] <= geometry["experts"] or geometry["layers"] < 1:
        raise ValueError("invalid route geometry")
    layout = root / manifest["expert_bytes_path"]
    if sha(layout) != manifest["expert_bytes_sha256"]:
        raise ValueError("expert byte layout SHA mismatch")
    all_sizes = json.loads(layout.read_text())
    sizes = {layer: all_sizes[str(layer)] for layer in layers}
    cases, train, identities = {}, [], set()
    for case in manifest["cases"]:
        path = (root / case["path"]).resolve()
        if case["split"] not in ("train", "heldout") or case["name"] in cases or case["sha256"] in identities:
            raise ValueError("duplicate trace/name or invalid split")
        if case["model_sha256"] != model_sha or sha(path) != case["sha256"]:
            raise ValueError("trace identity or SHA mismatch")
        if type(case["prompt_tokens"]) is not int or case["prompt_tokens"] < 1 or type(case["decode_tokens"]) is not int or case["decode_tokens"] < 1:
            raise ValueError("invalid workload geometry")
        groups = read_routes(path, layers, geometry["layers"], geometry["top_k"], geometry["experts"], case["prompt_tokens"], case["decode_tokens"])
        cases[case["name"]] = groups
        identities.add(case["sha256"])
        if case["split"] == "train":
            train.append(groups)
    if not train or not any(case["split"] == "heldout" for case in manifest["cases"]):
        raise ValueError("separate TRAIN and HELDOUT traces required")
    allocations = [plan(train, sizes, budget, policy) for budget in manifest["budgets_bytes"]
                   for policy in ("uniform_frequency", "global_frequency_per_byte", "request_bundle")]
    # Freeze each plan before scoring held-out data.
    for allocation in allocations:
        allocation["scores"] = {name: score(groups, allocation, sizes) for name, groups in cases.items()}
    return {"schema_version": 1, "manifest_sha256": sha(manifest_path), "inputs": manifest, "allocations": allocations,
            "limits": "Static logical payload coverage only; no runtime cache, measured transfers, latency or speedup. Bundle greedy is not optimal. All-hit is per token/layer; all-layer coverage includes only selected layers. Payload excludes allocator metadata, in-flight slots and reserved memory."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    try:
        result = run(args.manifest)
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
    except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
