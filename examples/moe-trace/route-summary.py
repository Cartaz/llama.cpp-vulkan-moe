#!/usr/bin/env python3
"""Convert completed scheduler routing observations to an explicit replay trace."""

import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path


spec = importlib.util.spec_from_file_location("profile_summary", Path(__file__).with_name("profile-summary.py"))
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)
FIELDS = ["scheduler", "call", "event", "tensor", "n_expert", "n_tokens", "token", "rank", "expert", "host_us"]


def convert(path, phase_path, layers, top_k=8, n_experts=256, rep=0):
    """Convert one repetition, or return rows by repetition when rep is None."""
    if not layers or min(layers) < 0 or top_k < 1 or n_experts < top_k or (rep is not None and rep < 0):
        raise ValueError("invalid layers, top-k, expert count or repetition")
    phases = profile.read_phases(phase_path)
    starts = [phase["start_us"] for phase in phases]
    groups, schedulers, ended, calls = {}, set(), set(), {}
    observations = 0
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("unexpected scheduler route header")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("invalid scheduler route columns")
            values = {key: int(row[key]) for key in FIELDS if key not in ("event", "tensor")}
            scheduler, call = values["scheduler"], values["call"]
            if scheduler < 1 or call < calls.get(scheduler, 0) or scheduler in ended:
                raise ValueError("invalid scheduler or reused/unordered route stream")
            schedulers.add(scheduler)
            calls[scheduler] = call
            if row["event"] == "scheduler_end":
                if row["tensor"] or any(values[k] != v for k, v in dict(n_expert=0, n_tokens=0, token=-1, rank=-1, expert=-1, host_us=0).items()):
                    raise ValueError("invalid route completion marker")
                ended.add(scheduler)
                continue
            match = re.match(r"blk\.(\d+)\.ffn_(?:gate|up|down)_exps\.weight(?:$|[.# (])", row["tensor"])
            if row["event"] != "router" or not match or call < 1:
                raise ValueError("invalid route event, expert tensor or call")
            layer = int(match[1])
            if values["n_expert"] != n_experts or not (0 <= values["expert"] < n_experts):
                raise ValueError("invalid expert count or ID")
            if values["n_tokens"] < 1 or not (0 <= values["token"] < values["n_tokens"]) or not (0 <= values["rank"] < top_k):
                raise ValueError("invalid route token count, position or rank")
            index = profile.phase_for_scope(phases, starts, values["host_us"], 0)
            if index is None:
                raise ValueError("routing observation is outside a phase")
            if layer not in layers:
                continue
            if phases[index]["n_tokens"] != values["n_tokens"]:
                raise ValueError("routing conversion requires one complete scheduler call per phase")
            group = groups.setdefault((index, scheduler, call, layer), {})
            key = values["token"], values["rank"]
            if key in group and group[key] != values["expert"]:
                raise ValueError("conflicting duplicate routing observations")
            group[key] = values["expert"]
            observations += 1
    if schedulers != ended or not schedulers or not groups:
        raise ValueError("empty/incomplete scheduler route profile")
    by_rep = {}
    phase_calls = []
    for index, phase in enumerate(phases):
        selected = {key: value for key, value in groups.items() if key[0] == index}
        pairs = {(key[1], key[2]) for key in selected}
        if len(pairs) != 1 or {key[3] for key in selected} != set(layers):
            raise ValueError("phase has missing layers or ambiguous scheduler calls")
        scheduler, call = next(iter(pairs))
        expected = {(token, rank) for token in range(phase["n_tokens"]) for rank in range(top_k)}
        if any(set(value) != expected for value in selected.values()):
            raise ValueError("routing phase has missing tokens or ranks")
        phase_calls.append(dict(rep=phase["rep"], phase=phase["phase"], position=phase["position"], scheduler=scheduler, call=call))
        if rep is None or phase["rep"] == rep:
            for token in range(phase["n_tokens"]):
                for layer in sorted(layers):
                    for rank in range(top_k):
                        by_rep.setdefault(phase["rep"], []).append((phase["phase"], phase["position"] + token, layer, rank, selected[(index, scheduler, call, layer)][(token, rank)]))
    rows = by_rep if rep is None else by_rep.get(rep, [])
    if not rows:
        raise ValueError("selected repetition has no routing observations")
    return rows, dict(schema_version=1, route_profile=str(path), phase_profile=str(phase_path), rep=rep,
                     layers=sorted(layers), top_k=top_k, n_experts=n_experts, observations=observations,
                     output_rows=sum(map(len, by_rep.values())), phase_calls=phase_calls,
                     limits="Only naturally CPU-visible IDs are observed. Duplicate triplet observations must agree. Conversion supports one complete scheduler call per phase and requires paired process provenance.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile")
    parser.add_argument("--phases", required=True)
    parser.add_argument("--layers", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--rep", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    try:
        rows, report = convert(args.profile, args.phases, set(map(int, args.layers.split(","))), args.top_k, args.experts, args.rep)
        with open(args.output, "x", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["phase", "token", "layer", "rank", "expert"])
            writer.writerows(rows)
        with open(args.json, "x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
