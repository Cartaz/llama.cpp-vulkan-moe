#!/usr/bin/env python3
"""Summarize opt-in scheduler host scopes and requested copy payloads."""

import argparse
import csv
import json
from bisect import bisect_right
from collections import defaultdict


FIELDS = ["scheduler", "call", "split", "event", "source", "destination", "tensor",
          "start_us", "duration_us", "bytes", "padding_bytes", "buffer_bytes", "status"]
EVENTS = {"compute_splits", "split", "split_wait", "input_wait", "event_wait_enqueue", "source_wait",
          "router_readback", "routing_scan", "expert_upload", "tensor_copy_async", "copy_wait", "tensor_copy",
          "compute_call", "callback_wait", "scheduler_wait", "graph_allocate", "scheduler_end"}
PHASE_FIELDS = ["rep", "phase", "position", "n_tokens", "start_us", "end_us", "status"]


def read_phases(path):
    phases = []
    finished = False
    previous_rep = -2
    position = 0
    decoding = False
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != PHASE_FIELDS:
            raise ValueError("unexpected phase profile header")
        for row in reader:
            if finished or None in row or any(value is None for value in row.values()):
                raise ValueError("invalid phase columns or rows after profile_end")
            phase = row["phase"]
            values = {key: int(row[key]) for key in PHASE_FIELDS if key != "phase"}
            if phase == "profile_end":
                if values != dict(rep=-1, position=0, n_tokens=0, start_us=0, end_us=0, status=0):
                    raise ValueError("invalid phase completion marker")
                finished = True
                continue
            rep, start, end = values["rep"], values["start_us"], values["end_us"]
            if phase not in ("prefill", "decode") or rep < -1 or values["status"] != 0:
                raise ValueError("invalid phase, repetition or failed evaluation")
            if start < 0 or end <= start or values["n_tokens"] < 1 or values["position"] < 0:
                raise ValueError("invalid phase interval or token count")
            if phases and start < phases[-1]["end_us"]:
                raise ValueError("overlapping or unordered phase intervals")
            if rep != previous_rep:
                if rep != previous_rep + 1 and not (not phases and rep == 0):
                    raise ValueError("repetition IDs must start at -1 or 0 and increase by one")
                previous_rep, position, decoding = rep, 0, False
            if values["position"] != position or (decoding and phase == "prefill"):
                raise ValueError("invalid phase order or token position")
            if position == 0 and phase != "prefill":
                raise ValueError("each repetition must start with prefill")
            decoding = phase == "decode"
            position += values["n_tokens"]
            phases.append({"phase": phase, **values, "scheduler_calls": []})
    if not finished or not phases:
        raise ValueError("empty or incomplete phase profile")
    return phases


def phase_for_scope(phases, starts, start, duration):
    end = start + duration
    index = bisect_right(starts, start) - 1
    if index >= 0 and end <= phases[index]["end_us"]:
        if duration == 0 and index > 0 and start == phases[index - 1]["end_us"]:
            raise ValueError("ambiguous zero-duration scope at a phase boundary")
        return index
    if (index >= 0 and start < phases[index]["end_us"]) or (index + 1 < len(phases) and end > starts[index + 1]):
        raise ValueError("scheduler scope crosses a phase boundary")
    return None


def summarize(path, phase_path=None):
    phases = read_phases(phase_path) if phase_path else []
    starts = [phase["start_us"] for phase in phases]
    phase_totals = defaultdict(lambda: defaultdict(lambda: {"count": 0, "host_duration_us": 0, "bytes": 0, "padding_bytes": 0}))
    phase_calls = defaultdict(int)
    totals = defaultdict(lambda: {"count": 0, "host_duration_us": 0, "bytes": 0, "padding_bytes": 0})
    buffers = defaultdict(int)
    calls = defaultdict(int)
    last_call = defaultdict(int)
    completed_call = defaultdict(int)
    ended = set()
    schedulers = set()
    envelope_us = 0
    rows = 0
    with open(path, newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != FIELDS:
            raise ValueError("unexpected scheduler profile header")
        for row in reader:
            rows += 1
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"line {reader.line_num}: expected {len(FIELDS)} columns")
            values = {key: int(row[key]) for key in ("scheduler", "call", "split", "start_us", "duration_us", "bytes", "padding_bytes", "buffer_bytes", "status")}
            scheduler, call = values["scheduler"], values["call"]
            event = row["event"]
            if event not in EVENTS or scheduler < 1 or values["split"] < -1:
                raise ValueError(f"line {reader.line_num}: invalid scheduler, split or event")
            if min(values[key] for key in ("call", "start_us", "duration_us", "bytes", "padding_bytes", "buffer_bytes")) < 0:
                raise ValueError(f"line {reader.line_num}: negative counter")
            if values["padding_bytes"] > values["bytes"] or values["status"] != 0:
                raise ValueError(f"line {reader.line_num}: invalid payload or failed scheduler operation")
            if scheduler in ended or call < last_call[scheduler]:
                raise ValueError("profile reuses a finished scheduler or decreases call IDs; use one fresh file per process")
            last_call[scheduler] = call
            schedulers.add(scheduler)
            if event == "scheduler_end":
                ended.add(scheduler)
                continue
            if event == "compute_splits":
                if call < 1 or call <= completed_call[scheduler]:
                    raise ValueError("duplicate or invalid compute_splits call")
                completed_call[scheduler] = call
                calls[scheduler] += 1
                envelope_us += values["duration_us"]
            if event == "split":
                buffers[(scheduler, row["destination"])] = max(buffers[(scheduler, row["destination"])], values["buffer_bytes"])
            key = (event, row["source"], row["destination"], row["tensor"])
            entry = totals[key]
            entry["count"] += 1
            entry["host_duration_us"] += values["duration_us"]
            entry["bytes"] += values["bytes"]
            entry["padding_bytes"] += values["padding_bytes"]
            if phases:
                index = phase_for_scope(phases, starts, values["start_us"], values["duration_us"])
                group = (phases[index]["rep"], phases[index]["phase"]) if index is not None else None
                entry = phase_totals[group][key]
                entry["count"] += 1
                entry["host_duration_us"] += values["duration_us"]
                entry["bytes"] += values["bytes"]
                entry["padding_bytes"] += values["padding_bytes"]
                if event == "compute_splits":
                    phase_calls[group] += 1
                    if index is not None:
                        phases[index]["scheduler_calls"].append({"scheduler": scheduler, "call": call})
    if not rows or not sum(calls.values()) or schedulers != ended:
        raise ValueError("empty/incomplete profile; every scheduler needs a scheduler_end record and some compute work")
    result = {
        "schema_version": 1, "profile": str(path), "schedulers": len(schedulers), "compute_calls": sum(calls.values()),
        "compute_splits_inclusive_us": envelope_us,
        "events": [{"event": key[0], "source": key[1], "destination": key[2], "tensor": key[3], **value}
                   for key, value in sorted(totals.items())],
        "scheduler_reserved_buffer_max": [{"scheduler": key[0], "backend": key[1], "bytes": value}
                                          for key, value in sorted(buffers.items())],
        "limits": ["Host scopes, not GPU active durations. Nested split/compute_splits scopes must not be added to leaf scopes.",
                   "Copy bytes are requested API payloads, not measured PCIe traffic; padding is included in bytes.",
                   "Router readback can originate on CPU; inspect the source backend.",
                   "Buffer maxima cover scheduler allocation only, not all weights/KV/recurrent/backend allocations.",
                   "No implicit PP/decode assignment, overlap estimate, tokens/s or output correctness verdict."]}
    if phases:
        if any(not phase["scheduler_calls"] for phase in phases):
            raise ValueError("phase evaluation has no complete scheduler compute call; check file pairing")
        grouped = defaultdict(lambda: {"eval_calls": 0, "n_tokens": 0, "elapsed_us": 0})
        for phase in phases:
            group = (phase["rep"], phase["phase"])
            grouped[group]["eval_calls"] += 1
            grouped[group]["n_tokens"] += phase["n_tokens"]
            grouped[group]["elapsed_us"] += phase["end_us"] - phase["start_us"]

        def events_ms(group):
            return [{"event": key[0], "source": key[1], "destination": key[2], "tensor": key[3],
                     **value, "host_duration_ms": value["host_duration_us"] / 1000}
                    for key, value in sorted(phase_totals[group].items())]

        result.update(schema_version=2, phase_profile=str(phase_path),
                      phase_evaluations=[{**phase, "elapsed_ms": (phase["end_us"] - phase["start_us"]) / 1000} for phase in phases],
                      phase_summary=[{"rep": group[0], "phase": group[1], **value, "elapsed_ms": value["elapsed_us"] / 1000,
                                      "compute_calls": phase_calls[group], "events": events_ms(group)}
                                     for group, value in sorted(grouped.items())],
                      unattributed_compute_calls=phase_calls[None], unattributed_events=events_ms(None))
        result["limits"].append("Phase correlation requires paired files from the same process and host clock; timestamps alone cannot authenticate file identity. Warmup is rep=-1. Out-of-window scopes remain unattributed.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile")
    parser.add_argument("--phases", help="paired MOE_REPLAY_PROFILE CSV from the same process")
    parser.add_argument("--json", required=True, dest="json_path")
    args = parser.parse_args()
    try:
        result = summarize(args.profile, args.phases)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    with open(args.json_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
