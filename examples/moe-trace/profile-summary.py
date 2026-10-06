#!/usr/bin/env python3
"""Summarize opt-in scheduler host scopes and requested copy payloads."""

import argparse
import csv
import json
from collections import defaultdict


FIELDS = ["scheduler", "call", "split", "event", "source", "destination", "tensor",
          "start_us", "duration_us", "bytes", "padding_bytes", "buffer_bytes", "status"]
EVENTS = {"compute_splits", "split", "split_wait", "input_wait", "event_wait_enqueue", "source_wait",
          "router_readback", "routing_scan", "expert_upload", "tensor_copy_async", "copy_wait", "tensor_copy",
          "compute_call", "callback_wait", "scheduler_wait", "graph_allocate", "scheduler_end"}


def summarize(path):
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
    if not rows or not sum(calls.values()) or schedulers != ended:
        raise ValueError("empty/incomplete profile; every scheduler needs a scheduler_end record and some compute work")
    return {
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile")
    parser.add_argument("--json", required=True, dest="json_path")
    args = parser.parse_args()
    try:
        result = summarize(args.profile)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    with open(args.json_path, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
