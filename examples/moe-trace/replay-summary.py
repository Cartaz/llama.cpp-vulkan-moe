#!/usr/bin/env python3
"""Validate replay calls and aggregate every prefill chunk before computing PP rate."""

import argparse
import csv
import json
import re
from pathlib import Path

TOKEN_FIELDS = ["phase", "token", "id"]
CALL_FIELDS = ["rep", "phase", "position", "n_tokens", "elapsed_us", "logits_hash"]


def read_tokens(path):
    prompt, decode = 0, 0
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != TOKEN_FIELDS:
            raise ValueError("unexpected token header")
        for position, row in enumerate(reader):
            if None in row or any(value is None for value in row.values()):
                raise ValueError("invalid token columns")
            if int(row["token"]) != position or int(row["id"]) < 0:
                raise ValueError("invalid token position or ID")
            if row["phase"] == "prefill" and not decode:
                prompt += 1
            elif row["phase"] == "decode" and prompt:
                decode += 1
            else:
                raise ValueError("invalid token phase order")
    if not prompt or not decode:
        raise ValueError("replay needs prefill and decode tokens")
    return prompt, decode


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    index = int(position)
    return ordered[index] + (ordered[min(index + 1, len(ordered) - 1)] - ordered[index]) * (position - index)


def read_calls(path):
    calls = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != CALL_FIELDS:
            raise ValueError("unexpected replay header")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("invalid replay columns")
            call = {key: int(row[key]) for key in ["rep", "position", "n_tokens", "elapsed_us"]}
            if min(call.values()) < 0 or call["n_tokens"] == 0 or call["elapsed_us"] == 0:
                raise ValueError("invalid replay counters")
            if not re.fullmatch(r"[0-9a-f]{16}", row["logits_hash"]):
                raise ValueError("invalid logits hash")
            calls.append(call | {"phase": row["phase"], "logits_hash": row["logits_hash"]})
    if not calls:
        raise ValueError("empty replay")
    return calls


def summarize(path, tokens, step, reference=None):
    if step < 1:
        raise ValueError("step must be positive")
    prompt, decode = read_tokens(tokens)
    calls = read_calls(path)
    reference_hashes = None
    if reference:
        expected = read_calls(reference)
        if any(row["rep"] != 0 for row in expected):
            raise ValueError("reference must contain one repetition")
        reference_hashes = {(row["phase"], row["position"], row["n_tokens"]): row["logits_hash"] for row in expected}
        if len(reference_hashes) != len(expected):
            raise ValueError("duplicate reference calls")
    repetitions = []
    index = 0
    while index < len(calls):
        rep = len(repetitions)
        prefill, generation = [], []
        position = 0
        while position < prompt:
            row = calls[index] if index < len(calls) else None
            count = min(step, prompt - position)
            if row is None or (row["rep"], row["phase"], row["position"], row["n_tokens"]) != (rep, "prefill", position, count):
                raise ValueError("incomplete or invalid prefill chunks")
            prefill.append(row)
            position += count
            index += 1
        for token in range(decode):
            row = calls[index] if index < len(calls) else None
            if row is None or (row["rep"], row["phase"], row["position"], row["n_tokens"]) != (rep, "decode", prompt + token, 1):
                raise ValueError("incomplete or invalid decode calls")
            generation.append(row)
            index += 1
        if reference_hashes is not None:
            keys = {(row["phase"], row["position"], row["n_tokens"]) for row in prefill + generation}
            if keys != set(reference_hashes) or any(row["logits_hash"] != reference_hashes[(row["phase"], row["position"], row["n_tokens"])] for row in prefill + generation):
                raise ValueError("per-call logits mismatch")
        pp_us = sum(row["elapsed_us"] for row in prefill)
        tg_us = sum(row["elapsed_us"] for row in generation)
        repetitions.append({"rep": rep, "prefill_calls": len(prefill), "decode_calls": len(generation),
                            "pp_elapsed_ms": pp_us / 1000, "tg_elapsed_ms": tg_us / 1000,
                            "pp_tps": prompt * 1e6 / pp_us, "tg_tps": decode * 1e6 / tg_us,
                            "tg_p50_ms": percentile([row["elapsed_us"] for row in generation], 0.5) / 1000,
                            "tg_p95_ms": percentile([row["elapsed_us"] for row in generation], 0.95) / 1000,
                            "first_pp_chunk_ms": prefill[0]["elapsed_us"] / 1000,
                            "last_pp_chunk_ms": prefill[-1]["elapsed_us"] / 1000})
    return {"schema_version": 1, "prefill_tokens": prompt, "decode_tokens": decode, "step": step,
            "calls": len(calls), "repetitions": repetitions,
            "timing_scope": "decode plus synchronize; model load, inter-call hashes and raw I/O excluded; not application TTFT"}


def check_logits(path, calls, vocab):
    if vocab < 1:
        raise ValueError("vocabulary must be positive")
    import hashlib
    import numpy as np
    source = Path(path)
    if source.stat().st_size != calls * vocab * 4:
        raise ValueError("unexpected raw logits size")
    values = np.memmap(source, dtype=np.float32, mode="r").reshape(calls, vocab)
    for vector in values:
        if not np.isfinite(vector).all() or not np.any(vector != 0):
            raise ValueError("nonfinite or all-zero logits vector")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for data in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            digest.update(data)
    return {"bytes": source.stat().st_size, "vectors": calls, "vocab": vocab, "finite_nonzero_each_vector": True, "sha256": digest.hexdigest()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("replay", type=Path)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--logits", type=Path)
    parser.add_argument("--vocab", type=int, default=248320)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = summarize(args.replay, args.tokens, args.step, args.reference)
        if args.logits:
            result["raw"] = check_logits(args.logits, result["calls"], args.vocab)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.error(str(error))
