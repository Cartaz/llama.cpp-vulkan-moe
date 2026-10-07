#!/usr/bin/env python3
"""Check offline profiles and manifest capture without loading a model."""

import csv
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


analyze = module("analyze")
manifest = module("manifest")
layer_budget = module("layer-budget")


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def run_trace(self, groups, *args, sizes=None, valid=True):
        trace = self.root / "trace.csv"
        with trace.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["phase", "token", "layer", "rank", "expert"])
            for phase, token, layer, experts in groups:
                for rank, expert in enumerate(experts):
                    writer.writerow([phase, token, layer, rank, expert])
        argv = [sys.executable, str(HERE / "analyze.py"), str(trace), "--json", str(self.root / "report.json"), *args]
        if sizes is not None:
            size_file = self.root / "sizes.json"
            size_file.write_text(json.dumps(sizes))
            argv.extend(["--expert-bytes", str(size_file)])
        result = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0 if valid else 2, result.stderr)
        if valid:
            return json.loads((self.root / "report.json").read_text())
        self.assertNotIn("Traceback", result.stderr)

    def test_legacy_counts_and_duplicate_experts(self):
        groups = [("prefill", 0, 0, [1, 2]), ("decode", 1, 0, [1, 3]),
                  ("decode", 2, 0, [3, 4]), ("decode", 3, 0, [4, 4])]
        for phase, expected in {"prefill": (2, [0, 0, 0]), "decode": (5, [0, 2, 2]), "all": (7, [0, 3, 3])}.items():
            layer = self.run_trace(groups, "--phase", phase, "--slots", "0,2,8", "--window", "2")["layers"][0]
            self.assertEqual((layer["activations"], [entry["hits"] for entry in layer["cache"]]), expected)
        layer = self.run_trace(groups, "--slots", "2", "--prefill-policies")["layers"][0]
        self.assertEqual([entry["hits"] for entry in layer["prefill_cache"]], [3, 1])

    def test_atomic_reuse_and_entropy(self):
        groups = [("decode", token, 0, experts) for token, experts in enumerate(([0, 1], [1, 2], [0, 2]))]
        layer = self.run_trace(groups, "--windows", "1,2,4")["layers"][0]
        self.assertEqual(layer["cold_activations"], 3)
        self.assertEqual(layer["reuse_token_gap_histogram"], {"0": 2, "1": 1})
        self.assertEqual(layer["reuse_distinct_intervening_histogram"], {"0": 2, "2": 1})
        self.assertAlmostEqual(layer["entropy_bits"], math.log2(3))
        self.assertAlmostEqual(layer["entropy_observed_normalized"], 1.0)
        windows = layer["working_set_windows"]
        self.assertEqual([entry["unique_mean"] for entry in windows], [2.0, 3.0, None])
        self.assertEqual([entry["full_windows"] for entry in windows], [3, 2, 0])

    def test_one_expert_and_sparse_token_positions(self):
        layer = self.run_trace([("decode", 10, 0, [5]), ("decode", 20, 0, [5])])["layers"][0]
        self.assertEqual(layer["entropy_bits"], 0)
        self.assertEqual(layer["entropy_observed_normalized"], 0)
        self.assertEqual(layer["reuse_token_gap_histogram"], {"0": 1})

    def test_layer_major_groups_and_byte_weighting(self):
        groups = [("decode", 0, 0, [0]), ("decode", 1, 0, [0]),
                  ("decode", 0, 1, [0]), ("decode", 1, 1, [1])]
        report = self.run_trace(groups, "--slots", "0,1", sizes={"0": 10, "1": 90})
        entry = report["cache_summary"][1]
        self.assertEqual((entry["activations"], entry["hits"]), (4, 1))
        self.assertEqual((entry["capacity_bytes"], entry["requested_bytes"], entry["hit_bytes"], entry["miss_bytes"]), (100, 200, 10, 190))
        self.assertAlmostEqual(entry["byte_hit_rate"], 0.05)

    def test_whole_request_coverage_and_layer_budget(self):
        groups = [("decode", token, 0, experts) for token, experts in enumerate(([0, 1], [1, 2], [2, 3], [2, 3]))]
        groups += [("decode", 0, 1, [9])]
        report = self.run_trace(groups, "--layers", "0", "--slots", "0,1,2", sizes={"0": 64, "1": 1024})
        self.assertEqual(report["selected_layers"], [0])
        entries = report["cache_summary"]
        self.assertEqual([entry["capacity_bytes"] for entry in entries], [0, 64, 128])
        self.assertEqual(entries[0]["whole_request"], dict(requests=4, all_hit=0, all_miss=4, mixed=0, exceeds_capacity=4, all_hit_rate=0))
        self.assertEqual(entries[2]["whole_request"], dict(requests=4, all_hit=1, all_miss=1, mixed=2, exceeds_capacity=0, all_hit_rate=0.25))
        self.assertEqual(entries[2]["hits"], 4)
        self.assertEqual(entries[1]["whole_request"]["exceeds_capacity"], 4)
        self.run_trace(groups, "--layers", "2", valid=False)

    def test_warm_and_static_byte_summaries(self):
        groups = [("prefill", 0, 0, [1, 2]), ("decode", 1, 0, [1, 3]), ("decode", 2, 0, [3, 4])]
        entries = self.run_trace(groups, "--prefill-policies", "--slots", "2", sizes={"0": 64})["cache_summary"]
        self.assertEqual([entry["hit_bytes"] for entry in entries], [64, 128, 64])
        self.assertEqual([entry["capacity_bytes"] for entry in entries], [128, 128, 128])

    def test_invalid_sizes_and_parameters(self):
        groups = [("decode", 0, 0, [0])]
        for args in (("--slots", "abc"), ("--slots", "-1"), ("--windows", "0"), ("--window", "0")):
            self.run_trace(groups, *args, valid=False)
        for sizes in ({}, {"0": True}, {"0": 0}, {"0": 1.5}, {"-1": 10}, []):
            self.run_trace(groups, sizes=sizes, valid=False)

    def test_reject_replayed_groups_and_bad_ids(self):
        for groups in ([ ("decode", 1, 0, [0]), ("decode", 0, 0, [1]) ],
                       [ ("decode", 0, 0, [0]), ("decode", 1, 0, [1]), ("decode", 0, 0, [2]) ],
                       [ ("other", 0, 0, [0]) ], [ ("decode", 0, 0, [-1]) ]):
            self.run_trace(groups, valid=False)

    def test_reject_malformed_columns_and_ranks(self):
        trace = self.root / "bad.csv"
        header = "phase,token,layer,rank,expert\n"
        for row in ("decode,0,0,1,0\n", "decode,0,0,0\n", "decode,0,0,0,0,extra\n"):
            trace.write_text(header + row)
            result = subprocess.run([sys.executable, str(HERE / "analyze.py"), str(trace)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn("Traceback", result.stderr)

    def test_reuse_against_independent_history(self):
        rng = random.Random(6800)
        history = [rng.sample(range(16), rng.randint(1, 8)) for _ in range(100)]
        profile = analyze.RoutingProfile([1, 8, 32, 128])
        expected_tokens, expected_distinct = {}, {}
        cold = 0
        for token, selected in enumerate(history):
            for expert in selected:
                previous = next((i for i in range(token - 1, -1, -1) if expert in history[i]), None)
                if previous is None:
                    cold += 1
                else:
                    gap = token - previous - 1
                    distinct = len(set().union(*history[previous + 1:token]))
                    expected_tokens[gap] = expected_tokens.get(gap, 0) + 1
                    expected_distinct[distinct] = expected_distinct.get(distinct, 0) + 1
            profile.access(selected)
        self.assertEqual(profile.cold, cold)
        self.assertEqual(profile.token_gaps, expected_tokens)
        self.assertEqual(profile.distinct_gaps, expected_distinct)


class ManifestTests(unittest.TestCase):
    def test_freeze_rejects_changed_libraries_and_present_zero(self):
        expected = {"binary": {"sha256": "binary"}, "environment": {},
                    "repository": {"commit": "commit", "tracked_diff": ""},
                    "libraries": [{"path": "/tmp/libggml.so", "sha256": "original"}]}
        manifest.verify_frozen(copy.deepcopy(expected), expected)
        for field, value in (("libraries", [{"path": "/tmp/libggml.so", "sha256": "changed"}]),
                             ("environment", {"GGML_VK_FA_Q8_SYNC": "0"}), ("binary", {"sha256": "changed"})):
            changed = copy.deepcopy(expected)
            changed[field] = value
            with self.assertRaises(ValueError):
                manifest.verify_frozen(changed, expected)

    def test_fingerprint_and_replaced_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact"
            path.write_bytes(b"moe\0artifact")
            record = manifest.fingerprint(path)
            self.assertEqual(record["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(record["bytes"], 12)
            with self.assertRaises(ValueError):
                manifest.fingerprint(path, (-1, -1))

    def test_mapped_libraries_reject_deleted_and_wrong_inode(self):
        with self.assertRaises(ValueError):
            manifest.mapped_libraries("0-1 r-xp 0 00:00 1 /tmp/libtest.so (deleted)")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "libtest.so"
            path.write_bytes(b"test")
            stat = path.stat()
            device = f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}"
            line = f"0-1 r-xp 0 {device} {stat.st_ino} {path}"
            self.assertEqual(len(manifest.mapped_libraries(line + "\n" + line)), 1)
            with self.assertRaises(ValueError):
                manifest.mapped_libraries(f"0-1 r-xp 0 {device} {stat.st_ino + 1} {path}")

    def test_live_snapshot_without_model(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "CMakeCache.txt"
            cache.write_text("CMAKE_BUILD_TYPE:STRING=Release\nGGML_CPU_MOE_COMPACT:BOOL=OFF\n")
            args = SimpleNamespace(pid=os.getpid(), repo=HERE.parent.parent, cmake_cache=cache,
                                   workload=[HERE / "analyze.py"], model=None, log=None, require_vulkan=False, strict_map_device=False)
            report = manifest.snapshot(args)
            self.assertEqual(report["build"]["GGML_CPU_MOE_COMPACT"], "OFF")
            self.assertEqual(report["pid"], os.getpid())
            self.assertEqual(report["binary"]["sha256"], manifest.fingerprint(Path("/proc/self/exe"))["sha256"])
            self.assertTrue(report["libraries"])
            self.assertNotIn("model", report)
            args.require_vulkan = True
            with self.assertRaisesRegex(ValueError, "no mapped libggml-vulkan"):
                manifest.snapshot(args)


class LayerBudgetTests(unittest.TestCase):
    def test_heterogeneous_cost_and_determinism(self):
        traces = [{(0, 0): frozenset([1]), (0, 1): frozenset([2]), (1, 0): frozenset([1]), (1, 1): frozenset([2])}]
        sizes = {0: 3, 1: 7}
        for policy in ("uniform_frequency", "global_frequency_per_byte", "request_bundle"):
            for budget in (0, 2, 3, 7, 9, 10):
                result = layer_budget.plan(traces, sizes, budget, policy)
                self.assertEqual(result, layer_budget.plan(traces, sizes, budget, policy))
                self.assertLessEqual(result["payload_bytes"], budget)
                self.assertEqual(result["payload_bytes"] + result["slack_bytes"], budget)
                self.assertEqual(result["payload_bytes"], sum(v["slots"] * sizes[int(k)] for k, v in result["layers"].items()))
        result = layer_budget.plan(traces, sizes, 9, "global_frequency_per_byte")
        self.assertEqual(result["layers"]["0"]["experts"], [1])
        self.assertEqual(result["zero_quota_layers"], [1])

    def test_indivisible_bundle_and_shared_gain(self):
        traces = [{(0, 0): frozenset([0, 1]), (1, 0): frozenset([0, 1]), (2, 0): frozenset([1, 2])}]
        self.assertEqual(layer_budget.plan(traces, {0: 4}, 7, "request_bundle")["payload_bytes"], 0)
        allocation = layer_budget.plan(traces, {0: 4}, 8, "request_bundle")
        self.assertEqual(allocation["layers"]["0"]["experts"], [0, 1])
        stats = layer_budget.score(traces[0], allocation, {0: 4})
        self.assertEqual((stats["all_hit"], stats["mixed"], stats["all_miss"]), (2, 1, 0))
        self.assertEqual(stats["hits"], 5)
        self.assertEqual(stats["all_layer_hit_tokens"], 2)

    def test_whole_token_and_partition(self):
        groups = {(0, 0): frozenset([1, 2]), (0, 1): frozenset([1, 2]), (1, 0): frozenset([3, 4]), (1, 1): frozenset([2, 3])}
        allocation = {"layers": {"0": {"experts": [1, 2]}, "1": {"experts": [1, 2]}}}
        stats = layer_budget.score(groups, allocation, {0: 2, 1: 5})
        self.assertEqual((stats["all_hit"], stats["all_miss"], stats["mixed"]), (2, 1, 1))
        self.assertEqual(stats["hit_bytes"], 19)
        self.assertEqual(stats["all_layer_hit_tokens"], 1)
        with self.assertRaises(ValueError):
            layer_budget.score({(0, 0): groups[0, 0]}, allocation, {0: 2, 1: 5})

    def test_manifest_geometry_identity_and_heldout_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sizes = root / "sizes.json"
            sizes.write_text('{"0": 4}')
            cases = []
            for name, split, ids in (("train", "train", [0, 1]), ("heldout", "heldout", [2, 3])):
                path = root / (name + ".csv")
                with path.open("w", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["phase", "token", "layer", "rank", "expert"])
                    for token, phase in ((0, "prefill"), (1, "decode")):
                        for rank, expert in enumerate(ids):
                            writer.writerow([phase, token, 0, rank, expert])
                cases.append(dict(name=name, split=split, path=path.name, sha256=layer_budget.sha(path), model_sha256="a" * 64, prompt_tokens=1, decode_tokens=1))
            data = dict(model_sha256="a" * 64, geometry=dict(layers=1, experts=4, top_k=2), selected_layers=[0],
                        expert_bytes_path=sizes.name, expert_bytes_sha256=layer_budget.sha(sizes), cases=cases, budgets_bytes=[8])
            path = root / "manifest.json"
            def run():
                path.write_text(json.dumps(data))
                return layer_budget.run(path)
            before = run()
            heldout = root / cases[1]["path"]
            heldout.write_text(heldout.read_text().replace(",2", ",0"))
            cases[1]["sha256"] = layer_budget.sha(heldout)
            after = run()
            self.assertEqual([{k: v for k, v in item.items() if k != "scores"} for item in before["allocations"]],
                             [{k: v for k, v in item.items() if k != "scores"} for item in after["allocations"]])
            self.assertNotEqual(before["allocations"][0]["scores"]["heldout"], after["allocations"][0]["scores"]["heldout"])
            cases[1]["model_sha256"] = "b" * 64
            with self.assertRaisesRegex(ValueError, "identity"):
                run()
            cases[1]["model_sha256"] = "a" * 64
            cases[1]["sha256"] = "c" * 64
            with self.assertRaisesRegex(ValueError, "SHA"):
                run()
            cases[1]["sha256"] = layer_budget.sha(heldout)
            heldout.write_text(heldout.read_text().replace("decode,1,0,1,3", "decode,1,0,1,0"))
            cases[1]["sha256"] = layer_budget.sha(heldout)
            with self.assertRaisesRegex(ValueError, "top-k"):
                run()


if __name__ == "__main__":
    unittest.main()
