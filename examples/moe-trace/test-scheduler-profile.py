#!/usr/bin/env python3
"""Verify scheduler counters and profiler OFF/ON correctness on small graphs."""

import argparse
import csv
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("profile_summary", HERE / "profile-summary.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
route_spec = importlib.util.spec_from_file_location("route_summary", HERE / "route-summary.py")
route_summary = importlib.util.module_from_spec(route_spec)
route_spec.loader.exec_module(route_summary)
CHECK = None
VULKAN = False


class SummaryTests(unittest.TestCase):
    def phases(self, rows, finish=True):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "phases.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=summary.PHASE_FIELDS)
            writer.writeheader()
            for updates in rows:
                row = dict(rep=0, phase="prefill", position=0, n_tokens=1, start_us=1000, end_us=1500, status=0)
                row.update(updates)
                writer.writerow(row)
            if finish:
                writer.writerow(dict(rep=-1, phase="profile_end", position=0, n_tokens=0, start_us=0, end_us=0, status=0))
        return path

    def profile(self, rows):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "profile.csv"
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=summary.FIELDS)
            writer.writeheader()
            for updates in rows:
                row = dict.fromkeys(summary.FIELDS, 0)
                row.update(scheduler=1, call=1, split=-1, event="compute_splits", source="", destination="", tensor="")
                row.update(updates)
                writer.writerow(row)
        return path

    def test_keep_envelopes_separate_and_parse_quotes(self):
        path = self.profile([{"event": "expert_upload", "split": 0, "destination": "Vulkan0", "tensor": 'blk.0,"expert"',
                              "duration_us": 3, "bytes": 100, "padding_bytes": 20},
                             {"event": "split", "duration_us": 10, "buffer_bytes": 64},
                             {"duration_us": 12}, {"event": "scheduler_end"}])
        report = summary.summarize(path)
        self.assertEqual(report["compute_splits_inclusive_us"], 12)
        upload = next(entry for entry in report["events"] if entry["event"] == "expert_upload")
        self.assertEqual((upload["host_duration_us"], upload["bytes"], upload["padding_bytes"]), (3, 100, 20))
        self.assertEqual(upload["tensor"], 'blk.0,"expert"')

    def test_correlate_vulkan_queries_with_exact_call_and_phase(self):
        profile = self.profile([{"start_us": 100, "duration_us": 300},
                                {"event": "compute_call", "destination": "Vulkan0", "start_us": 120, "duration_us": 100},
                                {"event": "scheduler_end"}])
        phases = self.phases([dict(start_us=100, end_us=500)])
        log = profile.with_name("vulkan.log")
        content = ("Vulkan graph: backend=Vulkan0,start_us=130,end_us=200,queries=2,concurrent=0\n"
                   "----------------\nVulkan Timings:\n"
                   "MUL_MAT_ID q4_K: 1 x 12.5 us = 12.5 us (10 GFLOPS/s)\n"
                   "RMS_NORM(8,1,1,1): 1 x 1.5 us = 1.5 us\nTotal time: 14 us.\n")
        log.write_text(content)
        report = summary.summarize(profile, phases, log)
        self.assertEqual(report["schema_version"], 3)
        graph = report["gpu_graphs"][0]
        self.assertEqual((graph["scheduler"], graph["call"], graph["rep"], graph["phase"]), (1, 1, 0, "prefill"))
        self.assertEqual(graph["gpu_timestamp_ms"], 0.014)
        self.assertEqual(sum(op["count"] for op in report["gpu_phase_operations"]), 2)
        for invalid in (content.replace("queries=2", "queries=3"), content.replace("14 us.", "15 us."),
                        content.replace("end_us=200", "end_us=221"), content.replace("concurrent=0", "concurrent=1"),
                        content[:content.index("Total time:")], content + content):
            log.write_text(invalid)
            with self.assertRaises(ValueError):
                summary.summarize(profile, phases, log)
        with self.assertRaises(ValueError):
            summary.summarize(profile, vulkan_path=log)

    def test_completed_routes_and_conflicting_or_missing_observations(self):
        phases = self.phases([dict(start_us=100, end_us=500)])
        path = phases.with_name("routes.csv")
        valid = ("scheduler,call,event,tensor,n_expert,n_tokens,token,rank,expert,host_us\n"
                 "1,1,router,blk.0.ffn_up_exps.weight,8,1,0,0,7,200\n"
                 "1,1,router,blk.0.ffn_up_exps.weight,8,1,0,1,7,200\n"
                 "1,1,router,blk.0.ffn_down_exps.weight,8,1,0,0,7,250\n"
                 "1,1,router,blk.0.ffn_down_exps.weight,8,1,0,1,7,250\n"
                 "1,1,scheduler_end,,0,0,-1,-1,-1,0\n")
        path.write_text(valid)
        rows, report = route_summary.convert(path, phases, {0}, 2, 8)
        self.assertEqual(rows, [("prefill", 0, 0, 0, 7), ("prefill", 0, 0, 1, 7)])
        self.assertEqual(report["observations"], 4)
        for invalid in (valid.replace("0,0,7,250", "0,0,6,250"), valid.replace("0,1,7,200", "0,2,7,200"),
                        valid.replace(",200\n", ",600\n"), valid[:valid.index("1,1,scheduler_end")],
                        valid.replace("8,1,0", "8,2,0"), valid + valid.splitlines()[-1] + "\n"):
            path.write_text(invalid)
            with self.assertRaises(ValueError):
                route_summary.convert(path, phases, {0}, 2, 8)

    def test_reject_invalid_or_incomplete_profiles(self):
        for rows in ([{}], [{}, {"event": "scheduler_end"}, {}], [{}, {}, {"event": "scheduler_end"}],
                     [{"duration_us": -1}], [{"status": -2}], [{"bytes": 1, "padding_bytes": 2}],
                     [{"event": "unknown"}], [{"event": "scheduler_end"}]):
            with self.assertRaises(ValueError):
                summary.summarize(self.profile(rows))

    def test_explicit_phases_warmup_and_milliseconds(self):
        phases = self.phases([dict(rep=-1, start_us=300, end_us=500), {},
                              dict(phase="decode", position=1, start_us=1700, end_us=1900)])
        profile = self.profile([dict(call=0, event="scheduler_wait", start_us=20, duration_us=5),
                                dict(call=1, start_us=450, duration_us=40),
                                dict(call=2, event="expert_upload", start_us=1105, duration_us=20, bytes=80, padding_bytes=16),
                                dict(call=2, start_us=1100, duration_us=50),
                                dict(call=3, start_us=1750, duration_us=50),
                                dict(call=3, event="scheduler_end")])
        report = summary.summarize(profile, phases)
        self.assertEqual([(entry["rep"], entry["phase"], entry["elapsed_ms"], entry["compute_calls"])
                          for entry in report["phase_summary"]], [(-1, "prefill", 0.2, 1), (0, "decode", 0.2, 1), (0, "prefill", 0.5, 1)])
        prefill = next(entry for entry in report["phase_summary"] if entry["rep"] == 0 and entry["phase"] == "prefill")
        upload = next(entry for entry in prefill["events"] if entry["event"] == "expert_upload")
        self.assertEqual((upload["host_duration_ms"], upload["bytes"], upload["padding_bytes"]), (0.02, 80, 16))
        self.assertEqual(report["phase_evaluations"][2]["scheduler_calls"], [{"scheduler": 1, "call": 3}])
        self.assertEqual(report["unattributed_events"][0]["host_duration_us"], 5)

    def test_reject_invalid_phase_data_and_partial_scope_matches(self):
        for rows in ([dict(phase="decode")], [dict(status=1)], [dict(end_us=999)], [dict(n_tokens=0)],
                     [{}, dict(start_us=1200)], [{}, dict(rep=2, start_us=1600)],
                     [{}, dict(phase="decode", position=2, start_us=1600)],
                     [{}, dict(phase="profile_end"), {}]):
            with self.assertRaises(ValueError):
                summary.read_phases(self.phases(rows))
        with self.assertRaises(ValueError):
            summary.read_phases(self.phases([{}], finish=False))
        for start, duration in ((900, 200), (1490, 20), (2000, 10)):
            profile = self.profile([dict(start_us=start, duration_us=duration), dict(event="scheduler_end")])
            with self.assertRaises(ValueError):
                summary.summarize(profile, self.phases([{}]))
        phases = summary.read_phases(self.phases([{}, dict(phase="decode", position=1, start_us=1500, end_us=1800)]))
        with self.assertRaises(ValueError):
            summary.phase_for_scope(phases, [1000, 1500], 1500, 0)


class RuntimeTests(unittest.TestCase):
    def test_output_and_exact_copy_counters(self):
        if CHECK is None:
            self.skipTest("provide --check for compiled scheduler graphs")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profile = root / "profile.csv"
            argv = [str(CHECK)] + (["--vulkan"] if VULKAN else [])
            environment = dict(os.environ)
            environment.pop("GGML_SCHED_PROFILE", None)
            environment.pop("MOE_REPLAY_PROFILE", None)
            subprocess.run(argv + ["--output-bin", str(root / "off.bin")], env=environment, capture_output=True, check=True)
            self.assertFalse(profile.exists())
            environment["GGML_SCHED_PROFILE"] = str(profile)
            subprocess.run(argv + ["--output-bin", str(root / "on.bin")], env=environment, capture_output=True, check=True)
            self.assertEqual((root / "off.bin").read_bytes(), (root / "on.bin").read_bytes())
            route_environment = dict(environment)
            route_environment.pop("GGML_SCHED_PROFILE", None)
            route_environment.pop("MOE_REPLAY_PROFILE", None)
            route_environment["GGML_SCHED_ROUTE_PROFILE"] = str(root / "routes.csv")
            subprocess.run(argv + ["--output-bin", str(root / "routes.bin")], env=route_environment, capture_output=True, check=True)
            self.assertEqual((root / "off.bin").read_bytes(), (root / "routes.bin").read_bytes())
            with (root / "routes.csv").open() as stream:
                route_rows = list(csv.DictReader(stream))
            self.assertEqual({int(row["scheduler"]) for row in route_rows if row["event"] == "scheduler_end"}, set(range(1, 17)))
            expected_routes = {}
            selection = ((0, 1), (1, 4), (4, 7))
            for scheduler in range(1, 17):
                case = (scheduler - 1) % 8
                if VULKAN and case in (2, 6):
                    continue  # Full-copy fallback has no naturally host-read IDs.
                for call in (1, 2):
                    for token in range(1 if case < 4 else 3):
                        for rank in range(2):
                            expected_routes[(scheduler, call, token, rank)] = 7 if call == 2 else selection[token][rank]
            actual_routes = {}
            for row in route_rows:
                if row["event"] == "router":
                    self.assertEqual(int(row["n_expert"]), 8)
                    key = tuple(int(row[k]) for k in ("scheduler", "call", "token", "rank"))
                    self.assertNotIn(key, actual_routes)
                    actual_routes[key] = int(row["expert"])
            self.assertEqual(actual_routes, expected_routes)
            for path in (str(root / "missing" / "routes.csv"), "/dev/full" if Path("/dev/full").exists() else ""):
                route_environment["GGML_SCHED_ROUTE_PROFILE"] = path
                result = subprocess.run(argv + ["--output-bin", str(root / "failed-routes.bin")], env=route_environment, capture_output=True, check=True)
                if path:
                    self.assertIn(b"route profile", result.stderr)
                self.assertEqual((root / "off.bin").read_bytes(), (root / "failed-routes.bin").read_bytes())
            route_environment["GGML_SCHED_ROUTE_PROFILE"] = ""
            subprocess.run(argv + ["--output-bin", str(root / "empty-routes.bin")], env=route_environment, capture_output=True, check=True)
            self.assertEqual((root / "off.bin").read_bytes(), (root / "empty-routes.bin").read_bytes())
            report = summary.summarize(profile)
            self.assertEqual((report["schedulers"], report["compute_calls"]), (16, 32))
            with profile.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            uploads = [row for row in rows if row["event"] == "expert_upload"]
            if VULKAN:
                self.assertTrue(any(row["event"] == "compute_call" and row["destination"] == "Vulkan0" for row in rows))
                self.assertTrue(any(row["event"] == "event_wait_enqueue" for row in rows))
                self.assertTrue(any('ids,"strided"' in row["tensor"] for row in rows))
                for scheduler in range(1, 17):
                    for call in (1, 2):
                        selected = [row for row in uploads if int(row["scheduler"]) == scheduler and int(row["call"]) == call]
                        case = (scheduler - 1) % 8
                        small = case < 4
                        fallback = case in (2, 6)
                        expected = (0, 0, 0) if fallback else ((1024, 0, 1) if call == 2 else ((2560, 512, 1) if small else (5120, 1024, 3)))
                        actual = (sum(int(row["bytes"]) for row in selected), sum(int(row["padding_bytes"]) for row in selected), len(selected))
                        self.assertEqual(actual, expected, (scheduler, call))
                        if fallback:
                            copies = [row for row in rows if int(row["scheduler"]) == scheduler and int(row["call"]) == call
                                      and row["tensor"] == "blk.0.ffn_up_exps.weight" and row["event"] in ("tensor_copy", "tensor_copy_async")]
                            self.assertEqual(sum(int(row["bytes"]) for row in copies), 8192)
                        else:
                            readback = [row for row in rows if int(row["scheduler"]) == scheduler and int(row["call"]) == call and row["event"] == "router_readback"]
                            expected_span = 8 if small else (136 if case == 7 else 24)
                            self.assertEqual(sum(int(row["bytes"]) for row in readback), expected_span)
            else:
                self.assertFalse(uploads)
            environment["MOE_REPLAY_PROFILE"] = str(root / "phases.csv")
            environment["GGML_SCHED_PROFILE"] = str(root / "paired.csv")
            paired_process = subprocess.run(argv + ["--output-bin", str(root / "paired.bin")], env=environment, capture_output=True, check=True)
            self.assertEqual((root / "off.bin").read_bytes(), (root / "paired.bin").read_bytes())
            paired = summary.summarize(root / "paired.csv", root / "phases.csv")
            self.assertEqual(len(paired["phase_evaluations"]), 32)
            self.assertEqual(paired["unattributed_compute_calls"], 0)
            self.assertEqual(sum(entry["compute_calls"] for entry in paired["phase_summary"]), 32)
            for index, evaluation in enumerate(paired["phase_evaluations"]):
                self.assertEqual(evaluation["phase"], "prefill" if index % 2 == 0 else "decode")
                self.assertEqual(evaluation["scheduler_calls"], [{"scheduler": index // 2 + 1, "call": index % 2 + 1}])
            if VULKAN:
                uploads_by_phase = {phase: sum(event["bytes"] for entry in paired["phase_summary"] if entry["phase"] == phase
                                              for event in entry["events"] if event["event"] == "expert_upload")
                                    for phase in ("prefill", "decode")}
                self.assertEqual(uploads_by_phase, {"prefill": 46080, "decode": 12288})
                if "GGML_VK_PERF_LOGGER" in environment:
                    log = root / "vulkan.log"
                    log.write_bytes(paired_process.stderr)
                    gpu = summary.summarize(root / "paired.csv", root / "phases.csv", log)
                    self.assertEqual(len(gpu["gpu_graphs"]), 32)
                    for graph in gpu["gpu_graphs"]:
                        self.assertEqual(graph["phase"], "prefill" if graph["call"] == 1 else "decode")
            # A phase file is single-run output; refuse to overwrite a previous run.
            result = subprocess.run(argv, env=environment, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b"cannot create fresh replay phase profile", result.stderr)
            environment.pop("MOE_REPLAY_PROFILE")
            if sys.platform == "linux":
                import resource
                import signal

                def fail_file_writes():
                    signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
                    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))

                failed_environment = dict(environment)
                failed_environment.pop("GGML_SCHED_PROFILE")
                failed_environment["MOE_REPLAY_PROFILE"] = str(root / "failed-phases.csv")
                result = subprocess.run(argv, env=failed_environment, capture_output=True, preexec_fn=fail_file_writes)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(b"replay phase profile write failed", result.stderr)
                with self.assertRaises(ValueError):
                    summary.read_phases(root / "failed-phases.csv")
            environment["GGML_SCHED_PROFILE"] = str(root / "missing" / "profile.csv")
            result = subprocess.run(argv + ["--output-bin", str(root / "invalid-path.bin")], env=environment, capture_output=True, check=True)
            self.assertIn(b"cannot open scheduler profile", result.stderr)
            self.assertEqual((root / "off.bin").read_bytes(), (root / "invalid-path.bin").read_bytes())
            environment["GGML_SCHED_PROFILE"] = ""
            environment["MOE_REPLAY_PROFILE"] = ""
            subprocess.run(argv + ["--output-bin", str(root / "empty-env.bin")], env=environment, capture_output=True, check=True)
            self.assertEqual((root / "off.bin").read_bytes(), (root / "empty-env.bin").read_bytes())
            if Path("/dev/full").exists():
                environment["GGML_SCHED_PROFILE"] = "/dev/full"
                result = subprocess.run(argv + ["--output-bin", str(root / "write-failure.bin")], env=environment, capture_output=True, check=True)
                self.assertIn(b"scheduler profile write failed", result.stderr)
                self.assertEqual((root / "off.bin").read_bytes(), (root / "write-failure.bin").read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=Path)
    parser.add_argument("--vulkan", action="store_true")
    args, remaining = parser.parse_known_args()
    CHECK = args.check.resolve() if args.check else None
    VULKAN = args.vulkan
    unittest.main(argv=[__file__, *remaining])
