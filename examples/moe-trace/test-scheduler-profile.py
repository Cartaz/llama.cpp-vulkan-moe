#!/usr/bin/env python3
"""Verify scheduler counters and profiler OFF/ON correctness on small graphs."""

import argparse
import csv
import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("profile_summary", HERE / "profile-summary.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)
CHECK = None
VULKAN = False


class SummaryTests(unittest.TestCase):
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

    def test_reject_invalid_or_incomplete_profiles(self):
        for rows in ([{}], [{}, {"event": "scheduler_end"}, {}], [{}, {}, {"event": "scheduler_end"}],
                     [{"duration_us": -1}], [{"status": -2}], [{"bytes": 1, "padding_bytes": 2}],
                     [{"event": "unknown"}], [{"event": "scheduler_end"}]):
            with self.assertRaises(ValueError):
                summary.summarize(self.profile(rows))


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
            subprocess.run(argv + ["--output-bin", str(root / "off.bin")], env=environment, capture_output=True, check=True)
            self.assertFalse(profile.exists())
            environment["GGML_SCHED_PROFILE"] = str(profile)
            subprocess.run(argv + ["--output-bin", str(root / "on.bin")], env=environment, capture_output=True, check=True)
            self.assertEqual((root / "off.bin").read_bytes(), (root / "on.bin").read_bytes())
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
            environment["GGML_SCHED_PROFILE"] = str(root / "missing" / "profile.csv")
            result = subprocess.run(argv + ["--output-bin", str(root / "invalid-path.bin")], env=environment, capture_output=True, check=True)
            self.assertIn(b"cannot open scheduler profile", result.stderr)
            self.assertEqual((root / "off.bin").read_bytes(), (root / "invalid-path.bin").read_bytes())
            environment["GGML_SCHED_PROFILE"] = ""
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
