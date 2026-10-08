#!/usr/bin/env python3
"""Snapshot a live Linux benchmark process outside the inference timer."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ENV_PREFIXES = ("GGML_", "RADV_", "VK_", "MESA_", "MOE_", "LLAMA_MOE_", "OMP_")
ENV_KEYS = ("LD_LIBRARY_PATH", "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", "GGML_VK_FA_Q8_SYNC",
            "RADV_PERFTEST")


def fingerprint(path, expected_identity=None):
    path = Path(path)
    before = path.stat()
    if expected_identity is not None and (before.st_dev, before.st_ino) != expected_identity:
        raise ValueError(f"file replaced before hashing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"file changed while hashing: {path}")
    return {"path": str(path.resolve()), "bytes": after.st_size, "sha256": digest.hexdigest()}


def command(argv, strip=True):
    result = subprocess.run(argv, capture_output=True, text=True, timeout=10, check=True)
    return result.stdout.strip() if strip else result.stdout


def process_start(proc):
    # The command name in stat can contain spaces and parentheses.
    return (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]


def mapped_libraries(maps):
    libraries = {}
    for line in maps.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 6 or not fields[5].startswith("/") or ".so" not in Path(fields[5]).name:
            continue
        path = fields[5]
        if path in libraries:
            continue
        if path.endswith(" (deleted)"):
            raise ValueError(f"mapped library deleted: {path}")
        stat = Path(path).stat()
        major, minor = (int(value, 16) for value in fields[3].split(":"))
        if stat.st_ino != int(fields[4]):
            raise ValueError(f"mapped library replaced: {path}")
        libraries[path] = fingerprint(path, (stat.st_dev, stat.st_ino))
        libraries[path].update({"maps_inode": int(fields[4]), "maps_device": fields[3],
                                "stat_device": f"{os.major(stat.st_dev):02x}:{os.minor(stat.st_dev):02x}",
                                "device_matches": (os.major(stat.st_dev), os.minor(stat.st_dev)) == (major, minor)})
    return list(libraries.values())


def device_snapshot(card):
    device = card / "device"
    values = {}
    stable_names = ("vendor", "device", "subsystem_vendor", "subsystem_device", "mem_info_vram_total",
                    "mem_info_vis_vram_total", "max_link_speed", "max_link_width")
    for name in stable_names + ("mem_info_vram_used", "mem_info_vis_vram_used", "mem_info_gtt_used",
                                "gpu_busy_percent", "current_link_speed", "current_link_width"):
        path = device / name
        if path.exists():
            values[name] = path.read_text().strip()
    identity = {name: values[name] for name in stable_names if name in values}
    identity["device_address"] = device.resolve().name
    driver = device / "driver"
    if driver.exists():
        identity["driver"] = driver.resolve().name
    resource = device / "resource"
    if resource.exists():
        resources = []
        for index, line in enumerate(resource.read_text().splitlines()):
            start, end, flags = (int(value, 16) for value in line.split())
            if end < start:
                raise ValueError(f"invalid PCI resource range: {resource}")
            resources.append({"index": index, "bytes": end - start + 1 if flags else 0, "flags": flags})
        identity["pci_resources"] = resources
    return {"card": card.name, "values": values, "identity": identity}


def verify_frozen(report, expected):
    if not isinstance(expected, dict):
        raise ValueError("expected a manifest JSON object")
    fields = ("binary", "cmake_cache", "build", "environment", "environment_explicit_absent", "workloads", "model", "command", "affinity", "hardware")
    for field in fields:
        if report.get(field) != expected.get(field):
            raise ValueError(f"frozen manifest mismatch: {field}")
    for field in ("commit", "tracked_diff"):
        if report["repository"][field] != expected["repository"][field]:
            raise ValueError(f"frozen manifest mismatch: repository {field}")
    current_libraries = {entry["path"]: entry["sha256"] for entry in report["libraries"]}
    expected_libraries = {entry["path"]: entry["sha256"] for entry in expected["libraries"]}
    if current_libraries != expected_libraries:
        raise ValueError("frozen manifest mismatch: loaded libraries")


def snapshot(args):
    proc = Path("/proc") / str(args.pid)
    start = process_start(proc)
    environment = dict(entry.split("=", 1) for entry in (proc / "environ").read_bytes().decode().split("\0") if "=" in entry)
    maps = (proc / "maps").read_text()
    libraries = mapped_libraries(maps)
    repo = str(Path(args.repo).resolve())
    diff = command(["git", "-C", repo, "diff", "HEAD", "--binary"], strip=False)
    cache_path = Path(args.cmake_cache)
    build = {}
    for line in cache_path.read_text().splitlines():
        if "=" not in line or ":" not in line or line.startswith(("#", "//")):
            continue
        key, value = line.split("=", 1)
        key = key.split(":", 1)[0]
        if key.startswith(("GGML_", "LLAMA_", "CMAKE_C_", "CMAKE_CXX_")) or key in ("CMAKE_BUILD_TYPE", "CMAKE_GENERATOR"):
            build[key] = value
    report = {
        "schema_version": 2,
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "runtime_snapshot",
        "pid": args.pid,
        "process_start_ticks": start,
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), None),
        "command": (proc / "cmdline").read_bytes().decode().rstrip("\0").split("\0"),
        "affinity": sorted(os.sched_getaffinity(args.pid)),
        "environment": {key: value for key, value in environment.items() if key.startswith(ENV_PREFIXES) or key in ENV_KEYS},
        "environment_explicit_absent": [key for key in ENV_KEYS if key not in environment],
        "binary": fingerprint(proc / "exe"),
        "libraries": libraries,
        "library_device_identity_complete": all(library["device_matches"] for library in libraries),
        "repository": {"path": repo, "commit": command(["git", "-C", repo, "rev-parse", "HEAD"]),
                       "status": command(["git", "-C", repo, "status", "--porcelain", "--untracked-files=normal"]),
                       "tracked_diff": diff},
        "cmake_cache": fingerprint(cache_path),
        "build": build,
        "workloads": [fingerprint(path) for path in args.workload],
        "process_status": (proc / "status").read_text(),
        "meminfo": Path("/proc/meminfo").read_text(),
        "devices": [],
        "tool_versions": {},
        "optional_capture_errors": [],
        "limits": ["Snapshot only; no performance or correctness verdict.",
                   "Capture after backend loading; dynamically loaded libraries can change later.",
                   "Hashing performs I/O; pause benchmark timing during capture.",
                   "Untracked source contents are not archived; retain them separately.",
                   "DRM memory values are snapshots, not transfer counts or peaks."],
    }
    if args.model:
        report["model"] = fingerprint(args.model)
    for name, argv in (("installed_cxx", [build.get("CMAKE_CXX_COMPILER", "c++"), "--version"]), ("cmake", ["cmake", "--version"]),
                       ("ninja", ["ninja", "--version"]), ("driver_packages", ["pacman", "-Q", "mesa", "vulkan-radeon"])):
        try:
            report["tool_versions"][name] = command(argv)
        except (OSError, subprocess.SubprocessError) as error:
            report["optional_capture_errors"].append(f"{name}: {error}")
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*")):
        if "-" in card.name:
            continue
        report["devices"].append(device_snapshot(card))
    report["hardware"] = {"kernel": report["kernel"], "machine": report["machine"], "cpu": report["cpu"],
                          "devices": sorted((entry["identity"] for entry in report["devices"]), key=lambda entry: entry["device_address"])}
    if args.log:
        log = Path(args.log)
        report["initialization_log"] = fingerprint(log)
        report["effective_initialization"] = [line for line in log.read_text(errors="replace").splitlines()
                                               if any(marker in line for marker in ("buffer size", "offloaded", "fit_params", "n_ctx", "n_batch", "n_ubatch", "n_threads", "Vulkan", "RADV"))]
    if process_start(proc) != start:
        raise ValueError("process changed during snapshot")
    if args.require_vulkan and not any("libggml-vulkan" in library["path"] for library in libraries):
        raise ValueError("target has no mapped libggml-vulkan; backend may not have loaded yet")
    mismatches = [library["path"] for library in libraries if not library["device_matches"]]
    if mismatches:
        report["optional_capture_errors"].append(f"maps/stat device IDs differ; device identity unverified: {mismatches}")
        if args.strict_map_device:
            raise ValueError("maps/stat device IDs differ; cannot verify complete library identity")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--cmake-cache", required=True)
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--model", help="hash model bytes; performs substantial I/O")
    parser.add_argument("--log", help="initialization log from a paused benchmark process")
    parser.add_argument("--require-vulkan", action="store_true", help="require a mapped Vulkan backend library; this does not prove GPU execution")
    parser.add_argument("--strict-map-device", action="store_true", help="fail if maps/stat device IDs differ instead of recording incomplete identity")
    parser.add_argument("--expect", help="require the frozen source, artifacts, environment and loaded library set of this manifest")
    parser.add_argument("--json", required=True, dest="json_path")
    args = parser.parse_args()
    try:
        report = snapshot(args)
        if args.expect:
            with open(args.expect, encoding="utf-8") as stream:
                verify_frozen(report, json.load(stream))
            report["frozen_reference"] = fingerprint(args.expect)
    except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError) as error:
        parser.error(str(error))
    with open(args.json_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
