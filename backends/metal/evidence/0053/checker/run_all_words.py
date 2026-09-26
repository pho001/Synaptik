#!/usr/bin/env python3
"""Run independent MPFR checker shards as processes and merge deterministic evidence."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile

SUM_FIELDS = [
    "words", "nanInputs", "infinityInputs", "zeroInputs", "subnormalInputs",
    "normalInputs", "ordinaryExpResults", "expClassFailures",
    "expDistanceFailures", "sigmoidFailures", "addSiteFailures",
    "divSiteFailures", "unresolved",
]


def word(value: str) -> int:
    return int(value, 16)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit("usage: run_all_words.py CHECKER SHARDS REPORT.json")
    checker = Path(sys.argv[1]).resolve()
    shards = int(sys.argv[2])
    report_path = Path(sys.argv[3]).resolve()
    if not checker.is_file() or not 1 <= shards <= 64:
        raise SystemExit("checker path or shard count is invalid")
    constants_path = Path(__file__).resolve().parents[1] / "generated" / "constants.json"
    constants = json.loads(constants_path.read_text(encoding="utf-8"))
    magnitude_count = 0x80000000
    with tempfile.TemporaryDirectory(prefix="task0053-checker-") as directory:
        root = Path(directory)
        processes = []
        paths = []
        for index in range(shards):
            begin = magnitude_count * index // shards
            end = magnitude_count * (index + 1) // shards
            path = root / f"shard-{index:02d}.json"
            paths.append(path)
            processes.append(subprocess.Popen([
                str(checker), "1", str(path), hex(begin), hex(end)
            ]))
        statuses = [process.wait() for process in processes]
        if any(status not in (0, 1) for status in statuses):
            raise SystemExit(f"checker shard infrastructure failure: {statuses}")
        reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    merged = {
        "checker": "task0053-mpfr-partitioned-all-binary32-v2",
        "processShards": shards,
        "beginMagnitude": "0x00000000",
        "endMagnitude": "0x80000000",
        "referencePartitions": {
            "positiveRoundsToOneBelow": constants["positiveExpNotOneFirst"],
            "negativeRoundsToOneBelow": constants["negativeExpNotOneFirst"],
            "positiveOverflowsAtOrAbove": constants["overflowFirst"],
            "negativeIsSubnormalOrZeroAtOrAbove": constants["subnormalFirstMagnitude"],
            "boundaries": "MPFR 4.2.2 binary searches retained in generate_constants.c",
            "interior": "direct adaptive MPFR exp through 1024-bit precision",
        },
    }
    for field in SUM_FIELDS:
        merged[field] = sum(report[field] for report in reports)
    maximum = min(
        reports,
        key=lambda report: (-report["maximumOrderedDistance"],
                            word(report["maximumDistanceWord"])),
    )
    merged["maximumOrderedDistance"] = maximum["maximumOrderedDistance"]
    merged["maximumDistanceWord"] = maximum["maximumDistanceWord"]
    merged["maximumDistanceActual"] = maximum["maximumDistanceActual"]
    merged["maximumDistanceReference"] = maximum["maximumDistanceReference"]
    failures = [report for report in reports if report["firstFailureCode"] != 0]
    if failures:
        first = min(failures, key=lambda report: word(report["firstFailureWord"]))
        merged["firstFailureWord"] = first["firstFailureWord"]
        merged["firstFailureCode"] = first["firstFailureCode"]
    else:
        merged["firstFailureWord"] = "0xffffffff"
        merged["firstFailureCode"] = 0
    merged["verdict"] = "PASS" if all(
        report["verdict"] == "PASS" for report in reports) else "FAIL"
    if merged["words"] != 0x100000000:
        raise SystemExit(f"checker coverage mismatch: {merged['words']}")
    expected_classes = {
        "nanInputs": 2 * ((1 << 23) - 1),
        "infinityInputs": 2,
        "zeroInputs": 2,
        "subnormalInputs": 2 * ((1 << 23) - 1),
        "normalInputs": 2 * 254 * (1 << 23),
    }
    for field, expected in expected_classes.items():
        if merged[field] != expected:
            raise SystemExit(
                f"checker raw partition mismatch for {field}: {merged[field]}")
    merged["rawPartitionCountsVerified"] = True
    report_path.write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if merged["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
