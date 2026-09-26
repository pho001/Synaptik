#!/usr/bin/env python3
"""Write or check the deterministic Task-0053 evidence hash manifest."""

import hashlib
import json
from pathlib import Path
import sys

EVIDENCE = Path(__file__).resolve().parents[1]
ROOT = EVIDENCE.parents[3]
TARGET = EVIDENCE / "manifests" / "top-manifest.json"
IMPLEMENTATION = [
    ROOT / "native/metal-macos-arm64/src/task0053_candidate.metal",
    ROOT / "native/metal-macos-arm64/src/task0053_integer_core.h",
    ROOT / "native/metal-macos-arm64/src/synaptik_task0053_candidate_kernels.h",
    ROOT / "native/metal-macos-arm64/generate-task0053-header.py",
    ROOT / "native/metal-macos-arm64/src/synaptik_metal_foundation.m",
    ROOT / "native/metal-macos-arm64/build.sh",
    ROOT / "backends/metal/src/test/java/io/github/pho001/synaptik/backend/metal/MetalMpsGraphRawAbiNativeTest.java",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render() -> str:
    evidence_files = sorted(
        path for path in EVIDENCE.rglob("*")
        if path.is_file() and path != TARGET and "__pycache__" not in path.parts
    )
    files = sorted(set(evidence_files + IMPLEMENTATION), key=lambda path: str(path.relative_to(ROOT)))
    report = {
        "schema": "synaptik-metal-task0053-top-manifest-v1",
        "selfExcluded": str(TARGET.relative_to(ROOT)),
        "files": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for path in files
        ],
    }
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def main() -> None:
    expected = render()
    if sys.argv[1:] == ["--check"]:
        if not TARGET.is_file() or TARGET.read_text(encoding="utf-8") != expected:
            raise SystemExit("Task-0053 top manifest is stale")
        return
    if sys.argv[1:]:
        raise SystemExit("usage: write_manifest.py [--check]")
    TARGET.write_text(expected, encoding="utf-8")


if __name__ == "__main__":
    main()
