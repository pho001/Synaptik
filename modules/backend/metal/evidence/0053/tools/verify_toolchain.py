#!/usr/bin/env python3
"""Verify the pinned local proof executables and libraries without network access."""

import hashlib
import json
from pathlib import Path
import subprocess

EVIDENCE = Path(__file__).resolve().parents[1]
MANIFEST = EVIDENCE / "manifests" / "toolchain.json"
CONSTANTS = EVIDENCE / "generated" / "constants.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    tools = manifest["tools"]
    for name, entry in tools.items():
        path = Path(entry.get("path", entry.get("library", "")))
        if not path.is_file():
            raise SystemExit(f"pinned {name} artifact is absent: {path}")
        actual = sha256(path)
        if actual != entry["sha256"]:
            raise SystemExit(f"pinned {name} artifact hash differs: {actual}")
    lean_version = subprocess.run(
        [tools["lean"]["path"], "--version"], check=True,
        text=True, capture_output=True).stdout
    if tools["lean"]["version"] not in lean_version:
        raise SystemExit(f"unexpected Lean version: {lean_version.strip()}")
    sollya_version = subprocess.run(
        [tools["sollya"]["path"], "--version"], check=True,
        text=True, capture_output=True).stdout
    if tools["sollya"]["version"] not in sollya_version:
        raise SystemExit(f"unexpected Sollya version: {sollya_version.strip()}")
    constants = json.loads(CONSTANTS.read_text(encoding="utf-8"))
    if constants["mpfrVersion"] != tools["mpfr"]["version"]:
        raise SystemExit("generated constants use a different MPFR version")
    if constants["gmpVersion"] != tools["gmp"]["version"]:
        raise SystemExit("generated constants use a different GMP version")


if __name__ == "__main__":
    main()
