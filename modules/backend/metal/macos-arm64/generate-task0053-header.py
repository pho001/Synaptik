#!/usr/bin/env python3
"""Deterministically embed the retained Task-0053 Metal source as an NSString."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "src" / "task0053_candidate.metal"
CORE = ROOT / "src" / "task0053_integer_core.h"
TARGET = ROOT / "src" / "synaptik_task0053_candidate_kernels.h"


def quote(line: str) -> str:
    escaped = line.replace("\\", "\\\\").replace('"', '\\"')
    return f'@"{escaped}\\n"'


def render() -> str:
    source = SOURCE.read_text(encoding="utf-8")
    core = CORE.read_text(encoding="utf-8")
    include = '#include "task0053_integer_core.h"'
    if not source.endswith("\n") or not core.endswith("\n"):
        raise SystemExit("Task-0053 source files must end with a newline")
    if source.count(include) != 1:
        raise SystemExit("Task-0053 Metal source must include the integer core exactly once")
    source = source.replace(include, core.rstrip("\n"))
    rows = [
        "#ifndef SYNAPTIK_TASK0053_CANDIDATE_KERNELS_H",
        "#define SYNAPTIK_TASK0053_CANDIDATE_KERNELS_H",
        "",
        "static NSString *const SynaptikTask0053CandidateKernelSource =",
        *(quote(line) for line in source.splitlines()),
        ";",
        "",
        "#endif",
        "",
    ]
    return "\n".join(rows)


def main() -> None:
    rendered = render()
    if sys.argv[1:] == ["--check"]:
        if not TARGET.is_file() or TARGET.read_text(encoding="utf-8") != rendered:
            raise SystemExit("Task-0053 embedded header is stale")
        return
    if sys.argv[1:]:
        raise SystemExit("usage: generate-task0053-header.py [--check]")
    TARGET.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
