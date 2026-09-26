#!/usr/bin/env python3
"""Generate exact Lean constants from the retained MPFR constants manifest."""

import json
from pathlib import Path
import sys

EVIDENCE = Path(__file__).resolve().parents[1]
SOURCE = EVIDENCE / "generated" / "constants.json"
TARGET = EVIDENCE / "proof" / "Task0053Generated.lean"


def render() -> str:
    constants = json.loads(SOURCE.read_text(encoding="utf-8"))
    table = constants["exp2Q31"]
    rows = [
        "import Lean",
        "",
        "namespace Task0053.Generated",
        "",
        "set_option maxRecDepth 100000",
        "",
        f"def ln2Over128Q48 : Nat := {int(constants['ln2Over128Q48'], 16)}",
        f"def overflowFirst : Nat := {int(constants['overflowFirst'], 16)}",
        f"def subnormalFirstMagnitude : Nat := {int(constants['subnormalFirstMagnitude'], 16)}",
        f"def positiveExpNotOneFirst : Nat := {int(constants['positiveExpNotOneFirst'], 16)}",
        f"def negativeExpNotOneFirst : Nat := {int(constants['negativeExpNotOneFirst'], 16)}",
        "",
        "def polynomialQ32 : Array Int := #[",
        "  4294967296, 4294967296, 2147483648, 715827883,",
        "  178956971, 35791394, 5965232",
        "]",
        "",
        "def exp2Q31 : Array Nat := #[",
    ]
    for offset in range(0, len(table), 4):
        values = ", ".join(str(int(value, 16)) for value in table[offset:offset + 4])
        suffix = "," if offset + 4 < len(table) else ""
        rows.append(f"  {values}{suffix}")
    rows.extend([
        "]",
        "",
        "theorem polynomial_size : polynomialQ32.size = 7 := by decide",
        "theorem table_size : exp2Q31.size = 128 := by decide",
        "",
        "end Task0053.Generated",
        "",
    ])
    return "\n".join(rows)


def main() -> None:
    expected = render()
    if sys.argv[1:] == ["--check"]:
        if not TARGET.is_file() or TARGET.read_text(encoding="utf-8") != expected:
            raise SystemExit("Task-0053 Lean constants are stale")
        return
    if sys.argv[1:]:
        raise SystemExit("usage: generate_lean_constants.py [--check]")
    TARGET.write_text(expected, encoding="utf-8")


if __name__ == "__main__":
    main()
