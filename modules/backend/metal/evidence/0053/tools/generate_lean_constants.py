#!/usr/bin/env python3
"""Generate exact Lean constants from the retained MPFR constants manifest."""

import json
from pathlib import Path
import sys

EVIDENCE = Path(__file__).resolve().parents[1]
SOURCE = EVIDENCE / "generated" / "constants.json"
TARGET = EVIDENCE / "proof" / "Task0053Generated.lean"

CERTIFICATES = EVIDENCE / "generated" / "directed-certificates.json"

def render() -> str:
    constants = json.loads(SOURCE.read_text(encoding="utf-8"))
    certificates = json.loads(CERTIFICATES.read_text(encoding="utf-8"))
    table = constants["exp2Q31"]
    rows = [
        "import Lean",
        "",
        "namespace Task0053.Generated",
        "",
        "set_option maxRecDepth 100000",
        "set_option exponentiation.threshold 512",
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
        "structure DirectedCertificate where",
        "  (lower upper rounded predecessor successor : Nat)",
        "",
        "def directedTable : List DirectedCertificate := [",
    ])
    for index, certificate in enumerate(certificates["table"]):
        suffix = "," if index + 1 < len(certificates["table"]) else ""
        rows.append(
            "  ⟨"
            f"{int(certificate['lowerNumerator'], 16)}, "
            f"{int(certificate['upperNumerator'], 16)}, "
            f"{int(certificate['rounded'], 16)}, "
            f"{int(certificate['predecessor'], 16)}, "
            f"{int(certificate['successor'], 16)}"
            f"⟩{suffix}")
    rows.extend([
        "]",
        "",
        "def directedCertificateChecks (certificate : DirectedCertificate) : Bool :=",
        "  let scale := 2^256",
        "  decide (certificate.lower ≤ certificate.upper) &&",
        "  decide (certificate.upper - certificate.lower ≤ 1) &&",
        "  decide (certificate.predecessor + 1 = certificate.rounded) &&",
        "  decide (certificate.rounded + 1 = certificate.successor) &&",
        "  decide ((2 * certificate.rounded - 1) * scale < 2 * certificate.lower) &&",
        "  decide (2 * certificate.upper < (2 * certificate.rounded + 1) * scale)",
        "",
        "def directedTableChecks : Bool :=",
        "  directedTable.all directedCertificateChecks",
        "def directedTableWordsMatch : Bool :=",
        "  directedTable.map (·.rounded) == exp2Q31.toList",
        "",
        "-- This checker removes MPFR from the trust root.  The directed bounds are",
        "-- replayed as exact rational bounds on the unique positive 128th root of 2^index.",
        "def algebraicTableCertificateChecks (entry : Nat × DirectedCertificate) : Bool :=",
        "  let index := entry.1",
        "  let certificate := entry.2",
        "  let scale := 2^(31 + 256)",
        "  let target := 2^index * scale^128",
        "  decide (certificate.lower^128 ≤ target) &&",
        "  decide (target ≤ certificate.upper^128)",
        "",
        "def algebraicTableChecks : Bool :=",
        "  (List.range 128 |>.zip directedTable).all algebraicTableCertificateChecks",
        "",
        "def widthCertificateChecks : Bool :=",
        "  decide (24973256993800192 < 2^55) &&",
        "  decide (16384 * ln2Over128Q48 < 2^55) &&",
        "  decide (11629080 < 2^24) &&",
        "  decide (4306612134 < 2^33) &&",
        "  decide (4271771996 * 4306612134 < 2^64)",
        "",
        "theorem polynomial_size : polynomialQ32.size = 7 := by decide",
        "theorem table_size : exp2Q31.size = 128 := by decide",
        "theorem directed_table_size : directedTable.length = 128 := by decide",
        "theorem directed_table_checks : directedTableChecks = true := by decide",
        "theorem algebraic_table_checks : algebraicTableChecks = true := by decide",
        "theorem directed_table_words_match : directedTableWordsMatch = true := by decide",
        "theorem width_certificate_checks : widthCertificateChecks = true := by decide",
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
