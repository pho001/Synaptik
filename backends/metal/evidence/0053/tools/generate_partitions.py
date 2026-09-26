#!/usr/bin/env python3
"""Generate compact sign/exponent/range-quotient partitions for Task-0053."""

import json
from pathlib import Path
import sys

EVIDENCE = Path(__file__).resolve().parents[1]
CONSTANTS = EVIDENCE / "generated" / "constants.json"
TARGET = EVIDENCE / "generated" / "quotient-partitions.json"


def round_div_even(value: int, divisor: int) -> int:
    negative = value < 0
    magnitude = abs(value)
    quotient, remainder = divmod(magnitude, divisor)
    if remainder > divisor - remainder or (
            remainder == divisor - remainder and quotient & 1):
        quotient += 1
    return -quotient if negative else quotient


def round_shift_even(value: int, shift: int) -> int:
    if shift == 0:
        return value
    if shift >= 64:
        return 0
    return round_div_even(value, 1 << shift)


def raw_to_q48(magnitude: int, negative: bool) -> int:
    exponent = magnitude >> 23
    significand = (magnitude & 0x7fffff) | 0x800000
    shift = exponent - 102
    value = significand << shift if shift >= 0 else round_shift_even(significand, -shift)
    return -value if negative else value


def quotient(magnitude: int, negative: bool, logarithm: int) -> int:
    return round_div_even(raw_to_q48(magnitude, negative), logarithm)


def partitions(negative: bool, end: int, logarithm: int) -> list[dict[str, int]]:
    result = []
    for exponent in range(1, 255):
        begin = max(0x00800000, exponent << 23)
        limit = min(end, (exponent + 1) << 23)
        if begin >= limit:
            continue
        cursor = begin
        while cursor < limit:
            selected = quotient(cursor, negative, logarithm)
            low, high = cursor + 1, limit
            while low < high:
                middle = (low + high) // 2
                current = quotient(middle, negative, logarithm)
                changed = current > selected if not negative else current < selected
                if changed:
                    high = middle
                else:
                    low = middle + 1
            next_cursor = low
            result.append({
                "sign": 1 if negative else 0,
                "exponent": exponent,
                "quotient": selected,
                "beginMagnitude": cursor,
                "endMagnitude": next_cursor,
            })
            cursor = next_cursor
    return result


def render() -> str:
    constants = json.loads(CONSTANTS.read_text(encoding="utf-8"))
    logarithm = int(constants["ln2Over128Q48"], 16)
    positive_end = int(constants["overflowFirst"], 16)
    negative_end = int(constants["subnormalFirstMagnitude"], 16)
    records = partitions(False, positive_end, logarithm)
    records.extend(partitions(True, negative_end, logarithm))
    report = {
        "schema": "task0053-sign-exponent-quotient-partitions-v1",
        "ln2Over128Q48": logarithm,
        "positiveBegin": 0x00800000,
        "positiveEnd": positive_end,
        "negativeBegin": 0x00800000,
        "negativeEnd": negative_end,
        "partitions": records,
    }
    return json.dumps(report, separators=(",", ":"), sort_keys=True) + "\n"


def main() -> None:
    expected = render()
    if sys.argv[1:] == ["--check"]:
        if not TARGET.is_file() or TARGET.read_text(encoding="utf-8") != expected:
            raise SystemExit("Task-0053 quotient partitions are stale")
        return
    if sys.argv[1:]:
        raise SystemExit("usage: generate_partitions.py [--check]")
    TARGET.write_text(expected, encoding="utf-8")


if __name__ == "__main__":
    main()
