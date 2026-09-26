#!/usr/bin/env python3
"""Check retained Task-0053 reports and fail closed while any proof gate is open."""

import json
from pathlib import Path

EVIDENCE = Path(__file__).resolve().parents[1]
STATUS = EVIDENCE / "manifests" / "proof-status.json"
ALL_WORDS = EVIDENCE / "generated" / "all-words-report.json"


def main() -> None:
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    all_words = json.loads(ALL_WORDS.read_text(encoding="utf-8"))
    if all_words["verdict"] != "PASS" or all_words["unresolved"] != 0:
        raise SystemExit("Task-0053 all-word model check is not complete PASS")
    if status["overall"] != "PASS":
        raise SystemExit(
            "Task-0053 proof gate is OPEN: " + status["smallestUnresolvedTheorem"])


if __name__ == "__main__":
    main()
