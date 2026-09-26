#!/usr/bin/env python3
"""Generate the kernel-replayed Lean quotient partition certificate."""

import json
from pathlib import Path
import sys

EVIDENCE = Path(__file__).resolve().parents[1]
SOURCE = EVIDENCE / "generated" / "quotient-partitions.json"
TARGET = EVIDENCE / "proof" / "Task0053Partitions.lean"


def render() -> str:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    partitions = data["partitions"]
    rows = [
        "import Task0053Integer",
        "",
        "namespace Task0053.Partitions",
        "",
        "open Task0053.Integer",
        "",
        "set_option maxRecDepth 2000000",
        "set_option maxHeartbeats 0",
        "",
        "structure Partition where",
        "  (negative : Bool)",
        "  (exponent beginMagnitude endMagnitude : Nat)",
        "  (quotient : Int)",
        "",
        "def certificate : List Partition := [",
    ]
    for index, partition in enumerate(partitions):
        suffix = "," if index + 1 < len(partitions) else ""
        rows.append(
            "  ⟨"
            f"{'true' if partition['sign'] else 'false'}, "
            f"{partition['exponent']}, {partition['beginMagnitude']}, "
            f"{partition['endMagnitude']}, {partition['quotient']}"
            f"⟩{suffix}")
    rows.extend([
        "]",
        "",
        "def partitionChecks (partition : Partition) : Bool :=",
        "  decide (partition.beginMagnitude < partition.endMagnitude) &&",
        "  decide (partition.beginMagnitude / 2^23 = partition.exponent) &&",
        "  decide ((partition.endMagnitude - 1) / 2^23 = partition.exponent) &&",
        "  decide (rangeQuotient partition.beginMagnitude partition.negative = partition.quotient) &&",
        "  decide (rangeQuotient (partition.endMagnitude - 1) partition.negative = partition.quotient)",
        "",
        "def coverageChecks (expected final : Nat) : List Partition → Bool",
        "  | [] => decide (expected = final)",
        "  | partition :: rest =>",
        "      decide (partition.beginMagnitude = expected) &&",
        "      decide (expected < partition.endMagnitude) &&",
        "      coverageChecks partition.endMagnitude final rest",
        "",
        "theorem coverageChecks_sound",
        "    (partitions : List Partition) (expected final word : Nat)",
        "    (checked : coverageChecks expected final partitions = true)",
        "    (lower : expected ≤ word) (upper : word < final) :",
        "    ∃ partition ∈ partitions,",
        "      partition.beginMagnitude ≤ word ∧ word < partition.endMagnitude := by",
        "  induction partitions generalizing expected with",
        "  | nil =>",
        "      simp [coverageChecks] at checked",
        "      omega",
        "  | cons partition rest induction =>",
        "      simp only [coverageChecks] at checked",
        "      rw [Bool.and_eq_true] at checked",
        "      rw [Bool.and_eq_true] at checked",
        "      have begin_eq : partition.beginMagnitude = expected := of_decide_eq_true checked.1.1",
        "      have nonempty : expected < partition.endMagnitude := of_decide_eq_true checked.1.2",
        "      have rest_checked := checked.2",
        "      by_cases in_first : word < partition.endMagnitude",
        "      · exact ⟨partition, by simp, by omega⟩",
        "      · obtain ⟨found, member, found_lower, found_upper⟩ :=",
        "          induction partition.endMagnitude rest_checked (by omega)",
        "        exact ⟨found, by simp [member], found_lower, found_upper⟩",
        "",
        "theorem coverageChecks_member_lower",
        "    (partitions : List Partition) (expected final : Nat)",
        "    (checked : coverageChecks expected final partitions = true)",
        "    (partition : Partition) (member : partition ∈ partitions) :",
        "    expected ≤ partition.beginMagnitude := by",
        "  induction partitions generalizing expected with",
        "  | nil => simp at member",
        "  | cons first rest induction =>",
        "      simp only [coverageChecks] at checked",
        "      rw [Bool.and_eq_true] at checked",
        "      rw [Bool.and_eq_true] at checked",
        "      have begin_eq : first.beginMagnitude = expected := of_decide_eq_true checked.1.1",
        "      have increasing : expected < first.endMagnitude := of_decide_eq_true checked.1.2",
        "      have rest_checked := checked.2",
        "      simp at member",
        "      rcases member with rfl | member",
        "      · omega",
        "      · have lower := induction first.endMagnitude rest_checked member",
        "        omega",
        "",
        "theorem coverageChecks_unique",
        "    (partitions : List Partition) (expected final word : Nat)",
        "    (checked : coverageChecks expected final partitions = true)",
        "    (left right : Partition)",
        "    (left_member : left ∈ partitions) (right_member : right ∈ partitions)",
        "    (left_contains : left.beginMagnitude ≤ word ∧ word < left.endMagnitude)",
        "    (right_contains : right.beginMagnitude ≤ word ∧ word < right.endMagnitude) :",
        "    left = right := by",
        "  induction partitions generalizing expected with",
        "  | nil => simp at left_member",
        "  | cons first rest induction =>",
        "      simp only [coverageChecks] at checked",
        "      rw [Bool.and_eq_true] at checked",
        "      rw [Bool.and_eq_true] at checked",
        "      have rest_checked := checked.2",
        "      rw [List.mem_cons] at left_member right_member",
        "      rcases left_member with rfl | left_member",
        "      · rcases right_member with rfl | right_member",
        "        · rfl",
        "        · have lower := coverageChecks_member_lower rest left.endMagnitude final",
        "              rest_checked right right_member",
        "          omega",
        "      · rcases right_member with rfl | right_member",
        "        · have lower := coverageChecks_member_lower rest right.endMagnitude final",
        "              rest_checked left left_member",
        "          omega",
        "        · exact induction first.endMagnitude rest_checked left_member right_member",
        "",
        "def positiveCertificate := certificate.takeWhile (fun partition => !partition.negative)",
        "def negativeCertificate := certificate.dropWhile (fun partition => !partition.negative)",
        "",
        f"def positiveCoverageChecks := coverageChecks 0x00800000 {data['positiveEnd']} positiveCertificate",
        f"def negativeCoverageChecks := coverageChecks 0x00800000 {data['negativeEnd']} negativeCertificate",
        "def positiveSignChecks := positiveCertificate.all (fun partition => !partition.negative)",
        "def negativeSignChecks := negativeCertificate.all (fun partition => partition.negative)",
        "",
        "",
        "def certificateChecks : Bool :=",
        f"  decide (certificate.length = {len(partitions)}) &&",
        "  positiveCertificate.all partitionChecks &&",
        "  negativeCertificate.all partitionChecks &&",
        "  positiveSignChecks && negativeSignChecks &&",
        "  positiveCoverageChecks && negativeCoverageChecks",
        "",
        "theorem quotient_partition_certificate : certificateChecks = true := by decide",
        "theorem positive_partition_records : positiveCertificate.all partitionChecks = true := by decide",
        "theorem negative_partition_records : negativeCertificate.all partitionChecks = true := by decide",
        "theorem positive_partition_coverage_check : positiveCoverageChecks = true := by decide",
        "theorem negative_partition_coverage_check : negativeCoverageChecks = true := by decide",
        "theorem positive_partition_signs : positiveSignChecks = true := by decide",
        "theorem negative_partition_signs : negativeSignChecks = true := by decide",
        "",
        "theorem positive_partition_covers (word : Nat)",
        f"    (lower : 0x00800000 ≤ word) (upper : word < {data['positiveEnd']}) :",
        "    ∃ partition ∈ positiveCertificate,",
        "      partition.beginMagnitude ≤ word ∧ word < partition.endMagnitude :=",
        "  coverageChecks_sound positiveCertificate 0x00800000 _ word",
        "    positive_partition_coverage_check lower upper",
        "",
        "theorem negative_partition_covers (word : Nat)",
        f"    (lower : 0x00800000 ≤ word) (upper : word < {data['negativeEnd']}) :",
        "    ∃ partition ∈ negativeCertificate,",
        "      partition.beginMagnitude ≤ word ∧ word < partition.endMagnitude :=",
        "  coverageChecks_sound negativeCertificate 0x00800000 _ word",
        "    negative_partition_coverage_check lower upper",
        "",
        "theorem positive_partition_disjoint (word : Nat) (left right : Partition)",
        "    (left_member : left ∈ positiveCertificate) (right_member : right ∈ positiveCertificate)",
        "    (left_contains : left.beginMagnitude ≤ word ∧ word < left.endMagnitude)",
        "    (right_contains : right.beginMagnitude ≤ word ∧ word < right.endMagnitude) :",
        "    left = right :=",
        "  coverageChecks_unique positiveCertificate 0x00800000 _ word",
        "    positive_partition_coverage_check left right left_member right_member",
        "    left_contains right_contains",
        "",
        "theorem negative_partition_disjoint (word : Nat) (left right : Partition)",
        "    (left_member : left ∈ negativeCertificate) (right_member : right ∈ negativeCertificate)",
        "    (left_contains : left.beginMagnitude ≤ word ∧ word < left.endMagnitude)",
        "    (right_contains : right.beginMagnitude ≤ word ∧ word < right.endMagnitude) :",
        "    left = right :=",
        "  coverageChecks_unique negativeCertificate 0x00800000 _ word",
        "    negative_partition_coverage_check left right left_member right_member",
        "    left_contains right_contains",
        "",
        "end Task0053.Partitions",
        "",
    ])
    return "\n".join(rows)


def main() -> None:
    expected = render()
    if sys.argv[1:] == ["--check"]:
        if not TARGET.is_file() or TARGET.read_text(encoding="utf-8") != expected:
            raise SystemExit("Task-0053 Lean quotient partitions are stale")
        return
    if sys.argv[1:]:
        raise SystemExit("usage: generate_lean_partitions.py [--check]")
    TARGET.write_text(expected, encoding="utf-8")


if __name__ == "__main__":
    main()
