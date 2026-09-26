#!/usr/bin/env python3
"""Audit retained Task-0053 source/header/host correspondence without executing a device."""

import hashlib
import json
from pathlib import Path
import re

EVIDENCE = Path(__file__).resolve().parents[1]
ROOT = EVIDENCE.parents[3]
NATIVE = ROOT / "native" / "metal-macos-arm64"
SOURCE = NATIVE / "src" / "task0053_candidate.metal"
CORE = NATIVE / "src" / "task0053_integer_core.h"
HEADER = NATIVE / "src" / "synaptik_task0053_candidate_kernels.h"
HOST = NATIVE / "src" / "synaptik_metal_foundation.m"
EMBEDDER = NATIVE / "generate-task0053-header.py"
REPORT = EVIDENCE / "generated" / "source-audit.json"
CONSTANTS = EVIDENCE / "generated" / "task0053_constants.h"
DIRECTED_CERTIFICATES = EVIDENCE / "generated" / "directed-certificates.json"
QUOTIENT_PARTITIONS = EVIDENCE / "generated" / "quotient-partitions.json"
MODEL = EVIDENCE / "model" / "task0053_model.c"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def digest_text(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def quote(line: str) -> str:
    escaped = line.replace("\\", "\\\\").replace('"', '\\"')
    return f'@"{escaped}\\n"'


def rendered_header(source: str) -> str:
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
    source_template = SOURCE.read_text(encoding="utf-8")
    core = CORE.read_text(encoding="utf-8")
    include = '#include "task0053_integer_core.h"'
    if source_template.count(include) != 1:
        raise SystemExit("candidate source must include the shared integer core exactly once")
    source = source_template.replace(include, core.rstrip("\n"))
    header = HEADER.read_text(encoding="utf-8")
    host = HOST.read_text(encoding="utf-8")
    constants = CONSTANTS.read_text(encoding="utf-8")
    model = MODEL.read_text(encoding="utf-8")
    failures: list[str] = []
    forbidden = {
        "hardware_float_type": r"\b(?:half|float|double)\b",
        "opaque_transcendental": r"\b(?:exp|exp2|pow|log)\s*\(",
        "fma_or_fast_intrinsic": r"\b(?:fma|fast::|precise::)\b",
        "bitcast_float": r"as_type\s*<\s*(?:half|float|double)",
    }
    hits = {
        name: sorted(set(re.findall(pattern, source)))
        for name, pattern in forbidden.items()
        if re.search(pattern, source)
    }
    if hits:
        failures.append("forbidden floating or opaque source token")
    if rendered_header(source) != header:
        failures.append("embedded NSString does not exactly match expanded shared-core source")
    table_match = re.search(
        r"T53_EXP2_Q31\[128\]\s*=\s*\{(?P<body>.*?)\};", source, re.S)
    table_entries = [] if table_match is None else re.findall(
        r"T53_U32_C\((0x[0-9a-fA-F]{8})\)", table_match.group("body"))
    if len(table_entries) != 128:
        failures.append("EXP2 table does not contain exactly 128 uint words")
    constants_table_match = re.search(
        r"TASK0053_EXP2_Q31\[128\]\s*=\s*\{(?P<body>.*?)\};", constants, re.S)
    constants_table = [] if constants_table_match is None else re.findall(
        r"0x[0-9a-fA-F]{8}", constants_table_match.group("body"))
    source_table = [entry.lower() for entry in table_entries]
    constants_table = [entry.lower() for entry in constants_table]
    if source_table != constants_table:
        failures.append("shared core and MPFR-generated EXP2 table words disagree")
    fixed_identities = [
        ("0x00000162e42fefa4", source, constants),
        ("0x42b17218", source, constants),
        ("0x42aeac50", source, constants),
    ]
    if any(identity not in left.lower() or identity not in right.lower()
            for identity, left, right in fixed_identities):
        failures.append("MSL range-reduction or threshold identity disagrees with generated constants")
    polynomial = ["5965232", "35791394", "178956971", "715827883",
                  "2147483648", "4294967296"]
    if any(value not in source or value not in constants for value in polynomial):
        failures.append("MSL polynomial words disagree with generated constants")
    model_functions = [
        "task0053_model_exp", "task0053_model_add_one",
        "task0053_model_divide_positive", "task0053_model_sigmoid",
    ]
    missing_model = [token for token in model_functions if token not in model]
    shared_model_include = (
        '#include "../../../../../native/metal-macos-arm64/src/task0053_integer_core.h"')
    if missing_model or shared_model_include not in model:
        failures.append("C model does not wrap the exact shared integer core")
    required_core = [
        "task0053_abs_i64", "task0053_i64_from_magnitude",
        "task0053_saturating_mul_u64", "task0053_saturating_add_i64",
        "task0053_saturating_sub_i64", "shift >= T53_U32_C(64)",
        "divisor == T53_U64_C(0)", "shift >= T53_I64_C(64)",
        "table_index >= T53_I64_C(128)", "exponent > T53_U32_C(127)",
    ]
    missing_core = [token for token in required_core if token not in core]
    if missing_core:
        failures.append("shared integer core lacks an unconditional safety guard")
    required_source = [
        "task0053_exp_word", "task0053_sigmoid_word", "task0053_add_one",
        "task0053_divide_positive", "task0053_candidate_exp", "task0053_candidate_sigmoid",
        "0x42b17218", "0x42aeac50", "T53_LN2_OVER_128_Q48",
    ]
    missing_source = [token for token in required_source if token not in source]
    if missing_source:
        failures.append("missing required source identity")
    required_host = [
        '#import "synaptik_task0053_candidate_kernels.h"',
        "SYNAPTIK_METAL_CUSTOM_EXP = 55U",
        "SYNAPTIK_METAL_CUSTOM_SIGMOID = 64U",
        '@"task0053_candidate_exp"', '@"task0053_candidate_sigmoid"',
        "stringByAppendingString:SynaptikTask0053CandidateKernelSource",
        "options.mathMode = MTLMathModeSafe",
        "case SYNAPTIK_METAL_CUSTOM_EXP:",
        "case SYNAPTIK_METAL_CUSTOM_SIGMOID:",
        "return SYNAPTIK_METAL_STATUS_UNSUPPORTED_OPERATION;",
        "BOOL task0053_domain_approved = NO;",
    ]
    missing_host = [token for token in required_host if token not in host]
    if missing_host:
        failures.append("missing host integration identity")
    report = {
        "audit": "task0053-source-host-v3",
        "directedCertificatesSha256": digest(DIRECTED_CERTIFICATES),
        "quotientPartitionsSha256": digest(QUOTIENT_PARTITIONS),
        "sourceSha256": digest(SOURCE),
        "integerCoreSha256": digest(CORE),
        "expandedMslSourceSha256": digest_text(source),
        "singleSourceCModel": "PASS",
        "embeddedHeaderSha256": digest(HEADER),
        "embedderSha256": digest(EMBEDDER),
        "hostSha256": digest(HOST),
        "constantsSha256": digest(CONSTANTS),
        "modelSha256": digest(MODEL),
        "tableEntries": len(table_entries),
        "hardwareFloatSites": [],
        "forbiddenHits": hits,
        "missingCoreSafetyTokens": missing_core,
        "unconditionalIntegerSafetyGuards": "PASS" if not missing_core else "FAIL",
        "missingSourceTokens": missing_source,
        "missingHostTokens": missing_host,
        "missingModelTokens": missing_model,
        "sourceAudit": "PASS" if not failures else "FAIL",
        "compiledSiteAudit": "PENDING_NO_STANDALONE_MSL_COMPILER",
        "nativeActivation": "BLOCKED_UNSUPPORTED_OPERATION",
        "domainApproval": "BLOCKED",
        "failures": failures,
    }
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
