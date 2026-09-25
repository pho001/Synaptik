#!/usr/bin/env python3
import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
HARNESS = ROOT / "gate3.mm"
KERNEL = ROOT.parent / "0052" / "oracle.metal"
EXPECTED_KERNEL_SHA256 = "ff15f63c9d2e54d62fa2157e1c424cc00acb90d5355101b18a07175e789a40c0"
EXPECTED_CANDIDATES = {
    "custom_cmp_gt", "custom_cmp_ge", "custom_cmp_lt", "custom_cmp_le", "custom_cmp_eq",
    "custom_cmp_ne", "custom_tensor_min", "custom_tensor_max", "custom_scalar_min",
    "custom_scalar_max", "custom_clamp_fused", "composed_custom_clamp",
    "custom_reduction_min", "custom_reduction_max", "custom_scan_sum", "custom_scan_prod",
}

def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

failures: list[str] = []
if sha256(KERNEL) != EXPECTED_KERNEL_SHA256:
    failures.append("retained kernel identity changed")
source = HARNESS.read_text()
if "MPSGraph" in source or "MetalPerformanceShadersGraph" in source:
    failures.append("opaque framework reference present")
execute = source[source.index("static void executeOnce"):source.index("static void requireWords")]
if "newBuffer" in execute or "setBytes:" in execute:
    failures.append("hot invocation allocates or uses implicit inline metadata")
encode = source[source.index("static void encodeCandidate"):source.index("static void executeOnce")]
if encode.count("dispatchThreads:") != 2:
    failures.append("dispatch sites are not exactly one unconditional plus one composed-only")
if "candidate.kind == RouteKind::CLAMP_COMPOSED" not in encode:
    failures.append("second dispatch is not composed-CLAMP guarded")
for required in (
    "candidate.dispatches", "candidate.temporaryBytes", "POINT_COUNT * 4",
    "WARMUP_ROUNDS=4", "RETAINED_ROUNDS=8", "BATCH_FLOOR_NS",
    "Gate 3 requires Apple M3 Max", "FAST_MATH=0", "VALIDATION=PASS",
):
    if required not in source:
        failures.append(f"missing required control: {required}")
found = set(re.findall(r'"((?:custom|composed_custom)_[a-z_]+)"', source))
missing = EXPECTED_CANDIDATES - found
if missing:
    failures.append("missing candidates: " + ",".join(sorted(missing)))
if "2, POINT_COUNT * 4" not in source:
    failures.append("composed CLAMP declaration is not two dispatches plus one FLOAT32 intermediate")

print(f"KERNEL_SHA256={sha256(KERNEL)}")
print(f"HARNESS_SHA256={sha256(HARNESS)}")
print(f"CANDIDATE_COUNT={len(EXPECTED_CANDIDATES)}")
if failures:
    for failure in failures:
        print(f"FAIL={failure}")
    sys.exit(1)
print("AUDIT=PASS")
