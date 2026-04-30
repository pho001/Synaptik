---
phase: 05
slug: accelerator-verification-and-documentation-closure
status: verified
threats_open: 0
asvs_level: 1
created: 2026-04-30
---

# Phase 05 - Security

Per-phase security contract: threat register, accepted risks, and audit trail.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Prepared execution -> benchmark report rendering | Runtime traces are summarized into human and JSON benchmark reports. | Accelerator path, fallback reasons, copy timings, materialization traces, storage residency |
| Planner/backend selection -> report diagnostics | Backend selection cost traces are exposed for observability. | Selected accelerator candidate, rejected finalists, cost summary, boundary count |
| Benchmark workload definition -> verification report | Closure workload must represent the architecture shift without committing local measurements. | Workload metadata, in-memory benchmark request, trace contract output |
| Metal/CUDA capability gates -> native execution claims | Native accelerator paths must remain explicit and capability-gated. | Metal buffer evidence, CUDA unavailable policy, native shim availability |
| Local workspace artifacts -> repository history | Scratch files and machine-local profile changes must not enter commits accidentally. | `.planning/tmp/`, generated class files, local tuning profile artifacts |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-5-01 | Repudiation | Accelerator report evidence | mitigate | `BenchmarkSessionTest.renderersExposeAcceleratorEvidenceContract` asserts `acceleratorBufferExecutionPath`, `acceleratorBufferReasonCode`, fallback reason, materialization/copy fields, and `DEVICE_OWNED` residency in text and JSON reports. | closed |
| T-5-02 | Repudiation | Backend selection report diagnostics | mitigate | `BenchmarkSessionTest.renderersExposeBackendSelectionCostDiagnostics` asserts selected backend, `rejectedFinalists`, final score, transfer/compute estimates, preset, and `boundaryCount` in text and JSON output. | closed |
| T-5-03 | Tampering | Benchmark evidence artifacts | mitigate | Phase 5 report tests construct in-memory `BenchmarkReport`/`BenchmarkRequest` fixtures and do not persist measured report output to the repository. | closed |
| T-5-04 | Coverage gap | Closure workload | mitigate | `StandardWorkloadsTest.transformerBlockClosureWorkloadCoversAcceleratorEvidenceFamilies` asserts transformer-block metadata, forward-backward mode, validation shape, and source-level stressors including linear, reshape, permute, attention, tanh, mean, and gradients. | closed |
| T-5-05 | Tampering | Adjacent device buffer handoff | mitigate | `PreparedMetalExecutableBufferBindingTest.adjacentDeviceOwnedInputUsesBufferBindingWithoutCpuMaterialization` asserts one buffer execution, zero tensor-array executions, `BUFFER_BINDING`, `BUFFER_BINDING_AVAILABLE`, `DEVICE_OWNED` output residency, and no pre-publication CPU materialization traces. | closed |
| T-5-06 | Availability | Native Metal verification | mitigate | Metal trace tests use native bridge assumptions and `metalTest`; docs keep Metal native verification separate from portable Java verification. | closed |
| T-5-07 | Spoofing | CUDA native support claim | mitigate | CUDA policy tests assert `REQUIRED_BUFFER_EXECUTION_UNAVAILABLE` for required buffer mode, and docs state CUDA remains capability-gated until a native shim exists. | closed |
| T-5-08 | Tampering | Benchmark/profile ownership documentation | mitigate | `docs/calibration-autotune.md` states benchmark reports are explain artifacts and report-contract evidence, not runtime sources of truth; autotune and calibration own persistence while benchmark commands remain read-only. | closed |
| T-5-09 | Spoofing | CUDA documentation | mitigate | `docs/metal-backend.md` explicitly says CUDA remains capability-gated until a native shim exists and does not claim production CUDA native buffer execution. | closed |
| T-5-10 | Tampering | Local scratch and profile artifacts | mitigate | `.gitignore` ignores `.planning/tmp/` and generated class files; `SourceTreeHygieneTest` asserts scratch is ignored/untracked and tracked local tuning fixtures stay explicit. Final Phase 5 summary records local profile changes left unstaged. | closed |

*Status: open / closed*
*Disposition: mitigate (implementation required) / accept (documented risk) / transfer (third-party)*

---

## Accepted Risks Log

No accepted risks.

---

## Summary Threat Flags

No `## Threat Flags` sections were present in Phase 5 summaries.

---

## Security Audit 2026-04-30

| Metric | Count |
|--------|-------|
| Threats found | 10 |
| Closed | 10 |
| Open | 0 |

## Evidence

| Threats | Evidence | Result |
|---------|----------|--------|
| T-5-01, T-5-02 | `BenchmarkSessionTest.java:377-403` asserts selected/rejected cost diagnostics; `BenchmarkSessionTest.java:407-497` asserts accelerator path, reason codes, fallback reasons, copy timings, and device residency. | PASS |
| T-5-01, T-5-02 | `AcceleratorTraceSummary.java`, `TextBenchmarkReportRenderer.java`, and `JsonBenchmarkReportRenderer.java` expose `reasonCodes`, `fallbackReasons`, `backendSelectionCost`, selected backend, rejected finalists, and boundary count. | PASS |
| T-5-03 | `BenchmarkSessionTest.java:452-470` builds an in-memory report fixture; `BenchmarkSessionTest.java:510-516` runs an in-memory benchmark request for the closure workload. | PASS |
| T-5-04 | `StandardWorkloadsTest.java:61-89` asserts transformer-block closure workload metadata, forward-backward profile, validation contract, and operation stressors. | PASS |
| T-5-05 | `PreparedMetalExecutableBufferBindingTest.java:185-209` asserts adjacent device-owned input uses buffer binding, no tensor-array execution, device-owned output residency, and no CPU materialization traces before publication. | PASS |
| T-5-06 | `MetalBufferTraceSmokeTest.java:40-68` and `MetalBufferTraceSmokeTest.java:71-90` verify buffer path evidence under native assumptions; `build.gradle:123` registers `metalTest`. | PASS |
| T-5-07, T-5-09 | `PreparedCudaExecutableBufferPolicyTest.java:28-78` asserts required CUDA buffer execution is unavailable; `docs/metal-backend.md:526-550` documents trace fields and CUDA capability gating. | PASS |
| T-5-08 | `docs/calibration-autotune.md:3328-3330` documents benchmark reports as explain artifacts/report-contract evidence and keeps persistence owned by autotune/calibration. | PASS |
| T-5-10 | `.gitignore` contains `.planning/tmp/`, `/*.class`, and `**/*.class`; `SourceTreeHygieneTest.java:15-44` asserts scratch/profile hygiene; `05-03-SUMMARY.md` records local profile artifacts left unstaged. | PASS |
| All | Final recorded verification passed: `./gradlew classes`, targeted Phase 5 tests, `./gradlew metalTest`, and source hygiene checks. | PASS |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-04-30 | 10 | 10 | 0 | Codex |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-04-30
