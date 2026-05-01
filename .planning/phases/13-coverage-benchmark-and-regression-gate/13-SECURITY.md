---
phase: 13
slug: coverage-benchmark-and-regression-gate
status: verified
threats_open: 0
asvs_level: 1
created: 2026-05-01
---

# Phase 13 - Security

Per-phase security contract: threat register, accepted risks, and audit trail for coverage benchmark and regression gate work.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Prepare/run traces to coverage report | Benchmark reports derive GPU coverage from execution metadata and materialization traces. | Backend labels, execution paths, selected/rejected decisions, materialization reasons, copy timings. |
| Accelerator buffer path to native coverage claim | Native buffer execution, tensor-array bridge execution, and CPU fallback must remain separate report facts. | `BUFFER_BINDING`, `TENSOR_ARRAY`, `CPU_FALLBACK`, reason codes, fallback reasons. |
| Suite workload evidence to baseline comparison | Representative workload reports compare coverage/materialization behavior without timing thresholds. | Workload names, selected region length, materialization/fallback/handoff counts. |
| Regression gate policy to test failures | Gate policy turns hidden exits into stable failure strings. | Coverage summary, policy thresholds, failure strings. |
| Native capability checks to docs/summary | Native Metal/CUDA evidence must distinguish pass, unavailable, and capability-skipped results. | Gradle native task result, capability-skip status, docs/summary wording. |
| Working tree to committed evidence | Local tuning/profile outputs are not canonical evidence. | `profiles/platform/.../tuning/abc/*` local JSON/JSONL files and checked-in docs/tests. |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-13-01 | Repudiation | GPU coverage reports | mitigate | `GpuCoverageSummary` computes `gpuCoverageRatio` from accelerator-buffer run steps divided by total traced run steps; renderer tests assert the field in text and JSON reports. | closed |
| T-13-02 | Repudiation / Tampering | Native coverage classification | mitigate | `GpuCoverageSummary` reports `tensorArrayStepCount` separately from `bufferBindingStepCount`; regression gates can reject tensor-array fallback when native buffer coverage is required. | closed |
| T-13-03 | Repudiation | Compact report rendering | mitigate | Coverage summary and renderers include `selectedRegionCount`, `maxSelectedRegionLength`, `averageSelectedRegionLength`, and `rejectedCandidateReasonCounts`; tests assert exact rendered fields. | closed |
| T-13-04 | Repudiation / Information Disclosure | CPU materialization and handoff reporting | mitigate | Coverage summary includes CPU materialization reason counts, bytes, duration, storage residency counts, and `deviceHandoffCount`; tests assert the materialization and handoff evidence. | closed |
| T-13-05 | Denial of Service / Repudiation | Baseline comparison | mitigate | `GpuCoverageComparison` compares selected region length, CPU materialization count, fallback count, and device handoff count only; docs state coverage/materialization behavior is the gate, not raw timing. | closed |
| T-13-06 | Tampering | Baseline source | mitigate | Baseline contracts are checked-in Java records/tests and suite report fields; docs mark `profiles/platform/.../tuning/abc/*` as non-canonical local output. | closed |
| T-13-07 | Repudiation | Representative workload suite | mitigate | `BenchmarkSuiteSessionTest.representativeCoverageSuiteNamesTransformerMlpAndConvOrNormalization` asserts `transformer_block_hot_path`, `mlp_classifier_small`, and `conv2d_resnet_3x3`. | closed |
| T-13-08 | Repudiation | Supported GPU target coverage | mitigate | `GpuCoverageRegressionGate` fails with `lost GPU coverage` when coverage ratio, selected region length, fallback count, or native-buffer requirements regress. | closed |
| T-13-09 | Repudiation | CPU materialization boundary | mitigate | `GpuCoverageRegressionGate` fails with `unexpected CPU materialization`; tests exercise the failure string. | closed |
| T-13-10 | Repudiation / Tampering | Tensor-array fallback | mitigate | `GpuCoverageRegressionGate` fails with `hidden tensor-array fallback` when tensor-array bridge steps exceed policy. | closed |
| T-13-11 | Repudiation | Native CUDA evidence | mitigate | `13-04-SUMMARY.md` records `native CUDA status: capability-skipped`, and validation records the capability-skipped CUDA gate separately from Metal pass status. | closed |
| T-13-12 | Tampering | Local tuning/profile artifacts | mitigate | `13-04-SUMMARY.md` and `13-VALIDATION.md` record `git status --short`; this audit confirms no staged `profiles/platform/.../tuning/abc/*` files. | closed |
| T-13-13 | Repudiation | Future regression evidence | mitigate | Docs name checked-in report fields/tests as the evidence contract and machine-local benchmark/calibration output as non-canonical artifacts. | closed |

Status: closed.
Disposition: mitigate.

---

## Evidence

| Threat Ref | Evidence |
|------------|----------|
| T-13-01 | `src/main/java/tuning/benchmark/report/GpuCoverageSummary.java`, `src/main/java/tuning/benchmark/report/TextBenchmarkReportRenderer.java`, `src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java`, `src/test/java/BenchmarkSessionTest.java` |
| T-13-02 | `src/main/java/tuning/benchmark/report/GpuCoverageSummary.java`, `src/main/java/tuning/benchmark/report/GpuCoverageRegressionGate.java`, `src/test/java/GpuCoverageRegressionGateTest.java` |
| T-13-03 | `src/main/java/tuning/benchmark/report/GpuCoverageSummary.java`, `src/main/java/tuning/benchmark/report/TextBenchmarkReportRenderer.java`, `src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java`, `src/test/java/BenchmarkSessionTest.java` |
| T-13-04 | `src/main/java/tuning/benchmark/report/GpuCoverageSummary.java`, `src/test/java/GpuCoverageSummaryTest.java`, `src/test/java/BenchmarkSessionTest.java` |
| T-13-05 | `src/main/java/tuning/benchmark/report/GpuCoverageBaseline.java`, `src/main/java/tuning/benchmark/report/GpuCoverageComparison.java`, `docs/compute-flow.md`, `docs/gpu-lowering-coverage.md` |
| T-13-06 | `src/main/java/tuning/benchmark/report/GpuCoverageBaseline.java`, `src/test/java/GpuCoverageSummaryTest.java`, `docs/development.md`, `docs/testing.md` |
| T-13-07 | `src/test/java/BenchmarkSuiteSessionTest.java`, `src/main/java/tuning/workload/StandardWorkloads.java`, `.planning/phases/13-coverage-benchmark-and-regression-gate/13-02-SUMMARY.md` |
| T-13-08 | `src/main/java/tuning/benchmark/report/GpuCoverageRegressionGate.java`, `src/test/java/GpuCoverageRegressionGateTest.java`, `src/test/java/BenchmarkSessionTest.java` |
| T-13-09 | `src/main/java/tuning/benchmark/report/GpuCoverageRegressionGate.java`, `src/test/java/GpuCoverageRegressionGateTest.java`, `src/test/java/BenchmarkSessionTest.java` |
| T-13-10 | `src/main/java/tuning/benchmark/report/GpuCoverageRegressionGate.java`, `src/test/java/GpuCoverageRegressionGateTest.java`, `docs/testing.md` |
| T-13-11 | `.planning/phases/13-coverage-benchmark-and-regression-gate/13-04-SUMMARY.md`, `.planning/phases/13-coverage-benchmark-and-regression-gate/13-VALIDATION.md` |
| T-13-12 | `.planning/phases/13-coverage-benchmark-and-regression-gate/13-04-SUMMARY.md`, `.planning/phases/13-coverage-benchmark-and-regression-gate/13-VALIDATION.md`, `git status --short`, `git diff --cached --name-only` |
| T-13-13 | `docs/compute-flow.md`, `docs/gpu-lowering-coverage.md`, `docs/development.md`, `docs/testing.md` |

---

## Accepted Risks Log

No accepted risks.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-05-01 | 13 | 13 | 0 | Codex inline auditor |

### Security Audit 2026-05-01

| Metric | Count |
|--------|-------|
| Threats found | 13 |
| Closed | 13 |
| Open | 0 |

Verification commands:

- `./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests CompiledGraphTraceTest` - PASS
- `rg -n "gpuCoverageRatio|tensorArrayStepCount|bufferBindingStepCount|selectedRegionCount|maxSelectedRegionLength|averageSelectedRegionLength|rejectedCandidateReasonCounts|cpuMaterializationReasonCounts|cpuMaterializationBytes|cpuMaterializationDurationNs|storageResidencyCounts|deviceHandoffCount|fallbackCount|coverage/materialization behavior|not raw timing|GpuCoverageBaseline|GpuCoverageComparison|hidden tensor-array fallback|lost GPU coverage|unexpected CPU materialization|unexpected device handoff|native CUDA status|capability-skipped|profiles/platform/.../tuning/abc/\* remained unstaged|machine-local benchmark/calibration output|checked-in evidence contract|do not commit local tuning artifacts" src/main/java/tuning/benchmark/report src/test/java docs .planning/phases/13-coverage-benchmark-and-regression-gate/13-04-SUMMARY.md .planning/phases/13-coverage-benchmark-and-regression-gate/13-VALIDATION.md` - PASS
- `rg -n "transformer_block_hot_path|mlp_classifier_small|conv2d_resnet_3x3|layer_norm_small|representativeCoverageSuiteNamesTransformerMlpAndConvOrNormalization" src/test/java/BenchmarkSuiteSessionTest.java src/main/java/tuning/workload/StandardWorkloads.java .planning/phases/13-coverage-benchmark-and-regression-gate/13-02-SUMMARY.md` - PASS
- `git status --short` - only pre-existing local `profiles/platform/.../tuning/abc/*` files were modified.
- `git diff --cached --name-only` - empty before staging this security report, confirming no local tuning artifacts were staged.

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-05-01
