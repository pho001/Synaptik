---
phase: 14-coverage-gap-triage-and-hot-path-targets
plan: "01"
status: complete
---

# 14-01 Summary - Coverage Gap Triage Model

## Coverage gap triage model

Added `GpuCoverageGapCategory`, `GpuCoverageGap`, and `GpuCoverageGapTriage` so v1.2 GPU coverage summaries can be converted into deterministic ranked gaps by workload, candidate, backend, category, reason, raw count, severity score, and downstream requirement family.

The model separates rejected candidates, CPU materialization, tensor-array fallback, CPU fallback, device handoff, low selected-region length, low GPU coverage, and non-device-owned storage residency.

## Verification

| Command | Status |
| --- | --- |
| `./gradlew test --tests GpuCoverageGapTriageTest --tests GpuCoverageSummaryTest` | passed |

## Requirement Coverage

- `GPUTRIAGE-01`: fallback, CPU materialization, tensor-array, and handoff reasons are explicit gap categories.
- `GPUTRIAGE-03`: deterministic severity scoring ranks measured exits before speculative coverage work.

## Artifact Hygiene

`profiles/platform/.../tuning/abc/* remained unstaged`.
