---
phase: 46-cross-backend-router-calibration-and-regression-gates
plan: 46-01-router-calibration-evidence-model
status: completed
completed: 2026-05-02
requirements:
  - BACKENDROUTE-01
---

# 46-01 Summary: Router Calibration Evidence Model

## Delivered

- Added `CrossBackendRouterEvidence`, a report-side model built from existing `ExecutionTrace` data.
- Aggregates common accelerator paths, Metal routes, CUDA execution paths, rejected route/capability reasons, native copy strategies, output-buffer write statuses, selected region length, lowered primitive count, fused subpatterns, layout materialization, dtype residency, CPU materialization, and device handoffs.
- Kept existing execution behavior and public `Tensor` API unchanged.

## Evidence

- `CrossBackendRouterEvidenceTest.summarizesMetalMpsGraphCopyRequiredEvidence`
- `CrossBackendRouterEvidenceTest.customMetalKernelTrueWriteRoutePassesStrictEvidenceGate`
- `CrossBackendRouterEvidenceTest.cudaCapabilityMissingCanBeRequiredAsExplicitFallbackEvidence`
