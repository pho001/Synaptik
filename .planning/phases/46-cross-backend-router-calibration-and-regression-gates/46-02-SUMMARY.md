---
phase: 46-cross-backend-router-calibration-and-regression-gates
plan: 46-02-representative-workload-gates
status: completed
completed: 2026-05-02
requirements:
  - BACKENDROUTE-02
---

# 46-02 Summary: Representative Workload Gates

## Delivered

- Added `CrossBackendRouterGatePolicy`, `CrossBackendRouterGateResult`, `CrossBackendRouterWorkloadExpectation`, and `CrossBackendRouterRegressionGate`.
- Gates fail hidden tensor-array replay, unexpected CPU fallback, CPU materialization, internal CPU materialization, lost native buffer binding, lost selected-region/lowered-primitive evidence, missing required routes/reasons, native-copy/write-status regressions, and unsupported route overclaims.
- Suite-target evaluation names workload and backend in failures.

## Evidence

- `CrossBackendRouterEvidenceTest.representativeGateFailsHiddenTensorArrayReplayAndCpuMaterialization`
- `CrossBackendRouterEvidenceTest.mpsGraphCannotOverclaimTrueOutputBufferWrite`
- `CrossBackendRouterEvidenceTest.suiteTargetFailuresNameWorkloadAndBackend`
