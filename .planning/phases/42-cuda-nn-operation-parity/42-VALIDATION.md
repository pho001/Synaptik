---
phase: 42
status: passed
validated: 2026-05-02
validation_gaps: 0
requirements:
  - CUDANN-01
  - CUDANN-02
  - CUDANN-03
---

# Phase 42 Nyquist Validation

## Result

PASSED. Phase 42 has positive evidence for semantic classification and negative evidence for unsupported CUDA native execution claims across SDPA, conv/pool, and dense loss.

## Requirement Coverage

| Requirement | Positive Evidence | Negative Evidence | Verdict |
|-------------|-------------------|-------------------|---------|
| CUDANN-01 | `CudaRegionLowererTest` covers unmasked, external BOOL masked, causal, and external+causal SDPA diagnostics with stable `maskMode` and target details. | The same tests prove CUDA SDPA remains `CAPABILITY_MISSING`; dtype/layout invalid cases reject before capability fallback. | PASS |
| CUDANN-02 | Conv/pool semantic checks validate scoped dense `FLOAT32` NCHW/OIHW forward contracts and report `CONV_POOL` blockers. | Grouped/depthwise, dilation, invalid shape/layout/dtype, and `AVG_POOL2D countIncludePad=true` remain explicit blockers; no native route is claimed. | PASS |
| CUDANN-03 | Dense `NLL_LOSS` and dense `CROSS_ENTROPY_LOSS` validate dense `FLOAT32` loss contracts and report `target=dense_loss_small`. | Dense loss remains `DAG_PRIMITIVE_UNSUPPORTED`; index-target loss remains a separate `UNSUPPORTED_INDEX_SEMANTICS` blocker. | PASS |

## Task Coverage

| Plan | Evidence | Verdict |
|------|----------|---------|
| 42-01 | `42-01-SUMMARY.md`, `CudaNnSemantics`, SDPA mask-mode tests. | PASS |
| 42-02 | `42-02-SUMMARY.md`, conv/pool semantic checks, conv/pool blocker tests. | PASS |
| 42-03 | `42-03-SUMMARY.md`, dense loss semantic checks, dense-vs-index loss tests. | PASS |
| 42-04 | `42-04-SUMMARY.md`, `CUDANN` target metadata, docs, coverage target tests. | PASS |

## Regression Gate

Validated commands:

```bash
./gradlew test --tests backend.cuda.lowering.CudaRegionLowererTest
./gradlew test --tests backend.cuda.lowering.CudaRegionLowererTest --tests backend.accelerator.lowering.GpuBackendParityReportTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuCoverageTriageReportTest --tests GpuHotPathCoverageTargetsTest --tests SourceTreeHygieneTest
./gradlew classes
git diff --check
```

All commands passed.

## Nyquist Findings

- No missing test dimensions were found for the claimed support-or-rejection scope.
- No native CUDA NN support claim was introduced without execution evidence.
- No hidden CPU fallback or tensor-array success path was counted as support.
- Local profile artifacts remain unstaged and non-canonical.

