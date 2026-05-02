# Phase 36 Verification: Scatter And Index Gradient Semantics

**Status:** Verified
**Date:** 2026-05-02

## Requirement Mapping

| Requirement | Verdict | Evidence |
|---|---|---|
| `METALSCATTER-01` | Complete | `MetalIndexWriteSemantics` validates `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` dtype/layout/rank/shape/static-bounds contracts, then returns stable `UNSUPPORTED_DUPLICATE_INDEX` when native duplicate accumulation parity is not proven. |
| `METALSCATTER-02` | Complete | CPU parity and bounds fixtures cover repeated index values, duplicate `TAKE_ALONG_AXIS_GRAD` accumulation, and OOB behavior in `ScatterAddExecutionTest`, `GatherExecutionTest`, and `TakeAlongAxisExecutionTest`. |
| `METALSCATTER-03` | Complete | Prepared-execution gates prove supported adjacent Metal producers can still be selected while `SCATTER_ADD` / index-gradient primitives remain explicit CPU prepared steps with stable rejection reasons. Coverage gates keep this separate from forward index support. |

## Coverage Evidence

- `GpuTargetCoverageTruth` reports Metal forward `GATHER` and `TAKE_ALONG_AXIS` as native executable, while `SCATTER_ADD`, `GATHER_GRAD`, and `TAKE_ALONG_AXIS_GRAD` remain `UNSUPPORTED_REJECTION`.
- `gather_take_small` remains the hard native forward-index target with INT32 residency evidence.
- `scatter_index_gradient_small` is a visible-blocker target requiring `UNSUPPORTED_DUPLICATE_INDEX` or named index-write/gradient operation evidence.

## Verification Commands

```bash
./gradlew test --tests ScatterAddExecutionTest --tests GatherExecutionTest --tests TakeAlongAxisExecutionTest
./gradlew test --tests PreparedExecutionBuildTest --tests backend.metal.lowering.MetalRegionLowererTest
./gradlew metalTest
./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests GpuHotPathCoverageTargetsTest --tests StandardWorkloadsTest
./gradlew test --tests SourceTreeHygieneTest
./gradlew classes
git diff --check
```

All commands passed during Phase 36-03 and Phase 36-04 execution.

## Residual Scope

Native Metal execution for duplicate-index write-add or gradient scatter remains future work. Promoting any Phase 36 operation to native support requires backend-owned duplicate accumulation semantics, CPU parity, trace evidence, and hard native-buffer gates.
