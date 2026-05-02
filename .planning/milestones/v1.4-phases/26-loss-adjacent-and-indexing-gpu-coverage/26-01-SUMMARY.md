# Summary 26-01: Loss And Indexing Semantics Coverage Contract

**Status:** Complete
**Date:** 2026-05-02

## Completed

- Added Phase 26 reason codes for index semantics, ignore-index, duplicate-index accumulation, and bounds-sensitive unsupported cases.
- Expanded Metal/CUDA coverage rows for `GATHER_GRAD`, `TAKE_ALONG_AXIS`, and `TAKE_ALONG_AXIS_GRAD`.
- Refined loss/index rows so unsupported status is operation-specific instead of a generic dtype fallback.
- Updated semantics contracts to state that INT32 index residency is not native index/loss compute.
- Updated `docs/gpu-lowering-coverage.md`.

## Evidence

- `GpuLoweringCoverageMatrixTest.phaseTwentySixMatrixCoversLossAndIndexingFamilyExplicitly`
- `GpuTargetSemanticsContractTest.phaseTwentySixContractsDistinguishResidencyFromNativeIndexCompute`

## Verification

Passed:

```bash
./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests GpuTargetSemanticsContractTest
```
