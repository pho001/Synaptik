---
phase: 41
status: passed
verified: 2026-05-02
threats_reviewed: 8
threats_open: 0
requirements:
  - CUDADTYPE-01
  - CUDADTYPE-02
  - CUDAINDEX-01
---

# Phase 41 Security Verification

## Result

PASSED. Phase 41 does not introduce public device tensors, unsafe native CUDA execution claims, or hidden fallback paths. The implementation keeps CUDA dtype/layout/index support conservative and report-visible.

## Threat Review

| Threat | Status | Mitigation Evidence |
|--------|--------|---------------------|
| T41-01: `INT32` index residency mistaken for generic CUDA arithmetic | Closed | `CudaDTypeRolePolicy` separates `INDEX_INPUT` from `COMPUTE_INPUT` / `COMPUTE_OUTPUT`; tests assert compute/output rejection for `INT32`. |
| T41-02: `BFLOAT16` residency reported as native CUDA BF16 compute | Closed | `RESIDENCY_ONLY_NOT_COMPUTE` reason and capability report tests keep BF16 residency separate from compute/output. |
| T41-03: Metadata-only CUDA views used directly by dense-only native compute | Closed | `CudaAcceleratorBufferBinder` rejects non-dense read bindings as materialization-required for native buffer compute. |
| T41-04: CUDA broadcast repair counted as supported because Metal supports it | Closed | `CudaDeviceLayoutMaterializer` rejects broadcast materialization with `CUDA_LAYOUT_BROADCAST_UNSUPPORTED`; tests lock this behavior. |
| T41-05: CUDA gather/take support claimed without CPU-compatible bounds behavior | Closed | `CudaPartitionSupport` checks static INT32 bounds before legal candidates reach `CAPABILITY_MISSING`. |
| T41-06: Forward index support confused with scatter/index-gradient support | Closed | `GATHER_GRAD`, `TAKE_ALONG_AXIS_GRAD`, and `SCATTER_ADD` remain `UNSUPPORTED_DUPLICATE_INDEX` in matrix/tests. |
| T41-07: Coverage gate treats residency-only dtype as native CUDA compute | Closed | Hot-path targets now tag CUDA dtype/index evidence separately; docs and tests distinguish role from compute/output support. |
| T41-08: Local benchmark/profile artifacts staged as phase evidence | Closed | `git status --short` confirmed profile artifacts remain unstaged; only source/docs/planning files were committed. |

## Security-Relevant Verification

```bash
./gradlew test --tests backend.cuda.CudaDTypeRolePolicyTest --tests backend.cuda.bridge.CudaCapabilityReportTest
./gradlew test --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest
./gradlew test --tests backend.cuda.buffer.CudaDeviceLayoutMaterializerTest --tests backend.cuda.exec.CudaLayoutTransformDeviceFlowTest
./gradlew test --tests backend.cuda.lowering.CudaRegionLowererTest
./gradlew test --tests backend.accelerator.lowering.GpuBackendParityReportTest --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest
./gradlew test --tests GpuCoverageTriageReportTest --tests GpuHotPathCoverageTargetsTest --tests SourceTreeHygieneTest
```

All relevant commands passed during Phase 41 execution.

## Residual Risk

- CUDA native forward gather/take execution is still absent and must not be treated as supported.
- CUDA BF16/BOOL/INT32 native compute/output remains unsupported beyond role-specific residency evidence.
- Optional native CUDA execution remains capability-gated by local CUDA hardware/toolchain availability.
