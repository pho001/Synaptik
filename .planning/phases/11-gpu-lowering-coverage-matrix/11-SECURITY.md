---
phase: 11
slug: gpu-lowering-coverage-matrix
status: verified
threats_open: 0
asvs_level: 1
created: 2026-04-30
---

# Phase 11 - Security

Per-phase security contract: threat register, accepted risks, and audit trail for GPU lowering coverage matrix work.

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Planner to shared coverage matrix | Metal and CUDA legality adapters consume backend-neutral lowering classifications before backend-specific gates. | Operation type, backend, support status, unsupported reason. |
| Shared matrix to docs/tests | Source-level coverage claims are mirrored in checked-in documentation and drift tests. | Supported/fallback/unsupported claims and reason-code text. |
| Layout/view planning to CUDA execution | CUDA may only accept non-dense compute when metadata-only view propagation or dense materialization makes the consumer layout legal. | Shape, stride, storage offset, dense/materialized buffer state. |
| Lowerer to native DAG ABI | Semantic operations are lowered into existing accelerator DAG primitives unless a native ABI opcode is deliberately added. | DAG node types, scalar attributes, output node ids. |
| Prepare trace to users/reports | Selected and rejected GPU candidates must expose stable reasons so GPU exits are auditable. | Backend selection decisions, rejection reasons, selected node ids. |
| Working tree to committed artifacts | Local tuning/profile files must stay out of committed phase closure artifacts. | Local profile JSON/JSONL files under `profiles/platform/.../tuning/abc/*`. |

---

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-11-01 | Tampering / Information Disclosure | Coverage matrix | mitigate | `GpuLoweringCoverageEntry` rejects supported rows with non-supported reasons and rejects non-supported rows with `SUPPORTED`; `GpuLoweringCoverageMatrixTest.nonSupportedEntriesHaveStableReasons` verifies concrete reason codes. | closed |
| T-11-02 | Tampering / Repudiation | Coverage docs | mitigate | `GpuLoweringCoverageMatrixTest.docsMatrixListsRequiredStatusesAndReasons` and `docs/gpu-lowering-coverage.md` pin required families, statuses, and reason-code phrases. | closed |
| T-11-03 | Tampering / Repudiation | Metal/CUDA planner allowlists | mitigate | `MetalPartitionSupport` and `CudaGpuRegionLegalityAdapter` consult `GpuLoweringCoverageMatrix.entryFor(...)`; backend tests assert matrix-backed support and stable rejection fragments. | closed |
| T-11-04 | Elevation of Privilege / Tampering | CUDA layout legality | mitigate | CUDA `plannerUnsupportedReason(...)` returns `UNSUPPORTED_LAYOUT` for direct non-dense compute until view propagation or dense materialization makes layout legal; CUDA layout tests cover this behavior. | closed |
| T-11-05 | Tampering / Denial of Service | Native DAG ABI | mitigate | `LOG_SOFTMAX` lowers as `SOFTMAX` followed by `LOG` in `AcceleratorSubgraphLowerer`; docs state no native ABI opcode was added. | closed |
| T-11-06 | Repudiation | Prepared backend selection | mitigate | `PreparedExecutionBuildTest` asserts selected Metal/CUDA `linear/matmul -> LOG_SOFTMAX` regions and visible rejection for unsupported loss/reduction candidates. | closed |
| T-11-07 | Repudiation / Information Disclosure | Trace evidence | mitigate | `CompiledGraphTraceTest` asserts selected supported `LOG_SOFTMAX` regions and rejected reduction traces contain `UNSUPPORTED_OPERATION`. | closed |
| T-11-08 | Tampering | Local profile artifacts | mitigate | `11-04-SUMMARY.md` records `git status --short` evidence that only pre-existing `profiles/platform/.../tuning/abc/*` files remain unstaged; current audit did not stage those files. | closed |

---

## Evidence

| Threat Ref | Evidence |
|------------|----------|
| T-11-01 | `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageEntry.java`, `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java`, `src/test/java/backend/accelerator/lowering/GpuLoweringCoverageMatrixTest.java` |
| T-11-02 | `docs/gpu-lowering-coverage.md`, `docs/native-bridges-and-blas.md`, `src/test/java/backend/accelerator/lowering/GpuLoweringCoverageMatrixTest.java` |
| T-11-03 | `src/main/java/backend/metal/lowering/MetalPartitionSupport.java`, `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java`, `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java`, `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java` |
| T-11-04 | `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java`, `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java`, `src/test/java/backend/cuda/exec/CudaLayoutTransformDeviceFlowTest.java` |
| T-11-05 | `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java`, `src/test/java/backend/accelerator/lowering/AcceleratorSubgraphLowererTest.java`, `docs/gpu-lowering-coverage.md` |
| T-11-06 | `src/test/java/PreparedExecutionBuildTest.java` |
| T-11-07 | `src/test/java/CompiledGraphTraceTest.java` |
| T-11-08 | `.planning/phases/11-gpu-lowering-coverage-matrix/11-04-SUMMARY.md`, `git status --short` |

---

## Accepted Risks Log

No accepted risks.

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-04-30 | 8 | 8 | 0 | Codex |

### Security Audit 2026-04-30

| Metric | Count |
|--------|-------|
| Threats found | 8 |
| Closed | 8 |
| Open | 0 |

Verification commands:

- `./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest` - PASS
- `./gradlew test --tests backend.metal.MetalLayoutAwareDeviceFlowTest --tests backend.cuda.exec.CudaLayoutTransformDeviceFlowTest` - PASS

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-04-30
