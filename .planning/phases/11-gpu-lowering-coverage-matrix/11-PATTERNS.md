# Phase 11 Pattern Map: GPU Lowering Coverage Matrix

## Purpose

Map the Phase 11 files to existing analogs so execution follows current source ownership and test style.

## New Or Modified Production Files

| Target file | Role | Closest analog | Pattern to follow |
|---|---|---|---|
| `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageStatus.java` | Shared coverage status enum | `src/main/java/backend/lowering/LoweringFamily.java` | Small enum with stable identifiers and no backend-specific state. |
| `src/main/java/backend/accelerator/lowering/GpuLoweringUnsupportedReason.java` | Stable reason code enum | `src/main/java/backend/accelerator/buffer/AcceleratorBufferReasonCode.java` | Stable source-level codes used by tests/docs. |
| `src/main/java/backend/accelerator/lowering/GpuLoweringOperationFamily.java` | Coverage matrix row grouping | `src/main/java/operations/Operation.java` | Keep taxonomy backend-neutral and semantic. |
| `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageEntry.java` | Immutable coverage row | `src/main/java/backend/accelerator/buffer/AcceleratorLayoutTransformDecision.java` | Java record with normalized null-safe fields. |
| `src/main/java/backend/accelerator/lowering/GpuLoweringCoverageMatrix.java` | Matrix source of truth | `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java` | Static helper consumed by both backend adapters and tests. |
| `src/main/java/backend/accelerator/lowering/AcceleratorSubgraphLowerer.java` | Shared DAG lowering | existing file | Add narrow `LOG_SOFTMAX` decomposition if implemented; do not duplicate backend-specific logic. |
| `src/main/java/backend/metal/lowering/MetalPartitionSupport.java` | Metal planner legality and reasons | existing file | Keep Metal dtype and SDPA semantic gates local while consuming shared coverage entries. |
| `src/main/java/backend/cuda/lowering/CudaGpuRegionLegalityAdapter.java` | CUDA planner legality and reasons | `src/main/java/backend/metal/lowering/MetalPartitionSupport.java` | Add `plannerUnsupportedReason(...)` symmetry and use shared coverage entries. |
| `docs/gpu-lowering-coverage.md` | Checked-in coverage matrix | `docs/metal-backend.md`, `docs/native-bridges-and-blas.md` | Human-readable matrix with statuses and reason codes that tests can grep. |
| `docs/native-bridges-and-blas.md` | Native ABI context | existing file | Link to coverage matrix and clarify no hidden CPU materialization. |
| `docs/development.md` | Verification commands | existing file | Add focused commands for Phase 11 lowering coverage. |

## New Or Modified Test Files

| Target file | Role | Closest analog | Pattern to follow |
|---|---|---|---|
| `src/test/java/backend/accelerator/lowering/GpuLoweringCoverageMatrixTest.java` | Matrix/source/docs drift guard | `src/test/java/backend/accelerator/buffer/AcceleratorLayoutTransformPlannerTest.java` | Pure unit tests over records and enums. |
| `src/test/java/backend/accelerator/lowering/AcceleratorSubgraphLowererTest.java` | Shared DAG lowering tests | `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java` | Build real tensors, snapshot compiled nodes, and inspect lowered DAG specs. |
| `src/test/java/backend/metal/lowering/MetalRegionLowererTest.java` | Metal selected/rejected planner coverage | existing file | Extend existing helper methods and assert exact unsupported reasons. |
| `src/test/java/backend/cuda/lowering/CudaRegionLowererTest.java` | CUDA selected/rejected planner coverage | existing file | Add real `CudaGpuRegionLegalityAdapter` candidate tests, not only manually assembled plans. |
| `src/test/java/PreparedExecutionBuildTest.java` | Backend selection selected/rejected flow evidence | existing file | Reuse backend assignment and prepare trace assertions. |
| `src/test/java/CompiledGraphTraceTest.java` | Trace closure | existing file | Assert reason metadata remains visible where current trace path supports it. |

## Implementation Boundaries

- Backend-neutral coverage contracts belong in `backend.accelerator.lowering`, not public `tensor`.
- Metal/CUDA native handles and capability checks remain backend-owned.
- CPU remains the correctness oracle; Phase 11 must not weaken CPU region support.
- Fused GPU compound execution remains Phase 12.
- Benchmark coverage ratios and regression gates remain Phase 13.

