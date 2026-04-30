# Phase 7: CUDA Buffer Execution And Materialization - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-04-30
**Phase:** 7-CUDA Buffer Execution And Materialization
**Areas discussed:** Native operation scope, Native ABI/resource ownership, Materialization boundaries, Adjacent CUDA handoff, Verification strategy

---

## Native operation scope

| Option | Description | Selected |
|--------|-------------|----------|
| Smallest dense FLOAT32 operation | Prove the buffer path with RELU or simple ADD before expanding operation coverage. | ✓ |
| Matmul/linear first | Start with heavier neural-network operations, closer to benchmark interest but higher native complexity. | |
| Broad operation subset | Implement many DAG node types immediately. | |

**User's choice:** Auto-selected recommended default: Smallest dense FLOAT32 operation.
**Notes:** This satisfies the Phase 7 requirement for a representative supported accelerator operation while keeping materialization and handoff in scope.

---

## Native ABI/resource ownership

| Option | Description | Selected |
|--------|-------------|----------|
| CUDA-owned backend package | Keep CUDA handles, access enums, allocators, materializers, and resource lifetimes under `backend.cuda.*`. | ✓ |
| Shared native buffer package | Move backend-specific native details into shared accelerator packages. | |
| Public Tensor device API | Expose CUDA residency directly through user-facing tensor APIs. | |

**User's choice:** Auto-selected recommended default: CUDA-owned backend package.
**Notes:** Carries forward the project decision that public tensors stay logical and shared accelerator records stay backend-neutral.

---

## Materialization boundaries

| Option | Description | Selected |
|--------|-------------|----------|
| ExecutionState materializer | Use `DeviceToCpuMaterializer` plus `ExecutionState.requireCpuReadable(...)` for graph-output and CPU-consumer reads. | ✓ |
| Direct tensor copy in CUDA executable | Copy CUDA bytes into output tensors directly from `PreparedCudaExecutable`. | |
| Defer materialization | Execute native buffers but leave CPU publication for a later phase. | |

**User's choice:** Auto-selected recommended default: ExecutionState materializer.
**Notes:** This keeps CUDA aligned with the Metal materialization contract and avoids publishing stale CPU storage.

---

## Adjacent CUDA handoff

| Option | Description | Selected |
|--------|-------------|----------|
| Narrow dense same-backend handoff | Reuse CUDA `DeviceBufferBinding` only for dense FLOAT32 same-backend compatible regions. | ✓ |
| General handoff across layouts/dtypes | Attempt broader handoff immediately. | |
| No handoff in Phase 7 | Materialize every CUDA region output to CPU before the next region. | |

**User's choice:** Auto-selected recommended default: Narrow dense same-backend handoff.
**Notes:** This directly targets `CUDA-05` without overclaiming unsupported layout/dtype coverage.

---

## Verification strategy

| Option | Description | Selected |
|--------|-------------|----------|
| Portable fake-bridge tests plus gated native tests | Verify Java policy/materialization/handoff portably and run native CUDA checks only when capability is present. | ✓ |
| Native-only CUDA tests | Require CUDA hardware/tooling for Phase 7 verification. | |
| Policy-only tests | Avoid native path tests entirely. | |

**User's choice:** Auto-selected recommended default: Portable fake-bridge tests plus gated native tests.
**Notes:** Local environments without CUDA must still pass the portable gate; CPU remains the correctness oracle.

---

## the agent's Discretion

- The exact first native operation is left to implementation planning. Prefer `RELU` or simple `ADD`; implement both only if it is low-cost after the first kernel exists.
- Class names and Java/native helper boundaries may follow existing Metal/CUDA conventions.

## Deferred Ideas

- CUDA trace/report parity and final docs belong to Phase 8.
- Broad CUDA operation coverage, additional dtypes, higher-rank ABI expansion, and profile/calibration updates are deferred.
