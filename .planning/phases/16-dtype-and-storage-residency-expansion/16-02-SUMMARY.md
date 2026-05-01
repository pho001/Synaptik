---
phase: 16-dtype-and-storage-residency-expansion
plan: "02"
status: complete
requirements-completed: [GPUSTORAGE-01, GPUSTORAGE-02]
completed: 2026-05-01
---

# Phase 16 Plan 02: Accelerator DType Residency Decisions Summary

Added a backend-neutral dtype residency decision contract for Metal/CUDA diagnostics without widening native dtype compute.

## Accelerator dtype residency decisions

- Added `AcceleratorDTypeResidencyDecision` and `AcceleratorDTypeResidencyPolicy`.
- Decisions separate `residentRepresentable`, `nativeInputLegal`, `nativeOutputLegal`, and `nativeComputeLegal`.
- Metal remains conservative: `FLOAT32` compute/output and `BOOL` only for predicate-style external input.
- CUDA dense native buffer execution remains `FLOAT32` only.
- Unsupported native Metal/CUDA dtype roles carry `GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE` plus stable backend/role/dtype detail.

## Verification

| Command | Result |
|---------|--------|
| `./gradlew test --tests backend.accelerator.residency.AcceleratorDTypeResidencyPolicyTest --tests backend.metal.MetalMpsCapabilitiesTest --tests backend.cuda.buffer.CudaBufferAllocatorTest` | Passed |

## Requirement Coverage

- `GPUSTORAGE-01`: Dtype residency decisions represent `BFLOAT16`, `INT32`, and `BOOL` metadata without native compute claims.
- `GPUSTORAGE-02`: Metal/CUDA unsupported dtype paths are capability-gated and produce stable backend-specific reasons.

## Guardrail

dtype residency is not native arithmetic support

## Deviations from Plan

None - plan executed exactly as written.

## Hygiene

profiles/platform/.../tuning/abc/* remained unstaged

## Self-Check: PASSED
