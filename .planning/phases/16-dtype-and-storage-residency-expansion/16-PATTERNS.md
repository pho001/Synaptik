# Phase 16 Pattern Map: DType And Storage Residency Expansion

**Phase:** 16 - DType And Storage Residency Expansion
**Date:** 2026-05-01
**Status:** Complete

## Pattern Mapping Complete

## Files To Modify Or Extend

| Planned file | Role | Closest existing analog | Notes |
|---|---|---|---|
| `src/main/java/graph/execution/RuntimeMemoryBinder.java` | Runtime region slot binding | Existing `FLOAT64`/`FLOAT32` maps in the same file | Extend typed slots to `short[]`, `int[]`, and `byte[]` without changing binding policy. |
| `src/test/java/graph/execution/RuntimeMemoryBinderTest.java` | Runtime binding regression tests | Existing workspace-sensitive binding test | Add focused tests for `BFLOAT16`, `INT32`, `BOOL`, alias/view skip, and `FLOAT32` preservation. |
| `src/main/java/backend/accelerator/residency/AcceleratorDTypeResidencyDecision.java` | Immutable dtype residency decision record | `GpuCoverageGap.java`, `AcceleratorBufferInputDecision.java` | New internal record if diagnostics need a shared contract. |
| `src/main/java/backend/accelerator/residency/AcceleratorDTypeResidencyPolicy.java` | Backend-neutral dtype residency helper | `AcceleratorLayoutTransformPlanner.java`, `GpuLoweringCoverageMatrix.java` | Role-specific decisions for external input, internal value, output, and compute. |
| `src/test/java/backend/accelerator/residency/AcceleratorDTypeResidencyPolicyTest.java` | Policy tests | `GpuLoweringCoverageMatrixTest.java` | Assert exact dtype/backend/role support and rejection reason strings. |
| `src/main/java/backend/metal/MetalMpsCapabilities.java` | Metal dtype source of truth | Existing `supportsComputeDType`, `supportsExternalInputDType` | Keep current conservative contract and expose stable diagnostics through policy/tests. |
| `src/test/java/backend/metal/MetalMpsCapabilitiesTest.java` | Metal capability regression tests | Existing backend lowering tests | Add if absent; assert `BOOL` predicate input is distinct from compute/output support. |
| `src/main/java/backend/cuda/buffer/CudaBufferAllocator.java` | CUDA buffer allocation guard | Existing `validateDenseFloat32(...)` | Preserve dense `FLOAT32` allocation guard until explicit non-F32 allocation support exists. |
| `src/test/java/backend/cuda/buffer/CudaBufferAllocatorTest.java` | CUDA dtype guard tests | CUDA binder/region lowerer tests | Add or extend to assert unsupported non-F32 residency decisions are explicit. |
| `src/main/java/backend/accelerator/lowering/GpuLoweredRegionManifestRenderer.java` | Compact manifest rendering | Existing manifest renderer | Include dtype residency assumptions/rejections if implementation records them there. |
| `src/main/java/tuning/benchmark/report/GpuCoverageSummary.java` | Coverage summary | Existing materialization and storage residency counters | Add dtype-specific counters only from existing trace data. |
| `src/test/java/GpuCoverageSummaryTest.java` | Coverage tests | Existing synthetic trace fixtures | Prove dtype-specific reason visibility without native hardware. |
| `docs/gpu-lowering-coverage.md` | Developer docs | Existing lowering coverage docs | Document dtype residency vs native dtype compute support. |
| `docs/compute-flow.md` | Execution docs | Existing runtime residency docs | Explain true materialization boundaries for dtype-resident values. |
| `.planning/phases/16-dtype-and-storage-residency-expansion/16-*-SUMMARY.md` | Execution evidence | Prior phase summaries | Record verification and profile artifact hygiene per wave. |

## Reusable Code Patterns

### Runtime Slot Maps

`RuntimeMemoryBinder` already uses one `Map<Integer, array>` per dtype. Extend that direct pattern instead of introducing a generic boxed storage abstraction on the CPU hot path.

### Immutable Diagnostic Records

Report and accelerator records use Java records with compact constructors and null normalization. Any new dtype residency decision record should follow this style.

### Stable Reason Codes Plus Detail Strings

`GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE` is the stable category. Backend-specific detail strings should carry exact role and backend information without creating free-form-only diagnostics.

### Synthetic Trace Fixtures

Trace/report tests should use synthetic Java fixtures where native Metal/CUDA hardware is unavailable. Native execution tests stay optional and capability-gated.

## Landmines

- Do not add public `Tensor` device residency APIs.
- Do not change native Metal/CUDA ABI unless a task explicitly proves it is required.
- Do not count BOOL predicate input support as BOOL compute/output support.
- Do not relax CUDA dense `FLOAT32` allocation guards without explicit allocation/materialization tests.
- Do not hide unsupported non-F32 dtype cases behind generic CPU replay.
- Do not commit `profiles/platform/.../tuning/abc/*`.
