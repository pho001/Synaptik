---
phase: 001-accelerator-buffer-layout-abi
plan: "02"
subsystem: accelerator-runtime
tags: [java, gradle, accelerator, buffer-abi, metal, cuda]

requires:
  - phase: 001-accelerator-buffer-layout-abi
    provides: Shared accelerator buffer layout/access ABI from plan 001-01
provides:
  - Layout-aware AcceleratorBufferRequest input/output contracts
  - Per-input/per-output buffer decisions carrying AcceleratorBufferLayout
  - Metal buffer preflight consuming shared layout metadata with conservative fallback
  - Metal allocator/materializer comparisons based on binding.layout()
  - CUDA required-buffer unavailable policy aligned to shared reason taxonomy
affects: [phase-001-plan-03, metal-runtime, cuda-runtime, accelerator-buffer-policy]

tech-stack:
  added: []
  patterns:
    - Canonical accelerator buffer requests carry node ids, dtypes, and matching layout lists
    - Metal native buffer preflight rejects non-dense layouts with layoutClass/storageOffset/strides diagnostics
    - CUDA keeps native buffers unavailable while reporting shared required-unavailable reason codes

key-files:
  created: []
  modified:
    - src/main/java/backend/accelerator/buffer/AcceleratorBufferRequest.java
    - src/main/java/backend/accelerator/buffer/AcceleratorBufferInputDecision.java
    - src/main/java/backend/accelerator/buffer/AcceleratorBufferOutputDecision.java
    - src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java
    - src/main/java/backend/metal/buffer/MetalBufferAllocator.java
    - src/main/java/backend/metal/buffer/MetalDeviceToCpuMaterializer.java
    - src/main/java/backend/metal/exec/PreparedMetalExecutable.java
    - src/main/java/backend/cuda/exec/PreparedCudaExecutable.java
    - src/main/java/backend/cuda/bridge/CudaGraphBridge.java
    - src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java
    - src/test/java/backend/metal/buffer/MetalBufferAllocatorTest.java
    - src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java
    - src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java

key-decisions:
  - "Keep Metal view/strided/native execution conservative in Phase 1; classify and report layout classes without executing them natively."
  - "Use NATIVE_BUFFER_ABI_UNAVAILABLE for Metal bridges that lack the native buffer ABI, not BACKEND_BUFFER_NOT_IMPLEMENTED."
  - "Use REQUIRED_BUFFER_EXECUTION_UNAVAILABLE only for CUDA REQUIRE mode; optional CUDA modes keep BACKEND_BUFFER_NOT_IMPLEMENTED fallback taxonomy."

patterns-established:
  - "AcceleratorBufferRequest layout lists must strictly match their node-id lists after null normalization."
  - "Metal allocator and materializer compare layout metadata through binding.layout(), including strides and storage offset."

requirements-completed: [ABI-01, ABI-02, ABI-03, ABI-04]

duration: 10min
completed: 2026-04-29
---

# Phase 1 Plan 02: Metal/CUDA Seam Adaptation Summary

**Metal and CUDA accelerator seams now consume the shared layout ABI while keeping Phase 1 native execution conservative.**

## Performance

- **Duration:** 10 min
- **Started:** 2026-04-29T19:14:27Z
- **Completed:** 2026-04-29T19:23:57Z
- **Tasks:** 3
- **Files modified:** 13

## Accomplishments

- Extended `AcceleratorBufferRequest`, `AcceleratorBufferInputDecision`, and `AcceleratorBufferOutputDecision` with canonical `AcceleratorBufferLayout` fields.
- Wired `PreparedMetalExecutable` and `MetalAcceleratorBufferBinder` to classify runtime/resolved layouts, reject non-dense layouts with stable diagnostics, and report native Metal ABI absence explicitly.
- Routed Metal output allocation and CPU materialization through `binding.layout()` metadata.
- Updated CUDA required-buffer mode to use `REQUIRED_BUFFER_EXECUTION_UNAVAILABLE` while leaving native CUDA buffer bindings unavailable.

## Task Commits

1. **Task 1 RED:** `755f325` test(001-02): add failing Metal layout seam tests
2. **Task 1 GREEN:** `303df18` feat(001-02): route Metal buffer decisions through layouts
3. **Task 2 RED:** `5f36649` test(001-02): add failing Metal allocator layout tests
4. **Task 2 GREEN:** `4c0f12c` feat(001-02): allocate Metal buffers from layouts
5. **Task 3 RED:** `c84cac3` test(001-02): add failing CUDA required-buffer policy test
6. **Task 3 GREEN:** `474053e` feat(001-02): align CUDA required buffer policy

## Files Created/Modified

- `src/main/java/backend/accelerator/buffer/AcceleratorBufferRequest.java` - Adds strict input/output layout list contracts.
- `src/main/java/backend/accelerator/buffer/AcceleratorBufferInputDecision.java` - Carries per-input layout metadata.
- `src/main/java/backend/accelerator/buffer/AcceleratorBufferOutputDecision.java` - Carries per-output layout metadata.
- `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` - Builds layout-aware buffer requests and preserves binding layouts after writes.
- `src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java` - Uses shared layout metadata for preflight, binding compatibility, and native ABI unavailable decisions.
- `src/main/java/backend/metal/buffer/MetalBufferAllocator.java` - Allocates outputs from `AcceleratorBufferLayout` and validates readback layout compatibility.
- `src/main/java/backend/metal/buffer/MetalDeviceToCpuMaterializer.java` - Checks dtype, shape, strides, storage offset, and element count before materialization.
- `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java` - Uses shared required-unavailable reason code in CUDA REQUIRE mode.
- `src/main/java/backend/cuda/bridge/CudaGraphBridge.java` - Documents the shared layout ABI requirement for future buffer support.
- Metal/CUDA targeted tests - Updated for canonical signatures, layout-class diagnostics, allocator layout routing, and CUDA reason taxonomy.

## Decisions Made

- No compatibility constructors, overloads, adapters, or legacy request/decision entry points were added.
- Metal still rejects `ZERO_OFFSET_VIEW`, `NON_ZERO_OFFSET_VIEW`, `PERMUTED_OR_STRIDED_VIEW`, `BROADCAST_ZERO_STRIDE_VIEW`, and `UNSUPPORTED` for native buffer execution in this phase.
- The Metal FFM bridge test was updated as a required allocator signature call site, even though the plan's key test list focused on allocator/executable/CUDA policy tests.

## Deviations from Plan

None - plan executed exactly as written.

## TDD Notes

All three tasks followed RED/GREEN gates. Each RED commit introduced failing tests before implementation, and each GREEN commit made the targeted verification pass.

## Verification

- `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest` - passed
- `./gradlew test --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.buffer.MetalBufferBindingTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest` - passed
- `./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` - passed
- `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.buffer.MetalBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` - passed
- `./gradlew classes` - passed

## Known Stubs

None.

## Threat Flags

None - new security-relevant surface was covered by the plan threat model.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 001-03 can add the remaining focused trace/report documentation and test closure on top of canonical layout-aware Metal/CUDA seams. The unrelated dirty profile tuning files and `.planning/tmp/` were intentionally left untouched.

## Self-Check: PASSED

- Summary file exists on disk.
- Task commits `755f325`, `303df18`, `5f36649`, `4c0f12c`, `c84cac3`, and `474053e` exist in git history.

---
*Phase: 001-accelerator-buffer-layout-abi*
*Completed: 2026-04-29*
