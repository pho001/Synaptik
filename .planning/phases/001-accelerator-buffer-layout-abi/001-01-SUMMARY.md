---
phase: 001-accelerator-buffer-layout-abi
plan: "01"
subsystem: accelerator-runtime
tags: [java, gradle, accelerator, buffer-abi, metal]

requires: []
provides:
  - Backend-neutral accelerator buffer layout descriptor and classifier
  - DeviceBufferBinding layout/access/native identity ABI
  - MetalBufferBinding migration to shared layout metadata
  - Append-only accelerator buffer reason-code expansion
affects: [phase-001-plan-02, phase-001-plan-03, metal-runtime, cuda-runtime]

tech-stack:
  added: []
  patterns:
    - Backend-neutral layout records under backend.accelerator.buffer
    - Backend-specific native handles remain under backend.metal.buffer
    - Metal logical metadata is read through binding.layout()

key-files:
  created:
    - src/main/java/backend/accelerator/buffer/AcceleratorBufferAccessMode.java
    - src/main/java/backend/accelerator/buffer/AcceleratorBufferLayout.java
    - src/main/java/backend/accelerator/buffer/AcceleratorBufferLayoutClass.java
    - src/main/java/backend/accelerator/buffer/AcceleratorBufferLayoutClassifier.java
    - src/test/java/backend/accelerator/buffer/AcceleratorBufferLayoutClassifierTest.java
  modified:
    - src/main/java/backend/accelerator/buffer/AcceleratorBufferReasonCode.java
    - src/main/java/backend/memory/DeviceBufferBinding.java
    - src/main/java/backend/metal/buffer/MetalBufferBinding.java
    - src/main/java/backend/metal/buffer/MetalBufferAllocator.java
    - src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java
    - src/main/java/backend/metal/buffer/MetalDeviceToCpuMaterializer.java
    - src/main/java/backend/metal/exec/PreparedMetalExecutable.java
    - src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java
    - src/test/java/backend/metal/buffer/MetalBufferBindingTest.java
    - src/test/java/backend/metal/buffer/MetalBufferAllocatorTest.java
    - src/test/java/graph/execution/ExecutionStateResidencyTest.java
    - src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java
    - src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java

key-decisions:
  - "Keep public Tensor API unchanged; derive accelerator layout metadata from existing tensor runtime facts."
  - "Keep native Metal handles backend-owned and expose only diagnostic native identity strings through DeviceBufferBinding."
  - "Keep AcceleratorBufferRequest and input/output decision signatures unchanged for plan 001-01."

patterns-established:
  - "AcceleratorBufferLayout defensively copies shape and strides and computes checked dtype byte lengths."
  - "MetalBufferBinding has no shortcut logical metadata accessors; call sites use binding.layout().*."

requirements-completed: [ABI-01, ABI-02, ABI-03, ABI-04]

duration: 6min
completed: 2026-04-29
---

# Phase 1 Plan 01: Accelerator Buffer Layout ABI Summary

**Backend-neutral accelerator layout metadata with Metal bindings migrated to shared layout/access/native identity contracts.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-29T19:06:28Z
- **Completed:** 2026-04-29T19:12:11Z
- **Tasks:** 3
- **Files modified:** 18

## Accomplishments

- Added `AcceleratorBufferLayout`, `AcceleratorBufferLayoutClass`, `AcceleratorBufferLayoutClassifier`, and `AcceleratorBufferAccessMode`.
- Extended `DeviceBufferBinding` with shared `layout()`, `accessMode()`, and `nativeHandleIdentity()` methods while keeping native handle types out of common code.
- Refactored `MetalBufferBinding` and Metal allocator/bridge/materializer/executable call sites so logical facts flow through `binding.layout().*`.
- Appended stable fallback reason codes without changing accelerator request or decision record signatures.

## Task Commits

1. **Task 1 RED:** `5100283` test(001-01): add failing accelerator layout classifier tests
2. **Task 1 GREEN:** `ef43177` feat(001-01): add accelerator buffer layout classifier
3. **Task 2:** `e679fa1` feat(001-01): extend accelerator buffer reason codes
4. **Task 3 RED:** `6a754cf` test(001-01): add failing binding ABI migration tests
5. **Task 3 GREEN:** `4beb9d9` feat(001-01): attach shared layout ABI to buffer bindings

## Files Created/Modified

- `src/main/java/backend/accelerator/buffer/AcceleratorBufferLayout.java` - Immutable shared layout descriptor with defensive arrays and checked byte length.
- `src/main/java/backend/accelerator/buffer/AcceleratorBufferLayoutClassifier.java` - Pure classifier for dense, view, strided, broadcast, and unsupported layout classes.
- `src/main/java/backend/memory/DeviceBufferBinding.java` - Shared runtime binding ABI for layout, access mode, logical bytes, native identity, and availability.
- `src/main/java/backend/metal/buffer/MetalBufferBinding.java` - Metal binding now composes shared layout metadata with backend-owned Metal handle/access state.
- `src/main/java/backend/metal/buffer/MetalBufferAllocator.java` - Creates layouts from tensors/output facts and materializes through `binding.layout()`.
- `src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java` - Validates Metal bindings through shared layout metadata.
- `src/main/java/backend/metal/buffer/MetalDeviceToCpuMaterializer.java` - Checks dtype/shape/count through `binding.layout()`.
- `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` - Rebuilds writable bindings without shortcut layout facts.
- `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java` - Validates buffer dtype through shared layout metadata.
- `src/test/java/backend/accelerator/buffer/AcceleratorBufferLayoutClassifierTest.java` - Coverage for all layout classes, defensive copies, dtype byte lengths, and tensor extraction.
- Metal/runtime tests - Updated fake bindings and Metal binding constructors to the canonical shared ABI.

## Decisions Made

- No public `Tensor` API changes were made.
- No compatibility constructors, overloads, adapters, or shortcut logical accessors were added to `MetalBufferBinding`.
- `AcceleratorBufferRequest`, `AcceleratorBufferInputDecision`, and `AcceleratorBufferOutputDecision` signatures were left unchanged for the planned 001-02 migration.
- `nativeHandleIdentity()` returns a diagnostic string containing backend id, owner, storage mode, and byte length without exposing FFM or Metal handle objects to common code.

## Deviations from Plan

None - plan executed exactly as written.

## TDD Notes

Task 1 and Task 3 followed RED/GREEN gates with failing test commits before implementation. Task 2 was an append-only enum taxonomy update with no test file in the plan; it was verified with `./gradlew classes` and the required grep checks.

## Verification

- `./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest` - passed
- `./gradlew classes` - passed
- `./gradlew test --tests backend.metal.buffer.MetalBufferBindingTest --tests graph.execution.ExecutionStateResidencyTest` - passed
- `./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests backend.metal.buffer.MetalBufferBindingTest --tests graph.execution.ExecutionStateResidencyTest` - passed
- `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest` - passed

## Known Stubs

None.

## Threat Flags

None - new security-relevant surface was covered by the plan threat model.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Plan 001-02 can now migrate accelerator request/decision layout metadata and later Metal/CUDA call sites against the shared ABI. The unrelated dirty profile tuning files and `.planning/tmp/` were intentionally left untouched.

## Self-Check: PASSED

- Created files exist on disk.
- Task commits `5100283`, `ef43177`, `e679fa1`, `6a754cf`, and `4beb9d9` exist in git history.

---
*Phase: 001-accelerator-buffer-layout-abi*
*Completed: 2026-04-29*
