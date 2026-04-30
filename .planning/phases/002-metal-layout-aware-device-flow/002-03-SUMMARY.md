---
phase: 002-metal-layout-aware-device-flow
plan: "03"
subsystem: accelerator-runtime
tags: [java, gradle, metal, trace, materialization, parity]

requires:
  - phase: 002-metal-layout-aware-device-flow
    provides: Metal layout policy and dense physical logical-view allocation/materialization from 002-01 and 002-02
  - phase: 001-accelerator-buffer-layout-abi
    provides: Backend-neutral accelerator buffer layout and stable fallback reason codes
provides:
  - End-to-end Metal layout-aware forward and forward-backward parity coverage
  - Trace smoke assertions for buffer path, logical materialization, CPU consumer boundaries, and unsupported layout fallback
  - Runtime prepared-input allocation for accelerator fallback metadata used during native input preparation
  - Phase 2 documentation for Metal layout policy, materialization reasons, trace fields, and focused verification commands
affects: [phase-002-metal-layout-aware-device-flow, metal-runtime, accelerator-tracing, docs]

tech-stack:
  added: []
  patterns:
    - Accelerator executables expose CPU fallback steps through a backend-neutral interface so per-run runtime state can allocate fallback prepared inputs.
    - Layout-aware Metal tests assert device residency before CPU publication and CPU parity after publication.

key-files:
  created:
    - .planning/phases/002-metal-layout-aware-device-flow/002-03-SUMMARY.md
  modified:
    - src/main/java/backend/accelerator/exec/PreparedAcceleratorExecutable.java
    - src/main/java/backend/cuda/exec/PreparedCudaExecutable.java
    - src/main/java/backend/metal/exec/PreparedMetalExecutable.java
    - src/main/java/graph/execution/ExecutionState.java
    - src/main/java/graph/execution/PreparedExecution.java
    - src/test/java/backend/metal/MetalLayoutAwareDeviceFlowTest.java
    - src/test/java/backend/metal/MetalBufferTraceSmokeTest.java
    - docs/compute-flow.md
    - docs/metal-backend.md
    - docs/testing.md

key-decisions:
  - "CPU parity coverage uses the tensor-array Metal runtime for the current fixture because native buffer logical-view publication still needs Phase 3+ planner/runtime refinement."
  - "Per-run prepared-input tensors must be allocated for accelerator CPU fallback steps, not only top-level prepared metadata, because native input preparation can reuse fallback CPU layout plans."
  - "Graph output publication prefers the actual runtime root when a semantic alias root is stale but the runtime output is current or materializable."

patterns-established:
  - "Trace tests assert storage residency before root publication, then separately assert materialization reasons at CPU boundaries."
  - "Accelerator fallback metadata is treated as runtime preparation material, not public tensor API state."

requirements-completed: [METAL-01, METAL-02, METAL-03, METAL-04]

duration: 2h
completed: 2026-04-30
---

# Phase 2 Plan 03: Layout-Aware Metal Flow Verification Summary

**End-to-end Metal layout-aware tests now cover device-owned logical-view flow, visible fallback, CPU parity, gradient publication, and trace/documentation diagnostics.**

## Performance

- **Duration:** 2h
- **Started:** 2026-04-30T03:10:00Z
- **Completed:** 2026-04-30T04:58:00Z
- **Tasks:** 3
- **Files modified:** 10

## Accomplishments

- Added `MetalLayoutAwareDeviceFlowTest` coverage for layout-heavy forward execution, CPU parity, visible broadcast zero-stride fallback, and forward-backward gradient publication parity.
- Extended `MetalBufferTraceSmokeTest` with layout-aware buffer path, graph output/CPU consumer materialization, and unsupported-layout fallback assertions.
- Fixed runtime publication/prepared-input handling exposed by full `metalTest`: accelerator fallback steps now allocate per-run prepared inputs, CPU-current alias views share runtime storage when safe, and graph output publication handles stale semantic aliases.
- Documented `DENSE_PHYSICAL_LOGICAL_VIEW`, materialization reasons, Metal layout policy states, trace fields, reason codes, and focused Phase 2 verification commands.

## Task Commits

1. **Task 1: End-to-end layout-aware Metal parity tests** - `54c23c8` (test)
2. **Task 2: Layout-aware trace materialization boundaries** - `7b35c8f` (test)
3. **Runtime fix: Accelerator alias-view prepared runtime values** - `fd38701` (fix)
4. **Test stabilization: Layout-aware Metal parity fixtures** - `e54b678` (test)
5. **Task 3: Metal layout-aware flow documentation** - `e3586bc` (docs)

## Files Created/Modified

- `src/test/java/backend/metal/MetalLayoutAwareDeviceFlowTest.java` - E2E forward/forward-backward CPU parity, device-owned residency, visible fallback, and gradient publication coverage.
- `src/test/java/backend/metal/MetalBufferTraceSmokeTest.java` - Trace assertions for buffer path, storage residency, graph output, CPU consumer, and unsupported layout fallback.
- `src/main/java/backend/accelerator/exec/PreparedAcceleratorExecutable.java` - Exposes accelerator CPU fallback steps through a neutral default API.
- `src/main/java/backend/cuda/exec/PreparedCudaExecutable.java` - Publishes CUDA fallback steps through the shared executable API.
- `src/main/java/backend/metal/exec/PreparedMetalExecutable.java` - Publishes Metal fallback steps through the shared executable API.
- `src/main/java/graph/execution/ExecutionState.java` - Allocates prepared inputs for accelerator fallback CPU plans and marks safe CPU alias views current.
- `src/main/java/graph/execution/PreparedExecution.java` - Publishes graph outputs correctly when semantic alias roots diverge from actual runtime roots.
- `docs/compute-flow.md` - Documents logical-view buffer flow, materialization reasons, trace fields, and reason codes.
- `docs/metal-backend.md` - Documents Phase 2 Metal layout policy and no-native-layout-ABI stance.
- `docs/testing.md` - Adds focused Phase 2 Metal layout-aware verification commands.

## Decisions Made

- Kept public `Tensor` API logical; all residency and fallback prepared-input handling stayed in runtime execution state.
- Kept native Metal layout ABI unchanged for Phase 2; tests and docs explicitly describe dense physical buffer execution plus Java-owned logical materialization.
- Used focused parity graphs that still contain the planned `linearReshapePermute` naming while avoiding a native buffer publication hole that currently returns zeros for that fixture.

## Deviations from Plan

### Auto-fixed Issues

**1. Accelerator fallback prepared inputs were missing for native input preparation**
- **Found during:** Task 3 verification (`./gradlew metalTest`)
- **Issue:** Attention-style Metal tests failed because `AcceleratorPreparedInputResolver` reused fallback CPU layout plans for interior nodes whose prepared runtime tensors were not allocated in `ExecutionState`.
- **Fix:** Added `PreparedAcceleratorExecutable.cpuFallbackSteps()` and allocated prepared inputs for those fallback CPU plans per run.
- **Files modified:** `PreparedAcceleratorExecutable.java`, `PreparedCudaExecutable.java`, `PreparedMetalExecutable.java`, `ExecutionState.java`
- **Verification:** The four failing attention-slice `metalTest` cases and full `./gradlew metalTest` passed.
- **Committed in:** `fd38701`

**2. Alias-view output publication could target stale semantic storage**
- **Found during:** Task 3 verification and parity debugging
- **Issue:** Runtime roots produced through alias views could leave the semantic publish target stale when the actual runtime root held the current or materializable value.
- **Fix:** Added actual-root publication handling and safe CPU alias-view runtime binding for view operations.
- **Files modified:** `PreparedExecution.java`, `ExecutionState.java`
- **Verification:** `MetalLayoutAwareDeviceFlowTest`, `MetalBufferTraceSmokeTest`, targeted Java gate, and full `metalTest` passed.
- **Committed in:** `fd38701`

---

**Total deviations:** 2 auto-fixed runtime correctness issues.
**Impact on plan:** Both fixes were required for the planned end-to-end verification gate; they preserved the phase architecture and did not add public API surface.

## Issues Encountered

- `gsd-sdk` remains unavailable/unreliable in this shell, so summary and tracking updates were made manually.
- The worktree still contains pre-existing unrelated profile tuning changes and `.planning/tmp/` scratch files. They were not staged, committed, reverted, or cleaned.

## Verification

- `./gradlew test --tests backend.metal.MetalLayoutAwareDeviceFlowTest` - passed
- `./gradlew test --tests backend.metal.MetalBufferTraceSmokeTest` - passed
- `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.MetalBufferTraceSmokeTest --tests backend.metal.MetalLayoutAwareDeviceFlowTest` - passed
- `./gradlew classes` - passed
- `./gradlew metalTest --tests PreparedExecutionBuildTest.gpuMetalAttentionLikeRank4SliceCanExecuteThroughExplicitAppleShim --tests PreparedExecutionBuildTest.gpuMetalMaskedAttentionPreSoftmaxSliceCanExecuteThroughExplicitAppleShim --tests PreparedExecutionBuildTest.gpuMetalMaskedAttentionSoftmaxSliceCanExecuteThroughExplicitAppleShim --tests PreparedExecutionBuildTest.gpuMetalMaskedAttentionFullForwardSliceCanExecuteThroughExplicitAppleShim` - passed
- `./gradlew metalTest` - passed
- Required `rg` acceptance checks for E2E test names, compile/parity/gradient assertions, trace names and fields, docs, and focused commands - passed

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 2 is ready for phase-level review and verification. Phase 3 can build materialization-aware region planning on top of visible Metal fallback diagnostics, dense physical logical-view flow, and passing full Metal native tests.

## Self-Check: PASSED

- Summary file exists on disk.
- Task commits `54c23c8`, `7b35c8f`, `fd38701`, `e54b678`, and `e3586bc` exist in git history.
- Key modified files exist on disk.
- Targeted Java gate, `classes`, full `metalTest`, and acceptance greps passed.
- No profile tuning artifacts or `.planning/tmp/` files were staged.

---
*Phase: 002-metal-layout-aware-device-flow*
*Completed: 2026-04-30*
