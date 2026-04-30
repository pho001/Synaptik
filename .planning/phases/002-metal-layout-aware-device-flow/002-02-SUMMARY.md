---
phase: 002-metal-layout-aware-device-flow
plan: "02"
subsystem: accelerator-runtime
tags: [java, gradle, metal, buffer-layout, materialization, native-abi]

requires:
  - phase: 002-metal-layout-aware-device-flow
    provides: Metal layout policy and binder preflight from 002-01
  - phase: 001-accelerator-buffer-layout-abi
    provides: Backend-neutral accelerator buffer layout metadata
provides:
  - Dense physical Metal output allocation for policy-approved logical-view layouts
  - Logical-view Metal device-to-CPU materialization with scatter readback
  - Bridge contract proof that no native layout ABI is required for this path
affects: [phase-002-metal-layout-aware-device-flow, metal-runtime, accelerator-materialization]

tech-stack:
  added: []
  patterns:
    - Policy-approved logical views use dense physical native bytes plus Java-owned logical layout metadata
    - Metal materializer support gates mirror allocator readback capability before claiming CPU visibility

key-files:
  created:
    - .planning/phases/002-metal-layout-aware-device-flow/002-02-SUMMARY.md
  modified:
    - src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java
    - src/main/java/backend/metal/buffer/MetalBufferAllocator.java
    - src/main/java/backend/metal/buffer/MetalDeviceToCpuMaterializer.java
    - src/main/java/backend/metal/bridge/MetalMpsGraphBridge.java
    - src/test/java/backend/metal/buffer/MetalBufferAllocatorTest.java
    - src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java
    - src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java

key-decisions:
  - "Use dense physical Metal buffers sized to logicalByteLength for legal logical-view outputs instead of adding native stride/offset ABI."
  - "Keep CPU-upload inputs dense contiguous only; existing compatible Metal device bindings can feed adjacent Metal work when policy-approved."
  - "Materializer support returns true only when binding layout matches the target tensor and the Metal layout policy accepts that layout class."

patterns-established:
  - "Logical-view readback first reads dense native bytes and scatters by target shape, strides, and storage offset."
  - "Bridge validation remains dtype/node/access focused; Java owns logical materialization semantics."

requirements-completed: [METAL-02, METAL-03, METAL-04]

duration: 6min
completed: 2026-04-30
---

# Phase 2 Plan 02: Dense Physical Logical-View Metal Flow Summary

**Metal now allocates dense physical buffers for legal logical-view outputs and materializes them through Java-owned layout scatter without native ABI changes.**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-30T02:54:57Z
- **Completed:** 2026-04-30T03:01:13Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments

- Enabled policy-approved `ZERO_OFFSET_VIEW`, `NON_ZERO_OFFSET_VIEW`, and `PERMUTED_OR_STRIDED_VIEW` outputs to allocate dense physical Metal buffers while preserving logical layout metadata.
- Reused compatible existing logical-view `MetalBufferBinding` inputs across adjacent Metal executables without CPU materialization.
- Added materializer gates and scatter readback for graph output, CPU consumer, and gradient publication reasons.
- Documented and tested that the existing shape-only buffer bridge ABI remains sufficient; no native layout symbol or stride/storage-offset parameter was added.

## Task Commits

1. **Task 1 RED: Logical-view buffer tests** - `c84187c` (test)
2. **Task 1 GREEN: Logical-view output allocation** - `27abf77` (feat)
3. **Task 2 RED: Logical-view materializer tests** - `22af75c` (test)
4. **Task 2 GREEN: Scatter materialization** - `8013b68` (feat)
5. **Task 3: Native ABI proof tests and bridge Javadoc** - `ff3a9d2` (test)
6. **Acceptance traceability: Binder logical-view comment** - `73b39fe` (docs)

## Files Created/Modified

- `src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java` - Accepts policy-approved logical-view output and existing-device input paths; documents `DENSE_PHYSICAL_LOGICAL_VIEW` allocation.
- `src/main/java/backend/metal/buffer/MetalBufferAllocator.java` - Allocates dense physical output bytes and scatters dense logical readback into destination storage.
- `src/main/java/backend/metal/buffer/MetalDeviceToCpuMaterializer.java` - Gates support through layout equality, availability, dtype, and `MetalLayoutPolicy.output(...)`.
- `src/main/java/backend/metal/bridge/MetalMpsGraphBridge.java` - Documents dense physical buffer bridge semantics and Java-owned logical materialization.
- `src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java` - Fake-bridge coverage for logical-view output allocation and adjacent device handoff.
- `src/test/java/backend/metal/buffer/MetalBufferAllocatorTest.java` - Materializer support and scatter readback coverage.
- `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java` - ABI proof text and bridge validation coverage.

## Decisions Made

- Kept native Metal ABI unchanged; Java supplies dense physical buffers and owns view/stride/storage-offset materialization.
- Preserved the direct readback path for dense contiguous zero-offset destinations and added scatter only for logical-view destinations.
- Rejected broadcast zero-stride and unsupported layouts before allocation/materialization support is claimed.

## Deviations from Plan

None - plan scope was executed as written.

## TDD Notes

- Task 1 RED/GREEN commits already existed at execution start and were verified instead of duplicated.
- Task 2 followed RED/GREEN: the new materializer tests failed against the old contiguous-only materializer, then passed after scatter support.
- Task 3 tests passed immediately because existing bridge validation was already layout-metadata agnostic after Task 1; the task still added proof coverage and Javadoc.

## Verification

- `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest` - passed
- `./gradlew test --tests backend.metal.buffer.MetalBufferAllocatorTest` - passed
- `./gradlew test --tests backend.metal.bridge.MetalMpsFfmBridgeTest` - passed
- `./gradlew test --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest` - passed
- `./gradlew classes` - passed
- Required `rg` acceptance checks for layout actions, rejected layout classes, scatter helper/rank gate, materializer gates, proof test text, bridge contract text, and absence of new native layout ABI symbols - passed

## Known Stubs

None.

## Threat Flags

None - security-relevant surfaces were covered by the plan threat model, and no new network endpoint, auth path, file access boundary, schema, or native ABI surface was introduced.

## Issues Encountered

- `gsd-sdk` is not available in this shell, so summary/state/roadmap/requirements updates were made manually.
- The worktree still contains the pre-existing unrelated profile tuning changes and untracked `.planning/tmp/`; they were not modified, staged, committed, reverted, or cleaned.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for `002-03-PLAN.md`: end-to-end Metal flow tests and docs can now assert dense physical logical-view execution, materialization reasons, trace visibility, and CPU parity on top of the unit-level allocator/materializer bridge contract.

## Self-Check: PASSED

- Summary file exists on disk.
- Task commits `c84187c`, `27abf77`, `22af75c`, `8013b68`, `ff3a9d2`, and `73b39fe` exist in git history.
- Key modified files exist on disk.
- Plan verification commands passed.

---
*Phase: 002-metal-layout-aware-device-flow*
*Completed: 2026-04-30*
