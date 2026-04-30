---
phase: 002-metal-layout-aware-device-flow
plan: "01"
subsystem: accelerator-runtime
tags: [java, gradle, metal, accelerator, buffer-layout, preflight]

requires:
  - phase: 001-accelerator-buffer-layout-abi
    provides: Backend-neutral accelerator buffer layout classes, reason codes, and Metal binding metadata
provides:
  - MetalLayoutPolicy with direct dense, dense physical logical-view, and rejected layout actions
  - Policy-driven Metal binder input/output preflight decisions
  - Fake-bridge regression coverage for legal device-owned inputs and unsupported layout rejection
affects: [phase-002-metal-layout-aware-device-flow, metal-runtime, accelerator-buffer-preflight]

tech-stack:
  added: []
  patterns:
    - Package-private Metal policy class consumed by Java-side binder preflight
    - Fake-bridge tests assert AcceleratorBufferDecision paths, reason codes, and policy reason fragments

key-files:
  created:
    - src/main/java/backend/metal/buffer/MetalLayoutPolicy.java
    - .planning/phases/002-metal-layout-aware-device-flow/002-01-SUMMARY.md
  modified:
    - src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java
    - src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java

key-decisions:
  - "Keep MetalLayoutPolicy package-private under backend.metal.buffer so the shared accelerator ABI stays backend-neutral."
  - "Allow existing Metal device bindings with legal logical-view layouts after policy classification and binding compatibility checks, while CPU-upload inputs remain dense-only."
  - "Keep dense physical logical-view outputs conservative in plan 002-01; they are classified and traceable but fall back until allocator/materializer support lands in plan 002-02."

patterns-established:
  - "Policy reasons include policyAction, layoutClass, shape, storageOffset, and strides for every non-blank decision reason."
  - "Unsupported broadcast and unsupported output layouts are rejected before native execution in both AUTO and REQUIRE buffer modes."

requirements-completed: [METAL-01, METAL-03]

duration: 5min
completed: 2026-04-30
---

# Phase 2 Plan 01: Metal Layout Policy And Binder Preflight Summary

**Metal buffer preflight now classifies direct dense, dense physical logical-view, and rejected layouts with stable fake-bridge diagnostics.**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-30T02:36:30Z
- **Completed:** 2026-04-30T02:41:32Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Added `MetalLayoutPolicy` with exact `DIRECT_DENSE_BUFFER`, `DENSE_PHYSICAL_LOGICAL_VIEW`, and `REJECT` actions.
- Replaced Metal binder dense-only layout preflight with policy calls for CPU uploads, existing device inputs, and outputs.
- Preserved conservative CPU-upload and output fallback behavior while allowing compatible existing Metal logical-view inputs to stay device-owned.
- Locked broadcast/unsupported layout rejection in optional and required buffer modes before native execution.

## Task Commits

1. **Task 1: Add MetalLayoutPolicy with explicit layout actions** - `c689ae7` (test), `67b8b94` (feat)
2. **Task 2: Use layout policy in Metal binder preflight decisions** - `466e2c7` (test), `c1a9c5f` (feat)
3. **Task 3: Lock unsupported-layout rejection and required-mode behavior** - `3a771f2` (test)

## Files Created/Modified

- `src/main/java/backend/metal/buffer/MetalLayoutPolicy.java` - Metal-specific layout policy and reason text contract.
- `src/main/java/backend/metal/buffer/MetalAcceleratorBufferBinder.java` - Policy-driven input/output decisions and removal of the old dense-only helper.
- `src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java` - Fake-bridge tests for policy classification, existing-device view inputs, conservative logical-view output fallback, and unsupported-layout rejection.

## Decisions Made

- Kept the new policy Metal-local and package-private; no public `Tensor` API, common accelerator ABI, or native Metal ABI changed.
- Existing Metal device bindings can pass logical-view policy classification only after `incompatibleBindingReason(...)` validates dtype, shape, strides, storage offset, logical element count, availability, and access.
- Dense physical logical-view outputs remain visible but conservative in this plan because allocator/materializer support is intentionally deferred to `002-02-PLAN.md`.

## Deviations from Plan

None - plan scope was executed as written.

## TDD Notes

- Task 1 and Task 2 followed RED/GREEN commits.
- Task 3's new regression tests passed immediately because Task 2's binder policy implementation already supplied the required unsupported-layout behavior. No production code change was needed for Task 3.

## Verification

- `./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest` - passed
- `./gradlew classes` - passed
- Required `rg` acceptance checks for policy actions, policy methods, binder policy calls, removed dense-only helper, bridge ABI reason text, required test names, and rejection reason assertions - passed

## Known Stubs

None.

## Threat Flags

None - this plan changed local Java preflight policy and tests only; it introduced no new network endpoint, auth path, file access boundary, schema, or native ABI surface.

## Issues Encountered

- `gsd-sdk` is not available in the current shell, matching the existing `STATE.md` note. Summary/state/roadmap/requirements updates were made directly.
- The worktree still contains the pre-existing unrelated profile tuning changes and untracked `.planning/tmp/`; they were not modified, staged, committed, reverted, or cleaned.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Ready for `002-02-PLAN.md`: allocator and materializer work can consume the new `DENSE_PHYSICAL_LOGICAL_VIEW` output classification without changing native ABI symbols.

## Self-Check: PASSED

- Summary file exists on disk.
- Task commits `c689ae7`, `67b8b94`, `466e2c7`, `c1a9c5f`, and `3a771f2` exist in git history.
- Key created/modified files exist on disk.
- Plan verification commands passed.

---
*Phase: 002-metal-layout-aware-device-flow*
*Completed: 2026-04-30*
