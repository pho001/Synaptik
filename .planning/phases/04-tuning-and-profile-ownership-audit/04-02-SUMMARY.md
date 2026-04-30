---
phase: 04-tuning-and-profile-ownership-audit
plan: "02"
subsystem: tuning
tags: [profile-io, calibration, cli, persistence]
requires:
  - phase: 04-tuning-and-profile-ownership-audit
    provides: 04-01 ownership matrix and candidate-space enforcement
provides:
  - Strict platform runtime profile loading APIs
  - Schema/version and accelerator buffer validation for platform profiles
  - Strict CLI/calibration loading for persisted platform profiles
affects: [phase-04, tuning-cli, platform-calibration, profile-persistence]
tech-stack:
  added: []
  patterns: [strict loader beside tolerant legacy loader, path-wrapped profile IO errors]
key-files:
  created: []
  modified:
    - src/main/java/config/profile/PlatformRuntimeProfileIO.java
    - src/main/java/tuning/calibration/run/CalibrationRunner.java
    - src/main/java/synaptik/app/TuningCli.java
    - src/test/java/PlatformCalibrationSessionTest.java
    - src/test/java/ExecutionProfileIoTest.java
key-decisions:
  - "Platform runtime profiles now have strict loading APIs while existing tolerant loading remains available for compatibility."
  - "CLI and calibration paths that consume persisted platform runtime profiles use strict loading."
patterns-established:
  - "Strict profile IO wraps file-path context around parser errors while retaining the offending key name."
  - "ExecutionProfileIO remains tolerant for legacy full execution profiles until a dedicated strict full-profile contract exists."
requirements-completed: [TUNE-03]
duration: 3 min
completed: 2026-04-30
---

# Phase 4 Plan 02: Strict Platform Profile IO Summary

**Strict platform runtime profile loading with schema and accelerator buffer validation**

## Performance

- **Duration:** 3 min
- **Started:** 2026-04-30T06:19:41Z
- **Completed:** 2026-04-30T06:22:27Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Added `loadStrict(...)` and `fromJsonStrict(...)` to `PlatformRuntimeProfileIO`.
- Added schema constants and strict rejection for unsupported `plannerSchemaVersion` / `persistenceSchemaVersion`.
- Added strict validation for invalid `cudaBufferBindingMode`, `openclBufferBindingMode`, and `metalBufferBindingMode`.
- Switched `TuningCli` and `CalibrationRunner` persisted platform profile loading to strict mode.
- Documented full `ExecutionProfileIO` tolerant behavior with a regression test.

## Task Commits

1. **Task 1: Add strict platform runtime profile IO APIs and tests** - `d106602`
2. **Task 2: Use strict loading in CLI/calibration paths that require persisted platform profiles** - `9ee0a82`

## Files Created/Modified

- `src/main/java/config/profile/PlatformRuntimeProfileIO.java` - Strict APIs, supported schema constants, and buffer enum validation.
- `src/main/java/tuning/calibration/run/CalibrationRunner.java` - Uses strict loading for latest platform seed profiles.
- `src/main/java/synaptik/app/TuningCli.java` - Uses strict loading for calibration profiles.
- `src/test/java/PlatformCalibrationSessionTest.java` - Strict schema/buffer validation and legacy missing-buffer default tests.
- `src/test/java/ExecutionProfileIoTest.java` - Documents unchanged tolerant full execution profile loading.

## Decisions Made

- Kept `loadOrDefault(...)` and `fromJsonOrDefault(...)` unchanged for legacy tolerant call sites.
- Scoped strictness to platform runtime profiles for Phase 4; full execution profile strictness remains a separate future contract.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Verification

- `./gradlew test --tests PlatformCalibrationSessionTest`
- `./gradlew test --tests PlatformCalibrationSessionTest --tests ExecutionProfileIoTest --tests synaptik.app.TuningCliParsingTest`

## Self-Check: PASSED

- `PlatformRuntimeProfileIO.java` contains `loadStrict`, `fromJsonStrict`, supported schema constants, and key-specific buffer validation.
- `TuningCli` and `CalibrationRunner` use `PlatformRuntimeProfileIO.loadStrict(...)`.
- Targeted Gradle profile IO suite passes.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Wave 3 can consume audited runtime/profile inputs through `RuntimeConfig` without relying on silent persisted profile fallback behavior.

---
*Phase: 04-tuning-and-profile-ownership-audit*
*Completed: 2026-04-30*
