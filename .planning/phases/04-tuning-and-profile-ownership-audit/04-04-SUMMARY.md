---
phase: 04-tuning-and-profile-ownership-audit
plan: "04"
subsystem: tuning-profile-boundaries
tags: [tuning, benchmark, persistence, docs, hygiene]
requires:
  - phase: 04-tuning-and-profile-ownership-audit
    provides: 04-01 ownership matrix, 04-02 strict profile IO, and 04-03 runtime-derived costs
provides:
  - Benchmark CLI command persistence roles
  - Profile-read-only benchmark command tests
  - Tuning ownership and persistence boundary documentation
  - Final Phase 4 targeted verification evidence
affects: [phase-04, phase-05, tuning-cli, benchmark, docs]
tech-stack:
  added: []
  patterns: [command-kind persistence roles, benchmark read-only profile boundary]
key-files:
  created: []
  modified:
    - src/main/java/synaptik/app/TuningCli.java
    - src/test/java/synaptik/app/TuningCliParsingTest.java
    - src/test/java/BenchmarkSessionTest.java
    - src/main/java/tuning/ARCHITECTURE.md
    - src/main/java/tuning/PERSISTENCE.md
    - docs/architecture.md
    - .planning/ROADMAP.md
key-decisions:
  - "Benchmark command kinds are explicitly profile-read-only."
  - "Autotune, calibration, and the full local flow are the only profile-writing CLI command kinds."
  - "Benchmark reports remain explain artifacts rather than runtime sources of truth."
requirements-completed: [TUNE-01, TUNE-02, TUNE-03, TUNE-04]
duration: 12 min
completed: 2026-04-30
---

# Phase 4 Plan 04: Benchmark Profile Boundary And Docs

**Benchmark commands now have explicit read-only profile roles, and Phase 4 ownership docs are closed.**

## Performance

- **Duration:** 12 min
- **Started:** 2026-04-30T06:26:30Z
- **Completed:** 2026-04-30T06:39:00Z
- **Tasks:** 3
- **Files modified:** 7

## Accomplishments

- Added `TuningCli.CommandKind.writesProfileArtifacts()` so CLI command kinds expose whether they intentionally persist profile artifacts.
- Added parsing tests proving `benchmark-winner`, `benchmark-graph-space`, `benchmark.run --scenario graph-space`, and `HELP` are profile-read-only.
- Added parsing tests proving `FULL`, `CALIBRATION`, and `AUTOTUNE` are the profile-writing CLI command kinds.
- Added a `BenchmarkSessionTest` regression showing benchmark sessions run and produce reports without `PersistencePolicy` or tuning store record components.
- Documented the tuning knob ownership matrix with `Graph/workload-owned`, `Platform/dtype-owned`, `Obsolete`, `ACCELERATOR_BUFFER_MODE`, `METAL_SELECTION`, `MetalTransferModel`, and the `RuntimeConfig` cost boundary.
- Documented that benchmark commands are profile-read-only and that benchmark reports are explain artifacts, not runtime sources of truth.
- Updated the high-level architecture docs to point to tuning ownership and persistence boundaries.
- Updated the Phase 4 roadmap plan list with all four plan files and their final purpose statements.

## Changed Files By Ownership Area

### Ownership

- `src/main/java/tuning/ARCHITECTURE.md` - Adds the tuning knob ownership matrix and RuntimeConfig cost boundary.
- `.planning/ROADMAP.md` - Names all four Phase 4 plan files and their ownership/persistence responsibilities.

### Profile IO

- `src/main/java/tuning/PERSISTENCE.md` - Distinguishes profile-writing flows from benchmark read-only/report flows.

### Cost Model

- `docs/architecture.md` - Updates the accelerator cost note from Phase 3 deferral to Phase 4 RuntimeConfig-derived cost behavior.

### Benchmark And Docs

- `src/main/java/synaptik/app/TuningCli.java` - Adds command-kind profile persistence role metadata.
- `src/test/java/synaptik/app/TuningCliParsingTest.java` - Covers read-only benchmark commands and profile-writing autotune/calibration commands.
- `src/test/java/BenchmarkSessionTest.java` - Guards benchmark session independence from profile persistence policy.

## Task Commits

1. **Task 1: Add testable CLI profile persistence roles** - `9b26b5e`
2. **Task 2: Update ownership documentation and roadmap plan list** - `fb68c53`
3. **Task 3: Run final targeted verification and preserve local artifact hygiene** - this metadata commit

## Decisions Made

- Command-level write roles are owned by `TuningCli.CommandKind`, which keeps CLI persistence behavior testable without executing the flows.
- Benchmark session remains observational: it validates and measures supplied entries, producing reports without store/persistence policy coupling.
- The SDK `state.update-progress` handler is unreliable for this repository's STATE body format, so STATE and ROADMAP tracking were updated with scoped manual edits.

## Deviations from Plan

None - plan executed as written. The known SDK state handler limitation was handled by the plan's manual-update fallback.

## Issues Encountered

None.

## Verification

- `rg -n "writesProfileArtifacts" src/main/java/synaptik/app/TuningCli.java src/test/java/synaptik/app/TuningCliParsingTest.java` - PASS
- `rg -n "benchmarkCommandsAreProfileReadOnly|autotuneAndCalibrationCommandsWriteProfileArtifacts" src/test/java/synaptik/app/TuningCliParsingTest.java` - PASS
- `rg -n "new JsonFileBestProfileStore|new JsonFileTuningHistoryStore|PersistencePolicy" src/main/java/tuning/benchmark src/main/java/tuning/benchmark/report` - PASS, no profile persistence use found
- `./gradlew test --tests synaptik.app.TuningCliParsingTest --tests TuningStoreTest --tests BenchmarkSessionTest` - PASS
- `rg -n "Tuning knob ownership matrix|Graph/workload-owned|Platform/dtype-owned|ACCELERATOR_BUFFER_MODE|METAL_SELECTION|MetalTransferModel|Profile-derived accelerator costs enter through RuntimeConfig" src/main/java/tuning/ARCHITECTURE.md` - PASS
- `rg -n "Benchmark commands are profile-read-only|benchmark-winner|benchmark-graph-space|Autotune and calibration are the only profile-writing CLI flows|Benchmark reports are explain artifacts" src/main/java/tuning/PERSISTENCE.md` - PASS
- `rg -n "04-01-PLAN.md|04-02-PLAN.md|04-03-PLAN.md|04-04-PLAN.md" .planning/ROADMAP.md` - PASS
- `./gradlew classes` - PASS
- `./gradlew test --tests CalibrationCandidateOwnershipTest --tests GraphAutotuneCandidateSpaceTest --tests TuningKnobOwnershipTest --tests PlatformCalibrationSessionTest --tests ExecutionProfileIoTest --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests PreparedExecutionBuildTest --tests synaptik.app.TuningCliParsingTest --tests TuningStoreTest --tests BenchmarkSessionTest` - PASS

## Artifact Hygiene

`git status --short` showed only the known unstaged local profile churn under `profiles/platform/mac_os_x-aarch64-oracle_corporation-16c/tuning/abc/*` and the untracked `.planning/tmp/` scratch directory after verification. No `profiles/platform/.../tuning/abc/*` files and no `.planning/tmp/` files were staged.

## Self-Check: PASSED

- Benchmark command kinds are explicitly profile-read-only.
- Autotune, calibration, and full local flow command kinds are profile-writing.
- Docs distinguish graph autotune, platform calibration, strict profile IO, runtime-derived cost inputs, and benchmark/report ownership.
- `./gradlew classes` and the Phase 4 targeted test suite passed.
- Local profile artifacts and `.planning/tmp/` remained unstaged.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

Phase 4 execution is complete and ready for phase-level review, security, validation, and verification gates.

---
*Phase: 04-tuning-and-profile-ownership-audit*
*Completed: 2026-04-30*
