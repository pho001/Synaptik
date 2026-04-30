# Phase 4: Tuning And Profile Ownership Audit - Pattern Map

**Created:** 2026-04-30

## Files To Modify Or Create

| Target | Role | Closest Existing Analog | Pattern To Follow |
|--------|------|-------------------------|-------------------|
| `src/main/java/tuning/ownership/TuningKnobOwner.java` | Ownership enum | `src/main/java/tuning/candidate/CandidateKind.java` | Small enum with stable all-caps values. |
| `src/main/java/tuning/ownership/TuningKnobOwnership.java` | Central ownership matrix | `src/main/java/tuning/calibration/family/CalibrationFamilyRegistry.java` | Static registry, immutable maps/sets, fail-fast validation helpers. |
| `src/test/java/TuningKnobOwnershipTest.java` | Matrix unit tests | `src/test/java/CalibrationCandidateOwnershipTest.java` | Iterate generated candidates and assert invalid ownership throws. |
| `src/main/java/tuning/candidate/graph/GraphRuntimePolicyVariant.java` | Graph knob assignment metadata | `src/main/java/tuning/calibration/runtime/RuntimeProfileCandidate.java` | Record carries `Map<String, String> knobAssignments`, normalized to `Map.of()`. |
| `src/main/java/tuning/candidate/graph/GraphAutotuneCandidateSpace.java` | Graph candidate enforcement | `src/main/java/tuning/calibration/DefaultPlatformCalibrationSession.java` | Validate candidate ownership before returning generated candidates. |
| `src/main/java/config/profile/PlatformRuntimeProfileIO.java` | Strict platform profile IO | `src/main/java/config/profile/ExecutionProfileIO.java` | Preserve existing tolerant API, add strict API and targeted tests. |
| `src/main/java/backend/accelerator/select/AcceleratorPlanCostModel.java` | Runtime-derived cost summaries | `src/main/java/graph/optimizer/partition/cost/AcceleratorPartitionScoreModel.java` | Keep static summary API, add runtime-aware overload/helper. |
| `src/main/java/synaptik/app/TuningCli.java` | CLI persistence boundary | `src/test/java/synaptik/app/TuningCliParsingTest.java` | Add small queryable command-kind property and assert it in parsing tests. |
| `src/main/java/tuning/ARCHITECTURE.md` and `src/main/java/tuning/PERSISTENCE.md` | Ownership docs | Existing tuning docs | Add exact ownership/read-only sections with source-of-truth wording. |

## Important Data Flow

1. Platform calibration creates `PlatformRuntimeProfile`.
2. `PlatformRuntimeProfile.toRuntimeConfig()` creates runtime defaults.
3. Graph autotune assembles explicit `ExecutionProfile` candidates from graph policy plus platform runtime.
4. Benchmark measures explicit `ExecutionProfile` entries and returns reports.
5. Best profile records store the measured graph winner but later rebase graph policy onto the current platform runtime profile.

## Constraints For Executors

- Read current source before editing; several Phase 3 cost/trace APIs already exist and should be reused.
- Keep legacy tolerant profile-loading APIs available unless a task explicitly replaces a call site with strict loading.
- Add tests before or alongside behavior changes for ownership and strict IO.
- Do not stage `.planning/tmp/` or `profiles/platform/.../tuning/abc/*`.
