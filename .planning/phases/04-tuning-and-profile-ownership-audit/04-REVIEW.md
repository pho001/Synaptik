---
phase: 04-tuning-and-profile-ownership-audit
status: clean
depth: standard
files_reviewed: 25
findings:
  critical: 0
  warning: 0
  info: 0
  total: 0
reviewed: 2026-04-30
---

# Phase 4 Code Review

## Scope

- `src/main/java/backend/accelerator/select/AcceleratorPlanCostModel.java`
- `src/main/java/backend/accelerator/select/ProfileDerivedAcceleratorCostFactors.java`
- `src/main/java/config/profile/PlatformRuntimeProfileIO.java`
- `src/main/java/synaptik/app/TuningCli.java`
- `src/main/java/tuning/ARCHITECTURE.md`
- `src/main/java/tuning/PERSISTENCE.md`
- `src/main/java/tuning/benchmark/BenchmarkRequest.java`
- `src/main/java/tuning/calibration/family/CalibrationFamilyRegistry.java`
- `src/main/java/tuning/calibration/run/CalibrationRunner.java`
- `src/main/java/tuning/candidate/graph/GraphAutotuneCandidateSpace.java`
- `src/main/java/tuning/candidate/graph/GraphPolicyMutators.java`
- `src/main/java/tuning/candidate/graph/GraphRuntimePolicyVariant.java`
- `src/main/java/tuning/ownership/TuningKnobOwner.java`
- `src/main/java/tuning/ownership/TuningKnobOwnership.java`
- `docs/architecture.md`
- `src/test/java/BenchmarkSessionTest.java`
- `src/test/java/CalibrationCandidateOwnershipTest.java`
- `src/test/java/ExecutionProfileIoTest.java`
- `src/test/java/GraphAutotuneCandidateSpaceTest.java`
- `src/test/java/PlatformCalibrationSessionTest.java`
- `src/test/java/PreparedExecutionBuildTest.java`
- `src/test/java/TuningKnobOwnershipTest.java`
- `src/test/java/graph/optimizer/partition/cost/AcceleratorPartitionScoreModelTest.java`
- `src/test/java/synaptik/app/TuningCliParsingTest.java`
- `src/test/java/tuning/candidate/graph/GraphAutotuneCandidateSpaceTest.java`

## Findings

No open issues found after review.

## Resolved During Review

- `fix(04-04): align tuning ownership docs` corrected the new tuning architecture docs so `ACCELERATOR_BUFFER_MODE` and `MetalTransferModel` are documented as graph/workload-owned, matching `TuningKnobOwnership` and its tests. `METAL_SELECTION` remains platform/dtype-owned accelerator opt-in calibration.

## Verification

- `./gradlew classes` - passed during 04-04 execution.
- `./gradlew test --tests CalibrationCandidateOwnershipTest --tests GraphAutotuneCandidateSpaceTest --tests TuningKnobOwnershipTest --tests PlatformCalibrationSessionTest --tests ExecutionProfileIoTest --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests PreparedExecutionBuildTest --tests synaptik.app.TuningCliParsingTest --tests TuningStoreTest --tests BenchmarkSessionTest` - passed during 04-04 execution.
- `rg -n "Graph/workload-owned|Platform/dtype-owned|ACCELERATOR_BUFFER_MODE|METAL_SELECTION|MetalTransferModel|Profile-derived accelerator costs enter through RuntimeConfig" src/main/java/tuning/ARCHITECTURE.md` - passed after the review fix.

## Residual Risk

- `./gradlew metalTest` was not run because Phase 4 changed tuning ownership, profile IO, cost selection, CLI metadata, docs, and tests rather than native Metal code.
- Existing local tuning profile files under `profiles/platform/.../tuning/abc/*` and `.planning/tmp/` scratch files remain outside this phase's staged changes.
