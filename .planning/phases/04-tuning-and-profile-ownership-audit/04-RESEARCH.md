# Phase 4: Tuning And Profile Ownership Audit - Research

**Researched:** 2026-04-30
**Status:** Complete

## Research Question

What needs to be known to plan Phase 4 so graph autotune owns graph/workload policy, platform calibration owns hardware/dtype thresholds, profile IO rejects invalid accelerator profile fields, benchmark commands remain read-only, and the Phase 3 cost model can safely consume audited profile/calibration inputs?

## Findings

### Current Ownership Shape

- `src/main/java/config/profile/PlatformRuntimeProfile.java` already states the desired contract: platform calibration produces runtime thresholds, graph autotune consumes that profile as a frozen runtime input, and graph autotune varies graph policy.
- `src/main/java/tuning/candidate/graph/GraphAutotuneCandidateSpace.java` already keeps standard graph policy variants small and production-eligible. It intentionally adds `acceleratorBuffer=off` and `acceleratorBuffer=auto` variants using `AcceleratorRuntimeOverrides.bufferBindingMode(...)`.
- `src/main/java/tuning/candidate/graph/GraphRuntimePolicyVariant.java` documents the important exception: accelerator buffer binding is a graph/workload-specific runtime decision, so a graph candidate may carry a small runtime override while still sharing calibrated platform thresholds.
- `src/main/java/tuning/calibration/family/CalibrationFamilyRegistry.java` already contains a calibration ownership registry, including `METAL_SELECTION` as an explicit opt-in accelerator family for FLOAT32.
- `src/test/java/CalibrationCandidateOwnershipTest.java` already validates calibration candidates against their family-owned knobs. This is the right pattern to generalize into a phase-wide ownership matrix.

### Profile IO Risks

- `PlatformRuntimeProfileIO.loadOrDefault(...)` and `fromJsonOrDefault(...)` currently catch broad exceptions and return the fallback profile. That is compatible with legacy behavior but conflicts with TUNE-03 for invalid accelerator buffer/layout fields.
- The platform profile metadata already stores `plannerSchemaVersion` and `persistenceSchemaVersion`, but the loader does not enforce supported versions.
- Invalid enum values such as an unsupported `metalBufferBindingMode` currently flow through `findEnum(...)` and silently select the fallback enum. Phase 4 should add strict APIs while preserving existing defaulting APIs only for known legacy/missing fields.
- `ExecutionProfileIO.fromJsonOrDefault(...)` uses the same defaulting style for full execution profiles and is used by `JsonFileBestProfileStore`. The first strictness target should be platform runtime profiles, because Phase 4 scope explicitly names platform calibration/profile ownership. Full execution profile strict loading can be added only where needed for best-profile promotion safety.

### Cost Model Integration

- Phase 3 introduced static `AcceleratorPartitionScoreModel.MaterializationCostSummary` and named `StaticCostPreset` values.
- `AcceleratorPlanCostModel.decide(...)` currently receives only `PartitionPlan` and `RuntimeConfig`. That is enough to consume calibrated runtime thresholds after assembly because `PlatformRuntimeProfile.toRuntimeConfig()` carries accelerator minimum work and CPU materialization thresholds into `RuntimeConfig`.
- The safe Phase 4 cost update should not read profile files directly from the cost model. It should derive cost factors only from the `RuntimeConfig` already supplied to prepare/backend selection.
- The cost model can keep Phase 3 compatibility by leaving `summarize(PartitionPlan)` as conservative/static and adding `summarize(PartitionPlan, RuntimeConfig)` or a helper record for profile-derived factors. Tests should prove runtime thresholds influence summaries without bypassing CPU safeguards.

### Benchmark And Persistence Boundary

- `BenchmarkRequest` already documents that benchmark is observational and does not persist profiles, best records, histories, calibration artifacts, or reports. The benchmark session only returns a `BenchmarkReport`.
- `TuningCli.runWinnerBenchmark(...)` and `runGraphSpaceBenchmark(...)` print reports and do not call stores. `runAutotune(...)` creates a `PersistencePolicy` with best-profile/history paths, and `runCalibration(...)` delegates to calibration save logic.
- `JsonFileBenchmarkReportStore` exists for explicit report persistence. That is acceptable because reports are explain artifacts, not profile sources of truth. TUNE-04 should focus on preventing benchmark commands from writing profile/best/history/calibration artifacts.
- A small command-kind persistence classification in `TuningCli` would make the boundary testable: `CALIBRATION`, `AUTOTUNE`, and `FULL` can write profiles; `BENCHMARK_WINNER` and `BENCHMARK_GRAPH_SPACE` must be read-only for profile artifacts.

## Recommended Plan Shape

1. Establish an explicit ownership matrix and enforce it for calibration and graph autotune candidates.
2. Harden `PlatformRuntimeProfileIO` strict loading for schema/version and invalid accelerator buffer fields.
3. Wire audited runtime/profile-derived cost factors into `AcceleratorPlanCostModel` through `RuntimeConfig`, not file reads.
4. Lock benchmark read-only profile behavior in CLI/tests and update tuning docs.

## Validation Architecture

### Test Targets

- `CalibrationCandidateOwnershipTest` and a new graph ownership test should prove all candidate-space knob mutations are classified and owned.
- `PlatformCalibrationSessionTest` or a new `PlatformRuntimeProfileIoTest` should prove strict platform profile IO rejects unsupported schema versions and invalid accelerator buffer modes while preserving legacy missing-field behavior where intentional.
- `AcceleratorPartitionScoreModelTest` and `PreparedExecutionBuildTest` should prove profile-derived cost factors influence summaries and still preserve minimum-work and CPU fallback safeguards.
- `TuningCliParsingTest` should prove benchmark command kinds are read-only for profile artifacts while calibration/autotune/full are allowed to write profile results.
- Documentation verification should use `rg` for exact sections and terms.

### Commands

- Quick ownership/IO command:
  `./gradlew test --tests CalibrationCandidateOwnershipTest --tests GraphAutotuneCandidateSpaceTest --tests PlatformCalibrationSessionTest --tests ExecutionProfileIoTest`
- Cost-model command:
  `./gradlew test --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests PreparedExecutionBuildTest`
- Persistence/read-only command:
  `./gradlew test --tests synaptik.app.TuningCliParsingTest --tests TuningStoreTest --tests BenchmarkSessionTest`
- Full Phase 4 targeted suite:
  `./gradlew test --tests CalibrationCandidateOwnershipTest --tests GraphAutotuneCandidateSpaceTest --tests PlatformCalibrationSessionTest --tests ExecutionProfileIoTest --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests PreparedExecutionBuildTest --tests synaptik.app.TuningCliParsingTest --tests TuningStoreTest --tests BenchmarkSessionTest`

## Risks And Constraints

- Do not move public `Tensor` APIs or expose device residency to users.
- Do not make benchmark reports an execution source of truth.
- Do not commit local `profiles/platform/.../tuning/abc/*` churn unless explicitly updating canonical profiles.
- Keep graph autotune production candidate space small. Research-only graph variants must stay explicit and non-production-eligible.
- Keep Metal/CUDA concepts backend-neutral where possible; Metal-specific fields should remain clearly named.

## RESEARCH COMPLETE
