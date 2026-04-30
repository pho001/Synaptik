---
phase: 3
slug: materialization-aware-region-planning
status: complete
created: 2026-04-30
---

# Phase 3: Materialization-Aware Region Planning - Research

## Research Goal

Answer: what has to be known to plan Phase 3 well?

Phase 3 should make `PART` and prepare-time backend selection prefer profitable device-owned accelerator regions without hiding CPU materialization costs or weakening CPU execution. The main implementation risk is split ownership: compile-time partition planning already has structural and transfer-aware scoring, while prepare-time selection currently accepts accelerator plans mostly by availability and minimum work.

## Phase Requirements

- `PLAN-01`: score CPU materialization, upload/download, tensor-array fallback, layout fallback, dispatch overhead, and estimated compute work.
- `PLAN-02`: prefer longer profitable device-owned regions over short accelerator islands when boundaries are reduced.
- `PLAN-03`: reject or split accelerator regions when materialization/layout costs erase compute benefit.
- `PLAN-04`: keep CPU natural regions and CPU fused execution available and competitive.

## Current Architecture

### Compile-Time Region Planning

- `src/main/java/graph/optimizer/partition/PartitionIntentRule.java` resolves planning jobs for accelerator and CPU regions. In auto/offload mode it can schedule accelerator jobs from CPU graphs with `PartitionSourcePolicy.CPU_OR_TARGET_BACKEND`, then adds CPU natural regions when CPU nodes are present and CPU region policy is not `OFF`.
- `src/main/java/graph/optimizer/partition/ScoredCandidatePartitionPlanner.java` explores candidate regions, scores structural candidates, and chooses the best lowered accelerator candidate. It already prefers larger candidates on score ties, which matches the user decision that ambiguous accelerator candidates may win after legality gates pass.
- `src/main/java/graph/optimizer/partition/GreedyMaxRegionPartitionPlanner.java` deterministically grows maximum legal regions. It is useful as a conservative baseline but does not have score breakdown visibility.
- `src/main/java/graph/optimizer/partition/cost/AcceleratorPartitionScoreModel.java` already has `CandidateMetrics`, `TransferMetrics`, `TransferPolicy`, and `PlannerPolicy`. Current transfer scoring covers input bytes, output bytes, and avoided intermediate bytes for Metal scored planning, but not dispatch overhead, layout fallback class, tensor-array fallback, boundary count, or named score summaries.
- `src/main/java/config/optimizer/PartitionConfig.java` carries generic score weights and `metalTransferModel`.
- `src/main/java/config/optimizer/MetalTransferModel.java` already provides named presets (`CONSERVATIVE`, `MEASURED`, `AGGRESSIVE`) with internal constants. This fits Phase 3's requirement to expose presets without profile-derived raw weights.

### Prepare-Time Backend Selection

- `src/main/java/backend/select/DefaultBackendSelectionPolicy.java` filters candidate plans by compatibility, runtime enablement, runtime availability, and `AcceleratorPlanCostModel.decide(...)`. Its trace records only selected/rejected, reason, backend, node ids, and estimated work.
- `src/main/java/backend/accelerator/select/AcceleratorPlanCostModel.java` currently rejects missing/non-positive work and plans below `RuntimeConfig.accelerator().forBackend(...).minimumEstimatedWork()`. It does not account for plan boundary size, fallback mode, dispatch overhead, or layout/materialization pressure.
- `src/main/java/config/runtime/AcceleratorBackendConfig.java` and `AcceleratorBufferConfig.java` hold runtime thresholds and buffer policy. Phase 3 should not write profile-derived costs into these configs; Phase 4 owns profile/calibration cost updates.

### Trace And Report Surfaces

- `src/main/java/graph/execution/trace/PartitionDecisionTrace.java` carries planner strategy, target, seed, reason, selected/structural node ids, op types, estimated work, selected score, structural score, explored candidates, and budget status. It has room for richer cost summaries but no top rejected finalist list.
- `src/main/java/graph/execution/trace/PartitionCompileTrace.java` aggregates compile-time partition decisions.
- `src/main/java/graph/execution/trace/BackendSelectionDecisionTrace.java` and `BackendSelectionTrace.java` carry prepare-time selected/rejected decisions but lack score breakdowns and finalist summaries.
- `src/main/java/graph/execution/trace/CompileTrace.java`, `PrepareTrace.java`, and `ExecutionTrace.java` already separate compile, prepare, and run trace concerns. Phase 3 should keep planner cost data in compile/prepare traces and not expand every runtime step trace.
- `src/main/java/tuning/benchmark/report/AcceleratorTraceSummary.java`, `TextBenchmarkReportRenderer.java`, and `JsonBenchmarkReportRenderer.java` already expose accelerator path, bytes, copy times, CPU fallback, and CPU materialization counts. They should add compact compile/prepare planning summaries instead of duplicating all candidate records.
- `src/test/java/BenchmarkSessionTest.java` already tests text/json renderer output for accelerator transfer diagnostics and CPU materialization trace fields.

### CPU Safeguards

- `src/main/java/graph/optimizer/partition/CpuNaturalExecutionRegionPlanner.java` and `src/test/java/graph/optimizer/partition/CpuNaturalExecutionRegionPlannerTest.java` prove CPU natural regions remain available and optimized into CPU execution units.
- `src/main/java/graph/optimizer/region/DefaultRegionOptimizer.java` and `src/test/java/OptimizerFuseTest.java` cover CPU elementwise fusion behavior.
- `src/test/java/PreparedExecutionBuildTest.java` covers accelerator offload selection, backend selection traces, and CPU fused/BLAS prepare behavior.
- `src/test/java/BFloat16BlasDispatchTest.java` and CPU planner tests protect BLAS dispatch. Phase 3 should add focused assertions that accelerator offload does not steal BLAS/fused CPU hot paths when static cost is not clearly better.

## Implementation Options

### Option A: Extend Existing Score Model And Traces In Place

Add backend-neutral cost summary records to `AcceleratorPartitionScoreModel`, compute them in `ScoredCandidatePartitionPlanner`, carry them through `PartitionDecisionTrace`, and let `DefaultBackendSelectionPolicy` reuse the summary or compute a prepare-time summary from `PartitionPlan`.

Pros:
- Reuses current partition scoring and trace architecture.
- Keeps public `Tensor` API untouched.
- Lets execution stay focused on compile/prepare artifacts.

Cons:
- `PartitionDecisionTrace` and benchmark renderers will need constructor/test updates.
- Prepare-time selection may need a small helper to avoid duplicating score math.

Recommendation: use this option.

### Option B: New Planner/Selection Cost Package

Introduce a separate package such as `backend.accelerator.cost` with a reusable cost estimator consumed by both `PART` and backend selection. Existing score model calls into it.

Pros:
- Cleaner long-term ownership if Phase 4 wires profile/calibration costs.
- Easier to test prepare-time cost decisions without graph planner setup.

Cons:
- Larger refactor in Phase 3.
- Higher chance of breaking existing planner tests.

Recommendation: do only the minimal reusable helper if needed. Do not move all scoring now.

### Option C: Runtime-Only Cost Gate

Leave `PART` mostly structural and make prepare-time selection reject costly accelerator regions.

Pros:
- Smaller compile-time change.

Cons:
- Does not satisfy `PLAN-02`: the planner itself would still prefer short islands in some graphs.
- Trace would explain rejection after compile but not why candidate regions were shaped.

Recommendation: do not use as the primary approach.

## Recommended Plan Shape

1. Add static materialization-aware cost summaries and finalist tracking to scored partition planning.
2. Apply the same summary vocabulary to prepare-time backend selection and benchmark reports.
3. Add CPU safeguard tests and documentation/state updates, including a Phase 4 deferral for profile/calibration-derived cost model updates.

## Validation Architecture

### Test Infrastructure

- Framework: JUnit 5 through Gradle.
- Quick commands:
  - `./gradlew test --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests PreparedExecutionBuildTest`
  - `./gradlew test --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest`
- Full targeted commands:
  - `./gradlew classes`
  - `./gradlew test --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests PreparedExecutionBuildTest --tests BenchmarkSessionTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest`
  - `./gradlew metalTest` only when native Metal execution paths are touched.

### Required Test Coverage

- Score model tests prove:
  - boundary count and input/output bytes reduce score,
  - avoided intermediate bytes increase score,
  - dispatch overhead penalizes tiny accelerator islands,
  - layout/tensor-array fallback penalties can reject or demote a region,
  - named presets resolve to internal constants and no profile-derived weights are required.
- Planner tests prove:
  - a longer legal accelerator region beats a shorter island when it avoids CPU materialization,
  - top rejected finalists are captured with score summaries and reason codes,
  - unsupported or costly candidates are rejected/split with stable reason strings.
- Backend selection tests prove:
  - selection trace includes selected candidate and top rejected finalists,
  - static cost rejection happens before execution,
  - fallback and runtime-unavailable reasons remain visible.
- CPU safeguard tests prove:
  - CPU natural regions remain planned for CPU graphs,
  - CPU fused execution remains available,
  - BLAS/fused CPU paths are not displaced by accelerator offload when static cost does not clearly win.
- Benchmark renderer tests prove:
  - text and JSON reports include compact planner/backend cost summaries,
  - runtime step traces stay focused on execution metadata.

### Manual Verification

No manual UAT is required for planning. Native Metal smoke tests may be environment-dependent; use `./gradlew metalTest` when native paths changed and record skips if the local shim is unavailable.

## Risks And Mitigations

- Risk: cost math becomes profile-driven too early. Mitigation: only named presets and internal constants in Phase 3; add explicit Phase 4 notes for profile/calibration-derived cost updates.
- Risk: planner trace becomes too verbose. Mitigation: selected candidate plus a bounded top rejected finalist list only.
- Risk: CPU hot paths regress. Mitigation: add CPU natural/fusion/BLAS guard tests in the Phase 3 plan and keep CPU alternatives intact.
- Risk: compile-time and prepare-time cost models drift. Mitigation: use the same summary record names and reason codes in both layers even if each layer computes its own available fields.

## Files Likely To Change

- `src/main/java/graph/optimizer/partition/cost/AcceleratorPartitionScoreModel.java`
- `src/main/java/graph/optimizer/partition/ScoredCandidatePartitionPlanner.java`
- `src/main/java/graph/execution/trace/PartitionDecisionTrace.java`
- `src/main/java/graph/execution/trace/BackendSelectionDecisionTrace.java`
- `src/main/java/backend/accelerator/select/AcceleratorPlanCostModel.java`
- `src/main/java/backend/select/DefaultBackendSelectionPolicy.java`
- `src/main/java/tuning/benchmark/report/AcceleratorTraceSummary.java`
- `src/main/java/tuning/benchmark/report/TextBenchmarkReportRenderer.java`
- `src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java`
- `src/test/java/graph/optimizer/partition/cost/AcceleratorPartitionScoreModelTest.java`
- `src/test/java/PreparedExecutionBuildTest.java`
- `src/test/java/BenchmarkSessionTest.java`
- `src/test/java/graph/optimizer/partition/CpuNaturalExecutionRegionPlannerTest.java`
- `src/test/java/OptimizerFuseTest.java`
- `src/test/java/BFloat16BlasDispatchTest.java`

## Deferred To Phase 4

Phase 4 must wire reliable profile/calibration-derived costs into the cost model after tuning/profile ownership is audited. Phase 3 should leave stable extension points and trace fields, but selection must ignore profile-derived speed evidence.
