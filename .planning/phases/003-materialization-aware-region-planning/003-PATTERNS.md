---
phase: 3
slug: materialization-aware-region-planning
status: complete
created: 2026-04-30
---

# Phase 3 - Pattern Map

## Purpose

Map Phase 3 target files to nearby implementation patterns so execution can edit with the existing architecture rather than inventing a parallel planner/reporting system.

## Planned File Roles

| File | Role | Closest Existing Pattern | Notes |
|------|------|--------------------------|-------|
| `src/main/java/graph/optimizer/partition/cost/AcceleratorPartitionScoreModel.java` | Static cost math and summary records | Existing `CandidateMetrics`, `TransferMetrics`, `TransferPolicy`, `PlannerPolicy` records | Add backend-neutral summaries adjacent to current score helpers; keep named presets and internal constants. |
| `src/main/java/graph/optimizer/partition/ScoredCandidatePartitionPlanner.java` | Candidate scoring, tie-breaks, finalists | Existing `SearchAccumulator`, `SearchOutcome`, `isBetterAccepted(...)` | Track bounded top rejected finalists while preserving deterministic traversal and tie-break order. |
| `src/main/java/graph/execution/trace/PartitionDecisionTrace.java` | Compile-time partition diagnostic record | Existing immutable trace records in `graph.execution.trace` | Prefer Java records with constructor normalization and `List.copyOf(...)`. |
| `src/main/java/backend/accelerator/select/AcceleratorPlanCostModel.java` | Prepare-time static accelerator cost gate | Existing `Decision` record and `decide(...)` method | Extend `Decision` with score summary/reason detail without changing public tensor APIs. |
| `src/main/java/backend/select/DefaultBackendSelectionPolicy.java` | Selection policy and prepare trace | Existing selected/rejected trace creation | Keep runtime enablement/availability gates first; add cost summary only after plan compatibility and availability are known. |
| `src/main/java/graph/execution/trace/BackendSelectionDecisionTrace.java` | Prepare-time selected/rejected diagnostics | Existing trace records and list normalization | Add fields for final score, boundary count, estimated transfers/compute, selected preset, and top finalist summary. |
| `src/main/java/tuning/benchmark/report/TextBenchmarkReportRenderer.java` | Human-readable benchmark output | Existing `appendAcceleratorSummary(...)`, `appendCpuMaterializations(...)` | Add compact compile/prepare planning lines near candidate details; avoid dumping every candidate. |
| `src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java` | Structured benchmark output | Existing manual JSON builders | Add simple primitives/arrays; preserve null/empty behavior. |
| `src/test/java/graph/optimizer/partition/cost/AcceleratorPartitionScoreModelTest.java` | Cost unit tests | Current transfer scoring tests | Add tests for dispatch, fallback, layout class, boundary count, and preset constants. |
| `src/test/java/PreparedExecutionBuildTest.java` | Compile/prepare integration tests | Existing accelerator offload and fused/BLAS assertions | Add assertions on planner/backend selection traces without requiring native Metal execution. |
| `src/test/java/BenchmarkSessionTest.java` | Renderer tests | Existing text/json accelerator diagnostics test | Add text/json checks for planning cost fields using hand-built traces. |

## Data Flow Pattern

1. `PartitionIntentRule` creates a `PartitionPlanningRequest`.
2. `ScoredCandidatePartitionPlanner` builds candidate metrics, calls `AcceleratorPartitionScoreModel`, and emits `PartitionDecisionTrace`.
3. `CompiledGraph` stores compile artifacts and backend selection candidates.
4. `DefaultBackendSelectionPolicy` checks runtime policy and `AcceleratorPlanCostModel`, then emits `BackendSelectionTrace`.
5. `PreparedExecution` exposes `PrepareTrace`.
6. Benchmark reports read `ExecutionTrace.compile()`, `ExecutionTrace.prepare()`, and `ExecutionTrace.run()` and render summaries.

Phase 3 should keep that flow. Do not push residency/cost fields into `Tensor`.

## Trace Record Pattern

Existing trace records normalize nulls and clamp invalid numeric values in compact constructors:

```java
public record BackendSelectionTrace(
        int totalCandidates,
        int selectedCount,
        int rejectedCount,
        List<BackendSelectionDecisionTrace> decisions
) {
    public BackendSelectionTrace {
        totalCandidates = Math.max(0, totalCandidates);
        selectedCount = Math.max(0, selectedCount);
        rejectedCount = Math.max(0, rejectedCount);
        decisions = List.copyOf(decisions == null ? List.of() : decisions);
    }
}
```

New cost/finalist trace records should follow the same style.

## Test Pattern

- Unit-test pure score records in `AcceleratorPartitionScoreModelTest`.
- Integration-test compile/prepare trace presence in `PreparedExecutionBuildTest`.
- Use renderer tests with hand-built `ExecutionTrace` objects in `BenchmarkSessionTest` when a full graph would be noisy.
- Keep native Metal tests optional unless native execution code changes.

## Guardrails

- Keep Phase 3 cost constants static and named; do not read or persist local profile files.
- Keep fallback visible through trace/report reason codes.
- Keep CPU natural region and CPU fusion tests in the execution plan, not as optional cleanup.
- Update Phase 4 planning notes so profile/calibration-derived cost model ownership is handled there.
