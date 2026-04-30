---
phase: 003-materialization-aware-region-planning
status: passed
score: 29/29
verified: 2026-04-30
human_verification_required: false
threats_open: 0
---

# Phase 3 Verification: Materialization-Aware Region Planning

## Result

**PASSED** — Phase 3 achieved its goal: accelerator region planning and prepare-time backend selection now expose static materialization-aware cost decisions, benchmark reports summarize selected and rejected candidates, CPU hot paths remain guarded, and Phase 4 explicitly owns profile/calibration-derived cost model updates.

## Must-Have Verification

| Area | Status | Evidence |
|---|---|---|
| Static cost vocabulary | VERIFIED | `AcceleratorPartitionScoreModel` defines `MaterializationSignals`, `StaticCostPreset`, and `MaterializationCostSummary`; tests cover dispatch overhead, boundary/transfer penalties, avoided intermediate credit, fallback rejection, and named presets. |
| Compile-time selected/top rejected finalists | VERIFIED | `ScoredCandidatePartitionPlanner` records cost summaries and bounded rejected finalists through `PartitionDecisionTrace.CandidateCostTrace`; compile-trace tests verify summaries and finalist limits. |
| Prepare-time backend selection cost summaries | VERIFIED | `AcceleratorPlanCostModel.decide(...)`, `DefaultBackendSelectionPolicy`, and `BackendSelectionDecisionTrace` carry static cost summaries and stable rejection reasons. |
| Benchmark report visibility | VERIFIED | Text reports include `backendSelectionCost:` / `selectedBackend=` / `rejectedFinalists:`; JSON reports include `trace.backendSelectionCost.selected` and `trace.backendSelectionCost.rejectedFinalists`. |
| Runtime step trace boundary | VERIFIED | `ExecutionStepTrace` has no `backendSelectionCost` field; cost summaries stay in compile/prepare/report surfaces. |
| CPU natural regions | VERIFIED | `CpuNaturalExecutionRegionPlannerTest.cpuNaturalRegionsRemainAvailableWhenAcceleratorBenefitIsAmbiguous` proves CPU natural planning remains available under accelerator offload policy. |
| CPU fusion | VERIFIED | `OptimizerFuseTest.cpuFusionRemainsAvailableWithAcceleratorOffloadPolicy` proves fused CPU execution remains available when runtime accelerator execution is disabled under accelerator-aware graph policy. |
| CPU BLAS | VERIFIED | `BFloat16BlasDispatchTest.blasDispatchRemainsAvailableWithAcceleratorOffloadPolicy` proves BF16 matmul keeps `CPU_MATMUL_BLAS` when BLAS is enabled. |
| Static cost rejection | VERIFIED | `PreparedExecutionBuildTest.staticCostDoesNotSelectAcceleratorWhenCpuPathIsClearlyCompetitive` verifies `rejected-materialization-cost` and CPU execution availability. |
| Phase 4 deferral | VERIFIED | `docs/architecture.md` and `.planning/ROADMAP.md` explicitly state profile/calibration-derived costs are deferred to Phase 4 and include the required Phase 4 cost-model update note. |

## Requirement Traceability

| Requirement | Status | Evidence |
|---|---|---|
| PLAN-01 | SATISFIED | Cost model includes materialization, transfer, fallback, dispatch, compute, and named preset fields. |
| PLAN-02 | SATISFIED | Scored planning prefers profitable longer candidates and records rejected finalists; CPU safeguards prove accelerator preference remains conditional. |
| PLAN-03 | SATISFIED | Static cost rejection is implemented in compile and prepare surfaces, with `rejected-materialization-cost` coverage. |
| PLAN-04 | SATISFIED | CPU natural region, CPU fusion, and BF16 BLAS regression tests pass under accelerator-aware policy. |

## Automated Checks

| Command | Result |
|---|---|
| `./gradlew classes` | PASS |
| `./gradlew test --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests PreparedExecutionBuildTest --tests BenchmarkSessionTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest` | PASS |
| `./gradlew test --tests backend.accelerator.buffer.AcceleratorBufferLayoutClassifierTest --tests graph.execution.ExecutionStateResidencyTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.buffer.MetalBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest --tests backend.metal.MetalLayoutAwareDeviceFlowTest --tests backend.metal.MetalBufferTraceSmokeTest` | PASS |
| `gsd-sdk query verify.schema-drift 003` | PASS — no drift detected |
| `gsd-sdk query verify.codebase-drift` | PASS — no action required |

`./gradlew metalTest` was skipped because Phase 3 did not change native Metal execution code or native ABI files.

## Code Review

`003-REVIEW.md` status is `clean`.

One issue was found and fixed during review:

- `fix(003-02): render rejected backend cost decisions` makes benchmark reports include rejected backend-selection decisions that carry a cost summary even when no explicit finalist list is present.

## Human Verification

None required. The phase is covered by automated tests and documentation checks.

## Residual Risk

- Native Metal smoke coverage was not rerun because no native Metal path changed.
- Existing local tuning profile files under `profiles/platform/.../tuning/abc/*` and `.planning/tmp/` scratch files remain outside Phase 3 commits.

## Verdict

Phase 3 is ready to mark complete and hand off to Phase 4 planning.
