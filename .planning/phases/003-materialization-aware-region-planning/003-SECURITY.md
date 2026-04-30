---
phase: 003
slug: materialization-aware-region-planning
status: verified
threats_open: 0
asvs_level: 1
created: 2026-04-30
updated: 2026-04-30
---

# Phase 003 — Security

Per-phase security contract: threat register, accepted risks, and audit trail.

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|---------------|
| Compile-time optimizer to prepare-time backend selection | Partition plans and static cost summaries created during compile are consumed when runtime backend plans are selected. | Node ids, backend ids, estimated work, static materialization score summaries, finalist diagnostics. |
| Planning traces to benchmark reports | Internal compile/prepare diagnostics are rendered into human-readable and JSON benchmark reports. | Selected candidate summaries, bounded rejected finalists, reason codes, transfer/work estimates. |
| Static model to tuning/profile ownership | Phase 3 introduced static named compile/prepare presets; Phase 4 now owns audited runtime/profile-derived prepare-time cost factors. | Static preset names, runtime-derived `PROFILE_DERIVED` summaries, and ownership-audited runtime thresholds; backend cost code does not read local profile/history/calibration files directly. |

## Threat Register

| Threat ID | Category | Component | Disposition | Mitigation | Status |
|-----------|----------|-----------|-------------|------------|--------|
| T-3-01 | Integrity | `AcceleratorPartitionScoreModel` | mitigate | Cost input records clamp counts/bytes/work with `Math.max(...)`; non-positive and non-finite scores are rejected with stable reason codes. Evidence: `AcceleratorPartitionScoreModel` constructors and `scoreMaterializationAware(...)`; `AcceleratorPartitionScoreModelTest`. | closed |
| T-3-02 | Availability | `ScoredCandidatePartitionPlanner`, trace records | mitigate | Top rejected finalists are sorted deterministically by score and capped to 3 in the planner; `PartitionDecisionTrace` and `BackendSelectionDecisionTrace` also cap finalist lists. Evidence: `SearchAccumulator.addFinalist(...)`, `PartitionDecisionTrace`, `BackendSelectionDecisionTrace`. | closed |
| T-3-03a | Integrity | Compile-time cost model | mitigate | Static cost construction uses named presets and internal constants only; docs state no profile/calibration-derived runtime evidence is used in Phase 3. Evidence: `StaticCostPreset`, `docs/architecture.md`, `003-VERIFICATION.md`. | closed |
| T-3-03b | Integrity | Prepare-time backend selection | mitigate | `AcceleratorPlanCostModel.decide(...)` computes a materialization cost summary and rejects non-profitable summaries with stable reason codes such as `estimated-work-below-minimum` and `rejected-materialization-cost`; tests cover rejection. Current prepare-time `PROFILE_DERIVED` factors are Phase 4-owned and come from audited `RuntimeConfig`, not direct profile IO. Evidence: `AcceleratorPlanCostModel`, `ProfileDerivedAcceleratorCostFactors`, `PreparedExecutionBuildTest`, `04-SECURITY.md`. | closed |
| T-3-04 | Availability / Information Disclosure | Benchmark report renderers | mitigate | Text and JSON reports show selected candidates plus at most 3 rejected finalists; rejected backend decisions are summarized compactly instead of dumping all candidate search state. Evidence: `TextBenchmarkReportRenderer`, `JsonBenchmarkReportRenderer`, `BenchmarkSessionTest.renderersExposeBackendSelectionCostDiagnostics`. | closed |
| T-3-05 | Availability / Performance Regression | CPU execution paths | mitigate | Focused regression tests prove CPU natural regions, CPU fused execution, and BF16 BLAS dispatch remain selectable under accelerator-aware policy. Evidence: `CpuNaturalExecutionRegionPlannerTest`, `OptimizerFuseTest`, `BFloat16BlasDispatchTest`. | closed |
| T-3-06 | Process Integrity | Phase 4 planning handoff | mitigate | ROADMAP and PROJECT explicitly carry the deferred profile/calibration-derived cost model update into Phase 4. Evidence: `.planning/ROADMAP.md`, `.planning/PROJECT.md`, `docs/architecture.md`. | closed |
| T-3-07 | Availability | Runtime execution trace | mitigate | Planner/backend cost fields are kept in compile/prepare/report surfaces; `ExecutionStepTrace` has no `backendSelectionCost` field and docs state runtime steps remain execution-focused. Evidence: `ExecutionStepTrace`, `docs/architecture.md`, renderer tests. | closed |
| T-3-08 | Integrity / Documentation | Developer documentation | mitigate | Architecture docs state Phase 3 uses static named presets only and profile/calibration-derived costs are deferred to Phase 4. Evidence: `docs/architecture.md` section `Materialization-aware region planning`. | closed |

Notes:
- The Phase 3 plans reused `T-3-03` for two distinct threats. This audit disambiguates them as `T-3-03a` and `T-3-03b` while preserving the original threat reference.

## Accepted Risks Log

No accepted risks.

## Evidence And Checks

| Check | Result |
|-------|--------|
| Phase 3 verification | `003-VERIFICATION.md` passed 29/29. |
| Code review | `003-REVIEW.md` status is `clean`. |
| Phase 3 targeted tests | `./gradlew test --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests PreparedExecutionBuildTest --tests BenchmarkSessionTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest` passed during phase verification. |
| Regression gate | Phase 1/2 focused Java regression tests passed during phase verification. |
| Native Metal gate | Skipped because Phase 3 did not change native Metal execution code or native ABI files. |

## Security Re-Audit 2026-04-30

| Metric | Count |
|--------|-------|
| Threats found | 9 |
| Closed | 9 |
| Open | 0 |

## Re-Audit Evidence

| Check | Result |
|-------|--------|
| Phase 3 threat models from `003-01-PLAN.md`, `003-02-PLAN.md`, and `003-03-PLAN.md` map to all 9 closed security rows. | PASS |
| `AcceleratorPartitionScoreModel` clamps counts, bytes, work, and cost factors with `Math.max(...)` and rejects non-finite final scores with `rejected-non-finite-score`. | PASS |
| `ScoredCandidatePartitionPlanner`, `PartitionDecisionTrace`, `BackendSelectionDecisionTrace`, and benchmark report renderers keep rejected finalist lists bounded to 3. | PASS |
| Compile-time Phase 3 scoring still uses static named presets; current prepare-time `PROFILE_DERIVED` summaries are Phase 4-owned through `RuntimeConfig` and covered by `04-SECURITY.md` threats `T-4-07` and `T-4-09`. | PASS |
| Backend accelerator cost packages contain no direct `PlatformRuntimeProfileIO`, `JsonFileBestProfileStore`, `CalibrationArtifactLayout`, or `profiles/platform` path references. | PASS |
| `ExecutionStepTrace` remains runtime-execution focused and has no backend selection cost field. | PASS |
| `./gradlew test --tests graph.optimizer.partition.cost.AcceleratorPartitionScoreModelTest --tests graph.optimizer.partition.CpuNaturalExecutionRegionPlannerTest --tests PreparedExecutionBuildTest --tests BenchmarkSessionTest --tests OptimizerFuseTest --tests BFloat16BlasDispatchTest` | PASS |

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-04-30 | 9 | 9 | 0 | Codex |
| 2026-04-30 re-audit | 9 | 9 | 0 | Codex |

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-04-30
