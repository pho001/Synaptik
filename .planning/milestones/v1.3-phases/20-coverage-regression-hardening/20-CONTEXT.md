# Phase 20: Coverage Regression Hardening - Context

**Gathered:** 2026-05-01
**Status:** Ready for planning
**Source:** Auto context from roadmap, requirements, Phase 14 targets, Phase 18/19 closure evidence, and codebase inspection.

<domain>
## Phase Boundary

Phase 20 closes v1.3 by turning GPU coverage evidence into hard, deterministic regression contracts. The phase should
make hot paths staying on GPU auditable through trace and benchmark report fields, portable Java gates, and
capability-gated native Metal/CUDA evidence.

This phase is not for adding new GPU operation coverage, vendor library routing, public GPU tensor APIs, or
benchmark-timing gates. The goal is to fail clearly when previously supported hot paths shorten, add CPU
materialization, hide tensor-array fallback, lose native-buffer coverage, lose lowered/fused metadata, or omit native
skip/pass evidence.

</domain>

<decisions>
## Implementation Decisions

### Gate Scope And Strictness
- **D-01:** Phase 20 gates must be coverage/materialization gates, not raw timing gates. Timing can be reported as
  supporting context, but pass/fail should depend on structured trace/report evidence.
- **D-02:** Gate policies should cover the Phase 14 target workloads: `transformer_block_hot_path`,
  `mlp_classifier_small`, `conv2d_resnet_3x3`, and `layer_norm_small`.
- **D-03:** Hard failures should include lost selected-region length, lost multi-op region evidence, lower
  `loweredPrimitiveCount`, missing/lower fused subpattern count where expected, unexpected CPU materialization,
  hidden tensor-array bridge execution, CPU fallback, unexpected device handoffs, missing native buffer binding when
  required, and missing coverage summary.
- **D-04:** Policies should be deterministic and reviewable. Prefer checked-in Java policy records or target
  expectation records over machine-local benchmark output, mutable profile state, or prose-only thresholds.

### Evidence Source
- **D-05:** Structured trace/report fields are the source of truth: `gpuCoverageRatio`, `selectedRegionCount`,
  `multiOpGpuRegionCount`, `maxSelectedRegionLength`, `averageSelectedRegionLength`, `loweredPrimitiveCount`,
  `rejectedCandidateReasonCounts`, `cpuMaterializationReasonCounts`, `nativeBufferStepCount`,
  `tensorArrayStepCount`, `cpuFallbackStepCount`, `deviceHandoffCount`, `storageResidencyCounts`,
  `gpuFusedSubpatternCount`, `gpuFusedSubpatternTypes`, `gpuFusedSubpatternOriginalNodeIds`,
  `gpuFusedSubpatternLoweredPrimitiveCount`, and `gpuFusedSubpatternReasons`.
- **D-06:** `tensor-array bridge execution is not native buffer GPU coverage`; `nativeBufferStepCount` must remain tied
  to `BUFFER_BINDING`, while `tensorArrayStepCount` is a separate failure source for native-buffer-required policies.
- **D-07:** A "hot path stayed on GPU" result must be trace/report based. A fast benchmark with CPU materialization,
  tensor-array fallback, or missing lowered/fused metadata is still a regression.

### Target-Specific Expectations
- **D-08:** `transformer_block_hot_path` should be the primary hardening target because it exercises lowered DAG,
  normalization/softmax-ish evidence, fusion metadata, multi-op region execution, and device handoff boundaries.
- **D-09:** `mlp_classifier_small` should be the secondary hardening target because it exercises linear/matmul,
  bias/activation epilogues, elementwise chains, dtype/storage residency, and native-buffer coverage.
- **D-10:** `conv2d_resnet_3x3` and `layer_norm_small` may still contain visible unsupported blockers, but the gate must
  preserve stable rejection/materialization evidence rather than allowing silent CPU replay or missing report fields.
- **D-11:** If a target is expected to remain partially blocked, the policy should assert the explicit blocker reason
  and report fields, not require unsupported native execution.

### Portable And Native Evidence
- **D-12:** Portable Java gates are mandatory and must prove schema, failure semantics, and fallback visibility on
  machines without CUDA hardware or native Metal/CUDA availability.
- **D-13:** Native Metal and CUDA checks remain capability-gated. A native skip is acceptable only when it is explicitly
  reported as skipped-by-environment evidence; required native mode should fail clearly if native evidence is absent.
- **D-14:** CUDA hardware absence remains an environment risk, not a reason to weaken portable gate contracts.

### Reporting And Closure
- **D-15:** Text and JSON benchmark/suite reports should make gate inputs and gate results reviewable enough for the
  milestone audit to answer whether the selected hot paths stayed on GPU.
- **D-16:** Docs must explain the v1.3 evidence contract, how to interpret coverage failures, how native skip evidence
  differs from native pass evidence, and why local tuning/profile artifacts are not canonical proof.
- **D-17:** Local `profiles/platform/.../tuning/abc/*` artifacts remain unstaged unless a plan explicitly promotes a
  stable fixture. Phase 20 should not commit local benchmark or calibration output as proof.

### the agent's Discretion
The agent may choose the exact Java type names and decomposition for target expectations, gate policy extensions,
renderer fields, and test helpers. The implementation must stay backend-neutral at the policy/report level while
allowing backend-specific evidence from Metal and CUDA traces.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Planning And Requirements
- `.planning/ROADMAP.md` — Phase 20 goal, success criteria, constraints, and dependency chain from Phases 14-19.
- `.planning/REQUIREMENTS.md` — `GPUHARDEN-01`, `GPUHARDEN-02`, and `GPUHARDEN-03`.
- `.planning/PROJECT.md` — project-level accelerator/runtime constraints, public `Tensor` boundary, fallback visibility, and artifact hygiene.
- `.planning/phases/14-coverage-gap-triage-and-hot-path-targets/14-HOT-PATH-TARGETS.md` — source-of-truth target workloads and gap categories for Phase 20.

### Prior Phase Contracts
- `.planning/phases/17-normalization-reduction-and-loss-adjacent-lowering/17-CONTEXT.md` — visible support/rejection contract for normalization, reductions, conv, softmax-ish, and loss-adjacent gaps.
- `.planning/phases/18-fused-elementwise-and-epilogue-subregions/18-01-SUMMARY.md` — shared GPU fusion subpattern metadata contract.
- `.planning/phases/18-fused-elementwise-and-epilogue-subregions/18-04-SUMMARY.md` — fused subpattern trace/report evidence and CPU fused isolation.
- `.planning/phases/19-multi-op-gpu-region-execution/19-CONTEXT.md` — multi-op GPU region execution decisions, evidence fields, and artifact hygiene.
- `.planning/phases/19-multi-op-gpu-region-execution/19-VERIFICATION.md` — verified Phase 19 evidence and final focused commands.

### Codebase Maps
- `.planning/codebase/TESTING.md` — JUnit/Gradle patterns, targeted test commands, native skip assumptions, and benchmark test caveats.
- `.planning/codebase/CONVENTIONS.md` — Java naming/style, source hygiene, and architecture guard conventions.
- `.planning/codebase/STRUCTURE.md` — tuning/report/backend package locations and test placement.

### Coverage And Benchmark Code
- `src/main/java/tuning/benchmark/report/GpuCoverageSummary.java` — backend coverage aggregate and trace-derived evidence fields.
- `src/main/java/tuning/benchmark/report/GpuCoverageRegressionGate.java` — current fail-fast gate evaluator.
- `src/main/java/tuning/benchmark/report/GpuCoverageGatePolicy.java` — current gate policy record.
- `src/main/java/tuning/benchmark/report/GpuCoverageGateResult.java` — gate result contract.
- `src/main/java/tuning/benchmark/report/GpuHotPathCoverageTargets.java` — checked v1.3 target registry.
- `src/main/java/tuning/benchmark/report/GpuCoverageGapTriage.java` — deterministic gap categories and severity ranking.
- `src/main/java/tuning/benchmark/report/TextBenchmarkReportRenderer.java` — text report renderer for per-candidate coverage evidence.
- `src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java` — JSON report renderer for per-candidate coverage evidence.
- `src/main/java/tuning/benchmark/report/TextBenchmarkSuiteReportRenderer.java` — suite-level text coverage rendering.
- `src/main/java/tuning/benchmark/report/JsonBenchmarkSuiteReportRenderer.java` — suite-level JSON coverage rendering.
- `src/main/java/tuning/workload/StandardWorkloads.java` — source of representative workload definitions.

### Focused Tests
- `src/test/java/GpuCoverageRegressionGateTest.java` — current gate failure semantics.
- `src/test/java/GpuCoverageSummaryTest.java` — trace-to-summary coverage behavior.
- `src/test/java/GpuHotPathCoverageTargetsTest.java` — target registry coverage.
- `src/test/java/BenchmarkSessionTest.java` — report rendering and per-candidate gate behavior.
- `src/test/java/BenchmarkSuiteSessionTest.java` — suite coverage aggregation and target suite construction.
- `src/test/java/CompiledGraphTraceTest.java` — run trace evidence for lowered/fused GPU regions.
- `src/test/java/SourceTreeHygieneTest.java` — source hygiene and artifact guard patterns.

### Documentation
- `docs/testing.md` — GPU coverage regression gate commands and native skip interpretation.
- `docs/development.md` — GPU lowering/fusion/multi-op change gates and local tuning artifact hygiene.
- `docs/compute-flow.md` — trace and coverage summary semantics.
- `docs/gpu-lowering-coverage.md` — conservative lowering coverage and explicit unsupported blockers.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `GpuCoverageSummary.fromTrace(...)` already derives coverage, region length, lowered primitive count, fused
  subpattern metadata, fallback counts, CPU materialization reasons, storage residency, and device handoffs.
- `GpuCoverageRegressionGate` already provides fail-fast gate evaluation and `requirePass(...)`; Phase 20 can extend
  it rather than inventing a second gate path.
- `GpuCoverageGatePolicy` already carries coverage ratio, max selected region length, materialization, fallback,
  tensor-array, device-handoff, and native-buffer requirements.
- `GpuHotPathCoverageTargets.defaults()` and `.planning/phases/14-coverage-gap-triage-and-hot-path-targets/14-HOT-PATH-TARGETS.md`
  already define the four target workloads Phase 20 should enforce.
- Benchmark report renderers already emit most Phase 20 evidence fields; hardening can focus on missing gate result,
  target expectation, and native skip/pass evidence.

### Established Patterns
- Portable JUnit tests prove coverage/report contracts first; optional native Metal/CUDA tasks add capability-gated
  evidence.
- Tests use real trace/report objects and small synthetic fixtures rather than mocks.
- Architecture and artifact hygiene are enforced through `SourceTreeHygieneTest`.
- Local tuning profile files are treated as non-canonical unless a plan intentionally promotes them.

### Integration Points
- Gate policy extensions connect through `GpuCoverageGatePolicy`, `GpuCoverageRegressionGate`, and
  `GpuCoverageGateResult`.
- Target-specific expectations should connect to `GpuHotPathCoverageTargets` and benchmark suite report aggregation,
  not ad hoc workload names.
- Report rendering changes belong in both text and JSON renderers for benchmark reports and suite reports.
- Documentation closure should update `docs/testing.md`, `docs/development.md`, `docs/compute-flow.md`, or
  `docs/gpu-lowering-coverage.md` as needed.

</code_context>

<specifics>
## Specific Ideas

- Extend gate policies with minimum `multiOpGpuRegionCount`, minimum `loweredPrimitiveCount`, minimum
  `gpuFusedSubpatternCount`, and possibly expected rejected/materialization reason assertions.
- Add a target expectation registry or checked policy builder for Phase 14 hot paths so Phase 20 gates are not
  hand-coded differently in each test.
- Add stable failure strings for each hard gate: lost selected region length, lost lowered primitive coverage, lost
  fused subpattern evidence, unexpected CPU materialization, hidden tensor-array fallback, CPU fallback, unexpected
  device handoff, missing native buffer binding, missing coverage summary, and missing native evidence when required.
- Keep partially unsupported conv/norm targets useful by asserting explicit blocker evidence rather than pretending
  the backend supports them.

</specifics>

<deferred>
## Deferred Ideas

- Vendor library routing through cuBLAS, cuDNN, MPSGraph, or similar libraries remains deferred to `GPULIB-*`.
- Backend-native primitive cost model remains deferred to `GPULIB-02`.
- Universal native support for reductions, normalizations, convolution, dynamic shape, sparse, high-rank, and advanced
  indexing remains outside v1.3.
- Public GPU tensor/device API remains out of scope.
- Treating local benchmark/profile artifacts under `profiles/platform/.../tuning/abc/*` as canonical proof remains out
  of scope unless a future plan explicitly promotes stable fixtures.

</deferred>

---

*Phase: 20-Coverage Regression Hardening*
*Context gathered: 2026-05-01*
