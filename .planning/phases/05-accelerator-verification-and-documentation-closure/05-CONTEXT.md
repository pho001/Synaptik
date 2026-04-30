# Phase 5: Accelerator Verification And Documentation Closure - Context

**Gathered:** 2026-04-30
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 5 closes the accelerator/runtime milestone by proving the new device-owned execution flow through focused tests, trace/report contract checks, benchmark-report evidence, documentation, and source/artifact hygiene. It must make accelerator path selection, buffer mode, fallback, materialization, copy timing, storage residency, and planner context visible enough that regressions are caught by tests and diagnosable in reports.

This phase does not add a public device tensor API, broaden native accelerator operation coverage, implement a production CUDA shim, or commit local measured benchmark/calibration output as canonical project state.

</domain>

<decisions>
## Implementation Decisions

### Evidence Bar

- **D-01:** Phase 5 tests must assert core accelerator evidence fields, not merely render them. Required trace/report evidence includes accelerator path, buffer mode, fallback reason, materialization count, materialization reason, copy-time field presence, and storage residency where available.
- **D-02:** Benchmark evidence is report-contract only. Tests should prove benchmark reports expose the accelerator evidence fields, but Phase 5 should not commit measured benchmark output from a specific machine.
- **D-03:** The evidence field set must include the ROADMAP minimum plus planner context: selected accelerator candidate, top rejected finalist, cost summary, and region boundary counts.
- **D-04:** Metal/native evidence should be capability-gated strict coverage. Java-side tests must assert the trace/report contract in portable default tests; Metal-specific meaningful values should be verified through `metalTest` and skipped when the native shim is unavailable.

### the agent's Discretion

- The planner may choose the exact benchmark workload structure, as long as it satisfies the roadmap requirement to stress matmul/linear, view/layout transforms, elementwise fusion, reductions, and backward/gradient publication.
- The planner may choose whether documentation closure is organized as updates to existing docs, a dedicated guide, or both, provided the final docs cover accelerator ABI, device-owned flow, layout/view handling, CPU materialization boundaries, tuning ownership, and fallback diagnostics.
- The planner may choose the exact hygiene mechanism, but it must preserve the Phase 5 rule that unintentional local generated classes and calibration/benchmark outputs do not enter commits.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope And Requirements

- `.planning/ROADMAP.md` - Phase 5 goal, dependencies, success criteria, and closure-scope notes.
- `.planning/REQUIREMENTS.md` - `OBS-01` through `OBS-04` and `DOC-01` through `DOC-04`.
- `.planning/PROJECT.md` - project constraints: logical public tensors, backend-neutral accelerator abstractions, CPU correctness/performance baseline, visible fallback, and local artifact hygiene.
- `.planning/STATE.md` - current milestone status and prior decisions through Phase 4.

### Prior Phase Contracts

- `.planning/phases/001-accelerator-buffer-layout-abi/001-VERIFICATION.md` - validated backend-neutral buffer layout ABI and reason-code behavior.
- `.planning/phases/002-metal-layout-aware-device-flow/002-VERIFICATION.md` - validated Metal layout-aware device flow, fallback visibility, and materialization boundaries.
- `.planning/phases/003-materialization-aware-region-planning/003-CONTEXT.md` - locked planner evidence expectations and CPU safeguard decisions.
- `.planning/phases/003-materialization-aware-region-planning/003-VERIFICATION.md` - validated static cost planning, selected/rejected candidate summaries, and CPU safeguards.
- `.planning/phases/04-tuning-and-profile-ownership-audit/04-CONTEXT.md` - locked tuning ownership and cost-model carry-forward decisions.
- `.planning/phases/04-tuning-and-profile-ownership-audit/04-VERIFICATION.md` - validated runtime-derived accelerator cost summaries and benchmark read-only behavior.
- `.planning/phases/04-tuning-and-profile-ownership-audit/04-SECURITY.md` - security verification for strict profile IO and benchmark/profile write boundaries.
- `.planning/phases/04-tuning-and-profile-ownership-audit/04-VALIDATION.md` - Nyquist validation coverage for Phase 4 ownership and cost behavior.

### Codebase Maps

- `.planning/codebase/TESTING.md` - test organization, targeted Gradle commands, Metal test task, benchmark/debug caveats, and source hygiene patterns.
- `.planning/codebase/ARCHITECTURE.md` - compile/prepare/execute architecture, trace records, accelerator scaffolding, benchmark/tuning layers, and runtime state boundaries.
- `.planning/codebase/CONCERNS.md` - known concerns around profile artifacts, root/generated artifact hygiene, benchmark tests in the default suite, native bridge visibility, and CPU hot paths.

### Trace, Benchmark, And Hygiene Entry Points

- `src/main/java/graph/execution/trace/ExecutionTrace.java` - top-level execution trace contract.
- `src/main/java/graph/execution/trace/PartitionDecisionTrace.java` - planner/partition trace context.
- `src/main/java/graph/execution/trace/BackendSelectionDecisionTrace.java` - selected/rejected backend candidate evidence.
- `src/main/java/tuning/benchmark/report/BenchmarkReport.java` - benchmark report data model.
- `src/main/java/tuning/benchmark/report/TextBenchmarkReportRenderer.java` - human-readable benchmark report rendering.
- `src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java` - machine-readable benchmark report rendering.
- `src/main/java/backend/accelerator/select/AcceleratorPlanCostModel.java` - runtime-derived accelerator cost summary source.
- `src/test/java/BenchmarkSessionTest.java` - benchmark session/report contract coverage.
- `src/test/java/PreparedExecutionBuildTest.java` - prepared execution and traced execution integration coverage.
- `src/test/java/graph/execution/ExecutionStateResidencyTest.java` - residency/materialization behavior coverage.
- `src/test/java/backend/metal/MetalLayoutAwareDeviceFlowTest.java` - Metal layout-aware device flow tests.
- `src/test/java/backend/metal/MetalBufferTraceSmokeTest.java` - Metal buffer trace smoke coverage.
- `src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java` - Metal buffer binding tests.
- `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java` - CUDA buffer-policy behavior and required-unavailable coverage.
- `src/test/java/SourceTreeHygieneTest.java` - source/package/benchmark persistence hygiene pattern.
- `build.gradle` - `metalTest`, `verifySourceTreeClean`, and native shim build task definitions.

### Documentation Targets

- `docs/architecture.md` - architecture overview that should remain aligned with accelerator/runtime closure.
- `docs/metal-backend.md` - Metal backend behavior, buffer ABI, tests, and fallback diagnostics.
- `docs/calibration-autotune.md` - tuning ownership, benchmark read-only behavior, and report interpretation.
- `docs/testing.md` - exact test commands, Metal test task, source hygiene, and benchmark/debug caveats.
- `docs/troubleshooting.md` - fallback, benchmark/profile, native bridge, and generated artifact diagnostics.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets

- Trace records under `src/main/java/graph/execution/trace/` already carry execution, partition, and backend-selection diagnostics; Phase 5 should extend/assert these rather than introduce a parallel diagnostics channel.
- Benchmark report models/renderers under `src/main/java/tuning/benchmark/report/` are the natural place to enforce report-contract fields without persisting measured local output.
- Existing Metal tests under `src/test/java/backend/metal/` already use optional-native assumptions and fit the capability-gated `metalTest` decision.
- `SourceTreeHygieneTest` and `verifySourceTreeClean` already encode file-walking hygiene checks; Phase 5 can extend these patterns for root generated classes and unintended profile/benchmark artifacts.

### Established Patterns

- Portable tests should run under targeted `./gradlew test --tests ...` filters; optional native Metal coverage belongs under `./gradlew metalTest`.
- Optional native dependencies are skipped through JUnit assumptions when unavailable, while Java-side contract tests remain portable.
- Benchmark commands are observational/read-only after Phase 4; benchmark output is evidence for report shape and diagnostics, not a source of runtime profile truth.
- Public tensor APIs remain logical; accelerator residency evidence should be gathered from prepared execution, execution state, trace records, and benchmark reports.

### Integration Points

- Trace/report assertion work connects `ExecutionTrace`, `PartitionDecisionTrace`, `BackendSelectionDecisionTrace`, `ExecutionState`, benchmark reports, and renderer tests.
- Metal strict-value coverage connects Metal buffer execution tests with `metalTest` and native shim availability.
- Documentation closure connects `docs/metal-backend.md`, `docs/calibration-autotune.md`, `docs/testing.md`, `docs/troubleshooting.md`, and project architecture docs.
- Hygiene closure connects `build.gradle`, `SourceTreeHygieneTest`, profile artifact paths under `profiles/platform/**`, and generated class artifact checks.

</code_context>

<specifics>
## Specific Ideas

- Prefer tests that assert the presence and basic meaning of accelerator evidence fields over tests that depend on exact local timing values.
- Keep benchmark evidence as a report contract. Generated reports may be used during verification, but measured outputs from this machine should not become committed milestone artifacts.
- Tie Phase 5 evidence back to Phase 3/4 planner/cost decisions by including selected candidate, top rejected finalist, cost summary, and region boundary counts in report assertions.

</specifics>

<deferred>
## Deferred Ideas

None - discussion stayed within phase scope.

</deferred>

---

*Phase: 05-accelerator-verification-and-documentation-closure*
*Context gathered: 2026-04-30*
