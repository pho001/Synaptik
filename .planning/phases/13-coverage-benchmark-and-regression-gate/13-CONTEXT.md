# Phase 13: Coverage Benchmark And Regression Gate - Context

**Gathered:** 2026-04-30T20:14:16Z
**Status:** Ready for planning

<domain>
## Phase Boundary

Phase 13 closes v1.2 by making GPU region coverage measurable and regression-gated. It should extend trace, benchmark, and test/report contracts so Metal and CUDA coverage improvements from Phases 9-12 are auditable: GPU coverage ratio, selected region length, rejected candidate reasons, fallback counts, CPU materialization count/reason, copy timing, storage residency, and device handoffs. It should not add broad new GPU operation support, new fused kernels, or a public GPU tensor API; those belong to lowering/fusion phases or future accelerator coverage.

</domain>

<decisions>
## Implementation Decisions

### Coverage Evidence Contract
- **D-01:** Treat coverage as a first-class trace/report contract, not only a speed benchmark. The gate should inspect prepared/run metadata and benchmark reports for region length, GPU-owned step counts, fallback/rejection reasons, materialization counts, copy timing, residency, and handoff counts.
- **D-02:** Prefer backend-neutral metric names for shared Metal/CUDA evidence, with backend-specific fields allowed only where provider behavior differs. Existing Metal/CUDA trace fields must remain backward-compatible.
- **D-03:** Coverage metrics should distinguish "selected GPU region", "executed with native/buffer path", "executed through tensor-array bridge", "CPU fallback", and "CPU materialization boundary" so hidden exits cannot look like GPU coverage.

### Representative Workloads
- **D-04:** Use existing workload infrastructure before creating new workload machinery. Candidate workloads should come from `StandardWorkloads` and related specs: transformer block/hot path, MLP classification, conv2d or normalization-heavy graphs.
- **D-05:** The success criterion is fewer GPU-to-CPU exits or longer GPU-covered regions than the v1.1 baseline, not a universal raw-speed win. Raw timing can be reported, but coverage/materialization behavior is the milestone gate.
- **D-06:** Workloads must be small and deterministic enough for focused regression tests. Longer local benchmark runs may produce richer reports, but they must not become mandatory for every default test invocation.

### Regression Gates
- **D-07:** Add focused automated gates that fail when supported target workloads lose GPU coverage, add unexpected CPU materialization boundaries, or hide fallback behind tensor-array execution.
- **D-08:** Required-mode failures remain useful for proving no hidden CPU fallback, but default AUTO-mode reports should still expose fallback reason codes instead of failing ordinary portability paths.
- **D-09:** Native Metal and CUDA execution checks remain capability-gated. Portable Java tests must validate report schema, trace aggregation, fallback visibility, and capability-skip behavior on hosts without CUDA.

### Artifact And Profile Hygiene
- **D-10:** Do not commit machine-local benchmark/calibration output from `build/` or `profiles/platform/.../tuning/abc/*` unless planning intentionally creates stable fixtures. Phase 13 evidence should prefer checked-in tests, docs, and deterministic report fixtures.
- **D-11:** If benchmark report fixtures are needed, keep them minimal and deterministic under `src/test/resources` or a clearly named docs/test fixture path, not under local tuning/profile output directories.
- **D-12:** Regression gates should document which outputs are evidence contracts and which are local measurement artifacts.

### the agent's Discretion
- Exact metric container names, JSON field names, and report renderer placement are left to planning. Prefer existing packages under `tuning.benchmark.report`, `graph.execution.trace`, and accelerator execution metadata rather than adding a separate reporting subsystem.
- Planning may choose whether to implement a dedicated coverage summary record or extend existing report/trace summary types, provided the result is testable, documented, and backward-compatible.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase Scope
- `.planning/ROADMAP.md` — Phase 13 goal, success criteria, dependencies, and v1.2 milestone closure.
- `.planning/REQUIREMENTS.md` — GPUCOV-01, GPUCOV-02, and GPUCOV-03 acceptance scope.
- `.planning/PROJECT.md` — Project-level accelerator architecture constraints and trace/benchmark coverage milestone intent.
- `.planning/STATE.md` — Current milestone state, completed Phase 12 status, and local profile artifact hygiene notes.

### Prior Phase Contracts
- `.planning/phases/09-native-layout-abi-v2/09-VERIFICATION.md` — Layout ABI v2 capability/fallback contract that Phase 13 reports must preserve.
- `.planning/phases/10-gpu-layout-transform-and-view-path/10-VERIFICATION.md` — Layout/view residency and CPU materialization boundary contracts.
- `.planning/phases/11-gpu-lowering-coverage-matrix/11-VERIFICATION.md` — Lowering coverage matrix and selected/rejected candidate evidence.
- `.planning/phases/12-fused-gpu-region-execution/12-CONTEXT.md` — Decision that Phase 13 owns coverage ratios, workload comparisons, and regression gates.
- `.planning/phases/12-fused-gpu-region-execution/12-VALIDATION.md` — Phase 12 validated focused gates and CUDA capability-gated native evidence.

### Trace, Benchmark, And Report Code
- `src/main/java/graph/execution/PreparedExecution.java` — Execute/trace lifecycle and run trace production.
- `src/main/java/graph/execution/ExecutionState.java` — Device residency and CPU materialization traces.
- `src/main/java/graph/execution/trace/RunTrace.java` — Run-level trace aggregation.
- `src/main/java/graph/execution/trace/CpuMaterializationTrace.java` — Materialization reason/count/duration evidence.
- `src/main/java/backend/select/DefaultBackendSelectionPolicy.java` — Selected/rejected backend candidate decisions and cost reasons.
- `src/main/java/tuning/benchmark/report/AcceleratorTraceSummary.java` — Existing accelerator trace summary surface.
- `src/main/java/tuning/benchmark/report/BenchmarkReport.java` — Per-candidate benchmark report contract.
- `src/main/java/tuning/benchmark/report/BenchmarkCandidateReport.java` — Candidate-level report fields.
- `src/main/java/tuning/benchmark/report/JsonBenchmarkReportRenderer.java` — JSON report rendering contract.
- `src/main/java/tuning/benchmark/report/TextBenchmarkReportRenderer.java` — Text report rendering contract.
- `src/main/java/tuning/store/JsonFileBenchmarkReportStore.java` — Benchmark report persistence boundary.

### Workloads And Gates
- `src/main/java/tuning/workload/StandardWorkloads.java` — Existing representative workload catalog.
- `src/main/java/tuning/workload/TransformerBlockHotPathWorkloadSpec.java` — Transformer-block target workload.
- `src/main/java/tuning/workload/MlpClassificationWorkloadSpec.java` — MLP target workload.
- `src/main/java/tuning/workload/Conv2dWorkloadSpec.java` — Conv-ish target workload.
- `src/main/java/tuning/workload/NormalizationWorkloadSpec.java` — Normalization-heavy target workload.
- `src/test/java/BenchmarkSessionTest.java` — Existing benchmark session behavior tests.
- `src/test/java/BenchmarkSuiteSessionTest.java` — Existing suite-level benchmark behavior tests.
- `src/test/java/CompiledGraphTraceTest.java` — Trace metadata and materialization evidence tests.
- `src/test/java/ReportingDiffRendererTest.java` — Report/diff renderer expectations.
- `src/test/java/EtalonPerformanceRegressionTest.java` — Existing benchmark-style regression fixture pattern.

### Documentation
- `docs/compute-flow.md` — Trace fields, CPU materialization semantics, accelerator buffer reason codes, and benchmark report interpretation.
- `docs/metal-backend.md` — Metal trace reading and warning that timing alone is not proof of useful GPU residency.
- `docs/gpu-lowering-coverage.md` — Phase 11/12 operation and compound coverage contract.
- `docs/development.md` — Focused Gradle/native verification commands and artifact hygiene expectations.
- `docs/testing.md` — Test strategy and native capability-skip interpretation.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `PreparedExecution` and `RunTrace` already expose per-step execution metadata and CPU materialization traces.
- `ExecutionState` already records `CpuMaterializationTrace` entries with reason, residency, logical bytes, duration, completion, and diagnostic detail.
- `DefaultBackendSelectionPolicy` already records selected/rejected candidate information that can feed rejected reason counts.
- `AcceleratorTraceSummary`, `BenchmarkReport`, `BenchmarkCandidateReport`, and JSON/text report renderers already provide benchmark reporting surfaces to extend.
- `StandardWorkloads` and workload specs already cover transformer, MLP, conv2d, normalization, and related benchmark targets.

### Established Patterns
- Public `Tensor` remains logical; device residency evidence comes from runtime state, prepared metadata, trace records, and benchmark reports.
- Optional native Metal/CUDA checks stay capability-gated; portable Java tests prove schema, fallback, and trace contracts.
- Benchmark/profile output under local hardware paths is not committed unless intentionally promoted to a stable fixture.
- Focused Gradle filters are preferred over full default `./gradlew test` because default tests can include benchmark/debug slices.

### Integration Points
- Trace aggregation: `PreparedExecution`, `RunTrace`, `ExecutionStepTrace`, `StepExecutionMetadata`, and `CpuMaterializationTrace`.
- Backend coverage source data: backend selection traces, lowered region metadata, accelerator executable metadata, buffer decision attributes, and materialization traces.
- Benchmark reporting: `DefaultBenchmarkSession`, `DefaultBenchmarkSuiteSession`, `BenchmarkReport`, renderer classes, and report store boundaries.
- Workload selection: `StandardWorkloads`, transformer/MLP/conv/normalization specs, and existing debug benchmark tests.
- Regression gates: focused JUnit tests under `src/test/java` plus optional `metalTest` and CUDA capability-gated tasks.

</code_context>

<specifics>
## Specific Ideas

- "Hot path stayed on GPU" should become an auditable output: count selected accelerator regions, region length, CPU materializations, fallback reasons, and device handoffs.
- Compare coverage behavior to v1.1 baseline semantics, but avoid brittle local timing thresholds unless using stable fixtures.
- A useful first report can be per candidate/workload: `gpuCoverageRatio`, `maxGpuRegionLength`, `fallbackCount`, `cpuMaterializationCount`, `cpuMaterializationReasons`, `deviceHandoffCount`, and backend-specific execution path counts.
- If CUDA native execution is unavailable locally, tests should still prove CUDA report fields preserve capability-skip/fallback evidence rather than pretending native CUDA ran.

</specifics>

<deferred>
## Deferred Ideas

- Broad new GPU operation support beyond the Phase 11 matrix is future `ACCEL-01` scope.
- Larger fused GPU kernels beyond Phase 12's safe compound subset are future `ACCEL-02` scope.
- Higher-rank native ABI expansion beyond current workload needs is future `ACCEL-03` scope.

</deferred>

---

*Phase: 13-coverage-benchmark-and-regression-gate*
*Context gathered: 2026-04-30T20:14:16Z*
