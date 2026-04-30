# Phase 5: Accelerator Verification And Documentation Closure - Research

## Research Complete

Phase 5 is a closure phase. It should not introduce a new accelerator architecture. The implementation work should assert and document the observable contracts already created by Phases 1-4.

## Key Findings

### Trace And Report Evidence

- `PreparedExecution` already publishes backend-neutral accelerator buffer attributes on step metadata: `acceleratorBufferMode`, `acceleratorBufferBackend`, `acceleratorBufferExecutionPath`, `acceleratorBufferReasonCode`, `acceleratorBufferReason`, `acceleratorBufferPreparedInputUsed`, input/output counts, and storage residency fields.
- Metal-specific execution stats are already available in step attributes: `metalExecutionPath`, `metalFallbackReason`, copy counters, input/output bytes, native execution time, and buffer support flags.
- `BackendSelectionDecisionTrace` and `PartitionDecisionTrace` already carry cost summaries and bounded rejected finalists, which directly satisfy the Phase 5 planner-context evidence decision.
- `TextBenchmarkReportRenderer` and `JsonBenchmarkReportRenderer` already render accelerator summary, CPU materializations, backend selection cost, steps, and metadata attributes. The highest-value work is contract tests and small renderer/summary additions where field names or fallback-reason aggregation are missing.

### Benchmark Workload Coverage

- `TransformerBlockHotPathWorkloadSpec` already combines projection matmuls/linear ops, reshape/permute layout transforms, elementwise residual/feed-forward chains, attention, mean reduction, and forward-backward gradient publication when run with `ExecutionMode.FORWARD_BACKWARD`.
- The closure benchmark should use report-contract tests and generated in-test reports, not committed measured output. This matches the Phase 5 context decision and Phase 4 benchmark read-only boundary.
- Existing `BenchmarkSessionTest` has report renderer contract tests and manual trace fixtures. It is the best portable place to assert report fields without requiring a native Metal library.

### Metal Correctness And Device Handoff

- `MetalLayoutAwareDeviceFlowTest` already covers linear -> reshape -> permute device-owned flow, CPU parity, visible fallback for unsupported layouts, and forward-backward gradient publication.
- `MetalBufferTraceSmokeTest` already covers native buffer path, CPU boundary materialization, fallback reason, and layout-aware trace attributes under `metalTest`.
- `PreparedMetalExecutableBufferBindingTest` already has unit-level fake bridge coverage for buffer execution, prepared input usage, and residency transitions. Phase 5 should add/strengthen adjacent-region or prepared-input handoff assertions here and keep real native checks capability-gated.
- CUDA should remain policy-only for this phase. `PreparedCudaExecutableBufferPolicyTest` proves required CUDA buffer execution remains unavailable without a native shim; Phase 5 docs should not imply production CUDA support.

### Documentation And Hygiene

- Documentation targets are `docs/metal-backend.md`, `docs/calibration-autotune.md`, `docs/testing.md`, `docs/troubleshooting.md`, and `docs/architecture.md`.
- Existing hygiene checks live in `SourceTreeHygieneTest` and Gradle task `verifySourceTreeClean`. They scan `src/` and `test/`; Phase 5 should extend hygiene to root generated class artifacts, `.planning/tmp/`, and unintentional profile/benchmark outputs without deleting the current intentionally tracked profile fixtures.
- `.gitignore` currently ignores root `*.class` and `**/*.class`, but does not ignore `.planning/tmp/`. Phase 5 should add `.planning/tmp/` explicitly and add a test asserting that scratch path remains ignored.

## Recommended Plan Shape

1. Trace/report evidence contract:
   - Strengthen renderer/summary tests around accelerator path, buffer mode, fallback reason, materialization count/reason, copy-time fields, storage residency, selected backend candidate, rejected finalists, cost summary, and boundary counts.
   - Keep tests portable by using constructed trace fixtures where native execution is not required.

2. Closure workload and Metal correctness:
   - Add tests proving the transformer-block workload is the closure benchmark stressor for matmul/linear, view/layout transforms, elementwise chains, reductions, and backward/gradient publication.
   - Extend capability-gated Metal tests for meaningful values in real native traces.
   - Add unit-level adjacent device handoff/prepared-input assertions where fake bridges make the behavior deterministic.

3. Docs and hygiene:
   - Document the accelerator ABI, device-owned flow, layout/view handling, CPU materialization boundaries, tuning ownership, fallback diagnostics, and benchmark report interpretation.
   - Add hygiene checks for `.planning/tmp/` and root/local generated artifacts.
   - Run final focused verification and ensure local profile tuning outputs and `.planning/tmp/` are not staged.

## Validation Architecture

- Use JUnit Jupiter tests and Gradle tasks already configured in `build.gradle`.
- Portable quick checks:
  - `./gradlew test --tests BenchmarkSessionTest --tests StandardWorkloadsTest`
  - `./gradlew test --tests SourceTreeHygieneTest`
- Native Metal checks:
  - `./gradlew metalTest`
  - Must remain capability-gated through JUnit assumptions when the local Metal shim is unavailable.
- Final closure:
  - `./gradlew classes`
  - `./gradlew test --tests BenchmarkSessionTest --tests StandardWorkloadsTest --tests SourceTreeHygieneTest --tests graph.execution.ExecutionStateResidencyTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest`
  - `./gradlew metalTest`

## Risks And Constraints

- Do not commit measured benchmark output from the local machine.
- Do not stage `profiles/platform/.../tuning/abc/*` or `.planning/tmp/`.
- Do not make default `./gradlew test` depend on local Metal availability.
- Do not broaden accelerator operation coverage or CUDA native support in this phase.

## RESEARCH COMPLETE
