# Phase 05 - Pattern Map

## Closest Existing Analogs

| Planned Area | Files To Touch | Existing Analog | Pattern To Reuse |
|--------------|----------------|-----------------|------------------|
| Trace/report evidence tests | `src/test/java/BenchmarkSessionTest.java` | `renderersExposeBackendSelectionCostDiagnostics`, `renderersExposeMetalBridgeTransferDiagnostics` | Build explicit `ExecutionTrace` fixtures and assert exact text/JSON report keys. |
| Runtime trace evidence | `src/main/java/graph/execution/PreparedExecution.java`, `src/main/java/tuning/benchmark/report/AcceleratorTraceSummary.java` | Existing acceleratorBuffer and Metal stats attributes | Keep backend-neutral `acceleratorBuffer*` attributes as source of truth; aggregate Metal copy counters opportunistically. |
| Closure workload proof | `src/main/java/tuning/workload/TransformerBlockHotPathWorkloadSpec.java`, `src/test/java/StandardWorkloadsTest.java` | `transformerBlockHotPathInstantiatesProjectionAttentionAndFeedForwardGraph` | Assert workload metadata and graph behavior instead of committing measured benchmark output. |
| Metal capability-gated checks | `src/test/java/backend/metal/MetalLayoutAwareDeviceFlowTest.java`, `src/test/java/backend/metal/MetalBufferTraceSmokeTest.java` | `assumeNativeBufferBridge(...)` helper | Use JUnit assumptions for native shim availability; assert meaningful values when tests run. |
| Adjacent buffer handoff | `src/test/java/backend/metal/exec/PreparedMetalExecutableBufferBindingTest.java` | Fake bridge tests for buffer execution and prepared input use | Use deterministic fake bridge/state assertions instead of requiring native Metal for handoff mechanics. |
| Hygiene checks | `src/test/java/SourceTreeHygieneTest.java`, `build.gradle`, `.gitignore` | File-walking source hygiene and `verifySourceTreeClean` | Add repository-local artifact rules without deleting intentionally tracked fixtures. |
| Documentation closure | `docs/metal-backend.md`, `docs/calibration-autotune.md`, `docs/testing.md`, `docs/troubleshooting.md`, `docs/architecture.md` | Phase 4 docs updates and generated doc style | Update existing docs in place; keep benchmark reports as explain artifacts, not runtime sources of truth. |

## Concrete Patterns

- Tests should use exact string assertions for renderer/report contracts, e.g. `assertTrue(text.contains("backendSelectionCost:"))` and `assertTrue(json.contains("\"rejectedFinalists\": ["))`.
- Native Metal tests should require `System.getProperty("synaptik.metal.mps.lib")` and bridge availability through `assumeTrue(...)`, not hard-fail portable test runs.
- Hygiene checks should be file-walking or source/string assertions that can run under `./gradlew test --tests SourceTreeHygieneTest`.
- Plans must not stage or commit `profiles/platform/.../tuning/abc/*` or `.planning/tmp/`.
