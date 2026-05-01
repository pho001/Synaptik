<!-- generated-by: gsd-doc-writer -->
# Testing

Navigation: [Index](index.md#recommended-reading-paths) | [Development](development.md#local-setup) | [Tensor API](tensor-api.md#operation-catalog) | [Graph Optimizer](graph-optimizer.md#adding-or-changing-optimizer-behavior) | [Native Bridges & BLAS](native-bridges-and-blas.md#openblas-bridge-lifecycle) | [Metal Backend](metal-backend.md#tests) | [Compute Flow](compute-flow.md#traces) | [Troubleshooting](troubleshooting.md#openblas-missing-or-unavailable)

Chapters: [Test Framework And Setup](#test-framework-and-setup) | [Test Organization](#test-organization) | [Exact Commands](#exact-commands) | [Full Suite Duration And Heap Behavior](#full-suite-duration-and-heap-behavior) | [Targeted Test Patterns](#targeted-test-patterns) | [Debug And Benchmark Tests](#debug-and-benchmark-tests) | [Native And Optional Backend Tests](#native-and-optional-backend-tests) | [Source Hygiene Tests](#source-hygiene-tests) | [How To Interpret Failures](#how-to-interpret-failures)

Synaptik uses JUnit Jupiter through Gradle. Test configuration is in `build.gradle`, and tests live under `src/test/java`.

## Table Of Contents

- [Test Framework And Setup](#test-framework-and-setup)
- [Test Organization](#test-organization)
- [Exact Commands](#exact-commands)
- [Full Suite Duration And Heap Behavior](#full-suite-duration-and-heap-behavior)
- [Targeted Test Patterns](#targeted-test-patterns)
- [Debug And Benchmark Tests](#debug-and-benchmark-tests)
- [Native And Optional Backend Tests](#native-and-optional-backend-tests)
- [Source Hygiene Tests](#source-hygiene-tests)
- [How To Interpret Failures](#how-to-interpret-failures)

## Test Framework And Setup

Verified setup:

- Test dependency: `org.junit.jupiter:junit-jupiter:5.11.2`
- Runtime launcher: `org.junit.platform:junit-platform-launcher`
- Gradle test tasks call `useJUnitPlatform()`
- Test JVM args include `--add-modules=jdk.incubator.vector` and `--enable-native-access=ALL-UNNAMED`
- Default test heap is `2g`, unless overridden with `-Dsynaptik.testMaxHeap=<size>`
- Java compilation also passes `--add-modules jdk.incubator.vector`

The project has 163 `*Test.java` files, including 31 files under `src/test/java/debug`.

## Test Organization

| Path | Coverage |
|---|---|
| `src/test/java/*Test.java` | Core tensor ops, graph compilation, optimizer rules, execution, tuning, source hygiene |
| `src/test/java/backend` | Backend contracts and accelerator bridge/lowering tests |
| `src/test/java/backend/cpu` | CPU partition/lowering/fused/kernel planning tests |
| `src/test/java/backend/cuda` | CUDA bridge and lowering scaffolding tests |
| `src/test/java/backend/metal` | Metal bridge and lowering scaffolding tests |
| `src/test/java/config` | Optimizer/config behavior |
| `src/test/java/graph` | Graph execution, codegen, optimizer, memory, region behavior |
| `src/test/java/synaptik/app` | CLI parsing and app entry point tests |
| `src/test/java/tuning` | Tuning integration and workload isolation |
| `src/test/java/debug` | JUnit-driven benchmark and profile comparison tests |
| `src/test/resources/tuning/etalon/inference-performance-baseline.properties` | Baseline medians for `EtalonPerformanceRegressionTest` |

Many root tests are in the default package. Package-qualified tests exist under paths such as `synaptik.app`, `backend.cuda.bridge`, `backend.metal.bridge`, `graph.optimizer`, and `debug`.

## Exact Commands

Compile main and test code:

```bash
./gradlew classes
./gradlew testClasses
```

Run the full suite:

```bash
./gradlew test --no-daemon
```

Run the full suite from a clean test execution:

```bash
./gradlew test --no-daemon --rerun-tasks
```

Override heap when the suite or benchmark-style tests need more room:

```bash
./gradlew test --no-daemon -Dsynaptik.testMaxHeap=4g
```

Run one default-package test class:

```bash
./gradlew test --no-daemon --tests MatMulTest
```

Run one package-qualified test class:

```bash
./gradlew test --no-daemon --tests synaptik.app.TuningCliParsingTest
```

Run one debug benchmark test:

```bash
./gradlew test --no-daemon --tests debug.AbcCurrentBestProfileBenchmarkTest
```

Run one test method:

```bash
./gradlew test --no-daemon --tests "MatMulTest.matMulShapeMismatchThrows"
```

Run source hygiene checks through the JUnit class:

```bash
./gradlew test --no-daemon --tests SourceTreeHygieneTest
```

Run the Phase 5 accelerator closure verification slices:

```bash
./gradlew test --tests BenchmarkSessionTest --tests PreparedExecutionBuildTest
./gradlew test --tests StandardWorkloadsTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest
./gradlew metalTest
./gradlew test --tests SourceTreeHygieneTest
```

Run the Gradle source artifact task directly:

```bash
./gradlew verifySourceTreeClean
```

Remove generated artifacts that the build knows how to clean from `src/` or `test/`:

```bash
./gradlew cleanSourceArtifacts
```

Build the optional macOS Metal MPS shim:

```bash
./gradlew buildMetalMpsShim
```

Run the explicit optional native build lifecycle:

```bash
./gradlew nativeBuild
```

Run the Metal/MPS test slice with the freshly built shim:

```bash
./gradlew metalTest
```

`metalTest` is intentionally separate from `test`, `check`, and `build`. It is for macOS machines with the Metal toolchain available; Java-only verification remains portable.

## Full Suite Duration And Heap Behavior

A recent local verification in this workspace ran:

```bash
./gradlew test --no-daemon --rerun-tasks
```

and took roughly 25 minutes after the default test heap was set to `2g`. Treat that duration as an environment-specific observation, not a stable performance contract.

This is consistent with the codebase shape: the default test suite includes execution tests, optimizer tests, tuning/calibration tests, source hygiene tests, native bridge availability tests, debug benchmark tests, and etalon performance regression tests.

Heap behavior is controlled in `build.gradle`:

```text
System property: synaptik.testMaxHeap
Default when absent: 2g
```

Use `-Dsynaptik.testMaxHeap=4g` for heap failures. Do not use `org.gradle.jvmargs` as the first fix for test heap problems; the project has a dedicated test-task property.

## Targeted Test Patterns

Use these after changing specific areas:

| Change area | Commands |
|---|---|
| Tensor public API or basic ops | `./gradlew test --no-daemon --tests AllOpsTest --tests TensorAddTest --tests TensorUnaryCanonicalizationTest` |
| Broadcasting | `./gradlew test --no-daemon --tests BroadcastContractMatrixTest --tests BroadcastBinaryOpsTest --tests BroadcastPlannerTest` |
| DType/storage behavior | `./gradlew test --no-daemon --tests DataTypeExecutionCoverageTest --tests TensorStorageDataTypeTest --tests Int32IndexDtypeTest` |
| Matmul/linear | `./gradlew test --no-daemon --tests MatMulTest --tests LinearExecutionTest --tests LinearLoweringRuleTest` |
| Conv/pool | `./gradlew test --no-daemon --tests Conv2dExecutionTest --tests Conv2dLoweringRuleTest --tests Pool2dExecutionTest` |
| Losses | `./gradlew test --no-daemon --tests CrossEntropyLossExecutionTest --tests IndexTargetCrossEntropyLossExecutionTest --tests NllLossExecutionTest` |
| Gradients | `./gradlew test --no-daemon --tests GradientEngineRegressionTest --tests BroadcastContractMatrixTest` |
| Optimizer rewrite/CSE/fusion/memory | `./gradlew test --no-daemon --tests AlgebraicRewritingPowTest --tests CommonSubexpressionEliminationRuleTest --tests graph.optimizer.GraphOptimizerSinglePassTest --tests graph.optimizer.region.RegionOptimizationRuleTest --tests graph.optimizer.memory.MemoryPlannerRegionViewTest` |
| Backend boundaries | `./gradlew test --no-daemon --tests SourceTreeHygieneTest --tests backend.ComputeBackendTest` |
| Metal layout-aware device flow | `./gradlew test --no-daemon --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest`<br>`./gradlew test --no-daemon --tests backend.metal.buffer.MetalBufferAllocatorTest --tests backend.metal.bridge.MetalMpsFfmBridgeTest`<br>`./gradlew test --no-daemon --tests backend.metal.MetalLayoutAwareDeviceFlowTest --tests backend.metal.MetalBufferTraceSmokeTest`<br>`./gradlew classes`<br>`./gradlew metalTest` |
| CPU planning | `./gradlew test --no-daemon --tests CpuExecutionPlannerDispatchHeuristicsTest --tests backend.cpu.kernels.ElementwiseDispatchPlanningTest --tests backend.cpu.kernels.FusedDispatchPlanningTest` |
| Tuning/calibration | `./gradlew test --no-daemon --tests CalibrationFamilyRegistryTest --tests PlatformCalibrationDefaultsTest --tests TuningStoreTest --tests tuning.integration.SessionWorkloadIsolationTest` |
| CLI parsing | `./gradlew test --no-daemon --tests synaptik.app.TuningCliParsingTest --tests AppEntryPointTest` |

Gradle accepts multiple `--tests` filters in one command, as shown above.

## Debug And Benchmark Tests

`src/test/java/debug` contains benchmark/profile comparison tests in the `debug` package. They are not separate Gradle tasks. Run them explicitly when you need their output:

```bash
./gradlew test --no-daemon --tests "debug.*"
./gradlew test --no-daemon --tests debug.TransformerHotPathCurrentBestProfileBenchmarkTest
./gradlew test --no-daemon --tests debug.MetalMatMulBenchmarkTest
```

Some debug tests require existing profile artifacts. For example, `debug.AbcCurrentBestProfileBenchmarkTest` looks for:

```text
profiles/platform/<platform-id>/tuning/abc/<dtype>-best-profile.json
build/tuning/best-profiles/abc-<dtype>-best-profile.json
```

If those files are missing, run the CLI flow first:

```bash
./gradlew run --args="calibrate --dtype f64 --families all --preset quick --progress lines --color never"
./gradlew run --args="autotune f64"
```

`EtalonPerformanceRegressionTest` is tagged with `@Tag("benchmark")`, but `build.gradle` does not configure tag exclusion. The default `test` task can run it. It writes:

```text
build/tuning-etalon-regression/current-inference-suite.json
```

and compares current medians to:

```text
src/test/resources/tuning/etalon/inference-performance-baseline.properties
```

## Native And Optional Backend Tests

OpenBLAS-related tests use `OpenBlasFfmBridge.isAvailable()` assumptions in files such as:

- `src/test/java/MatMulTest.java`
- `src/test/java/LinearExecutionTest.java`
- `src/test/java/BFloat16BlasDispatchTest.java`
- `src/test/java/ComputeModeTraceTest.java`

OpenBLAS lookup order:

```text
-Dopenblas.lib=<path>
OPENBLAS_LIB
openblas
```

OpenBLAS tests validate an optional Java FFM bridge. A skipped OpenBLAS test usually means the local library or symbols
were not available; it does not mean the Java fallback path is broken. For the exact BLAS/GEMM dispatch model, bridge
symbol lookup, and fallback behavior, see [Native Bridges & BLAS: OpenBLAS Bridge Lifecycle](native-bridges-and-blas.md#openblas-bridge-lifecycle).

Metal tests use `-Dsynaptik.metal.mps.lib=<path>` or `SYNAPTIK_METAL_MPS_LIB` for explicit shim loading. CUDA tests use `-Dsynaptik.cuda.graph.lib=<path>` or `SYNAPTIK_CUDA_GRAPH_LIB`.

Build and run Metal bridge tests on macOS:

```bash
./gradlew buildMetalMpsShim
./gradlew test --no-daemon --tests backend.metal.bridge.MetalMpsFfmBridgeTest -Dsynaptik.metal.mps.lib=build/native/apple/libsynaptik_apple_mps.dylib
```

Preferred Metal slice for day-to-day native verification:

```bash
./gradlew metalTest
```

The task filters to Metal-specific tests, including `backend.metal.*` and `PreparedExecutionBuildTest.gpuMetal*`, and injects the `synaptik.metal.mps.lib` system property. If you need one isolated class or method, keep using `./gradlew test --tests ... -Dsynaptik.metal.mps.lib=...`.

For what each Metal test proves, including the native buffer ABI and adjacent-region buffer handoff, see
[Metal Backend: Tests](metal-backend.md#tests).

Build and run the optional CUDA graph shim tests when `nvcc` and CUDA hardware are available:

```bash
./gradlew buildCudaGraphShim cudaTest
```

The task writes `build/native/cuda/libsynaptik_cuda_graph.*`, sets `-Dsynaptik.cuda.graph.lib=` for the CUDA-focused
test slice, and uses the same lookup path as `SYNAPTIK_CUDA_GRAPH_LIB`. Native CUDA tests skip when nvcc or CUDA
hardware is unavailable. A skip is acceptable only when portable Java gates such as
`backend.cuda.bridge.CudaFfmBridgeTest`, `backend.cuda.buffer.CudaAcceleratorBufferBinderTest`, and
`backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest` pass.

CUDA fallback interpretation in test output should use `acceleratorBufferReasonCode`, `cudaFallbackReason`,
`NATIVE_BUFFER_ABI_UNAVAILABLE`, `REQUIRED_BUFFER_EXECUTION_UNAVAILABLE`, and `NATIVE_BUFFER_EXECUTION_FAILED` before
treating a fallback as a regression.

## GPU Coverage Regression Gates

Phase 13 coverage tests verify the GPU coverage summary and regression-gate contract with portable Java tests first.
Use `GpuCoverageSummaryTest`, `GpuCoverageRegressionGateTest`, `BenchmarkSessionTest`, `BenchmarkSuiteSessionTest`, and
`CompiledGraphTraceTest` to prove report schema, fallback visibility, and gate failures even when native CUDA is
capability-skipped.

The coverage report fields include `gpuCoverageRatio`, `selectedRegionCount`, `maxSelectedRegionLength`,
`rejectedCandidateReasonCounts`, `cpuMaterializationReasonCounts`, and `deviceHandoffCount`. These fields are the
checked-in evidence contract for coverage/materialization behavior, not raw timing. Native Metal and CUDA tasks add
native capability-gated evidence, but portable coverage gate behavior must not depend on local GPU availability.

Run the focused gate with:

```bash
./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests CompiledGraphTraceTest
```

For native slices, run:

```bash
./gradlew metalTest
./gradlew buildCudaGraphShim cudaTest
```

A hidden tensor-array fallback is a regression-gate failure when the policy requires native buffer binding. Local
`profiles/platform/.../tuning/abc/*` files and other machine-local benchmark/calibration output are not canonical test
fixtures; do not commit local tuning artifacts unless a plan explicitly says to promote them.

### Phase 20 coverage regression hardening

Phase 20 coverage regression hardening makes the hard GPU residency gate a report/trace evidence contract:
`hot path stayed on GPU is trace/report evidence, not timing-only`. Portable Java tests verify coverage gate behavior
and report rendering first; native Metal/CUDA checks are additive capability-gated evidence through fields such as
`targetCoverageGates`, `nativeEvidence`, and `capabilitySkipped`.

`tensor-array bridge execution is not native buffer GPU coverage`. A fast benchmark is not enough if the trace shows a
CPU fallback, hidden tensor-array bridge, missing native buffer binding, unexpected CPU materialization, or unexpected
device handoff. Phase 20 closure also records that `profiles/platform/.../tuning/abc/* remained unstaged`.

### Phase 14 GPU coverage triage

Phase 14 GPU coverage triage is portable Java tests and report-contract evidence. It does not require a native Metal
or CUDA device to prove the target registry, gap ranking, text renderer, or JSON renderer contract.

Run:

```bash
./gradlew test --tests GpuCoverageGapTriageTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageTriageReportTest
```

Use `BenchmarkSuiteSessionTest` and `GpuCoverageSummaryTest` with those tests for final phase closure. Native Metal and
CUDA results are capability-gated native evidence and should be treated as additive proof, not as a replacement for
the portable Java tests.

## Source Hygiene Tests

`SourceTreeHygieneTest` checks architecture and migration boundaries. Notable checks include:

- generated artifacts in `src/` or `test/`: `.class`, `.java.txt`, `.java.bak`, `.java.orig`, `.java.tmp`, `.tmp*`, `*.tmp*`, `.DS_Store`, `*~`
- root-level concrete backend helpers under `src/main/java/backend`
- legacy package paths such as `graph.fused`, `graph.codegen`, `graph.optimizer.fusion`, `operations.fused`
- CPU kernels outside `src/main/java/backend/cpu/kernels`
- CUDA/OpenCL kernels under old `src/main/java/backend/kernels/...` paths
- graph autotune candidates importing runtime/backend config
- production CLI references to removed stage-order profile mutators

Gradle also registers `verifySourceTreeClean`, which checks generated artifacts under `src/` and `test/`, and wires it into `check`.

## How To Interpret Failures

Use the failure location to classify the issue:

- `OutOfMemoryError: Java heap space`: rerun with `-Dsynaptik.testMaxHeap=4g`; inspect debug/benchmark tests if it happens only in full suite.
- `module jdk.incubator.vector not found`: the active JDK is not JDK 25 with the incubator vector module available.
- Native library unavailable with skipped tests: an assumption skipped optional native coverage; configure the relevant library only if the change targets that path.
- Native library unavailable with a hard failure: a runtime/profile required `OPENBLAS_FFM`, Metal, or CUDA instead of falling back.
- `Broadcast mismatch at dim ...`: inspect the shape builder before backend code; this comes from `BroadcastPlanner`.
- Shape assertion failures after prepare/execute: inspect operation descriptor parameters and prepared layout/broadcast metadata.
- Forward output mismatch: compare no-optimization execution with optimized execution to isolate rewrite/fusion/memory changes.
- Gradient mismatch or `getGradient()` null: verify `setRequiresGrad(true)` on leaf tensors and backward wiring in the relevant `tensor.ops.*` builder.
- `UnsupportedOperationException: <Kernel> does not support <dtype>`: implement the dtype entry point or adjust dtype routing in prepared metadata.
- `Cannot resolve CPU kernel for UNKNOWN operation type` or missing switch case: update `CpuKernelResolver`.
- Source hygiene failure: move code to the owner package rather than adding suppressions.
- Performance regression failure: read `build/tuning-etalon-regression/current-inference-suite.json`, compare the reported candidate/workload to the baseline resource, then rerun targeted debug benchmarks to reduce noise.
