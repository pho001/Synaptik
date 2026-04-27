<!-- generated-by: gsd-doc-writer -->
# Testing

Navigation: [Index](index.md) | [Development](development.md) | [Tensor API](tensor-api.md) | [Graph Optimizer](graph-optimizer.md) | [Compute Flow](compute-flow.md) | [Troubleshooting](troubleshooting.md)

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
./gradlew test --no-daemon --tests synaptik.app.MainCliParsingTest
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
| CPU planning | `./gradlew test --no-daemon --tests CpuExecutionPlannerDispatchHeuristicsTest --tests backend.cpu.kernels.ElementwiseDispatchPlanningTest --tests backend.cpu.kernels.FusedDispatchPlanningTest` |
| Tuning/calibration | `./gradlew test --no-daemon --tests CalibrationFamilyRegistryTest --tests PlatformCalibrationDefaultsTest --tests TuningStoreTest --tests tuning.integration.SessionWorkloadIsolationTest` |
| CLI parsing | `./gradlew test --no-daemon --tests synaptik.app.MainCliParsingTest --tests AppEntryPointTest` |

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

Metal tests use `-Dsynaptik.metal.mps.lib=<path>` or `SYNAPTIK_METAL_MPS_LIB` for explicit shim loading. CUDA tests use `-Dsynaptik.cuda.graph.lib=<path>` or `SYNAPTIK_CUDA_GRAPH_LIB`.

Build and run Metal bridge tests on macOS:

```bash
./gradlew buildMetalMpsShim
./gradlew test --no-daemon --tests backend.metal.bridge.MetalMpsFfmBridgeTest -Dsynaptik.metal.mps.lib=build/native/apple/libsynaptik_apple_mps.dylib
```

Needs verification: CUDA shim build instructions are not present in the repository; only the Java bridge and tests were found.

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
