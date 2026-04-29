# Testing Patterns

**Analysis Date:** 2026-04-29

## Test Framework

**Runner:**
- JUnit Jupiter `5.11.2`.
- Config: `build.gradle`.
- Gradle wrapper: `gradle/wrapper/gradle-wrapper.properties` uses Gradle `9.4.1`.
- Test classes live under `src/test/java`.
- Test resources live under `src/test/resources`.

**Assertion Library:**
- JUnit Jupiter assertions from `org.junit.jupiter.api.Assertions`.
- JUnit Jupiter assumptions from `org.junit.jupiter.api.Assumptions` for optional native/backend tests.
- JUnit Jupiter parameterized tests from `org.junit.jupiter.params.ParameterizedTest` and providers such as `EnumSource`.
- No Mockito, AssertJ, Hamcrest, or custom mocking framework is detected in `build.gradle` or `src/test/java`.

**Run Commands:**
```bash
./gradlew test              # Run all default JUnit tests
./gradlew test --no-daemon  # Run all tests with stable daemon behavior for long suites
./gradlew test --no-daemon --rerun-tasks  # Force a clean test execution
./gradlew test --no-daemon -Dsynaptik.testMaxHeap=4g  # Override test heap
./gradlew test --no-daemon --tests MatMulTest  # Run one default-package test class
./gradlew test --no-daemon --tests synaptik.app.TuningCliParsingTest  # Run one package-qualified test class
./gradlew test --no-daemon --tests "MatMulTest.matMulShapeMismatchThrows"  # Run one method
./gradlew verifySourceTreeClean  # Run Gradle source-artifact hygiene check
./gradlew metalTest  # Build and run the macOS Metal/MPS test slice
```

**Gradle Test Settings:**
- `build.gradle` calls `useJUnitPlatform()` for all `Test` tasks.
- `build.gradle` passes `--add-modules=jdk.incubator.vector` and `--enable-native-access=ALL-UNNAMED` to test JVMs.
- `build.gradle` sets test heap to `2g` unless `-Dsynaptik.testMaxHeap=<size>` is supplied.
- `build.gradle` does not configure JUnit tag exclusions, so `@Tag("benchmark")` tests can run under the default `test` task.
- `build.gradle` registers `metalTest` as a separate `Test` task that builds the local Metal shim and filters Metal-related tests.

## Test File Organization

**Location:**
- Tests are under `src/test/java`.
- Broad behavior tests are often in the default package, for example `src/test/java/AllOpsTest.java`, `src/test/java/MatMulTest.java`, `src/test/java/BroadcastContractMatrixTest.java`, and `src/test/java/PreparedExecutionBuildTest.java`.
- Package-qualified tests mirror production packages for newer or more scoped areas, for example `src/test/java/config/optimizer/OptimizerConfigTest.java`, `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java`, `src/test/java/graph/execution/ExecutionStateResidencyTest.java`, and `src/test/java/tuning/api/SynaptikTuningApiTest.java`.
- Debug and benchmark-style tests live under `src/test/java/debug`.
- The only detected test resource file is `src/test/resources/tuning/etalon/inference-performance-baseline.properties`.

**Naming:**
- Test files use `*Test.java`.
- Test methods usually use descriptive lower camelCase names, such as `matMulShapeMismatchThrows` in `src/test/java/MatMulTest.java`, `calibrationDslBuildsCalibrationCommand` in `src/test/java/tuning/api/SynaptikTuningApiTest.java`, and `explicitShimLibrarySupportsBufferAllocatorRoundtrip` in `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java`.
- Older tests may use `public void test...` style, as in `src/test/java/BroadcastPlannerTest.java`; prefer descriptive package-local methods for new tests.

**Structure:**
```text
src/test/java/
├── *Test.java                         # default-package broad tensor/graph/runtime tests
├── backend/**                         # backend contracts, CPU, CUDA, Metal, memory, lowering
├── config/**                          # config and profile tests
├── graph/**                           # graph execution, optimizer, memory, partition, region
├── synaptik/app/**                    # CLI parsing and app shape tests
├── tuning/**                          # tuning API, candidate, and integration tests
└── debug/**                           # benchmark/profile comparison tests
```

**Observed Scale:**
- `src/test/java` contains 178 `*Test.java` files.
- `src/test/java/debug` contains 31 `*Test.java` files.
- `src/main/java` contains 962 Java source files.

## Test Structure

**Suite Organization:**
```java
class OptimizerConfigTest {
    @Test
    void trainingDefaultsIncludePartFuseAndMem() {
        assertEquals(
                List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.PART, OptimizerStage.FUSE, OptimizerStage.MEM),
                OptimizerConfig.trainingDefaults().stageOrder()
        );
    }

    @Test
    void rejectsFuseWithoutPartitionStage() {
        assertThrows(IllegalArgumentException.class, () -> new OptimizerConfig(...));
    }
}
```

Use this focused arrange-act-assert pattern for small unit tests, following `src/test/java/config/optimizer/OptimizerConfigTest.java`.

**Execution Test Pattern:**
```java
Tensor out = a.matmul(b);
CompiledGraph.compile(out, OptimizerConfig.noOptimization())
        .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

assertArrayEquals(new double[]{5, 11, 39, 53}, out.toDoubleArrayCopy(), 1e-9);
```

Use compiled-graph execution for behavior that depends on runtime kernels, following `src/test/java/MatMulTest.java`, `src/test/java/Conv2dExecutionTest.java`, `src/test/java/Pool2dExecutionTest.java`, and `src/test/java/BroadcastContractMatrixTest.java`.

**Patterns:**
- Build tensors directly with explicit data, shape, label, and `DataType`, as in `src/test/java/MatMulTest.java`.
- Compile graph outputs with `CompiledGraph.compile(...)` and execute with explicit `RuntimeConfig` and `ExecutionMode`, as in `src/test/java/MatMulTest.java` and `src/test/java/BroadcastContractMatrixTest.java`.
- Use `OptimizerConfig.noOptimization()` for baseline semantic checks and `OptimizerConfig.inferenceDefaults()` or `OptimizerConfig.trainingDefaults()` when testing optimizer/runtime integration, as in `src/test/java/MatMulTest.java`.
- Use expected-value helper methods in matrix-style tests to avoid duplicating large expected arrays, as in `src/test/java/BroadcastContractMatrixTest.java`.
- Use file-walking assertions for architectural hygiene, as in `src/test/java/SourceTreeHygieneTest.java` and `src/test/java/LowercasePackageNamingTest.java`.

## Mocking

**Framework:** None detected.

**Patterns:**
```java
Assumptions.assumeTrue(OpenBlasFfmBridge.isAvailable(), "OpenBLAS FFM is unavailable");
```

Optional system capabilities are skipped with assumptions rather than mocked, as in `src/test/java/MatMulTest.java`, `src/test/java/BFloat16BlasDispatchTest.java`, `src/test/java/LinearExecutionTest.java`, `src/test/java/ComputeModeTraceTest.java`, `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java`, and `src/test/java/backend/cuda/bridge/CudaFfmBridgeTest.java`.

**What to Mock:**
- No repository pattern exists for mocking Java objects. Prefer small real fixtures, explicit config objects, temporary files, and direct domain value construction.
- For optional native libraries, use `assumeTrue(...)` and verify availability/unavailability behavior explicitly.

**What NOT to Mock:**
- Do not mock `Tensor`, `CompiledGraph`, `RuntimeConfig`, or backend bridge types when the test is validating execution behavior. Existing execution tests use real tensors and real compile/execute flows.
- Do not fake source-tree architecture checks. `src/test/java/SourceTreeHygieneTest.java` reads real files from `src/main/java` and `src/test/java`.

## Fixtures and Factories

**Test Data:**
```java
private static ExecutionProfile profile(String name, DataType dtype, RuntimeConfig runtime) {
    return new ExecutionProfile(
            name,
            name,
            dtype,
            ExecutionMode.FORWARD_BACKWARD,
            OptimizerConfig.trainingDefaults(),
            runtime
    );
}
```

Use private static helper methods inside the test class for repeated domain fixtures, following `src/test/java/tuning/api/SynaptikTuningApiTest.java`, `src/test/java/BroadcastContractMatrixTest.java`, and `src/test/java/MatMulTest.java`.

**Location:**
- Inline fixtures live inside each test class.
- Temporary filesystem fixtures use `Files.createTempDirectory(...)` or `Files.createTempFile(...)`, as in `src/test/java/CalibrationRunStoreTest.java`, `src/test/java/TuningStoreTest.java`, `src/test/java/HistoryAwareSearchStrategyTest.java`, and `src/test/java/PlatformCalibrationReportStoreTest.java`.
- Persistent performance baseline data lives in `src/test/resources/tuning/etalon/inference-performance-baseline.properties`.
- Checked-in platform profile artifacts live under `profiles/platform/...` and are consumed by tuning/debug flows documented in `docs/testing.md`.

**Fixture Caveat:**
- `@TempDir` is not used. Tests that create temp files manually should avoid relying on repository-relative cleanup unless the files intentionally belong under `build/`, as in `src/test/java/EtalonPerformanceRegressionTest.java`.

## Coverage

**Requirements:** None enforced.

**View Coverage:**
```bash
# Not configured: no JaCoCo or coverage task is detected in build.gradle.
```

**Coverage Approach:**
- Coverage is behavioral rather than metric-driven. The suite covers tensor operations, dtype/storage behavior, graph compilation, optimizer rules, runtime execution, backend lowering, native bridge availability, tuning/calibration stores, and source hygiene through focused JUnit classes under `src/test/java`.
- Source hygiene is part of verification through `src/test/java/SourceTreeHygieneTest.java` and the Gradle `verifySourceTreeClean` task in `build.gradle`.

## Test Types

**Unit Tests:**
- Config and value-object tests validate constructors, defaults, ordering rules, and invalid inputs. Examples: `src/test/java/config/optimizer/OptimizerConfigTest.java`, `src/test/java/config/profile/WorkloadProfilePresetTest.java`, and `src/test/java/TensorDataFactoryTest.java`.
- Planner and support tests validate pure algorithmic behavior. Examples: `src/test/java/BroadcastPlannerTest.java`, `src/test/java/backend/cpu/kernels/ElementwiseDispatchPlanningTest.java`, and `src/test/java/backend/cpu/kernels/layout/StridedLayoutPlanningTest.java`.

**Integration Tests:**
- Graph and execution tests compile tensors and execute through runtime backends. Examples: `src/test/java/MatMulTest.java`, `src/test/java/LinearExecutionTest.java`, `src/test/java/Conv2dExecutionTest.java`, `src/test/java/PreparedExecutionBuildTest.java`, and `src/test/java/GradientEngineRegressionTest.java`.
- Optimizer and memory tests validate multi-stage graph behavior. Examples: `src/test/java/CommonSubexpressionEliminationRuleTest.java`, `src/test/java/graph/optimizer/GraphOptimizerSinglePassTest.java`, `src/test/java/graph/optimizer/region/RegionOptimizationRuleTest.java`, and `src/test/java/graph/optimizer/memory/MemoryPlannerRegionViewTest.java`.
- Tuning and persistence tests exercise file IO and profile workflows. Examples: `src/test/java/TuningStoreTest.java`, `src/test/java/CalibrationRunStoreTest.java`, `src/test/java/tuning/integration/SessionWorkloadIsolationTest.java`, and `src/test/java/tuning/api/SynaptikTuningApiTest.java`.

**E2E Tests:**
- No browser or separate E2E framework is used.
- CLI parsing and application entry tests live in `src/test/java/synaptik/app/TuningCliParsingTest.java`, `src/test/java/synaptik/app/TuningCliShapeTest.java`, and `src/test/java/AppEntryPointTest.java`.

**Native/Optional Backend Tests:**
- OpenBLAS tests use `OpenBlasFfmBridge.isAvailable()` assumptions in files such as `src/test/java/MatMulTest.java`, `src/test/java/LinearExecutionTest.java`, `src/test/java/BFloat16BlasDispatchTest.java`, and `src/test/java/ComputeModeTraceTest.java`.
- Metal tests use explicit `synaptik.metal.mps.lib` assumptions in `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java`, `src/test/java/backend/metal/MetalBufferTraceSmokeTest.java`, and many Metal paths in `src/test/java/PreparedExecutionBuildTest.java`.
- CUDA bridge tests use explicit library assumptions in `src/test/java/backend/cuda/bridge/CudaFfmBridgeTest.java`.
- Use `./gradlew metalTest` for the maintained Metal slice configured in `build.gradle`.

**Benchmark and Debug Tests:**
- `src/test/java/debug` contains JUnit-driven benchmark/profile comparison tests with stdout report output.
- `src/test/java/EtalonPerformanceRegressionTest.java` is tagged `@Tag("benchmark")`, writes `build/tuning-etalon-regression/current-inference-suite.json`, and compares against `src/test/resources/tuning/etalon/inference-performance-baseline.properties`.
- Because `build.gradle` does not exclude tags, benchmark-tagged tests can run in `./gradlew test`.

## Common Patterns

**Parameterized Testing:**
```java
@ParameterizedTest
@EnumSource(value = DataType.class, names = {"FLOAT64", "FLOAT32", "BFLOAT16"})
void allBroadcastAwareOpsSupportRightAlignedLowerRankRightOperand(DataType dataType) {
    ...
}
```

Use `@ParameterizedTest` and `@EnumSource` for dtype matrices, following `src/test/java/BroadcastContractMatrixTest.java`, `src/test/java/DataTypeExecutionCoverageTest.java`, `src/test/java/MemoryOptimizerRuleDataTypeTest.java`, and `src/test/java/TransformOpsTest.java`.

**Async Testing:**
```java
# Not applicable: no async test helper pattern is detected.
```

The suite is synchronous. Parallelism is exercised through runtime configs and execution planners, not asynchronous test APIs.

**Error Testing:**
```java
IllegalArgumentException ex = assertThrows(
        IllegalArgumentException.class,
        () -> BroadcastPlanner.plan(new int[]{2, 3}, new int[]{3, 1}, new int[]{2, 4}, new int[]{4, 1})
);
assertTrue(ex.getMessage().contains("Broadcast mismatch"));
```

Use `assertThrows(...)` for invalid inputs and assert important message fragments when the message is part of the contract. Examples include `src/test/java/BroadcastPlannerTest.java`, `src/test/java/config/optimizer/OptimizerConfigTest.java`, `src/test/java/synaptik/app/TuningCliParsingTest.java`, and `src/test/java/backend/cuda/exec/PreparedCudaExecutableBufferPolicyTest.java`.

**Source Hygiene Testing:**
```java
try (Stream<Path> paths = Files.walk(root)) {
    List<String> offenders = paths
            .filter(Files::isRegularFile)
            .filter(path -> path.toString().endsWith(".java"))
            .flatMap(path -> ...)
            .sorted()
            .toList();
    assertTrue(offenders.isEmpty(), () -> "message: " + offenders);
}
```

Use real file scans for package and architecture boundary rules, following `src/test/java/SourceTreeHygieneTest.java` and `src/test/java/LowercasePackageNamingTest.java`.

**Known Test Caveats:**
- The default `test` task can run benchmark-style tests because `build.gradle` has no tag exclusion.
- The full suite is large and includes debug/profile tests under `src/test/java/debug`; use targeted `--tests` filters for iteration.
- Test heap defaults to `2g`; use `-Dsynaptik.testMaxHeap=4g` for heap-sensitive runs.
- Optional native tests skip through assumptions when OpenBLAS, Metal, or CUDA libraries are unavailable.
- `metalTest` is macOS-gated through `build.gradle` and builds `build/native/apple/libsynaptik_apple_mps.dylib` before running Metal filters.
- Some tests create temp files through `Files.createTempFile(...)` or `Files.createTempDirectory(...)` without JUnit `@TempDir`; avoid assuming automatic per-test cleanup in new filesystem tests.

---

*Testing analysis: 2026-04-29*
