# Coding Conventions

**Analysis Date:** 2026-04-29

## Naming Patterns

**Files:**
- Use Java standard `PascalCase.java` for normal production classes and records such as `src/main/java/tensor/Tensor.java`, `src/main/java/tensor/ops/binary/TensorBinaryOps.java`, `src/main/java/backend/cpu/kernels/elementwise/binary/CpuAddKernel.java`, and `src/main/java/config/optimizer/OptimizerConfig.java`.
- Use lower camel-case descriptor filenames only under `src/main/java/operations/**`, matching the operation class name exactly, such as `src/main/java/operations/elementwise/binary/add.java`, `src/main/java/operations/nn/conv/conv2d.java`, and `src/main/java/operations/linalg/scaledDotProductAttention.java`.
- Name tests as `*Test.java`. Root/default-package tests cover broad behavior, for example `src/test/java/MatMulTest.java`, `src/test/java/BroadcastContractMatrixTest.java`, and `src/test/java/SourceTreeHygieneTest.java`. Package-scoped tests mirror production packages, for example `src/test/java/config/optimizer/OptimizerConfigTest.java` and `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java`.
- Keep package roots lowercase. `src/test/java/LowercasePackageNamingTest.java` rejects uppercase root package references such as `Backend`, `Graph`, `Operations`, or `Tensor`.

**Functions:**
- Use lower camelCase for public API and helper methods, as in `TensorOps.add(...)` in `src/main/java/tensor/TensorOps.java`, `TensorBinaryOps.add(...)` in `src/main/java/tensor/ops/binary/TensorBinaryOps.java`, and `OptimizerConfig.trainingDefaults()` in `src/main/java/config/optimizer/OptimizerConfig.java`.
- Use domain-specific factory/default methods for immutable config objects: `defaults()`, `trainingDefaults()`, `inferenceDefaults()`, `noOptimization()`, and `withX(...)` appear in `src/main/java/config/optimizer/OptimizerConfig.java` and `src/main/java/tensor/options/Conv2dOptions.java`.
- Use descriptive test method names that state behavior, for example `matMulShapeMismatchThrows` in `src/test/java/MatMulTest.java`, `trainingDefaultsIncludePartFuseAndMem` in `src/test/java/config/optimizer/OptimizerConfigTest.java`, and `sourceTreeDoesNotContainCompiledOrTempArtifacts` in `src/test/java/SourceTreeHygieneTest.java`. Some older tests use `public void test...` naming in `src/test/java/BroadcastPlannerTest.java`; prefer the descriptive lower camelCase style for new tests.

**Variables:**
- Use lower camelCase for locals and fields, such as `computedShape`, `prevTensors`, and `forcedBackend` in `src/main/java/tensor/Tensor.java`.
- Use short mathematical names only where the algorithm is dense and local, for example `a`, `b`, and `out` in execution tests such as `src/test/java/MatMulTest.java`.
- Use `expected`, `actual`, `baseline`, and `out` names in tests to make assertions readable, as in `src/test/java/BroadcastContractMatrixTest.java` and `src/test/java/EtalonPerformanceRegressionTest.java`.
- Use `var` sparingly for obvious fluent builder outputs or temporary files in tests, as in `src/test/java/tuning/api/SynaptikTuningApiTest.java` and `src/test/java/CalibrationRunStoreTest.java`.

**Types:**
- Use `PascalCase` for normal classes, records, and enums, such as `Tensor`, `DataType`, `RuntimeConfig`, `OptimizerConfig`, `CpuKernelResolver`, and `MetalMpsFfmBridge`.
- Use lowercase operation descriptor classes under `operations.*` by design, such as `add`, `pow`, `matmul`, and `conv2d`; these implement `operations.Operation` and return an uppercase `Operation.OpType` from `opType()`.
- Use `SCREAMING_SNAKE_CASE` for constants and enum values, such as `Tensor.SYSTEM_FORWARD_OUTPUT_LABEL` in `src/main/java/tensor/Tensor.java`, `Operation.OpType.ADD` in `src/main/java/operations/Operation.java`, and static singleton kernel fields in `src/main/java/backend/cpu/registry/CpuKernelResolver.java`.

## Code Style

**Formatting:**
- No formatter config is detected. There is no `.editorconfig`, Checkstyle, PMD, SpotBugs, Spotless, or JaCoCo configuration in the repository root; `build.gradle` only applies `java` and `application`.
- Use 4-space indentation for Java code. `src/main/java/tensor/ops/binary/TensorBinaryOps.java`, `src/main/java/backend/cpu/kernels/elementwise/binary/CpuAddKernel.java`, and `src/test/java/MatMulTest.java` follow this style.
- Keep long constructor and method argument lists vertically aligned with one argument per line when they exceed normal line length, as in `src/main/java/config/optimizer/OptimizerConfig.java`, `src/main/java/backend/cpu/kernels/elementwise/binary/CpuAddKernel.java`, and `src/test/java/BroadcastContractMatrixTest.java`.
- Prefer Java 25-compatible modern language features already present in the codebase: records in `src/main/java/config/optimizer/OptimizerConfig.java`, switch expressions in `src/main/java/backend/cpu/registry/CpuKernelResolver.java`, `List.of(...)` in `src/test/java/config/optimizer/OptimizerConfigTest.java`, and `toList()` stream terminals in `src/test/java/SourceTreeHygieneTest.java`.
- Keep Gradle-managed generated artifacts out of source. `build.gradle` registers `verifySourceTreeClean` and `cleanSourceArtifacts`, and `src/test/java/SourceTreeHygieneTest.java` enforces the same source-tree hygiene from JUnit.

**Linting:**
- Not detected. The practical lint layer is test-based, especially `src/test/java/SourceTreeHygieneTest.java`, `src/test/java/LowercasePackageNamingTest.java`, and architecture tests such as `src/test/java/CpuKernelFamilyArchitectureTest.java`.
- Do not rely on a formatter or linter to catch style drift. Match nearby code in the edited package and add focused hygiene assertions when introducing a new architectural boundary.

## Import Organization

**Order:**
1. Production/package imports first, grouped by domain, as in `src/test/java/MatMulTest.java` and `src/main/java/graph/optimizer/cse/CommonSubexpressionEliminationRule.java`.
2. JUnit imports before static assertions in tests, as in `src/test/java/BroadcastContractMatrixTest.java` and `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java`.
3. `java.*` imports before static imports in many tests, as in `src/test/java/SourceTreeHygieneTest.java` and `src/test/java/tuning/api/SynaptikTuningApiTest.java`.
4. Static assertion and assumption imports last, as in `src/test/java/MatMulTest.java`, `src/test/java/config/optimizer/OptimizerConfigTest.java`, and `src/test/java/backend/metal/bridge/MetalMpsFfmBridgeTest.java`.

**Path Aliases:**
- Not applicable. Java packages map directly to `src/main/java/**` and `src/test/java/**`; there are no Gradle source-set aliases in `build.gradle`.

**Wildcard Imports:**
- Avoid wildcard imports for new code unless matching a dense local registry or ASM emitter pattern. Existing examples include `java.util.*` in `src/main/java/tensor/Tensor.java`, kernel-family imports in `src/main/java/backend/cpu/registry/CpuKernelResolver.java`, and ASM opcode static imports in `src/main/java/backend/cpu/fused/codegen/FusedClassEmitter.java`.
- Prefer explicit static assertion imports in tests. A few older tests use `import static org.junit.jupiter.api.Assertions.*;`, such as `src/test/java/BroadcastPlannerTest.java`, `src/test/java/AllOpsTest.java`, and `src/test/java/TensorAddTest.java`.

## Error Handling

**Patterns:**
- Use `IllegalArgumentException` for invalid public inputs, shape mismatches, unsupported option values, and parser errors. Examples include `src/main/java/tensor/TensorMetadata.java`, `src/main/java/tensor/options/Conv2dOptions.java`, `src/main/java/tensor/ops/conv/TensorConvOps.java`, and `src/main/java/synaptik/app/TuningCli.java`.
- Use `UnsupportedOperationException` when a dtype, backend, or mutating operation is intentionally unsupported. Examples include dtype conversion guards in `src/main/java/tensor/Tensor.java`, storage conversion guards in `src/main/java/tensor/TensorStorageSupport.java`, and accelerator backend stubs in `src/main/java/backend/cuda/CudaBackend.java` and `src/main/java/backend/opencl/OpenClBackend.java`.
- Use `IllegalStateException` for missing runtime state, unavailable native symbols after a path is selected, impossible branches, and invalid execution phases. Examples include `src/main/java/graph/execution/ExecutionState.java`, `src/main/java/backend/blas/OpenBlasFfmBridge.java`, and `src/main/java/backend/cpu/registry/CpuKernelResolver.java`.
- Prefer messages that include the failing domain value, such as shape/rank/dtype/path/node id. Examples are `src/main/java/tensor/TensorLayoutTransform.java`, `src/main/java/graph/execution/ExecutionState.java`, and `src/main/java/synaptik/app/TuningCli.java`.
- Null handling is mixed. Public tensor/config code often throws `IllegalArgumentException` with explicit messages, while some builder code relies on access or validation paths. For new public API code, perform explicit null checks near the boundary and use messages consistent with nearby files.

## Logging

**Framework:** `System.out` / `System.err` and injectable `PrintStream`-style listeners.

**Patterns:**
- CLI and report rendering write to stdout directly in `src/main/java/synaptik/app/TuningCli.java`, `src/main/java/synaptik/app/Main.java`, `src/main/java/numerics/NumericsCli.java`, and `src/main/java/tuning/etalon/FrameworkEtalonCli.java`.
- Progress reporting uses output-injected listener objects, such as `src/main/java/tuning/autotune/LoggingAutotuneProgressListener.java`, `src/main/java/tuning/calibration/progress/LoggingPlatformCalibrationProgressListener.java`, `src/main/java/tuning/calibration/progress/TerminalCalibrationProgressRenderer.java`, and `src/main/java/tuning/calibration/run/CalibrationRunner.java`.
- Low-level BLAS fallback warnings write to stderr in `src/main/java/backend/cpu/kernels/linalg/matmul/blas/MatMulBlasBackend.java`.
- Avoid adding general-purpose logging framework dependencies unless a phase explicitly introduces observability infrastructure. Follow the existing pattern of renderer/listener classes for user-facing progress and explicit stderr for exceptional fallback diagnostics.

## Comments

**When to Comment:**
- Use Javadoc on public API, config records, and important operation builders. `src/main/java/tensor/Tensor.java`, `src/main/java/tensor/ops/binary/TensorBinaryOps.java`, `src/main/java/tensor/options/Conv2dOptions.java`, and `src/main/java/config/optimizer/OptimizerConfig.java` are reference patterns.
- Keep comments focused on contracts, invariants, dtype restrictions, shape assumptions, and optimizer/backend safety boundaries. Avoid narrating obvious assignments.
- Use short inline comments only for non-obvious numerical or backend behavior. Prefer extracting support methods with descriptive names in packages such as `src/main/java/tensor/ops/*Support.java` and `src/main/java/backend/cpu/kernels/**`.

**JSDoc/TSDoc:**
- Not applicable. This is a Java codebase; use Java Javadoc.

**Javadoc Pattern:**
```java
/**
 * Adds two tensors elementwise with NumPy-style broadcasting.
 *
 * @param first left operand; must be non-null and floating numeric
 * @param second right operand; must be non-null and floating numeric
 * @return broadcasted sum tensor with promoted floating dtype
 * @throws NullPointerException if either input is null
 * @throws IllegalArgumentException if inputs are non-floating or not broadcast-compatible
 */
public static Tensor add(Tensor first, Tensor second) {
    ...
}
```

Use this style for new public operation methods near `src/main/java/tensor/ops/binary/TensorBinaryOps.java` and public facades in `src/main/java/tensor/TensorOps.java`.

## Function Design

**Size:** Keep new methods focused on one validation/build/execute concern. Long existing files such as `src/main/java/tensor/Tensor.java`, `src/main/java/synaptik/app/TuningCli.java`, and `src/test/java/PreparedExecutionBuildTest.java` are central surfaces rather than a target style for new code.

**Parameters:** Use explicit domain parameters instead of generic maps. Config records in `src/main/java/config/**`, option records in `src/main/java/tensor/options/**`, and execution methods in `src/main/java/graph/execution/**` model this pattern.

**Return Values:** Prefer immutable values and defensive copies at API boundaries. Examples include `List.copyOf(...)` in `src/main/java/config/optimizer/OptimizerConfig.java`, array clones in `src/main/java/tensor/TensorMetadata.java`, and immutable operation descriptors in `src/main/java/operations/**`.

**Validation:** Validate at construction or API entry. Compact record constructors in `src/main/java/tensor/options/Conv2dOptions.java`, `src/main/java/config/optimizer/OptimizerConfig.java`, and related config records are the preferred pattern for config invariants.

**Operation Builders:** For tensor operations, keep public facade methods thin and delegate to family builders:
```java
public static Tensor add(Tensor first, Tensor second) {
    return TensorBinaryOps.add(first, second);
}
```

Then place validation, broadcast planning, descriptor construction, and autograd setup in `src/main/java/tensor/ops/<family>/`, as shown by `src/main/java/tensor/ops/binary/TensorBinaryOps.java`.

## Module Design

**Exports:** Use final utility/facade classes with private constructors for static-only APIs, such as `src/main/java/tensor/TensorOps.java`, `src/main/java/tensor/ops/binary/TensorBinaryOps.java`, and `src/main/java/backend/cpu/registry/CpuKernelResolver.java`.

**Barrel Files:** Java package barrels are not used. Public aggregation happens through facade classes such as `src/main/java/tensor/Tensor.java`, `src/main/java/tensor/TensorOps.java`, `src/main/java/backend/ComputeEngine.java`, and `src/main/java/synaptik/app/TuningCli.java`.

**Layering Rules:**
- Keep public tensor methods in `src/main/java/tensor/Tensor.java` and static wrappers in `src/main/java/tensor/TensorOps.java`.
- Put family-specific operation construction and backward formulas under `src/main/java/tensor/ops/<family>/`.
- Put immutable operation descriptors under `src/main/java/operations/<family>/`.
- Put CPU kernel implementations under `src/main/java/backend/cpu/kernels/<family>/` and register them in `src/main/java/backend/cpu/registry/CpuKernelResolver.java`.
- Keep optimizer rule ownership in domain packages under `src/main/java/graph/optimizer/**`; `src/test/java/SourceTreeHygieneTest.java` rejects legacy `graph.optimizer.rules` ownership.
- Keep graph autotune candidates graph-policy-only; `src/test/java/SourceTreeHygieneTest.java` enforces this boundary for `src/main/java/tuning/candidate/graph`.

**Architecture Guard Tests:** When moving code across layers, update or add source-hygiene assertions in `src/test/java/SourceTreeHygieneTest.java`, package naming checks in `src/test/java/LowercasePackageNamingTest.java`, or focused architecture tests such as `src/test/java/CpuKernelFamilyArchitectureTest.java`.

---

*Convention analysis: 2026-04-29*
