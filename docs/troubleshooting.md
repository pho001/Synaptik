<!-- generated-by: gsd-doc-writer -->
# Troubleshooting

Navigation: [Index](index.md) | [Configuration](configuration.md) | [Testing](testing.md) | [Compute Flow](compute-flow.md) | [Tensor API](tensor-api.md) | [Calibration & Autotune](calibration-autotune.md)

Chapters: [Java Heap Space](#java-heap-space) | [Incubator Vector API](#incubator-vector-api) | [Missing Native Access](#missing-native-access) | [OpenBLAS Missing Or Unavailable](#openblas-missing-or-unavailable) | [Metal MPS Shim Missing](#metal-mps-shim-missing) | [CUDA Shim Missing](#cuda-shim-missing) | [Validation Mismatch In Benchmark Or Autotune](#validation-mismatch-in-benchmark-or-autotune) | [Shape And Broadcast Errors](#shape-and-broadcast-errors) | [Optimizer Rewrite Bugs](#optimizer-rewrite-bugs) | [Gradients Missing Or Wrong](#gradients-missing-or-wrong) | [Unsupported DType In A Kernel](#unsupported-dtype-in-a-kernel) | [CPU Kernel Resolution Failure](#cpu-kernel-resolution-failure) | [Performance Regressions](#performance-regressions) | [Generated Artifacts In Source Tree](#generated-artifacts-in-source-tree) | [Source Hygiene Architecture Failures](#source-hygiene-architecture-failures) | [Stale Or Missing Profile Artifacts](#stale-or-missing-profile-artifacts)

This document lists concrete Synaptik failure modes, where they usually come from, and the shortest verified fix path.

## Table Of Contents

- [Java Heap Space](#java-heap-space)
- [Incubator Vector API](#incubator-vector-api)
- [Missing Native Access](#missing-native-access)
- [OpenBLAS Missing Or Unavailable](#openblas-missing-or-unavailable)
- [Metal MPS Shim Missing](#metal-mps-shim-missing)
- [CUDA Shim Missing](#cuda-shim-missing)
- [Validation Mismatch In Benchmark Or Autotune](#validation-mismatch-in-benchmark-or-autotune)
- [Shape And Broadcast Errors](#shape-and-broadcast-errors)
- [Optimizer Rewrite Bugs](#optimizer-rewrite-bugs)
- [Gradients Missing Or Wrong](#gradients-missing-or-wrong)
- [Unsupported DType In A Kernel](#unsupported-dtype-in-a-kernel)
- [CPU Kernel Resolution Failure](#cpu-kernel-resolution-failure)
- [Performance Regressions](#performance-regressions)
- [Generated Artifacts In Source Tree](#generated-artifacts-in-source-tree)
- [Source Hygiene Architecture Failures](#source-hygiene-architecture-failures)
- [Stale Or Missing Profile Artifacts](#stale-or-missing-profile-artifacts)

## Java Heap Space

Symptom:

```text
java.lang.OutOfMemoryError: Java heap space
```

Likely context:

- Full `./gradlew test` run
- Debug benchmark tests under `src/test/java/debug`
- Tuning/calibration tests that instantiate fresh workloads repeatedly
- Etalon performance regression suite

Fix:

```bash
./gradlew test --no-daemon -Dsynaptik.testMaxHeap=4g
```

Why this works: `build.gradle` reads `System.getProperty('synaptik.testMaxHeap')` for every `Test` task. If absent, it sets `maxHeapSize = '2g'`.

If the failure persists, narrow the class:

```bash
./gradlew test --no-daemon --tests EtalonPerformanceRegressionTest -Dsynaptik.testMaxHeap=4g
./gradlew test --no-daemon --tests "debug.*" -Dsynaptik.testMaxHeap=4g
```

## Incubator Vector API

Symptom:

```text
module not found: jdk.incubator.vector
```

or runtime errors around `jdk.incubator.vector.*`.

Verified project behavior:

- `build.gradle` configures Java toolchain `25`.
- Compile tasks add `--add-modules jdk.incubator.vector`.
- Test/run/application JVMs add `--add-modules=jdk.incubator.vector`.

Fix:

```bash
./gradlew --version
./gradlew classes
```

Confirm Gradle is using a JDK 25 toolchain. `gradle.properties` contains only a commented Java 21 path:

```text
#org.gradle.java.home=/usr/lib/jvm/java-21-openjdk
```

Do not uncomment that Java 21 path for this project.

## Missing Native Access

Symptom:

```text
IllegalCallerException
Illegal native access
Foreign Function & Memory API access failure
```

Verified project behavior:

- `build.gradle` adds `--enable-native-access=ALL-UNNAMED` for `Test`, `JavaExec`, and `applicationDefaultJvmArgs`.

Fix for Gradle-managed commands:

```bash
./gradlew test --no-daemon --tests backend.metal.bridge.MetalMpsFfmBridgeTest
./gradlew run --args="benchmark-winner f64"
```

Fix for direct `java` commands:

```bash
java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED ...
```

Needs verification: direct `java` entry-point classpath commands are not documented as first-class workflow in the repository.

## OpenBLAS Missing Or Unavailable

Symptoms:

```text
OpenBLAS FFM is unavailable
Prepared conv2d GEMM plan requires OPENBLAS_FFM, but the bridge is not available.
OpenBLAS FFM bridge unavailable: ...
```

Lookup order in `src/main/java/backend/blas/OpenBlasFfmBridge.java`:

```text
-Dopenblas.lib=<path>
OPENBLAS_LIB
openblas
```

Fix:

```bash
./gradlew test --no-daemon --tests MatMulTest -Dopenblas.lib=/absolute/path/to/libopenblas.dylib
```

or:

```bash
OPENBLAS_LIB=/absolute/path/to/libopenblas.dylib ./gradlew test --no-daemon --tests MatMulTest
```

If OpenBLAS is not part of the change, run the Java fallback path with `BlasConfig.disabled()` in the test or choose tests that do not force `BlasProvider.OPENBLAS_FFM`.

## Metal MPS Shim Missing

Symptoms:

```text
Missing or unavailable Metal MPS bridge
synaptik.metal.mps.lib is not configured
```

Lookup order in `src/main/java/backend/metal/bridge/MetalMpsFfmBridge.java`:

```text
-Dsynaptik.metal.mps.lib=<path>
SYNAPTIK_METAL_MPS_LIB
synaptik_apple_mps
```

Build the shim on macOS:

```bash
./gradlew buildMetalMpsShim
```

Build all optional native components for the current platform:

```bash
./gradlew nativeBuild
```

The build script writes:

```text
build/native/apple/libsynaptik_apple_mps.dylib
```

Run the Metal test slice with the explicit library configured by Gradle:

```bash
./gradlew metalTest
```

Run one bridge test manually with the explicit library:

```bash
./gradlew test --no-daemon --tests backend.metal.bridge.MetalMpsFfmBridgeTest -Dsynaptik.metal.mps.lib=build/native/apple/libsynaptik_apple_mps.dylib
```

If the task says it is only supported on macOS, that is expected: `buildMetalMpsShim` checks `os.name` for `mac`.

Current dtype boundary: the Metal MPS FFM bridge uses `_f32` native compile/execute symbols. The Java planner and bridge currently accept `FLOAT32` compute/output tensors, `FLOAT32` data inputs, and `BOOL` only for predicate inputs such as the `where` condition. Direct SDPA remains CPU/fallback until the native MPSGraph scale contract is aligned with the framework's CPU semantics. Masked decomposed attention should stay as generic `WHERE`/`SOFTMAX`/`MATMUL` DAG operations rather than native `SDPA(maskTensor=bool)`, because the verified native MPSGraph SDPA mask operand expects a floating tensor. `FLOAT64`, `BFLOAT16`, and `INT32` graphs should remain on CPU unless a later native ABI/storage path is implemented.

Native buffer ABI boundary: a current shim should also export `synaptik_apple_mps_create_buffer`,
`synaptik_apple_mps_read_buffer`, `synaptik_apple_mps_destroy_buffer`, and
`synaptik_apple_mps_execute_partition_f32_buffers`. When those symbols are present,
`MetalMpsFfmBridge.supportsBufferBindings()` can return `true` and run Metal regions through
`BUFFER_BINDING`. If traces still show `TENSOR_ARRAY_COPY`, check `metalBufferBindingDecision`: it should explain
whether the bridge lacked symbols, input allocation failed, dtype/layout was unsupported, or native buffer execution
failed and fell back.

For successful buffer execution, outputs are marked `DEVICE_OWNED` until root/gradient publication reads the Metal
buffer back through the registered materializer. Seeing `metalNativeToJavaCopyNs=0` with a later
`CpuMaterializationTrace` is expected: there was no Java array round-trip between Metal regions, but public tensors
still become CPU-readable before `compute()` returns.

## CUDA Shim Missing

Symptoms:

```text
CUDA bridge unavailable
synaptik.cuda.graph.lib is not configured
```

Lookup order in `src/main/java/backend/cuda/bridge/CudaFfmBridge.java`:

```text
-Dsynaptik.cuda.graph.lib=<path>
SYNAPTIK_CUDA_GRAPH_LIB
synaptik_cuda_graph
```

Run the bridge availability test:

```bash
./gradlew test --no-daemon --tests backend.cuda.bridge.CudaFfmBridgeTest
```

Run it with an explicit library:

```bash
./gradlew test --no-daemon --tests backend.cuda.bridge.CudaFfmBridgeTest -Dsynaptik.cuda.graph.lib=/absolute/path/to/libsynaptik_cuda_graph.so
```

Needs verification: no CUDA native build script was found under `scripts/` or `src/main/native`.

## Validation Mismatch In Benchmark Or Autotune

Symptoms:

```text
validation failed
forward mismatch
candidate failed validation
```

Likely sources:

- Wrong dtype tolerance.
- A candidate profile changed runtime knobs outside its calibration family.
- A workload reused mutated graph state instead of creating a fresh `WorkloadInstance`.
- A rewrite/fusion/memory change altered semantics.

Immediate isolation:

```bash
./gradlew test --no-daemon --tests FrameworkEtalonTest
./gradlew test --no-daemon --tests tuning.integration.SessionWorkloadIsolationTest
./gradlew test --no-daemon --tests BenchmarkSessionTest --tests AutotuneSessionTest
```

Direct validation coverage exists in `src/test/java/ValidationEngineTest.java`, with additional integration coverage in benchmark, autotune, and session tests.

For optimizer changes, compare no-opt and optimized execution:

```bash
./gradlew test --no-daemon --tests GradientEngineRegressionTest
./gradlew test --no-daemon --tests CompiledGraphIdempotencyTest
```

## Shape And Broadcast Errors

Symptom:

```text
IllegalArgumentException: Broadcast mismatch at dim <n>: <a> vs <b>
```

Source:

- `src/main/java/tensor/BroadcastPlanner.java`

Fix path:

1. Check the operation builder in `src/main/java/tensor/ops/...`.
2. Confirm it calls the correct planner, such as `TensorBroadcastOps.planBinary(...)`.
3. Confirm the descriptor preserves the plan when the backend needs it.
4. Add or run broadcast coverage:

```bash
./gradlew test --no-daemon --tests BroadcastPlannerTest
./gradlew test --no-daemon --tests BroadcastBinaryOpsTest
./gradlew test --no-daemon --tests BroadcastContractMatrixTest
```

If the shape is valid before compile but wrong after execution, inspect:

- `src/main/java/backend/cpu/kernels/layout/plan/ResolvedBroadcastPlan.java`
- `src/main/java/backend/cpu/kernels/layout/plan/ResolvedWhereBroadcastPlan.java`
- `src/main/java/backend/cpu/kernels/layout/BroadcastPlanResolver.java`
- `src/main/java/backend/cpu/prepare/CpuNodePreparer.java`

## Optimizer Rewrite Bugs

Symptoms:

```text
optimized result differs from noOptimization
missing node after compile
stage order is invalid
```

Stage mapping is in `src/main/java/graph/optimizer/OptimizerFactory.java`:

```text
AR   -> graph.optimizer.rewrite.RewriteRule
CSE  -> graph.optimizer.cse.CommonSubexpressionEliminationRule
PART -> graph.optimizer.partition.PartitionIntentRule
FUSE -> graph.optimizer.region.RegionOptimizationRule
MEM  -> graph.optimizer.memory.MemoryOptimizerRule
```

`OptimizerConfig` enforces:

```text
FUSE requires PART
PART before FUSE
MEM requires FUSE
```

Fix path:

```bash
./gradlew test --no-daemon --tests AlgebraicRewritingPowTest
./gradlew test --no-daemon --tests AlgebraicRewritingSigmoidTest
./gradlew test --no-daemon --tests CommonSubexpressionEliminationRuleTest
./gradlew test --no-daemon --tests graph.optimizer.GraphOptimizerSinglePassTest
./gradlew test --no-daemon --tests graph.optimizer.region.RegionOptimizationRuleTest
```

If a new operation has parameters, update `CommonSubexpressionEliminationRule.parameterKey(...)`; otherwise CSE may treat parameterized nodes incorrectly.

For memory-related optimizer failures, disable memory reuse to isolate:

```bash
./gradlew test --no-daemon --tests MemoryPlannerSummaryTest -Dcg.optimizer.enableMemoryReuse=false
```

The property is read by `MemoryOptimizerRule`.

## Gradients Missing Or Wrong

Symptoms:

```text
expected gradient but was null
gradient array mismatch
FORWARD_BACKWARD does not update leaf gradients
```

Fix path:

1. Verify leaf tensors call `setRequiresGrad(true)` in the test or workload.
2. Verify the builder under `src/main/java/tensor/ops/...` attaches backward logic with `TensorInternalAccess.setBackwardFunction(...)`.
3. Verify broadcast gradients call `TensorBroadcastOps.sumToShape(...)` when an operand was broadcast.
4. Verify backend intent propagation for special gradient primitives when needed, as used by min/max gradient builders.
5. Run:

```bash
./gradlew test --no-daemon --tests GradientEngineRegressionTest
./gradlew test --no-daemon --tests BroadcastContractMatrixTest
./gradlew test --no-daemon --tests AllOpsTest
```

For loss/index gradients:

```bash
./gradlew test --no-daemon --tests IndexTargetNllLossExecutionTest
./gradlew test --no-daemon --tests IndexTargetCrossEntropyLossExecutionTest
./gradlew test --no-daemon --tests IgnoreIndexLossExecutionTest
```

## Unsupported DType In A Kernel

Symptom:

```text
UnsupportedOperationException: CpuXKernel does not support FLOAT32
UnsupportedOperationException: CpuXKernel does not support BFLOAT16
```

Source:

- Default methods in `src/main/java/backend/cpu/kernels/CpuKernel.java`

Fix path:

1. Implement the relevant `forwardF64`, `forwardF32`, `forwardBF16`, `forwardBOOL`, or `forwardI32` method.
2. Confirm `ResolvedCpuComputeContract` and planner output route to the expected dtype.
3. Register the kernel in `src/main/java/backend/cpu/registry/CpuKernelResolver.java`.
4. Add dtype coverage:

```bash
./gradlew test --no-daemon --tests DataTypeExecutionCoverageTest
./gradlew test --no-daemon --tests TensorStorageDataTypeTest
```

## CPU Kernel Resolution Failure

Symptoms:

```text
Cannot resolve CPU kernel for UNKNOWN operation type
CONST_SCALAR is an internal fused-plan op and has no standalone CPU kernel
Missing CPU kernel for opType=<TYPE>
```

Fix:

- Add the new `Operation.OpType` to `src/main/java/operations/Operation.java`.
- Add a singleton and switch branch in `src/main/java/backend/cpu/registry/CpuKernelResolver.java`.
- If the op is intentionally internal to fused planning, do not route it as a standalone compiled node.

Run:

```bash
./gradlew test --no-daemon --tests CpuKernelFamilyArchitectureTest
./gradlew test --no-daemon --tests SourceTreeHygieneTest
```

## Performance Regressions

Symptoms:

```text
Inference etalon performance regression check.
regressed beyond 40% tolerance and 0.050ms noise floor
```

Source:

- `src/test/java/EtalonPerformanceRegressionTest.java`

The test writes:

```text
build/tuning-etalon-regression/current-inference-suite.json
```

and compares against:

```text
src/test/resources/tuning/etalon/inference-performance-baseline.properties
```

Fix path:

1. Rerun the exact test to check noise:

```bash
./gradlew test --no-daemon --tests EtalonPerformanceRegressionTest
```

2. Run a targeted debug benchmark:

```bash
./gradlew test --no-daemon --tests debug.TransformerHotPathCurrentBestProfileBenchmarkTest
./gradlew test --no-daemon --tests debug.AbcCurrentBestProfileBenchmarkTest
```

3. Inspect whether the change touched planner thresholds in `CpuKernelConfig`, `CpuExecutionPlanner`, matmul/conv planners, fused dispatch, or materialization logic.
4. If the performance change is intended, update the baseline resource in the same change after recording the current report path.

Needs verification: the repository does not define a separate stable performance CI profile; local hardware and background load can affect these tests.

## Generated Artifacts In Source Tree

Symptoms:

```text
Generated artifacts found in source tree: [...]
Source tree contains generated artifacts: [...]
```

Sources:

- Gradle task `verifySourceTreeClean`
- `src/test/java/SourceTreeHygieneTest.java`

Rejected artifacts under `src/` or `test/` include:

```text
*.class
*.java.txt
*.java.bak
*.java.orig
*.java.tmp
*~
.tmp*
*.tmp*
.DS_Store
```

Fix:

```bash
./gradlew cleanSourceArtifacts
./gradlew verifySourceTreeClean
```

If the artifact is not covered by `cleanSourceArtifacts`, remove it manually only after confirming it is generated and not user-authored source.

## Source Hygiene Architecture Failures

Symptoms:

```text
CPU kernels belong under backend.cpu.kernels
Root backend package must not gain concrete helpers or wrappers
Legacy graph.optimizer.fusion references remain
Graph autotune candidates must not import runtime/backend config
```

Fix by moving code to the owner package:

| Failure area | Owner path |
|---|---|
| CPU kernels | `src/main/java/backend/cpu/kernels` |
| CPU preparation | `src/main/java/backend/cpu/prepare` |
| CPU partition legality | `src/main/java/backend/cpu/partition` |
| CPU fused planning/codegen | `src/main/java/backend/cpu/fused` |
| Generic backend selection | `src/main/java/backend/select` |
| Generic lowering contracts | `src/main/java/backend/lowering` |
| Generic prepare orchestration | `src/main/java/backend/prepare` |
| Optimizer rewrite | `src/main/java/graph/optimizer/rewrite` |
| Optimizer region policy | `src/main/java/graph/optimizer/region` plus CPU-specific policy under `backend.cpu.fused` |
| Graph autotune policy candidates | `src/main/java/tuning/candidate/graph` without runtime/backend imports |

Run:

```bash
./gradlew test --no-daemon --tests SourceTreeHygieneTest
```

## Stale Or Missing Profile Artifacts

Symptoms:

```text
Missing calibration profile: profiles/platform/<platform-id>/calibration/<dtype>-forward-backward.json
Missing best profile: profiles/platform/<platform-id>/tuning/abc/<dtype>-best-profile.json
Missing best profile for <dtype>
```

Fix:

```bash
./gradlew run --args="calibrate --dtype f64 --families all --preset quick --progress lines --color never"
./gradlew run --args="autotune f64"
./gradlew run --args="benchmark-winner f64"
```

For all dtypes:

```bash
./gradlew run --args="calibrate --dtypes all --families all --preset quick --progress lines --color never"
```

Profile paths are platform-dependent because `PlatformCalibrationPaths.platformId(HardwareFingerprint.capture())` includes OS, architecture, vendor, and CPU count information.
