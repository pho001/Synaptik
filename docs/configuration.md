<!-- generated-by: gsd-doc-writer -->
# Configuration

Navigation: [Index](index.md) | [Calibration & Autotune](calibration-autotune.md) | [Compute Flow](compute-flow.md) | [Development](development.md) | [Testing](testing.md) | [Troubleshooting](troubleshooting.md)

Chapters: [Build Requirements](#build-requirements) | [OptimizerConfig](#optimizerconfig) | [RuntimeConfig](#runtimeconfig) | [Execution Profiles](#execution-profiles) | [Platform Runtime Profiles](#platform-runtime-profiles) | [Tuning And Calibration Persistence](#tuning-and-calibration-persistence) | [System Properties And Environment Variables](#system-properties-and-environment-variables) | [CLI Configuration Behavior](#cli-configuration-behavior) | [Verification Notes](#verification-notes)

This document covers build/runtime requirements, optimizer and runtime configuration, backend knobs, profile persistence, tuning/calibration artifacts, system properties, and CLI configuration behavior.

## Build Requirements

| Requirement | Value | Source |
|---|---:|---|
| JDK toolchain | Java 25 | `build.gradle` `java.toolchain.languageVersion = 25` |
| Gradle wrapper | Use bundled `./gradlew` or `gradlew.bat` | `gradlew`, `gradlew.bat`, `gradle/wrapper/gradle-wrapper.properties` |
| Java modules | `jdk.incubator.vector` | Added for compile, test, and JavaExec tasks in `build.gradle` |
| Native access | `--enable-native-access=ALL-UNNAMED` | Added for test, run, and application default JVM args |
| Main class | `synaptik.app.Main` | `build.gradle` `application.mainClass` |

Core commands:

```bash
./gradlew classes
./gradlew test
./gradlew run
```

The `test` task defaults to `maxHeapSize = 2g`. Override it with:

```bash
./gradlew test -Dsynaptik.testMaxHeap=4g
```

## OptimizerConfig

**Source:** `src/main/java/config/optimizer/OptimizerConfig.java`

`OptimizerConfig` controls compile-time graph stages and stage-specific settings.

```java
new OptimizerConfig(
        List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.PART, OptimizerStage.FUSE, OptimizerStage.MEM),
        RewriteConfig.defaults(),
        CseConfig.strictDefaults(),
        FuseConfig.trainingDefaults(),
        MemoryConfig.defaults(),
        PartitionConfig.defaults()
)
```

Preset methods:

| Preset | Stage order | Notable settings |
|---|---|---|
| `OptimizerConfig.noOptimization()` | Empty list | Strict CSE config object is still present, but no stages run. |
| `OptimizerConfig.trainingDefaults()` | `AR, CSE, PART, FUSE, MEM` | Strict CSE and training fusion defaults. |
| `OptimizerConfig.inferenceDefaults()` | `AR, CSE, PART, FUSE, MEM` | Aggressive CSE and inference fusion defaults. |

Validation:

- `stageOrder` cannot be null, contain null stages, or contain duplicates.
- `FUSE` requires `PART`.
- `PART` must come before `FUSE`.
- `MEM` requires `FUSE`.

### Optimizer Stages

| Stage | Meaning |
|---|---|
| `AR` | Algebraic and lowering rewrites. |
| `CSE` | Common subexpression elimination. |
| `PART` | Backend partition planning. |
| `FUSE` | Fused CPU execution cluster planning. |
| `MEM` | Runtime memory reuse planning. |

### RewriteConfig

**Source:** `src/main/java/config/optimizer/RewriteConfig.java`

Fields:

| Field | Type | Default |
|---|---|---|
| `algebraic` | `AlgebraicRewriteConfig` | enabled |
| `linearLowering` | `LinearLoweringConfig` | enabled |
| `conv2dLowering` | `Conv2dLoweringConfig` | `HEURISTIC` |
| `piecewiseLowering` | `PiecewiseLoweringConfig` | all false |

`Conv2dLoweringMode` values:

- `OFF`
- `ALWAYS`
- `HEURISTIC`

`PiecewiseLoweringConfig.aggressiveDefaults()` enables canonical sigmoid, ReLU-like where lowering, and clamp-like where lowering.

### CseConfig

**Source:** `src/main/java/config/optimizer/CseConfig.java`

| Preset | `strictSafety` |
|---|---:|
| `CseConfig.strictDefaults()` | `true` |
| `CseConfig.aggressiveDefaults()` | `false` |

### FuseConfig

**Source:** `src/main/java/config/optimizer/FuseConfig.java`

| Field | Training default | Inference default |
|---|---:|---:|
| `maxClusterNodes` | `64` | `96` |
| `scoreThreshold` | `0.55` | `0.0` |
| `internalEdgeBonus` | `0.30` | `0.50` |
| `externalInputPenalty` | `0.20` | `0.10` |
| `sharedExpensivePenalty` | `1.00` | `0.50` |
| `nonCheapBonus` | `0.35` | `0.35` |
| `preserveSharedExpensiveNodes` | `true` | `false` |

Validation:

- `maxClusterNodes` must be greater than zero.
- Scoring values must be finite and non-negative.

### MemoryConfig

**Source:** `src/main/java/config/optimizer/MemoryConfig.java`

Default:

```java
new MemoryConfig(
        true,  // separateForwardBackwardPools
        false, // allowCrossPhaseReuse
        false, // allowLargerBufferReuse
        1      // minReusableBufferSize
)
```

Validation:

- `minReusableBufferSize` must be at least `1`.
- `allowCrossPhaseReuse` cannot be enabled when forward/backward pools are separated.

### PartitionConfig

**Source:** `src/main/java/config/optimizer/PartitionConfig.java`

Default:

```java
new PartitionConfig(
        16,     // maxSearchNodes
        512,    // maxVisitedCandidates
        1000.0, // nodeWeight
        120.0,  // internalEdgeWeight
        450.0,  // mergeNodeBonus
        80.0,   // tailDepthWeight
        60.0,   // externalInputPenalty
        1.0,    // workWeight
        PartitionPlannerStrategy.GREEDY_MAX_REGION,
        PartitionTarget.AUTO
)
```

`maxSearchNodes` and `maxVisitedCandidates` are clamped to at least `1`. Null strategy and target values default to `GREEDY_MAX_REGION` and `AUTO`.

## RuntimeConfig

**Source:** `src/main/java/config/runtime/RuntimeConfig.java`

`RuntimeConfig` controls runtime kernel tuning, approximations, BLAS, conv2d, fused execution, and accelerator enablement.

Fields:

| Field | Type | Null behavior |
|---|---|---|
| `kernel` | `KernelTuningConfig` | Required |
| `approximation` | `ApproximationConfig` | Defaults to `ApproxMode.OFF`, exact flag false |
| `blas` | `BlasConfig` | Defaults to disabled |
| `conv2d` | `Conv2dConfig` | Defaults from BLAS config |
| `fused` | `FusedExecutionPolicy` | Defaults to training fused policy |
| `accelerator` | `AcceleratorConfig` | Defaults to training accelerator config |

Preset methods:

| Preset | Purpose |
|---|---|
| `RuntimeConfig.trainingDefaults()` | CPU training defaults, disabled BLAS/conv2d BLAS, default fused and accelerator configs. |
| `RuntimeConfig.inferenceDefaults()` | CPU inference defaults, disabled BLAS/conv2d BLAS, default fused and accelerator configs. |
| `RuntimeConfig.noOptNoVecNoPar()` | Baseline runtime that effectively disables vectorization, parallelism, and BLAS through very high thresholds. |

### KernelTuningConfig

**Source:** `src/main/java/config/backend/KernelTuningConfig.java`

Fields:

- `cpu`: `CpuKernelConfig`
- `cuda`: `CudaKernelConfig`
- `opencl`: `OpenClKernelConfig`

Defaults:

| Config | Training default | Inference default |
|---|---|---|
| `CpuKernelConfig` | loop unroll `1`, matmul tile `16,0,0` | loop unroll `4`, matmul tile `32,32,32` |
| `CudaKernelConfig` | loop unroll `4`, matmul tile `16,16,16` | loop unroll `8`, matmul tile `32,32,32` |
| `OpenClKernelConfig` | loop unroll `1`, matmul tile `0,0,0` | loop unroll `4`, matmul tile `32,32,16` |

Important CPU defaults:

| Setting | Default |
|---|---:|
| cheap/transcendental/reduction vector min size | `1024` |
| cheap/transcendental/reduction parallel min size | `100000` |
| matmul parallel min size | `2000000` |
| contiguous/materialization thresholds | `1000000000` |
| low/medium/high cost target chunks per worker | `4`, `2`, `1` |
| min scalar/vector/reduction chunk size | `4096`, `8192`, `16384` |
| common pool low-cost max work per worker | `16384` |
| fused ASM vector widths | normalized to `1`, `2`, `4`, or `8` |
| sum accuracy mode | `FAST` |
| attention matmul policy | `AUTO` |
| matmul microkernel | `AUTO` |

### BlasConfig

**Source:** `src/main/java/config/runtime/BlasConfig.java`

Fields:

| Field | Default |
|---|---|
| `provider` | `BlasProvider.NONE` |
| `matmulMinWork` | `2000000` |
| `f32RequireMgeK` | `true` |
| `f32MaxNOverK` | `3.0` |
| `f32WideRequireMgeK` | same as `f32RequireMgeK` |
| `f32WideMaxNOverK` | same as `f32MaxNOverK` |
| `debug` | `false` |
| `threads` | normalized to `0` |

Supported providers:

- `NONE`
- `OPENBLAS_FFM`

`BlasConfig.disabled()` returns `NONE` with default thresholds.

### Conv2dConfig

**Source:** `src/main/java/config/runtime/Conv2dConfig.java`

Fields:

| Field | Default |
|---|---|
| `provider` | `BlasProvider.NONE` |
| `f64MinWork` | `2000000` |
| `f32MinWork` | `2000000` |
| `f32RequireMgeK` | `true` |
| `f32MaxNOverK` | `3.0` |
| `bf16MinWork` | `2000000` |
| `bf16RequireMgeK` | `true` |
| `bf16MaxNOverK` | `3.0` |

`Conv2dConfig.fromBlasConfig(blas)` copies the BLAS provider and work thresholds into conv2d defaults.

### ApproximationConfig

**Source:** `src/main/java/config/runtime/ApproximationConfig.java`

Fields:

| Field | Default | Values |
|---|---|---|
| `approxMode` | `ApproxMode.OFF` | `OFF`, `TRAINING_ONLY`, `ALWAYS` |
| `forceExactTranscendentals` | `false` | boolean |

`useFastExp(backwardEnabled)` and `useFastTanh(backwardEnabled)` return false when exact transcendentals are forced.

### FusedExecutionPolicy

**Source:** `src/main/java/config/runtime/FusedExecutionPolicy.java`

Defaults:

```java
new FusedExecutionPolicy(FusedPrimaryBackend.ASM, true)
```

Only `FusedPrimaryBackend.ASM` exists in the current source tree.

### AcceleratorConfig

**Sources:**

- `src/main/java/config/runtime/AcceleratorConfig.java`
- `src/main/java/config/runtime/AcceleratorBackendConfig.java`

Default backend config:

```java
new AcceleratorBackendConfig(
        true,  // enabled
        false, // requireRuntimeAvailability
        0L     // minimumEstimatedWork
)
```

`AcceleratorConfig.defaults()` enables CUDA, OpenCL, and Metal configs in policy. Runtime availability is not required by default. `AcceleratorConfig.disabled()` disables all three.

Current capability note: the CPU backend is the only fully implemented execution backend. Metal, CUDA, and OpenCL packages expose scaffolding and selected bridge/lowering code, but they are not documented here as production-ready runtimes.

## Execution Profiles

### ExecutionProfile

**Source:** `src/main/java/config/profile/ExecutionProfile.java`

Fields:

| Field | Required | Default behavior |
|---|---|---|
| `profileName` | No | Null becomes `"default"` |
| `candidateName` | No | Null/blank becomes `profileName` |
| `dataType` | Yes | Must be non-null |
| `mode` | Yes | Must be non-null |
| `optimizer` | Yes | Must be non-null |
| `runtime` | Yes | Must be non-null |
| `workload` | No | Null becomes `WorkloadProfile.none()` |

Example:

```java
ExecutionProfile profile = new ExecutionProfile(
        "demo",
        "demo",
        DataType.FLOAT64,
        ExecutionMode.FORWARD,
        OptimizerConfig.inferenceDefaults(),
        RuntimeConfig.inferenceDefaults()
);
```

### GraphExecutionPolicy

**Source:** `src/main/java/config/profile/GraphExecutionPolicy.java`

`GraphExecutionPolicy` wraps `OptimizerConfig` for graph autotune candidate assembly.

Presets:

- `GraphExecutionPolicy.trainingDefaults()`
- `GraphExecutionPolicy.inferenceDefaults()`
- `GraphExecutionPolicy.noOptimization()`

## Platform Runtime Profiles

**Sources:**

- `src/main/java/config/profile/PlatformRuntimeProfile.java`
- `src/main/java/config/profile/PlatformRuntimeProfileIO.java`

`PlatformRuntimeProfile` is the persisted calibration shape for platform/runtime settings. It contains:

- `metadata`
- `matmul`
- `conv2d`
- `fused`
- `elementwiseDispatch`
- `reduction`
- `scheduler`
- `materialization`
- `numerics`
- `accelerator`

JSON written by `PlatformRuntimeProfileIO.toJson(...)` has matching top-level objects:

```json
{
  "metadata": {},
  "matmul": {},
  "conv2d": {},
  "fused": {},
  "elementwiseDispatch": {},
  "reduction": {},
  "scheduler": {},
  "materialization": {},
  "numerics": {},
  "accelerator": {}
}
```

IO behavior:

- `save(path, profile)` creates parent directories and writes UTF-8 JSON.
- `loadOrDefault(path, fallback)` returns fallback if the path is null, missing, unreadable, or invalid.
- `fromJsonOrDefault(json, fallback)` parses known keys by string lookup and falls back per field when keys are missing.

Converting to runtime:

```java
RuntimeConfig runtime = platformRuntimeProfile.toRuntimeConfig();
```

This maps profile thresholds and backend choices into `RuntimeConfig`; CUDA/OpenCL kernel tuning still uses training defaults in this conversion.

## Tuning And Calibration Persistence

### CLI Calibration Layout

**Source:** `src/main/java/tuning/calibration/store/CalibrationArtifactLayout.java`

Default root is `profiles`. The layout resolves to:

```text
profiles/
  platform/
    <platform-id>/
      calibration/
        schema-v2/
          manifest.json
          latest/
            <dtype>/
              <mode>/
                profile.json
                manifest.json
          history/
            <dtype>/
              <mode>/
                <family-id>.jsonl
          runs/
            <run-id>/
              manifest.json
              <dtype>/
                <mode>/
                  <family-id>/
                    result.json
                    result.txt
                    selected-profile.json
                    candidates.jsonl
```

The platform id is produced from normalized hardware properties as:

```text
<normalized-os.name>-<normalized-os.arch>-<normalized-java.vendor>-<cores>c
```

Normalization lowercases values, trims surrounding whitespace, and replaces spaces or tabs with underscores.

### CLI Autotune Layout

**Source:** `src/main/java/synaptik/app/Main.java`

CLI autotune for the built-in ABC workload writes:

```text
profiles/platform/<platform-id>/tuning/abc/<dtype>-best-profile.json
profiles/platform/<platform-id>/tuning/abc/<dtype>-history.jsonl
```

### Tensor Convenience Autotune Layout

**Source:** `src/main/java/tensor/TensorExecutionSupport.java`

`Tensor.compute(new ComputeOptions().autotune(...))` uses a generic tensor workload and writes:

```text
build/tuning/tensor/<platform-id>/<graph-signature>/<seed-signature>/<dtype>-<mode>-best-profile.json
build/tuning/tensor/<platform-id>/<graph-signature>/<seed-signature>/<dtype>-<mode>-history.jsonl
```

`AutotunePolicy.IF_MISSING` reuses a compatible existing best profile when present. `AutotunePolicy.FORCE` reruns the generic graph autotune before execution.

### PersistencePolicy

**Source:** `src/main/java/tuning/store/PersistencePolicy.java`

Fields:

```java
new PersistencePolicy(
        true,
        true,
        bestProfilePath,
        historyPath
)
```

`PersistencePolicy.disabled()` sets both persistence booleans to false and paths to null.

## System Properties And Environment Variables

This section covers build, core runtime, optimizer, native bridge, diagnostic, and benchmark CLI properties found in the repository.

### Gradle And Build Properties

| Property | Default | Effect | Source |
|---|---|---|---|
| `synaptik.testMaxHeap` | `2g` | Overrides Gradle test max heap when non-blank. | `build.gradle` |
| `os.name` | JVM value | Used by `buildMetalMpsShim` to run only on macOS. | `build.gradle` |

### Runtime System Properties

| Property | Default | Effect | Source |
|---|---|---|---|
| `cg.cpu.blas.provider` | `NONE` | Selects `BlasProvider`; valid known values are `NONE`, `OPENBLAS_FFM`. | `backend/blas/BlasRuntime.java` |
| `cg.cpu.blas.matmulMinWork` | `2000000` | Positive long threshold for BLAS matmul dispatch. | `backend/blas/BlasRuntime.java` |
| `cg.cpu.blas.debug` | `false` | Enables BLAS debug behavior where used. | `backend/blas/BlasRuntime.java` |
| `cg.cpu.blas.f32RequireMgeK` | `true` | F32 BLAS shape guard. | `backend/blas/BlasRuntime.java` |
| `cg.cpu.blas.f32MaxNOverK` | `3.0` | F32 BLAS shape ratio guard. | `backend/blas/BlasRuntime.java` |
| `openblas.lib` | library name `openblas` | Explicit OpenBLAS library path/name for FFM lookup. | `backend/blas/OpenBlasFfmBridge.java` |
| `synaptik.metal.mps.lib` | library name `synaptik_apple_mps` | Explicit Metal MPS shim library path/name for FFM lookup. | `backend/metal/bridge/MetalMpsFfmBridge.java` |
| `synaptik.cuda.graph.lib` | library name `synaptik_cuda_graph` | Explicit CUDA graph bridge library path/name for FFM lookup. | `backend/cuda/bridge/CudaFfmBridge.java` |
| `cg.cpu.fused.profile` | `false` | Enables fused execution profiler collection. | `backend/cpu/kernels/fused/FusedExecutionProfiler.java` |
| `cg.math.forceExactTranscendentals` | `false` | Forces utility fast transcendental methods to call exact `Math` functions. | `utils/FastTranscendentals.java` |
| `cg.optimizer.enableMemoryReuse` | `true` | Enables memory reuse in memory optimizer rule. | `graph/optimizer/memory/MemoryOptimizerRule.java` |
| `cg.optimizer.ar.disableAllTransforms` | `false` | Disables all algebraic rewrite transforms. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableRebuildTopologicalClosure` | `false` | Disables AR topological closure rebuild. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disablePow2ToMul` | `false` | Disables `pow(x, 2)` to multiply rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableAddSelfToMul2` | `false` | Disables `x + x` rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableAddNegToZero` | `false` | Disables add-negative-to-zero rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableAddNegNegToNegAdd` | `false` | Disables negative-add rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableAddLogLogToLogMul` | `false` | Disables log-add rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableSubNegToAdd` | `false` | Disables subtract-negative rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableDivConstToMulRecip` | `false` | Disables division-by-constant reciprocal rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableDivMulScalarByConst` | `false` | Disables scalar division rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableDivInvToMul` | `false` | Disables division-by-inverse rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableDivOneToInv` | `false` | Disables reciprocal rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableMulScalarAssoc` | `false` | Disables scalar multiplication association rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableMulScalarNegPush` | `false` | Disables scalar negative pushdown rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableMulScalarConstFold` | `false` | Disables scalar constant folding rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableAddSubFactorize` | `false` | Disables add/sub factorization rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableMulInvToOne` | `false` | Disables inverse multiplication rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableMulNegNegToMul` | `false` | Disables double-negative multiply rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableMulExpExpToExpAdd` | `false` | Disables exp multiplication rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableNegSubSwap` | `false` | Disables negative subtraction swap rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableNegMulScalarPush` | `false` | Disables negative scalar pushdown rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disablePowPowFlatten` | `false` | Disables power flattening rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disablePowInvToNegExp` | `false` | Disables inverse power rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableLogPowToMulLog` | `false` | Disables log-power rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableLogInvToNegLog` | `false` | Disables log-inverse rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableLogSqrtToHalfLog` | `false` | Disables log-sqrt rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableExpLogCancel` | `false` | Disables exp-log cancellation. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableInvSigmoidPattern` | `false` | Disables inverse sigmoid pattern rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableInvPowToNegExp` | `false` | Disables inverse power to negative exponent rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableInvExpToExpNeg` | `false` | Disables inverse exp rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableInvNegPush` | `false` | Disables inverse negative push rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableClampMinIdentity` | `false` | Disables clamp-min identity rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableClampMinFlatten` | `false` | Disables clamp-min flatten rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableClampMaxIdentity` | `false` | Disables clamp-max identity rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |
| `cg.optimizer.ar.disableClampMaxFlatten` | `false` | Disables clamp-max flatten rewrite. | `graph/optimizer/rewrite/AlgebraicRewrite.java` |

### Diagnostic And Benchmark CLI Properties

| Property | Default | Effect | Source |
|---|---|---|---|
| `numerics.size` | `200000` | Numerics harness input size. | `numerics/NumericsHarness.java`, `numerics/NumericsCli.java` |
| `numerics.graphBlocks` | `6` | Numerics harness graph block count. | `numerics/NumericsHarness.java`, `numerics/NumericsCli.java` |
| `numerics.broadcastB0` | `128` | Numerics harness broadcast dimension. | `numerics/NumericsHarness.java`, `numerics/NumericsCli.java` |
| `numerics.broadcastB1` | `8` | Numerics harness broadcast dimension. | `numerics/NumericsHarness.java`, `numerics/NumericsCli.java` |
| `numerics.broadcastF` | `128` | Numerics harness feature/broadcast factor. | `numerics/NumericsHarness.java`, `numerics/NumericsCli.java` |
| `numerics.seed` | `42` | Numerics harness random seed. | `numerics/NumericsHarness.java`, `numerics/NumericsCli.java` |
| `numerics.stageA` | `NONE` | First optimizer-stage selection for numerics comparison. | `numerics/NumericsCli.java` |
| `numerics.stageB` | `AR` | Second optimizer-stage selection for numerics comparison. | `numerics/NumericsCli.java` |
| `numerics.nameA` | `A` | Display name for the first numerics candidate. | `numerics/NumericsCli.java` |
| `numerics.nameB` | `B` | Display name for the second numerics candidate. | `numerics/NumericsCli.java` |
| `numerics.dtype` | `FLOAT32` | Numerics harness dtype. | `numerics/NumericsCli.java` |
| `etalon.suite` | `all` | Etalon benchmark suite selector. | `tuning/etalon/FrameworkEtalonCli.java` |
| `etalon.preset` | `BALANCED` | Etalon benchmark preset selector. | `tuning/etalon/FrameworkEtalonCli.java` |
| `etalon.outDir` | `build/tuning-etalon` | Etalon report output directory. | `tuning/etalon/FrameworkEtalonCli.java` |

### Environment Variables

| Variable | Effect | Source |
|---|---|---|
| `OPENBLAS_LIB` | Fallback OpenBLAS library path/name when `openblas.lib` is unset. | `backend/blas/OpenBlasFfmBridge.java` |
| `SYNAPTIK_METAL_MPS_LIB` | Fallback Metal MPS shim library path/name when `synaptik.metal.mps.lib` is unset. | `backend/metal/bridge/MetalMpsFfmBridge.java` |
| `SYNAPTIK_CUDA_GRAPH_LIB` | Fallback CUDA graph bridge library path/name when `synaptik.cuda.graph.lib` is unset. | `backend/cuda/bridge/CudaFfmBridge.java` |

## CLI Configuration Behavior

**Source:** `src/main/java/synaptik/app/Main.java` and `src/main/java/tuning/calibration/run/CalibrationCommand.java`

Commands accepted by `Main`:

```bash
./gradlew run
./gradlew run --args="full <f64|f32|bf16>"
./gradlew run --args="calibrate --dtype <f64|f32|bf16> --family <family-id>"
./gradlew run --args="calibrate --dtype <f64|f32|bf16> --families all"
./gradlew run --args="calibrate --dtypes all --families all"
./gradlew run --args="autotune <f64|f32|bf16>"
./gradlew run --args="benchmark-winner <f64|f32|bf16>"
./gradlew run --args="benchmark-graph-space <f64|f32|bf16>"
```

No arguments defaults to `full f64`.

Calibration options:

| Option | Values | Default | Behavior |
|---|---|---|---|
| `--dtype` | `f64`, `f32`, `bf16` | Required unless `--dtypes all` | Calibrates one dtype. |
| `--dtypes` | `all` | Required unless `--dtype` | Calibrates `FLOAT64`, `FLOAT32`, and `BFLOAT16`. |
| `--family` | family id | Required unless `--families all` | Calibrates one family. |
| `--families` | `all` | Required unless `--family` | Calibrates full suite. |
| `--preset` | `TuningPreset` enum name | `BALANCED` | Selects measurement/validation presets. |
| `--mode` | `forward`, `forward-backward`, `forward_backward`, `training` | `FORWARD_BACKWARD` | Selects execution mode. |
| `--measurement` | `warmup:measure:repeats` | Preset benchmark measurement | Overrides iteration counts. |
| `--color` | `auto`, `always`, `never` | `auto` | Controls report coloring. |
| `--progress` | `live`, `lines`, `quiet` | `live` | Controls progress reporting. |
| `--output-root` | path | `profiles` | Changes calibration artifact root. |
| `--include-accelerators` | flag | false | Includes accelerator calibration families. |

Validation:

- Use either `--dtype` or `--dtypes all`, not both.
- Use either `--family` or `--families all`, not both.
- Calibration supports only `f64`, `f32`, and `bf16`.
- `--measurement` must have exactly three integer parts separated by `:`.
- `--color` and `--progress` reject unknown values.

Autotune and benchmark behavior:

- `autotune <dtype>` loads the latest calibration profile for that dtype and `FORWARD_BACKWARD` mode from `profiles/platform/<platform-id>/calibration/schema-v2/latest/<dtype>/forward-backward/profile.json`.
- If the calibration profile is missing, autotune throws an error telling the user to run calibration first.
- `benchmark-winner <dtype>` loads `profiles/platform/<platform-id>/tuning/abc/<dtype>-best-profile.json`.
- If the best profile is missing, benchmark-winner throws an error telling the user to run autotune first.

## Verification Notes

This document was checked against:

```bash
./gradlew classes
./gradlew test --tests BroadcastBinaryOpsTest --tests CompiledGraphIdempotencyTest --tests PreparedExecutionBuildTest --tests ExecutionProfileIoTest
```
