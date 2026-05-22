<!-- generated-by: gsd-doc-writer -->
# Configuration

Navigation: [Index](index.md#recommended-reading-paths) | [Calibration & Autotune](calibration-autotune.md#runtime-and-graph-artifacts) | [Compute Flow](compute-flow.md#tensor-compute-api) | [Native Bridges & BLAS](native-bridges-and-blas.md#configuration-and-library-lookup) | [Metal Backend](metal-backend.md#supported-operations-and-dtypes) | [Development](development.md#local-setup) | [Testing](testing.md#exact-commands) | [Troubleshooting](troubleshooting.md#openblas-missing-or-unavailable)

Chapters: [Build Requirements](#build-requirements) | [CompileConfig](#compileconfig) | [PublicationPolicy](#publicationpolicy) | [RuntimeConfig](#runtimeconfig) | [Execution Profiles](#execution-profiles) | [Platform Runtime Profiles](#platform-runtime-profiles) | [Tuning And Calibration Persistence](#tuning-and-calibration-persistence) | [System Properties And Environment Variables](#system-properties-and-environment-variables) | [CLI Configuration Behavior](#cli-configuration-behavior) | [Verification Notes](#verification-notes)

This document covers build/runtime requirements, compile and runtime configuration, backend knobs, profile persistence, tuning/calibration artifacts, system properties, and CLI configuration behavior.

## Table Of Contents

- [Build Requirements](#build-requirements)
- [CompileConfig](#compileconfig)
- [PublicationPolicy](#publicationpolicy)
- [RuntimeConfig](#runtimeconfig)
- [Execution Profiles](#execution-profiles)
- [Platform Runtime Profiles](#platform-runtime-profiles)
- [Tuning And Calibration Persistence](#tuning-and-calibration-persistence)
- [System Properties And Environment Variables](#system-properties-and-environment-variables)
- [CLI Configuration Behavior](#cli-configuration-behavior)
- [Verification Notes](#verification-notes)

## Build Requirements

| Requirement | Value | Source |
|---|---:|---|
| JDK toolchain | Java 25 | `build.gradle` `java.toolchain.languageVersion = 25` |
| Gradle wrapper | Use bundled `./gradlew` or `gradlew.bat` | `gradlew`, `gradlew.bat`, `gradle/wrapper/gradle-wrapper.properties` |
| Java modules | `jdk.incubator.vector` | Added for compile, test, and JavaExec tasks in `build.gradle` |
| Native access | `--enable-native-access=ALL-UNNAMED` | Added for test, run, and application default JVM args |
| Main class | `synaptik.app.TuningCli` | `build.gradle` `application.mainClass` |

Core commands:

```bash
./gradlew classes
./gradlew test
./gradlew run
```

Optional native Metal commands:

```bash
./gradlew buildMetalMpsShim
./gradlew nativeBuild
./gradlew metalTest
```

`buildMetalMpsShim` is the low-level macOS shim builder. `nativeBuild` is the explicit optional-native lifecycle task. `metalTest` builds the Metal shim and runs the Metal/MPS test slice with `synaptik.metal.mps.lib` set to `build/native/apple/libsynaptik_apple_mps.dylib`.

The published Metal shim is a separate native artifact, `synaptik-metal-macos-arm64`, with the resource
`native/macos-arm64/libsynaptik_apple_mps.dylib`. The main Synaptik publication declares that artifact as a runtime
dependency, so a normal consumer dependency on Synaptik still gets the macOS ARM64 shim transitively. This avoids relying
on the machine that builds the core JAR being macOS. Explicit property/env configuration still wins when you want to test
a local native build.

The `test` task defaults to `maxHeapSize = 2g`. Override it with:

```bash
./gradlew test -Dsynaptik.testMaxHeap=4g
```

## CompileConfig

**Source:** `src/main/java/config/compile/CompileConfig.java`

`CompileConfig` is the compile-time source of truth. It separates graph simplification from backend ownership planning, region optimization, and memory planning:

```java
CompileConfig.training()
        .withGraphOptimization(GraphOptimizationConfig.trainingDefaults())
        .withBackendPlanning(BackendPlanningConfig.autoAccelerator());
```

Primary layers:

| Layer | Source | Responsibility |
|---|---|---|
| `SemanticCanonicalizationConfig` | `config.compile` | Required semantic forward canonicalization before compile artifacts are produced. |
| `GraphOptimizationConfig` | `config.compile` | Backend-neutral graph rewrite/simplification/lowering: AR, CF, CSE, DCE, optional LOWER. |
| `BackendPlanningConfig` | `config.compile` | Compile-time backend ownership planning: `CPU_ONLY`, `EXPLICIT`, or `AUTO`. |
| `RegionOptimizationConfig` | `config.compile` | Optimization inside already-owned execution regions, including CPU fusion policy. |
| `MemoryPlanningConfig` | `config.compile` | Compile-time memory reuse policy. |

Preset methods:

| Preset | Meaning |
|---|---|
| `CompileConfig.training()` | Training graph optimization, explicit backend intent planning, region optimization, memory planning. |
| `CompileConfig.inference()` | Inference graph optimization, explicit backend intent planning, region optimization, memory planning. |
| `CompileConfig.trainingAutoAccelerator()` | Training defaults with automatic accelerator region discovery. |
| `CompileConfig.inferenceAutoAccelerator()` | Inference defaults with automatic accelerator region discovery. |
| `CompileConfig.noGraphOptimization()` | Disables graph optimization only; backend planning and prepare invariants remain active. |
| `CompileConfig.noGraphOptimizationBaseline()` | Benchmark-friendly no-graph-optimization preset that still honors explicit backend intent. |
| `CompileConfig.cpuOnlyBaseline()` | Strict CPU-only baseline with graph optimization, CPU regions, region optimization, and memory planning disabled. |

`BackendPlanningConfig.cpuOnly()` means no accelerator ownership regions. `BackendPlanningConfig.explicitOnly()` honors explicit GPU backend intent without auto-discovering GPU regions from a CPU-owned graph. `BackendPlanningConfig.autoAccelerator()` may discover GPU regions from CPU-owned graphs according to legality and cost policy.

Terminology to keep clean:

| Current term | Meaning | Not the same as |
|---|---|---|
| `GraphOptimizationConfig` | Backend-neutral graph simplification and lowering. | Backend planning, region optimization, memory planning, runtime dispatch. |
| `BackendPlanningConfig` | Compile-time backend ownership planning. | Execute-time "offload" or runtime availability. |
| `PartitionSearchConfig` | Search limits and scoring weights. | Backend target selection. |
| `RegionOptimizationConfig` | Fusion/execution-unit policy inside already owned regions. | Backend ownership discovery. |
| `RuntimeConfig` | Runtime/hardware policy. | Compile-time graph or backend ownership policy. |

Legacy public names such as an optimizer-wide stage-order config, offload config, and monolithic partition config are no longer the architecture model. The implementation still has lower-level helper configs under `config.optimizer` for rewrite, CSE, fuse, memory, CPU region, and Metal transfer cost models, but `CompileConfig` is the public compile-policy composition point.

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

### Backend Planning Search

`PartitionSearchConfig` now carries only search and scoring limits. Backend target, discovery mode, planner strategy, CPU region policy, and Metal transfer cost profile live under `BackendPlanningConfig`.

For detailed examples of `CPU_ONLY`, `EXPLICIT`, `AUTO`, required accelerator planning, CPU natural regions, accelerator regions, region optimization, memory planning, and benchmark semantics, see [Backend Planning And Regions](backend-planning-and-regions.md#backend-planning-and-regions).

## PublicationPolicy

**Source:** `src/main/java/graph/execution/PublicationPolicy.java`

`PublicationPolicy` controls which values are copied from run-scoped execution state back to public `Tensor` objects after execution.

It is a runtime visibility policy, not compile policy and not backend planning.

Policies:

| Policy | Publishes back to public tensors | Typical use |
|---|---|---|
| `ALL` | Every forward value and gradients. | Debugging and full graph visibility. |
| `OUTPUT_AND_GRADIENTS` | Root output and gradients. | Default ordinary execution. |
| `OUTPUT_ONLY` | Root output only. | Default optimizer-step execution and benchmark paths that do not need gradients attached. |
| `NONE` | Nothing. | Low-overhead benchmark runs that inspect traces or native-side effects instead of public tensor storage. |

Example:

```java
PreparedExecution prepared = compiled.prepare(runtime);
RunTrace trace = prepared.executeTraced(
        ExecutionMode.FORWARD_BACKWARD,
        PublicationPolicy.OUTPUT_ONLY
);
```

This still executes the prepared forward/backward graph. It only changes which results are synchronized to user-visible `Tensor` objects after the run. For accelerator paths, lower-publication policies can avoid device-to-CPU copies that exist only to update public tensor storage.

## RuntimeConfig

**Source:** `src/main/java/config/runtime/RuntimeConfig.java`

`RuntimeConfig` controls runtime kernel tuning, approximations, CPU storage policy, BLAS, conv2d, fused execution, and accelerator enablement.

Fields:

| Field | Type | Null behavior |
|---|---|---|
| `kernel` | `KernelTuningConfig` | Required |
| `approximation` | `ApproximationConfig` | Defaults to `ApproxMode.OFF`, exact flag false |
| `blas` | `BlasConfig` | Defaults to disabled |
| `conv2d` | `Conv2dConfig` | Defaults from BLAS config |
| `fused` | `FusedExecutionPolicy` | Defaults to training fused policy |
| `accelerator` | `AcceleratorConfig` | Defaults to training accelerator config |
| `cpuStorageProfile` | `CpuStorageProfile` | Defaults to `CPU_ARRAY` |
| `nativeCpuFailurePolicy` | `NativeCpuFailurePolicy` | Defaults to `FALLBACK_TO_ARRAY` |

Preset methods:

| Preset | Purpose |
|---|---|
| `RuntimeConfig.trainingDefaults()` | CPU training defaults, disabled BLAS/conv2d BLAS, default fused and accelerator configs. |
| `RuntimeConfig.inferenceDefaults()` | CPU inference defaults, disabled BLAS/conv2d BLAS, default fused and accelerator configs. |
| `RuntimeConfig.noOptNoVecNoPar()` | Baseline runtime that effectively disables vectorization, parallelism, and BLAS through very high thresholds. |

`CpuStorageProfile` is the runtime-level vocabulary for CPU storage ownership:

| Profile | Meaning |
|---|---|
| `CPU_ARRAY` | Keep CPU compute on the existing Java-array storage path. This is the compatibility default. |
| `CPU_NATIVE` | Prefer `MemorySegment`-backed native CPU storage for supported operations. Unsupported operation handling is governed by `NativeCpuFailurePolicy`. |
| `AUTO` | Let the runtime planner choose between Java-array and native CPU storage. |

`NativeCpuFailurePolicy` says what should happen when native CPU execution was requested but the current operation or shape is unsupported:

| Policy | Meaning |
|---|---|
| `FALLBACK_TO_ARRAY` | Unsupported native CPU operations may fall back to Java-array CPU execution. |
| `REQUIRE_NATIVE` | Native CPU matmul segment fallback fails instead of silently using the Java-array path. Broader chain-aware native planner enforcement remains future scope. |

This is deliberately separate from `BlasStorageMode`. `CpuStorageProfile` describes the runtime CPU storage policy for the whole prepared execution. `BlasStorageMode` describes the storage route of an individual BLAS-capable kernel family.

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
| `storageMode` | `BlasStorageMode.CPU_ARRAY` |
| `debug` | `false` |
| `threads` | normalized to `0` |

Supported providers:

- `NONE`
- `OPENBLAS_FFM`

`BlasConfig.disabled()` returns `NONE` with default thresholds.

`OPENBLAS_FFM` means the CPU matmul/conv2d GEMM paths may call OpenBLAS through Java's Foreign Function and Memory API.
It does not force every matrix multiplication through BLAS. The CPU matmul planner still checks dtype, estimated work,
contiguity, and `FLOAT32`/`BFLOAT16` shape guards before setting BLAS metadata. For the full explanation of BLAS,
GEMM, Java FFM, lookup order, fallback behavior, and why `threads` is currently normalized to `0`, see
[Native Bridges & BLAS: Configuration And Library Lookup](native-bridges-and-blas.md#configuration-and-library-lookup).

`storageMode` controls the CPU BLAS storage route:

| Mode | Meaning |
|---|---|
| `CPU_ARRAY` | Existing Java-array OpenBLAS bridge. It may copy Java arrays into native call buffers and copy output back. This is the compatibility default. |
| `CPU_NATIVE` | Prefer `MemorySegment`-backed native CPU storage for legal dense rank-2 GEMM. |
| `AUTO` | Let the planner choose native segment route for supported large dense GEMM shapes. |

BF16 symbol availability is more specific than `OPENBLAS_FFM` availability:

```text
cblas_sbgemm: BF16 inputs -> F32 output/continuation
cblas_bgemm:  BF16 inputs -> BF16 output
```

So a runtime can have OpenBLAS and `sbgemm` while still lacking `bgemm`. In that case BF16-to-F32 continuation may be accelerated, but BF16-output matmul/linear must not be reported as OpenBLAS BF16-output.

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

Current capability note: CPU remains the broadest backend. Metal has a real MPSGraph FFM path for a tested `FLOAT32`
and `BFLOAT16` subset, including native buffer binding when the shim exports the current buffer ABI. CUDA now has CUDA dense FLOAT32
buffer execution for the narrow native-buffer path: `CudaBufferAllocator`, `CudaDeviceToCpuMaterializer`,
`StorageResidency.DEVICE_OWNED`, and adjacent CUDA handoff are supported when the CUDA graph shim exposes the required
buffer symbols. Unsupported CUDA buffer layouts and dtypes fall back visibly, and CUDA trace and benchmark reports
publish `GPU_CUDA`, `cudaExecutionPath`, `cudaFallbackReason`, `acceleratorBufferReasonCode`, `acceleratorInputBytes`,
`acceleratorNativeDeviceCopyNs`, and `cpuMaterializationCount` evidence for that path. The detailed Metal capability
boundary, supported dtypes, buffer ABI, and fallback rules are in
[Metal Backend: Supported Operations And DTypes](metal-backend.md#supported-operations-and-dtypes).

CPU BF16 is a separate topic from Metal BF16. On CPU, BF16 storage generally uses `short[]`, while many elementwise computations promote to F32 and reductions may accumulate wider. This can make CPU BF16 slower than F32 for elementwise-heavy workloads even though BF16 stores fewer bytes. See [CPU BF16 Runtime](cpu-bf16.md#cpu-bf16-runtime).

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
| `compile` | Yes | Must be non-null |
| `runtime` | Yes | Must be non-null |
| `workload` | No | Null becomes `WorkloadProfile.none()` |

Example:

```java
ExecutionProfile profile = new ExecutionProfile(
        "demo",
        "demo",
        DataType.FLOAT64,
        ExecutionMode.FORWARD,
        CompileConfig.inference(),
        RuntimeConfig.inferenceDefaults()
);
```

Convenience profiles exist for the common runtime comparison axes:

| Preset | Meaning |
|---|---|
| `ExecutionProfile.cpuArray()` | Inference profile with CPU array storage and accelerators disabled. |
| `ExecutionProfile.cpuNative()` | Inference profile requesting native CPU storage and native-required diagnostics. |
| `ExecutionProfile.cpuAuto()` | Inference profile letting the CPU planner choose array vs native storage. |
| `ExecutionProfile.metalAuto()` | Inference profile with auto accelerator planning and CPU storage left on auto fallback policy. |

### GraphExecutionPolicy

**Source:** `src/main/java/config/profile/GraphExecutionPolicy.java`

`GraphExecutionPolicy` wraps `CompileConfig` for graph/autotune candidate assembly. It is compile-side policy only; runtime hardware thresholds remain in `RuntimeConfig` or persisted platform runtime profiles.

Presets:

- `GraphExecutionPolicy.trainingDefaults()`
- `GraphExecutionPolicy.inferenceDefaults()`
- `GraphExecutionPolicy.noGraphOptimization()`

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

Default runtime resolution is dtype-aware:

```java
RuntimeConfig runtime = RuntimeConfig.trainingDefaults(DataType.FLOAT32);
RuntimeConfig inference = RuntimeConfig.inferenceDefaults(DataType.FLOAT64);
```

Those overloads first ask `PlatformRuntimeProfileResolver` for a compatible calibrated profile for the current
platform, dtype, and execution mode. If no profile is found, they return the hardcoded defaults from
`RuntimeConfig.trainingDefaults()` or `RuntimeConfig.inferenceDefaults()`. `Tensor.compute(...)` and
`CompiledGraph.prepare()` use the dtype-aware overloads when the caller does not pass an explicit runtime config.

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

The platform id is intentionally short and portable. New calibration and tuning artifacts use:

```text
<normalized-os>-<normalized-arch>
```

Examples: `macos-arm64`, `macos-x64`, `linux-x64`. JVM vendor and CPU count are still recorded in
`HardwareFingerprint` metadata, but they are no longer part of the directory name. Runtime profile resolution can still
read older local directories such as `mac_os_x-aarch64-oracle_corporation-16c` as a compatibility fallback.

Runtime profile lookup order:

1. Roots from `-Dsynaptik.profiles.root=<path>`; multiple roots can be separated with the platform path separator.
2. Roots from `SYNAPTIK_PROFILES_ROOT`.
3. `./profiles`.
4. `~/.synaptik/profiles`.
5. Bundled classpath resources under `profiles/platform/...`.

### CLI Autotune Layout

**Source:** `src/main/java/synaptik/app/TuningCli.java`

CLI autotune for the built-in ABC workload writes:

```text
profiles/platform/<platform-id>/tuning/abc/<dtype>-best-profile.json
profiles/platform/<platform-id>/tuning/abc/<dtype>-history.jsonl
```

### Tensor Convenience Autotune Layout

**Source:** `src/main/java/tensor/internal/TensorExecutionSupport.java`

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
| `os.name` | JVM value | Used by `buildMetalMpsShim`, `nativeBuild`, and `metalTest` to run only on macOS. | `build.gradle` |
| `nvcc` on `PATH` | absent | Enables optional `buildCudaGraphShim` and `cudaTest`; outputs the shim under `build/native/cuda`. | `build.gradle`, `scripts/build-cuda-graph-shim.sh` |

### Runtime System Properties

| Property | Default | Effect | Source |
|---|---|---|---|
| `cg.cpu.blas.provider` | `NONE` | Selects `BlasProvider`; valid known values are `NONE`, `OPENBLAS_FFM`. | `backend/blas/BlasRuntime.java` |
| `cg.cpu.blas.matmulMinWork` | `2000000` | Positive long threshold for BLAS matmul dispatch. | `backend/blas/BlasRuntime.java` |
| `cg.cpu.blas.debug` | `false` | Enables BLAS debug behavior where used. | `backend/blas/BlasRuntime.java` |
| `cg.cpu.blas.f32RequireMgeK` | `true` | F32 BLAS shape guard. | `backend/blas/BlasRuntime.java` |
| `cg.cpu.blas.f32MaxNOverK` | `3.0` | F32 BLAS shape ratio guard. | `backend/blas/BlasRuntime.java` |
| `openblas.lib` | bundled JavaCPP OpenBLAS, then library name `openblas` | Explicit OpenBLAS library path/name for FFM lookup. When unset, Synaptik tries the bundled `org.bytedeco:openblas-platform` dependency before the platform loader name. | `backend/blas/OpenBlasSymbols.java` |
| `synaptik.metal.mps.lib` | bundled JAR resource, then library name `synaptik_apple_mps` | Explicit Metal MPS shim library path/name for FFM lookup. When unset, Synaptik tries the bundled platform resource before the system library name. | `backend/metal/bridge/MetalNativeLibraryResolver.java` |
| `synaptik.native.cache.dir` | `~/.synaptik/native` | Root directory for extracted bundled native libraries. The Metal shim is cached below `metal-mps/<platform>/<sha256>/`. | `backend/metal/bridge/MetalNativeLibraryResolver.java` |
| `synaptik.cuda.graph.lib` | library name `synaptik_cuda_graph` | Explicit CUDA graph bridge library path/name for FFM lookup. | `backend/cuda/bridge/CudaFfmBridge.java` |
| `cg.cpu.fused.profile` | `false` | Enables fused execution profiler collection. | `backend/cpu/kernels/fused/FusedExecutionProfiler.java` |
| `cg.math.forceExactTranscendentals` | `false` | Forces utility fast transcendental methods to call exact `Math` functions. | `utils/FastTranscendentals.java` |
| `cg.optimizer.ar.disableAllTransforms` | `false` | Disables all algebraic rewrite transforms. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableRebuildTopologicalClosure` | `false` | Disables AR topological closure rebuild. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disablePow2ToMul` | `false` | Disables `pow(x, 2)` to multiply rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableAddSelfToMul2` | `false` | Disables `x + x` rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableAddNegToZero` | `false` | Disables add-negative-to-zero rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableAddNegNegToNegAdd` | `false` | Disables negative-add rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableAddLogLogToLogMul` | `false` | Disables log-add rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableSubNegToAdd` | `false` | Disables subtract-negative rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableDivConstToMulRecip` | `false` | Disables division-by-constant reciprocal rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableDivMulScalarByConst` | `false` | Disables scalar division rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableDivInvToMul` | `false` | Disables division-by-inverse rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableDivOneToInv` | `false` | Disables reciprocal rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableMulScalarAssoc` | `false` | Disables scalar multiplication association rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableMulScalarNegPush` | `false` | Disables scalar negative pushdown rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableMulScalarConstFold` | `false` | Disables scalar constant folding rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableAddSubFactorize` | `false` | Disables add/sub factorization rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableMulInvToOne` | `false` | Disables inverse multiplication rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableMulNegNegToMul` | `false` | Disables double-negative multiply rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableMulExpExpToExpAdd` | `false` | Disables exp multiplication rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableNegSubSwap` | `false` | Disables negative subtraction swap rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableNegMulScalarPush` | `false` | Disables negative scalar pushdown rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disablePowPowFlatten` | `false` | Disables power flattening rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disablePowInvToNegExp` | `false` | Disables inverse power rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableLogPowToMulLog` | `false` | Disables log-power rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableLogInvToNegLog` | `false` | Disables log-inverse rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableLogSqrtToHalfLog` | `false` | Disables log-sqrt rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableExpLogCancel` | `false` | Disables exp-log cancellation. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableInvSigmoidPattern` | `false` | Disables inverse sigmoid pattern rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableInvPowToNegExp` | `false` | Disables inverse power to negative exponent rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableInvExpToExpNeg` | `false` | Disables inverse exp rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableInvNegPush` | `false` | Disables inverse negative push rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableClampMinIdentity` | `false` | Disables clamp-min identity rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableClampMinFlatten` | `false` | Disables clamp-min flatten rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableClampMaxIdentity` | `false` | Disables clamp-max identity rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |
| `cg.optimizer.ar.disableClampMaxFlatten` | `false` | Disables clamp-max flatten rewrite. | `graph/optimizer/rewrite/algebraic/AlgebraicRewriteSwitches.java` |

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
| `OPENBLAS_LIB` | OpenBLAS library path/name used when `openblas.lib` is unset; it wins over bundled JavaCPP lookup. | `backend/blas/OpenBlasSymbols.java` |
| `SYNAPTIK_METAL_MPS_LIB` | Fallback Metal MPS shim library path/name when `synaptik.metal.mps.lib` is unset. It has priority over the bundled JAR resource. | `backend/metal/bridge/MetalNativeLibraryResolver.java` |
| `SYNAPTIK_CUDA_GRAPH_LIB` | Fallback CUDA graph bridge library path/name when `synaptik.cuda.graph.lib` is unset. | `backend/cuda/bridge/CudaFfmBridge.java` |

Optional CUDA native tasks:

```bash
./gradlew buildCudaGraphShim
./gradlew cudaTest
./gradlew buildCudaGraphShim cudaTest
```

`buildCudaGraphShim` writes `build/native/cuda/libsynaptik_cuda_graph.*`. `cudaTest` sets `synaptik.cuda.graph.lib` to that output path before running CUDA-focused tests, and is skipped when `nvcc` is unavailable.
Native CUDA tests skip when nvcc or CUDA hardware is unavailable. The portable Java gates for CUDA buffer execution do
not require CUDA hardware; use `./gradlew buildCudaGraphShim cudaTest` only when validating the optional native shim.
CUDA fallback interpretation uses `acceleratorBufferReasonCode`, `cudaFallbackReason`, and stable reason codes such as
`NATIVE_BUFFER_ABI_UNAVAILABLE`, `REQUIRED_BUFFER_EXECUTION_UNAVAILABLE`, and `NATIVE_BUFFER_EXECUTION_FAILED`.

## CLI Configuration Behavior

**Source:** `src/main/java/synaptik/app/TuningCli.java` and `src/main/java/tuning/calibration/run/CalibrationCommand.java`

Commands accepted by `TuningCli`:

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
