<!-- generated-by: gsd-doc-writer -->
# Development

Navigation: [Index](index.md#recommended-reading-paths) | [Architecture](architecture.md#system-overview) | [Modules](modules.md#package-map) | [Adding Tensor Operation](adding-tensor-operation.md#implementation-checklist) | [Native Bridges & BLAS](native-bridges-and-blas.md#java-ffm-step-by-step) | [Metal Backend](metal-backend.md#tests) | [Testing](testing.md#exact-commands) | [Configuration](configuration.md#build-requirements) | [Troubleshooting](troubleshooting.md#generated-artifacts-in-source-tree)

Chapters: [Local Setup](#local-setup) | [Repository Structure](#repository-structure) | [Coding Patterns](#coding-patterns) | [Adding Tensor Ops](#adding-tensor-ops) | [Adding Backend Kernels](#adding-backend-kernels) | [Adding Optimizer Rules](#adding-optimizer-rules) | [Adding Tuning Knobs And Families](#adding-tuning-knobs-and-families) | [Documentation Workflow](#documentation-workflow) | [Operational Risks](#operational-risks)

Synaptik is a Java tensor and compiled-graph runtime. The public tensor API builds semantic graph nodes, the graph layer compiles and optimizes them, and backend packages prepare and execute concrete kernels.

## Table Of Contents

- [Local Setup](#local-setup)
- [Repository Structure](#repository-structure)
- [Coding Patterns](#coding-patterns)
- [Adding Tensor Ops](#adding-tensor-ops)
- [Adding Backend Kernels](#adding-backend-kernels)
- [Adding Optimizer Rules](#adding-optimizer-rules)
- [Adding Tuning Knobs And Families](#adding-tuning-knobs-and-families)
- [Documentation Workflow](#documentation-workflow)
- [Operational Risks](#operational-risks)

## Local Setup

Requirements verified from `build.gradle`, `settings.gradle`, and `gradle/wrapper/gradle-wrapper.properties`:

- JDK 25 toolchain: `java { toolchain { languageVersion = JavaLanguageVersion.of(25) } }`
- Gradle wrapper distribution: Gradle `9.4.1`
- JUnit Jupiter `5.11.2` for tests
- Incubator Vector API enabled for compile, test, run, and application tasks
- Native access enabled for test/run/application tasks with `--enable-native-access=ALL-UNNAMED`

Use the wrapper from the repository root:

```bash
./gradlew classes
./gradlew test --no-daemon
./gradlew run --args="calibrate --dtype f64 --family matmul"
```

If the full suite runs out of heap, use the project-supported test heap property:

```bash
./gradlew test --no-daemon -Dsynaptik.testMaxHeap=4g
```

`build.gradle` sets `maxHeapSize = '2g'` for all `Test` tasks unless `-Dsynaptik.testMaxHeap=<size>` is provided.

## Repository Structure

The main code is under `src/main/java`:

| Path | Role |
|---|---|
| `src/main/java/tensor` | Public tensor API, graph-building helpers, dtype/storage/layout support |
| `src/main/java/tensor/ops` | Family-specific public operation builders and backward formulas |
| `src/main/java/operations` | Immutable primitive descriptors implementing `operations.Operation` |
| `src/main/java/graph` | Compilation, compile planning, optimizer state, prepared graph artifacts, execution trace metadata |
| `src/main/java/graph/optimizer` | Backend-neutral graph optimization: `AR`, `CF`, `CSE`, `DCE`, optional `LOWER` |
| `src/main/java/backend` | Backend-neutral contracts and backend-specific CPU/Metal/CUDA/OpenCL roots |
| `src/main/java/backend/cpu` | Complete CPU backend implementation, registry, prepare, lowering, kernels, fused execution |
| `src/main/java/backend/metal` | Metal bridge/lowering/prepare scaffolding using the local MPS shim |
| `src/main/java/backend/cuda` | CUDA bridge/lowering/prepare scaffolding |
| `src/main/java/config` | Runtime, optimizer, backend, and profile configuration records |
| `src/main/java/tuning` | Benchmark, calibration, autotune, workload, persistence, and reporting code |
| `src/main/java/tuning/api` | Fluent Java API over calibration, execution-profile construction, benchmark request/session flows, and benchmark report policy |
| `src/main/java/numerics` | Numerics comparison CLI and harness |
| `src/main/java/synaptik/app/TuningCli.java` | Tuning CLI entry point |
| `src/main/java/synaptik/app/Main.java` | Programmatic calibration and benchmark example using regular Java calls |
| `src/main/native/apple` | Metal MPS Objective-C shim source |
| `scripts/build-metal-mps-shim.sh` | Builds `build/native/apple/libsynaptik_apple_mps.dylib` on macOS; packaging copies it into generated JAR resources |
| `profiles/platform/...` | Checked-in platform calibration/report artifacts for the current known platform |

Root-level backend implementation classes are intentionally limited. `SourceTreeHygieneTest.backendRootContainsOnlyFacadeFiles` allows only `ApproxMode.java`, `ComputeBackend.java`, and `ComputeEngine.java` directly under `src/main/java/backend`.

## Coding Patterns

Follow the current layering:

```text
Tensor / TensorOps
  -> tensor.ops.*
  -> operations.*
  -> graph / graph.optimizer
  -> backend.prepare and backend.<target>.prepare
  -> backend.<target>.registry
  -> backend.<target>.kernels
```

Keep these boundaries intact:

- Public API delegation lives in `src/main/java/tensor/Tensor.java` and `src/main/java/tensor/TensorOps.java`.
- Shape validation, dtype choice, primitive construction, and backward graph formulas live in `src/main/java/tensor/ops/<family>/`.
- Primitive descriptors live in `src/main/java/operations/<family>/` and should carry immutable semantic parameters only.
- Backend kernel selection for CPU is centralized in `src/main/java/backend/cpu/kernels/CpuKernelRegistry.java`.
- CPU preparation and workspace decisions live in `src/main/java/backend/cpu/prepare/CpuNodePreparer.java`.
- Runtime threshold interpretation lives in CPU planning classes such as `src/main/java/backend/cpu/kernels/plan/CpuExecutionPlanner.java`.
- Optimizer stage wiring lives in `src/main/java/graph/optimizer/OptimizerFactory.java`.
- Graph autotune candidates must stay graph-policy-only; `SourceTreeHygieneTest.graphAutotuneCandidatePackageDoesNotImportRuntimeOrBackendConfig` rejects runtime/backend config imports from `src/main/java/tuning/candidate/graph`.

Source hygiene tests also reject legacy package paths such as `graph.fused`, `graph.codegen`, `graph.optimizer.fusion`, `operations.fused`, `backend/kernels/cpu`, `backend/kernels/cuda`, and `backend/kernels/opencl`.

## Adding Tensor Ops

For the full end-to-end guide, including descriptors, builders, public facades, CPU kernels, autograd formulas,
CSE signatures, fusion/accelerator integration, and tests, see [Adding A Tensor Operation: Implementation Checklist](adding-tensor-operation.md#implementation-checklist).

Use an existing family as the template. For a binary elementwise op, compare:

- Descriptor: `src/main/java/operations/elementwise/binary/add.java`
- Builder/backward: concrete operation classes under `src/main/java/tensor/ops/binary/`, such as `AddOp.java`
- Public static facade: `src/main/java/tensor/TensorOps.java`
- Instance method facade: `src/main/java/tensor/Tensor.java`
- CPU kernel: `src/main/java/backend/cpu/kernels/elementwise/binary/CpuAddKernel.java`
- CPU registry entry: `src/main/java/backend/cpu/kernels/CpuKernelRegistry.java`
- Coverage: `src/test/java/AllOpsTest.java`, family-specific execution tests, and dtype/broadcast tests when applicable

Checklist for a new primitive:

1. Add an `Operation.OpType` enum value in `src/main/java/operations/Operation.java`.
2. Add an immutable descriptor class under the correct `src/main/java/operations/...` family.
3. Add or extend the matching `src/main/java/tensor/ops/...` builder.
4. Use `TensorPrimitiveBuilder.unary`, `binary`, `ternary`, `nary`, or view helpers instead of directly mutating tensor internals.
5. Attach backward logic with `TensorInternalAccess.setBackwardFunction` when the op participates in autograd.
6. Add a static wrapper in `src/main/java/tensor/TensorOps.java`.
7. Add an instance wrapper in `src/main/java/tensor/Tensor.java` if the op is part of the user-facing fluent API.
8. Add CPU runtime support and register it in `CpuKernelRegistry` unless the op is compile-only or descriptor-only.
9. Add tests for forward values, gradients, dtype handling, shape validation, and optimizer interaction if a rewrite can see the op.

Broadcasting should use the existing planners. Binary ops call `TensorBroadcastOps.planBinary(...)`, which delegates to `BroadcastPlanner` and throws `IllegalArgumentException("Broadcast mismatch at dim ...")` when aligned dimensions are incompatible.

Two extra rules are easy to miss:

- If the operation descriptor has semantic parameters, update `CommonSubexpressionEliminationRule.parameterKey(...)`.
- If the concrete `Operation` descriptor returns `true` from `isFusable()`, the interpreted and ASM fused paths must support it before the flag is safe.

Not every public API addition needs a new primitive descriptor. If the requested behavior is an ergonomic composition of existing primitives, prefer a composition-first helper under the relevant `tensor.ops.*` family.

Examples:

- `Tensor.stack(axis, ...)` is implemented as `expandDims` plus `concat`; the new public surface is useful, but the math does not require a `STACK` operation descriptor.
- Masked `mean(axis, mask)` composes mask broadcasting, `where`, `sum`, valid-count reduction, and division; it does not need a separate masked-reduction kernel before performance evidence says otherwise.
- `Tensor.take(axis, int[])` is an ergonomic wrapper over `gatherAxis` semantics; the Java array overload constructs an index tensor and delegates.

Use this rule of thumb:

```text
If the graph semantics are naturally expressed as a small, readable DAG of existing primitives,
add a public helper and tests before adding a new operation id.
```

When you choose the composition path, still document it and test it like public API. For sequence-shaped tensor helpers, `NdTensorSequencePrimitivesTest` is the reference test class: it covers factories, shape helpers, N-D `linear`, stack/unstack, axis indexing, masked reductions, masked cross entropy, and gradients through these composed helpers.

## Adding Backend Kernels

The CPU backend is the complete execution backend. New CPU kernels should live under `src/main/java/backend/cpu/kernels/<family>/`, not under old root paths.

CPU kernel checklist:

1. Implement `src/main/java/backend/cpu/kernels/CpuKernel.java`.
2. Implement `execute(CpuKernelCall)` and keep dtype/storage/layout routing at the kernel-family boundary.
3. Use the existing family owner where possible, such as `StorageAwareBinaryElementwiseKernel`, `ElementwiseUnaryExecutor`, reduction executors, matmul executables, or conv/pool executors.
4. Read prepared metadata from `CpuKernelCall` / `CpuKernelContext`, not from ad hoc policy logic inside the hot loop.
5. Add a singleton field and switch case in `src/main/java/backend/cpu/kernels/CpuKernelRegistry.java`.
6. If the op needs workspace or special prepared metadata, update the relevant planner under `src/main/java/backend/cpu/prepare/...`.
7. Add execution tests that call `CompiledGraph.compile(...).execute(...)` rather than only testing helper methods.

For numeric binary elementwise kernels, use `CpuAddKernel` as the reference shape: concrete scalar/vector loops live in the final op kernel while storage/layout/native routing is handled by the storage-aware binary family base.

For native or accelerator-adjacent paths:

- OpenBLAS FFM lookup checks `-Dopenblas.lib=<path>`, then `OPENBLAS_LIB`, then bundled JavaCPP OpenBLAS, then library name `openblas`.
- Metal MPS lookup checks `-Dsynaptik.metal.mps.lib=<path>`, then `SYNAPTIK_METAL_MPS_LIB`, then the bundled classpath resource `native/<platform>/libsynaptik_apple_mps.dylib`, then library name `synaptik_apple_mps`.
- CUDA lookup checks `-Dsynaptik.cuda.graph.lib=<path>`, then `SYNAPTIK_CUDA_GRAPH_LIB`, then library name `synaptik_cuda_graph`.
- macOS Metal shim build commands:

```bash
./gradlew buildMetalMpsShim
./gradlew nativeBuild
./gradlew metalTest
```

- CUDA shim build/probe commands:

```bash
./gradlew buildCudaGraphShim
./gradlew cudaTest
```

`buildCudaGraphShim` calls `scripts/build-cuda-graph-shim.sh` and writes the optional native shim to `build/native/cuda/libsynaptik_cuda_graph.*`. Use it with `-Dsynaptik.cuda.graph.lib=<path>`, `SYNAPTIK_CUDA_GRAPH_LIB`, or the default library name `synaptik_cuda_graph`. CUDA native build is optional; default Java lifecycle tasks stay portable and do not require CUDA toolkit, CUDA hardware, or `nvcc`.

The general native-bridge model, including BLAS/GEMM terminology and Java FFM symbol binding, is documented in
[Native Bridges & BLAS: Java FFM Step-By-Step](native-bridges-and-blas.md#java-ffm-step-by-step). Read it before changing `backend.blas`, `OpenBlasRuntime`, `OpenBlasArrayGemm`, `OpenBlasSegmentGemm`,
or native dispatch thresholds.

`buildMetalMpsShim` is the low-level task that calls `scripts/build-metal-mps-shim.sh` and writes `build/native/apple/libsynaptik_apple_mps.dylib`. `refreshMetalMacosArm64Resource` rebuilds that shim on Apple Silicon and refreshes the committed native resource in `synaptik-metal-macos-arm64/src/main/resources/native/macos-arm64/`. `nativeBuild` is the user-facing optional-native lifecycle task. `metalTest` builds the shim, sets `-Dsynaptik.metal.mps.lib` to the freshly built dylib, and runs only Metal/MPS-focused tests.

The core Synaptik JAR does not compile or embed the Metal shim during `processResources`. The macOS ARM64 binary is published by the separate `synaptik-metal-macos-arm64` artifact and is a runtime dependency of the main artifact. This keeps JitPack/Linux core builds portable while still giving Apple Silicon consumers a classpath resource that `MetalNativeLibraryResolver` can extract. Use `nativeBuild` or `metalTest` when a change touches `src/main/native/apple`, `src/main/java/backend/metal`, or Metal partition/lowering behavior and you want an explicit native verification command. The native ABI and Objective-C call path are documented in [Metal Backend: Native Buffer ABI](metal-backend.md#native-buffer-abi) and [Metal Backend: Objective-C Native Shim](metal-backend.md#objective-c-native-shim).

Layout ABI v2 capability checks are optional-symbol gated for both Metal and CUDA. Portable bridge tests cover
missing-symbol behavior without requiring hardware; native tasks such as `./gradlew metalTest` and
`./gradlew buildCudaGraphShim cudaTest` verify local shim exports when the toolchain is available.

For CUDA bridge and buffer-policy changes, use focused portable checks first:

```bash
./gradlew classes
./gradlew test --tests backend.accelerator.lowering.GpuBackendParityReportTest
./gradlew test --tests backend.cuda.bridge.CudaCapabilityReportTest --tests backend.cuda.bridge.CudaFfmBridgeTest
./gradlew test --tests backend.cuda.buffer.CudaAcceleratorBufferBinderTest
./gradlew test --tests backend.cuda.buffer.CudaBufferAllocatorTest
./gradlew test --tests backend.cuda.buffer.CudaDeviceToCpuMaterializerTest
./gradlew test --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest
./gradlew test --tests GpuCoverageTriageReportTest --tests GpuHotPathCoverageTargetsTest
./gradlew test --tests SourceTreeHygieneTest
```

CUDA buffer decisions use stable reason code strings such as `NATIVE_BUFFER_ABI_UNAVAILABLE`, `INPUT_DTYPE_UNSUPPORTED`, `OUTPUT_LAYOUT_UNSUPPORTED`, `INPUT_BINDING_UNAVAILABLE`, `INPUT_NOT_CPU_CURRENT`, `NATIVE_BUFFER_EXECUTION_FAILED`, and `REQUIRED_BUFFER_EXECUTION_UNAVAILABLE`. Phase 7 proves CUDA dense FLOAT32 buffer execution, `CudaBufferAllocator`, `CudaDeviceToCpuMaterializer`, `StorageResidency.DEVICE_OWNED`, and adjacent CUDA handoff for compatible `CudaBufferBinding` instances. Unsupported CUDA buffer layouts and dtypes fall back visibly, and CPU remains the correctness oracle for every portable CUDA result. Phase 41 makes CUDA dtype decisions role-specific: `FLOAT32` is compute/output, `INT32` is index-input/residency only, `BOOL` is predicate-input/residency only, and `BFLOAT16` is residency-only. `dtype residency is not native dtype compute`. This is narrow dense `FLOAT32` CUDA buffer coverage, not broad CUDA operation coverage.

CUDA fallback interpretation starts with `acceleratorBufferReasonCode` and `cudaFallbackReason` in the run trace or benchmark report. The shared accelerator buffer ABI used by Metal and CUDA keeps public `Tensor` objects logical while `backend.cuda.*` owns CUDA handles and lifetimes.

For the GPU layout transform and view path, use the shared planner and trace checks before native-only gates:

```bash
./gradlew classes
./gradlew test --tests backend.accelerator.buffer.AcceleratorLayoutTransformPlannerTest --tests graph.execution.DeviceLayoutViewPropagationTest
./gradlew test --tests backend.metal.MetalLayoutAwareDeviceFlowTest --tests backend.cuda.exec.CudaLayoutTransformDeviceFlowTest
./gradlew metalTest
./gradlew buildCudaGraphShim cudaTest
```

Metadata-only views can preserve device bindings without Java array materialization. Dense GPU materialization covers
`contiguous()` and non-contiguous-source `reshape` only when backend capability and run-scoped service wiring exist.
Direct non-dense CUDA compute remains conservative until Phase 11 lowering coverage; fallback must keep
`acceleratorBufferReasonCode`, `cudaFallbackReason`, or `CpuMaterializationTrace` evidence visible. The valid CPU
materialization boundaries are graph output, a CPU consumer, and gradient publication.

### GPU lowering coverage checks

Use these focused gates after changing GPU operation coverage, shared accelerator DAG lowering, or Metal/CUDA legality:

```bash
./gradlew classes
./gradlew test --tests backend.accelerator.lowering.* --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest
./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest
./gradlew metalTest
./gradlew buildCudaGraphShim cudaTest
```

Optional native CUDA verification uses:

```bash
./gradlew buildCudaGraphShim cudaTest
```

Native CUDA tests skip when nvcc or CUDA hardware is unavailable. Do not commit local CUDA build outputs or local profile tuning files produced while running these checks.

For Phase 17 normalization, reduction, and loss-adjacent closure, use the focused portable command below. It proves that `LOG_SOFTMAX remains lowered as SOFTMAX followed by LOG`, `loss-adjacent fallback remained visible`, `CPU parity remained the correctness oracle`, and `native reduction and normalization support is not implied by a fallback row`.

```bash
./gradlew test --tests backend.accelerator.lowering.GpuLoweringCoverageMatrixTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest
```

### GPU compound region checks

Use these focused gates after changing GPU compound region lowering, including `LINEAR_BIAS_ACTIVATION`, `ELEMENTWISE_CHAIN`, `REDUCTION_ADJACENT`, or CPU fused rejection behavior. `Operation.OpType.FUSED remains CPU-only`; the public Tensor remains logical and device residency stays in ExecutionState and DeviceBufferBinding. Metal and CUDA coverage is backend-specific, so run both portable backend tests and optional native gates when the toolchain is available.

```bash
./gradlew classes
./gradlew test --tests backend.accelerator.lowering.* --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest
./gradlew test --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest
./gradlew test --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest
./gradlew metalTest
./gradlew buildCudaGraphShim cudaTest
```

For Phase 18 fused elementwise and epilogue subregions, `GPU fusion is region-internal lowering/fusion, not CPU fused ASM reuse`. Trace and benchmark coverage changes must preserve fallback/materialization evidence and render `gpuFusedSubpatternCount`, `gpuFusedSubpatternTypes`, `gpuFusedSubpatternOriginalNodeIds`, `gpuFusedSubpatternLoweredPrimitiveCount`, and `gpuFusedSubpatternReasons`.

Phase 18 closure also expects source hygiene gates proving accelerator, Metal, and CUDA packages do not import CPU fused internals. Local tuning files are not closure evidence; `profiles/platform/.../tuning/abc/* remained unstaged`.

For Phase 19 multi-op GPU region execution, a `selected GPU partition can execute as one backend-owned lowered region`
only when shared lowering, backend legality, dtype/layout, capability, and native-buffer binding gates accept the
candidate. `ExecutionState and device buffer bindings carry supported internal values`, while public `Tensor` remains a
logical API and CPU-readable publication stays at graph output, CPU consumer, or gradient publication boundaries.

Use the same gates plus coverage/report checks for Phase 19 changes:

```bash
./gradlew classes
./gradlew test --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest --tests PreparedExecutionBuildTest --tests CompiledGraphTraceTest --tests GpuCoverageSummaryTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests backend.metal.exec.PreparedMetalExecutableBufferBindingTest --tests backend.cuda.exec.PreparedCudaExecutableBufferPolicyTest --tests graph.execution.DeviceLayoutViewPropagationTest --tests SourceTreeHygieneTest
```

`tensor-array bridge execution is not native buffer GPU coverage`; preserve separate `nativeBufferStepCount` and
`tensorArrayStepCount` evidence in reports. `GPU fusion remains region-internal lowering/fusion, not CPU fused ASM
reuse`, and `vendor library routing is deferred to GPULIB-*`. Do not imply universal Metal/CUDA op support:
normalization, reduction, conv, and loss-adjacent blockers must continue to report visible support/rejection outcomes.
Local tuning files are not Phase 19 evidence; `profiles/platform/.../tuning/abc/* remained unstaged`.

### GPU coverage regression checks

Use these focused gates after changing GPU coverage summaries, benchmark report rendering, representative workload
baselines, or regression gate policy. They validate the checked-in evidence contract without depending on
machine-local benchmark/calibration output.

```bash
./gradlew classes
./gradlew test --tests GpuCoverageSummaryTest --tests GpuCoverageRegressionGateTest --tests BenchmarkSessionTest --tests BenchmarkSuiteSessionTest --tests CompiledGraphTraceTest
./gradlew metalTest
./gradlew buildCudaGraphShim cudaTest
git status --short
```

Treat `profiles/platform/.../tuning/abc/*` as local tuning output unless a plan explicitly promotes a stable fixture.
Do not commit local tuning artifacts from these paths as Phase 13 evidence. The canonical proof is the checked-in
tests, docs, and report fields such as `gpuCoverageRatio`, `cpuMaterializationReasonCounts`, and hidden tensor-array
fallback failures.

Phase 20 coverage regression hardening tightens that contract for v1.3 closure. `hot path stayed on GPU is trace/report evidence, not timing-only`, so coverage gates must read trace/report fields rather than benchmark medians.
Reports expose `targetCoverageGates`, `nativeEvidence`, and `capabilitySkipped` to distinguish portable Java proof from
native Metal/CUDA pass/skip evidence.

`tensor-array bridge execution is not native buffer GPU coverage`; preserve native buffer, tensor-array, fallback, CPU
materialization, and handoff counters as separate evidence. Local machine tuning output remains non-canonical, and
Phase 20 closure expects `profiles/platform/.../tuning/abc/* remained unstaged`.

### GPU coverage triage checks

Use these focused gates after changing Phase 14 GPU coverage triage, hot-path target selection, or triage report
rendering:

```bash
./gradlew test --tests GpuCoverageGapTriageTest --tests GpuHotPathCoverageTargetsTest --tests GpuCoverageTriageReportTest
./gradlew classes
```

These checks prove the portable triage/report contract. Native Metal and CUDA execution remains capability-gated and
does not replace the checked target list in `.planning/phases/14-coverage-gap-triage-and-hot-path-targets/14-HOT-PATH-TARGETS.md`.

### GPU lowered-region manifest checks

Use these focused Phase 15 gates after changing `GpuLoweredRegionManifest`, shared accelerator lowering manifests,
prepare/backend-selection trace metadata, benchmark manifest rendering, or source hygiene around local tuning output:

```bash
./gradlew test --tests backend.accelerator.lowering.GpuLoweredRegionManifestTest --tests backend.accelerator.lowering.AcceleratorSubgraphLowererTest --tests CompiledGraphTraceTest
./gradlew test --tests backend.metal.lowering.MetalRegionLowererTest --tests backend.cuda.lowering.CudaRegionLowererTest
./gradlew test --tests BenchmarkSessionTest --tests GpuCoverageSummaryTest
./gradlew test --tests SourceTreeHygieneTest
```

The manifest is Java-side trace/report metadata. Public `Tensor` remains logical, native Metal/CUDA ABI stays
backend-owned, and CPU `Operation.OpType.FUSED` remains CPU-only.

### GPU dtype residency checks

Use this focused Phase 16 gate after changing runtime typed slot binding, Metal/CUDA dtype residency policy, lowered-region dtype evidence, or benchmark dtype residency report fields:

```bash
./gradlew test --tests graph.execution.RuntimeMemoryBinderTest --tests backend.accelerator.residency.AcceleratorDTypeResidencyPolicyTest --tests GpuCoverageSummaryTest
```

`dtype residency is not native dtype compute`: `BFLOAT16`, `INT32`, `INT64`, and `BOOL` may be represented in runtime storage residency or trace evidence while Metal/CUDA still reject unsupported native compute/output roles with `UNSUPPORTED_DTYPE`. Metal BF16 is now native only for scoped operation families; generic INT32/INT64 compute, BOOL-producing compute outside scoped BOOL families, FLOAT64, and unsupported BF16 families remain explicit fallback/rejection cases.

## Adding Graph Optimization Or Planning Rules

Graph optimization is configured by `src/main/java/config/compile/GraphOptimizationConfig.java` and built by `src/main/java/graph/optimizer/OptimizerFactory.java`.

```text
CLEANUP_FIXPOINT(AR -> CF -> CSE -> DCE) -> optional LOWER
```

Current ownership:

- `GraphOptimizationConfig.trainingDefaults()` enables `AR`, `CF`, `CSE`, `DCE`, and `LOWER` with strict CSE.
- `GraphOptimizationConfig.inferenceDefaults()` enables the same graph stages with inference CSE defaults.
- `GraphOptimizationConfig.noGraphOptimization()` disables graph optimization only.
- Backend planning, region optimization, and memory planning are owned by `CompileConfig`, not the graph optimizer.

Add changes by ownership:

| Change | Target path |
|---|---|
| Algebraic identity or lowering | `src/main/java/graph/optimizer/rewrite` |
| Constant folding | `src/main/java/graph/optimizer/simplify` |
| Common subexpression behavior | `src/main/java/graph/optimizer/simplify/CommonSubexpressionEliminationRule.java` |
| Backend ownership planning | `src/main/java/graph/compile` and `src/main/java/graph/compile/planning/partition` |
| Region/fused execution units | `src/main/java/graph/compile/planning/region` and CPU-specific fused policy under `src/main/java/backend/cpu/fused` |
| Memory reuse or binding policy | `src/main/java/graph/compile/planning/memory` |

`OptimizerFactory.create(...)` maps graph optimization config to concrete rules:

- `AR` -> optional `PiecewiseCanonicalizationRule` and `AlgebraicSimplificationRule`
- `CF` -> `new ConstantFoldingRule()`
- `CSE` -> `new CommonSubexpressionEliminationRule(config.cse())`
- `DCE` -> `new DeadCodeEliminationRule()`
- `LOWER` -> optional `LinearLoweringRule`, loss lowering rules, and optional `Conv2dDagLoweringRule`

When a new operation has semantic parameters, update CSE signature handling in `CommonSubexpressionEliminationRule.parameterKey(...)`; otherwise structurally different instances may collapse incorrectly or identical instances may fail to collapse.

Use focused tests:

```bash
./gradlew test --no-daemon --tests AlgebraicRewritingPowTest
./gradlew test --no-daemon --tests CommonSubexpressionEliminationRuleTest
./gradlew test --no-daemon --tests graph.optimizer.GraphOptimizerSinglePassTest
./gradlew test --no-daemon --tests graph.compile.planning.region.DefaultRegionOptimizerServiceTest
./gradlew test --no-daemon --tests graph.compile.planning.memory.MemoryPlannerRegionViewTest
```

## Adding Tuning Knobs And Families

Runtime knobs are owned by config/profile/planner code, not optimizer rules.

Primary files:

- `src/main/java/config/backend/CpuKernelConfig.java`
- `src/main/java/config/runtime/RuntimeConfig.java`
- `src/main/java/config/profile/PlatformRuntimeProfile.java`
- `src/main/java/backend/cpu/kernels/plan/CpuExecutionPlanner.java`
- `src/main/java/tuning/calibration/family/CalibrationFamilyRegistry.java`
- `src/main/java/tuning/calibration/PlatformCalibrationDefaults.java`
- `src/main/java/tuning/autotune/TuningDefaults.java`
- `src/main/java/tuning/workload/StandardWorkloads.java`
- `src/main/java/tuning/workload/CalibrationWorkloads.java`

The standard calibration suite is defined in `CalibrationFamilyRegistry.standardSuite()`:

```text
scheduler
matmul
attention-matmul
elementwise-dispatch
fused-dispatch
fused-cheap-contiguous-width
fused-cheap-strided-width
fused-noncheap-contiguous-width
fused-noncheap-strided-width
reduction
attention-thresholds
materialization
```

`metal-selection` is only included by `CalibrationFamilyRegistry.fullSuite(true)` when the CLI receives `--include-accelerators`.

When adding a runtime knob:

1. Add the field and defaults to the appropriate config record/class.
2. Thread it through `PlatformRuntimeProfile` and profile IO if it must persist.
3. Thread it into planner policy, usually via `CpuExecutionPlanner.from(CpuKernelConfig)`.
4. Add it to the owning calibration family in `CalibrationFamilyRegistry`.
5. Add candidates in `PlatformCalibrationDefaults`.
6. Add tests that verify ownership and candidate generation.
7. Update `src/main/java/tuning/KNOBS.md` and any profile examples that expose the field.

Useful CLI commands:

```bash
./gradlew run --args="calibrate --dtype f64 --family matmul"
./gradlew run --args="calibrate --dtypes all --families all --preset quick --progress lines --color never"
./gradlew run --args="autotune f64"
./gradlew run --args="benchmark-winner f64"
./gradlew run --args="benchmark-graph-space f64"
```

## Documentation Workflow

Documentation is Markdown in the repository. There is no dedicated documentation Gradle task in `build.gradle`.

Update docs next to the code being changed:

- Project overview: `README.md`
- Public tensor API and package boundaries: `src/main/java/tensor/README.md`, `src/main/java/tensor/API.md`
- Primitive descriptors: `src/main/java/operations/README.md`
- Graph lifecycle: `src/main/java/graph/README.md`
- Optimizer stages: `src/main/java/graph/optimizer/*.md`
- Backend architecture: `src/main/java/backend/README.md`, `src/main/java/backend/cpu/README.md`, `src/main/java/backend/prepare/README.md`, `src/main/java/backend/lowering/README.md`, `src/main/java/backend/partition/README.md`
- Tuning: `src/main/java/tuning/*.md`
- Contributor-facing docs: `docs/development.md`, `docs/testing.md`, `docs/troubleshooting.md`

When changing behavior, update the nearest package doc in the same change. Keep examples executable against the current public API and prefer exact file paths over conceptual descriptions.

## Operational Risks

- Full test runs are slow. A recent local verification in this workspace took roughly 25 minutes after the default test heap was set to `2g`; treat that as an environment-specific observation, not a stable performance contract.
- Heap use is non-trivial because tests include compile/prepare/execute, tuning, debug benchmark, and performance regression paths.
- `src/test/java/debug` contains benchmark-style JUnit tests; they are still named `*Test.java` and are included by the default Gradle test task unless filtered.
- Native acceleration is optional in many tests. Tests that require OpenBLAS, Metal, or CUDA often use JUnit assumptions, but configured profiles can still fail if they require unavailable native libraries.
- Tuning writes and reads artifacts under `profiles/` and `build/`; stale best-profile files can make benchmark tests fail or measure the wrong candidate.
- Performance regression checks write reports under `build/tuning-etalon-regression` and compare against `src/test/resources/tuning/etalon/inference-performance-baseline.properties`.
- `verifySourceTreeClean` fails if generated artifacts appear under `src/` or `test/`.
- Shape/layout bugs often surface late in backend tests because compile-time descriptors may be valid while prepared metadata or strided execution is wrong.
