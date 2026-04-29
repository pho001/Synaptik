# Codebase Structure

**Analysis Date:** 2026-04-29

## Directory Layout

```text
Synaptik/
├── build.gradle                    # Java/application build, JDK 25 toolchain, Vector API flags, tests, native tasks
├── settings.gradle                 # Gradle root project and toolchain resolver plugin
├── gradle.properties               # Gradle project properties
├── gradlew                         # Gradle wrapper launcher
├── config/                         # Checked-in JSON configuration examples
├── docs/                           # Generated and hand-curated architecture, API, testing, and operation guides
├── gradle/wrapper/                 # Gradle wrapper distribution metadata
├── logotypes/                      # Project image assets
├── profiles/platform/              # Checked-in platform calibration/profile artifacts
├── scripts/                        # Build helper scripts such as Metal MPS shim build
├── src/main/java/                  # Main Java source packages
├── src/main/native/apple/          # Objective-C Metal MPS native shim source
├── src/test/java/                  # JUnit Jupiter test sources
├── src/test/resources/             # Test resources
├── todo/                           # Roadmap and review notes
└── .planning/codebase/             # GSD codebase map documents
```

## Directory Purposes

**Root Build Files:**
- Purpose: Define the single Java/Gradle application project.
- Contains: `build.gradle`, `settings.gradle`, `gradle.properties`, `gradlew`, and `gradlew.bat`.
- Key files: `build.gradle` declares Java/application plugins, JDK 25 toolchain, ASM dependencies, JUnit Jupiter dependencies, Vector API JVM flags, source-tree hygiene tasks, and Metal native tasks.

**Documentation:**
- Purpose: Explain architecture, APIs, compute flow, optimizer behavior, configuration, testing, and development workflows.
- Contains: `docs/index.md`, `docs/architecture.md`, `docs/compute-flow.md`, `docs/modules.md`, `docs/development.md`, `docs/testing.md`, `docs/tensor-api.md`, and `docs/adding-tensor-operation.md`.
- Key files: `README.md` provides the high-level reading guide; `docs/development.md` provides practical source placement and contribution guidance.

**Main Java Source:**
- Purpose: Production code for tensor graphs, graph compilation, backend execution, configuration, tuning, numerics, and application entry points.
- Contains: 962 Java files under `src/main/java`.
- Key files: `src/main/java/tensor/Tensor.java`, `src/main/java/graph/CompiledGraph.java`, `src/main/java/backend/ComputeEngine.java`, `src/main/java/backend/cpu/CpuBackend.java`, `src/main/java/config/profile/ExecutionProfile.java`, and `src/main/java/synaptik/app/TuningCli.java`.

**Tensor Package:**
- Purpose: Public tensor API, graph-building helpers, dtype/storage/layout support, and compute convenience bridge.
- Contains: `src/main/java/tensor/Tensor.java`, `src/main/java/tensor/TensorOps.java`, `src/main/java/tensor/TensorExecutionSupport.java`, `src/main/java/tensor/TensorPrimitiveBuilder.java`, storage classes such as `src/main/java/tensor/Float64Storage.java`, option records under `src/main/java/tensor/options`, and family builders under `src/main/java/tensor/ops`.
- Key files: `src/main/java/tensor/README.md`, `src/main/java/tensor/API.md`, `src/main/java/tensor/ops/binary/TensorBinaryOps.java`, `src/main/java/tensor/ops/reduction/TensorReduceOps.java`, and `src/main/java/tensor/ops/linalg/TensorMatMulOps.java`.

**Tensor Operation Builders:**
- Purpose: Keep public operation-family validation, graph construction, and backward formulas out of `Tensor.java`.
- Contains: `src/main/java/tensor/ops/unary`, `src/main/java/tensor/ops/binary`, `src/main/java/tensor/ops/compare`, `src/main/java/tensor/ops/bool`, `src/main/java/tensor/ops/select`, `src/main/java/tensor/ops/layout`, `src/main/java/tensor/ops/index`, `src/main/java/tensor/ops/reduction`, `src/main/java/tensor/ops/linalg`, `src/main/java/tensor/ops/conv`, `src/main/java/tensor/ops/pool`, `src/main/java/tensor/ops/normalization`, and `src/main/java/tensor/ops/loss`.
- Key files: Use `Tensor<Family>Ops.java` for public family builders and `<Family>Support.java` for local helpers, for example `src/main/java/tensor/ops/conv/TensorConvOps.java` and `src/main/java/tensor/ops/conv/ConvSupport.java`.

**Operations Package:**
- Purpose: Immutable primitive descriptors implementing the `operations.Operation` contract.
- Contains: `src/main/java/operations/Operation.java` and descriptor families under `src/main/java/operations/elementwise`, `src/main/java/operations/reduction`, `src/main/java/operations/layout`, `src/main/java/operations/index`, `src/main/java/operations/linalg`, `src/main/java/operations/loss`, `src/main/java/operations/nn`, and `src/main/java/operations/normalization`.
- Key files: `src/main/java/operations/README.md`, `src/main/java/operations/elementwise/binary/add.java`, `src/main/java/operations/linalg/matmul.java`, `src/main/java/operations/loss/crossEntropyLossIndices.java`, and `src/main/java/operations/nn/conv/conv2d.java`.

**Graph Package:**
- Purpose: Compile facade, compiled node snapshots, compile artifacts, execution artifacts, and traces.
- Contains: `src/main/java/graph/CompiledGraph.java`, `src/main/java/graph/CompiledNode.java`, `src/main/java/graph/CompiledGradientBinding.java`, `src/main/java/graph/SemanticForwardCanonicalizer.java`, compile support under `src/main/java/graph/compile`, execution support under `src/main/java/graph/execution`, and optimizer code under `src/main/java/graph/optimizer`.
- Key files: `src/main/java/graph/README.md`, `src/main/java/graph/compile/GraphCompiler.java`, `src/main/java/graph/compile/CompileArtifacts.java`, `src/main/java/graph/execution/PreparedExecution.java`, and `src/main/java/graph/execution/CompiledNodeExecutionMetadata.java`.

**Graph Optimizer Package:**
- Purpose: Ordered optimizer stages and their shared state.
- Contains: `src/main/java/graph/optimizer/rewrite`, `src/main/java/graph/optimizer/cse`, `src/main/java/graph/optimizer/partition`, `src/main/java/graph/optimizer/region`, `src/main/java/graph/optimizer/memory`, `src/main/java/graph/optimizer/intent`, and `src/main/java/graph/optimizer/state`.
- Key files: `src/main/java/graph/optimizer/OptimizerFactory.java`, `src/main/java/graph/optimizer/GraphOptimizer.java`, `src/main/java/graph/optimizer/rewrite/RewriteRule.java`, `src/main/java/graph/optimizer/cse/CommonSubexpressionEliminationRule.java`, `src/main/java/graph/optimizer/partition/PartitionIntentRule.java`, `src/main/java/graph/optimizer/region/RegionOptimizationRule.java`, and `src/main/java/graph/optimizer/memory/MemoryOptimizerRule.java`.

**Backend Root Package:**
- Purpose: Backend-neutral facade and dispatch contracts.
- Contains: `src/main/java/backend/ApproxMode.java`, `src/main/java/backend/ComputeBackend.java`, `src/main/java/backend/ComputeEngine.java`, plus subpackages for prepare, lowering, partition, selection, runtime, memory, BLAS, CPU, Metal, CUDA, OpenCL, and shared accelerator records.
- Key files: `src/main/java/backend/README.md`, `src/main/java/backend/ComputeEngine.java`, `src/main/java/backend/prepare/PreparedExecutionBuilder.java`, `src/main/java/backend/select/DefaultBackendSelectionPolicy.java`, and `src/main/java/backend/lowering/LoweringPipeline.java`.

**CPU Backend Package:**
- Purpose: Complete CPU backend implementation, preparation, lowering, partition legality, kernel registry, kernel families, plans, workspaces, fused execution, and fused code generation.
- Contains: `src/main/java/backend/cpu/CpuBackend.java`, `src/main/java/backend/cpu/prepare`, `src/main/java/backend/cpu/registry`, `src/main/java/backend/cpu/kernels`, `src/main/java/backend/cpu/fused`, `src/main/java/backend/cpu/lowering`, `src/main/java/backend/cpu/partition`, and `src/main/java/backend/cpu/plan`.
- Key files: `src/main/java/backend/cpu/README.md`, `src/main/java/backend/cpu/prepare/CpuNodePreparer.java`, `src/main/java/backend/cpu/registry/CpuKernelResolver.java`, `src/main/java/backend/cpu/kernels/plan/CpuExecutionPlanner.java`, and `src/main/java/backend/cpu/fused/exec/FusedExecutionBackendResolver.java`.

**CPU Kernel Families:**
- Purpose: Dtype-aware runtime kernels and prepared execution planners.
- Contains: `src/main/java/backend/cpu/kernels/elementwise`, `src/main/java/backend/cpu/kernels/reduction`, `src/main/java/backend/cpu/kernels/layout`, `src/main/java/backend/cpu/kernels/index`, `src/main/java/backend/cpu/kernels/linalg`, `src/main/java/backend/cpu/kernels/nn`, `src/main/java/backend/cpu/kernels/grad`, `src/main/java/backend/cpu/kernels/fused`, and `src/main/java/backend/cpu/kernels/plan`.
- Key files: `src/main/java/backend/cpu/kernels/CpuKernel.java`, `src/main/java/backend/cpu/kernels/CpuKernelContext.java`, `src/main/java/backend/cpu/kernels/CpuNodeExecutionPlan.java`, `src/main/java/backend/cpu/kernels/elementwise/binary/CpuAddKernel.java`, `src/main/java/backend/cpu/kernels/reduction/CpuSumKernel.java`, and `src/main/java/backend/cpu/kernels/linalg/CpuMatMulKernel.java`.

**Accelerator Backend Packages:**
- Purpose: Shared accelerator DAG/executable/buffer support and backend-specific Metal/CUDA/OpenCL implementations.
- Contains: `src/main/java/backend/accelerator`, `src/main/java/backend/metal`, `src/main/java/backend/cuda`, and `src/main/java/backend/opencl`.
- Key files: `src/main/java/backend/accelerator/exec/PreparedAcceleratorExecutable.java`, `src/main/java/backend/accelerator/dag/AcceleratorDagSpec.java`, `src/main/java/backend/metal/MetalBackend.java`, `src/main/java/backend/metal/prepare/MetalNodePreparer.java`, `src/main/java/backend/metal/buffer/MetalBufferBinding.java`, `src/main/java/backend/cuda/CudaGpuBackend.java`, and `src/main/java/backend/opencl/OpenClBackend.java`.

**Configuration Packages:**
- Purpose: Immutable optimizer/runtime/backend/profile records that drive compile, prepare, execution, and tuning.
- Contains: `src/main/java/config/optimizer`, `src/main/java/config/runtime`, `src/main/java/config/backend`, and `src/main/java/config/profile`.
- Key files: `src/main/java/config/optimizer/OptimizerConfig.java`, `src/main/java/config/optimizer/OptimizerStage.java`, `src/main/java/config/runtime/RuntimeConfig.java`, `src/main/java/config/backend/CpuKernelConfig.java`, `src/main/java/config/profile/ExecutionProfile.java`, and `src/main/java/config/profile/PlatformRuntimeProfile.java`.

**Tuning Package:**
- Purpose: Benchmark, calibration, graph autotune, measurement, search, validation, reports, persistence, workload definitions, and fluent APIs.
- Contains: `src/main/java/tuning/api`, `src/main/java/tuning/autotune`, `src/main/java/tuning/benchmark`, `src/main/java/tuning/calibration`, `src/main/java/tuning/candidate`, `src/main/java/tuning/etalon`, `src/main/java/tuning/measure`, `src/main/java/tuning/preset`, `src/main/java/tuning/reporting`, `src/main/java/tuning/search`, `src/main/java/tuning/store`, `src/main/java/tuning/trace`, `src/main/java/tuning/validate`, and `src/main/java/tuning/workload`.
- Key files: `src/main/java/tuning/README.md`, `src/main/java/tuning/api/Synaptik.java`, `src/main/java/tuning/autotune/AutotuneSession.java`, `src/main/java/tuning/benchmark/BenchmarkSession.java`, `src/main/java/tuning/calibration/run/CalibrationRunner.java`, `src/main/java/tuning/workload/StandardWorkloads.java`, and `src/main/java/tuning/store/JsonFileBestProfileStore.java`.

**Application Package:**
- Purpose: CLI and programmatic examples for calibration, autotune, benchmark, and reporting flows.
- Contains: `src/main/java/synaptik/app/TuningCli.java` and `src/main/java/synaptik/app/Main.java`.
- Key files: `src/main/java/synaptik/app/TuningCli.java` is the Gradle application main class configured by `build.gradle`.

**Numerics Package:**
- Purpose: Numerical drift comparison harness and CLI.
- Contains: `src/main/java/numerics/NumericsCli.java`, `src/main/java/numerics/NumericsHarness.java`, `src/main/java/numerics/NumericsGraphFactory.java`, `src/main/java/numerics/NumericsMetrics.java`, `src/main/java/numerics/NumericsPolicy.java`, and `src/main/java/numerics/NumericsReport.java`.
- Key files: `src/main/java/numerics/NumericsCli.java`.

**Utilities Package:**
- Purpose: Small support classes for specialized execution paths.
- Contains: `src/main/java/utils`.
- Key files: Use existing utility placement in `src/main/java/utils` for shared, backend-agnostic helpers only.

**Native Source:**
- Purpose: Optional macOS Metal MPS shim source.
- Contains: `src/main/native/apple/synaptik_apple_mps_stub.m`.
- Key files: `scripts/build-metal-mps-shim.sh` builds `build/native/apple/libsynaptik_apple_mps.dylib`.

**Tests:**
- Purpose: JUnit Jupiter test coverage for tensor API, execution, optimizer, backend, config, tuning, and package hygiene.
- Contains: 179 Java files under `src/test/java` plus resources under `src/test/resources`.
- Key files: `src/test/java/SourceTreeHygieneTest.java`, `src/test/java/TensorComputeConvenienceApiTest.java`, `src/test/java/PreparedExecutionBuildTest.java`, `src/test/java/AllOpsTest.java`, `src/test/java/backend/ComputeBackendTest.java`, and package-specific tests under `src/test/java/backend`, `src/test/java/graph`, `src/test/java/config`, and `src/test/java/tuning`.

**Planning And Roadmap Artifacts:**
- Purpose: GSD planning state, codebase maps, and roadmap/review notes.
- Contains: `.planning/codebase`, `.planning/tmp`, and `todo`.
- Key files: `.planning/codebase/ARCHITECTURE.md`, `.planning/codebase/STRUCTURE.md`, and roadmap notes such as `todo/44-cpu-execution-regions-fusion-and-graph-autotune.md`.

## Key File Locations

**Entry Points:**
- `src/main/java/synaptik/app/TuningCli.java`: Gradle application main class for calibration, autotune, benchmark, and full tuning flows.
- `src/main/java/synaptik/app/Main.java`: Programmatic tuning API example and direct Java main.
- `src/main/java/numerics/NumericsCli.java`: Numerics comparison CLI using `numerics.*` system properties.
- `src/main/java/tensor/Tensor.java`: Public fluent API for user graph construction and convenience compute/compile/prepare calls.
- `src/main/java/tensor/TensorOps.java`: Public static operation facade.
- `src/main/java/graph/CompiledGraph.java`: Compile and prepare facade.
- `src/main/java/graph/execution/PreparedExecution.java`: Runtime execution entry for prepared forward/backward steps.
- `src/main/java/backend/ComputeEngine.java`: Per-step backend dispatcher.

**Configuration:**
- `build.gradle`: Build, test, application, source hygiene, and native task configuration.
- `settings.gradle`: Root project name `Synaptik` and Gradle toolchain resolver plugin.
- `config/optimizer-profile.json`: Checked-in optimizer profile JSON example.
- `src/main/java/config/optimizer/OptimizerConfig.java`: Optimizer stage order and policy record.
- `src/main/java/config/runtime/RuntimeConfig.java`: Runtime/backend policy record.
- `src/main/java/config/profile/ExecutionProfile.java`: Runnable profile combining optimizer and runtime settings.
- `src/main/java/config/profile/PlatformRuntimeProfile.java`: Machine-oriented runtime default profile.

**Core Logic:**
- `src/main/java/tensor/TensorPrimitiveBuilder.java`: Central primitive tensor construction helpers.
- `src/main/java/tensor/BroadcastPlanner.java`: Binary broadcast compatibility planning.
- `src/main/java/operations/Operation.java`: Operation taxonomy and fusable flags.
- `src/main/java/graph/compile/GraphCompiler.java`: Compile session implementation.
- `src/main/java/graph/optimizer/OptimizerFactory.java`: Stage-to-rule mapping.
- `src/main/java/backend/prepare/PreparedExecutionBuilder.java`: Compile artifact to prepared execution conversion.
- `src/main/java/backend/select/DefaultBackendSelectionPolicy.java`: Backend plan filtering and decision trace.
- `src/main/java/backend/lowering/LoweringPipeline.java`: Optimized region lowering orchestration.
- `src/main/java/backend/cpu/prepare/CpuNodePreparer.java`: CPU prepare-time kernel/plan/workspace/fused metadata resolution.
- `src/main/java/backend/cpu/registry/CpuKernelResolver.java`: CPU operation-to-kernel registry.
- `src/main/java/backend/cpu/CpuBackend.java`: CPU runtime execution.

**Testing:**
- `src/test/java/SourceTreeHygieneTest.java`: Architectural package boundary and source hygiene assertions.
- `src/test/java/LowercasePackageNamingTest.java`: Package naming convention guard.
- `src/test/java/TensorComputeConvenienceApiTest.java`: Tensor compute convenience behavior.
- `src/test/java/CompiledGraphIdempotencyTest.java`: Compile behavior coverage.
- `src/test/java/PreparedExecutionBuildTest.java`: Prepared execution build coverage.
- `src/test/java/CommonSubexpressionEliminationRuleTest.java`: CSE behavior coverage.
- `src/test/java/OptimizerFuseTest.java`: Fusion optimizer coverage.
- `src/test/java/DataTypeExecutionCoverageTest.java`: dtype execution coverage.
- `src/test/java/backend`: Backend-specific tests.
- `src/test/java/graph`: Graph and optimizer tests.
- `src/test/java/tuning`: Tuning, calibration, candidate, and integration tests.

**Documentation:**
- `README.md`: Repository overview and reading guide.
- `docs/architecture.md`: Detailed implementation architecture.
- `docs/compute-flow.md`: Compile/prepare/execute lifecycle walkthrough.
- `docs/modules.md`: Package-by-package source guide.
- `docs/development.md`: Local setup, source placement, and change checklists.
- `docs/adding-tensor-operation.md`: End-to-end checklist for adding operations.
- `docs/testing.md`: Test commands and practices.
- `src/main/java/tensor/README.md`: Tensor package ownership.
- `src/main/java/operations/README.md`: Operation descriptor ownership.
- `src/main/java/graph/README.md`: Graph lifecycle ownership.
- `src/main/java/backend/README.md`: Backend package ownership.
- `src/main/java/tuning/README.md`: Tuning package ownership.

## Naming Conventions

**Files:**
- Public classes and most implementation classes use Java `PascalCase.java`, for example `src/main/java/tensor/Tensor.java`, `src/main/java/graph/CompiledGraph.java`, and `src/main/java/backend/cpu/kernels/elementwise/binary/CpuAddKernel.java`.
- Operation descriptor classes intentionally use lower camel case filenames and class names, for example `src/main/java/operations/elementwise/binary/add.java`, `src/main/java/operations/layout/reshape.java`, and `src/main/java/operations/linalg/scaledDotProductAttention.java`.
- Tensor operation families use `Tensor<Family>Ops.java` plus optional `<Family>Support.java`, for example `src/main/java/tensor/ops/reduction/TensorReduceOps.java` and `src/main/java/tensor/ops/reduction/ReductionSupport.java`.
- CPU kernel classes use `Cpu<Operation>Kernel.java`, for example `src/main/java/backend/cpu/kernels/reduction/CpuSumKernel.java` and `src/main/java/backend/cpu/kernels/nn/CpuConv2dKernel.java`.
- Planner/config records use descriptive nouns ending in `Config`, `Policy`, `Plan`, `Hints`, `Profile`, `Request`, `Result`, `Trace`, or `Metadata`, for example `src/main/java/config/optimizer/PartitionConfig.java`, `src/main/java/backend/cpu/kernels/CpuNodeExecutionPlan.java`, and `src/main/java/graph/execution/trace/PrepareTrace.java`.
- Test classes use `*Test.java`, for example `src/test/java/MatMulTest.java` and `src/test/java/backend/ComputeBackendTest.java`.
- Package README files are named `README.md`, for example `src/main/java/backend/cpu/README.md`.

**Directories:**
- Java packages are lowercase, enforced by `src/test/java/LowercasePackageNamingTest.java`.
- Source directories follow Gradle defaults: `src/main/java`, `src/test/java`, and `src/test/resources`.
- Semantic operation families mirror across `src/main/java/tensor/ops`, `src/main/java/operations`, and `src/main/java/backend/cpu/kernels`.
- Backend-specific implementation belongs under backend roots such as `src/main/java/backend/cpu`, `src/main/java/backend/metal`, `src/main/java/backend/cuda`, and `src/main/java/backend/opencl`.
- Backend-neutral contracts belong under `src/main/java/backend/prepare`, `src/main/java/backend/lowering`, `src/main/java/backend/partition`, `src/main/java/backend/select`, `src/main/java/backend/runtime`, and `src/main/java/backend/memory`.
- Graph optimizer stages live in stage-owned packages: `src/main/java/graph/optimizer/rewrite`, `src/main/java/graph/optimizer/cse`, `src/main/java/graph/optimizer/partition`, `src/main/java/graph/optimizer/region`, and `src/main/java/graph/optimizer/memory`.
- Avoid legacy paths guarded by `src/test/java/SourceTreeHygieneTest.java`, including `src/main/java/graph/fused`, `src/main/java/graph/codegen`, `src/main/java/graph/optimizer/fusion`, `src/main/java/operations/fused`, and `src/main/java/backend/kernels/cpu`.

## Where to Add New Code

**New Public Tensor Operation:**
- Primary descriptor: Add `Operation.OpType` in `src/main/java/operations/Operation.java` and descriptor under `src/main/java/operations/<family>`.
- Tensor builder: Add or update `src/main/java/tensor/ops/<family>/Tensor<Family>Ops.java`.
- Builder helpers: Add family-local helpers under `src/main/java/tensor/ops/<family>/<Family>Support.java` when validation or shape logic is nontrivial.
- Public static facade: Add method to `src/main/java/tensor/TensorOps.java`.
- Public fluent facade: Add method to `src/main/java/tensor/Tensor.java` when part of the user-facing API.
- CPU kernel: Add kernel under `src/main/java/backend/cpu/kernels/<family>` and register it in `src/main/java/backend/cpu/registry/CpuKernelResolver.java`.
- CPU prepare/planning: Add workspace or special plan logic in `src/main/java/backend/cpu/prepare/CpuNodePreparer.java` and relevant planner under `src/main/java/backend/cpu/kernels/<family>/plan` or `src/main/java/backend/cpu/kernels/plan`.
- Optimizer/CSE: Update `src/main/java/graph/optimizer/cse/CommonSubexpressionEliminationRule.java` when descriptors carry semantic parameters.
- Tests: Add value, dtype, shape validation, gradient, optimizer, and backend execution coverage under `src/test/java`.

**New Optimizer Rewrite Or Stage Behavior:**
- Algebraic identity or semantic lowering: Add under `src/main/java/graph/optimizer/rewrite`.
- CSE signature behavior: Update `src/main/java/graph/optimizer/cse/CommonSubexpressionEliminationRule.java`.
- Backend partition intent: Add under `src/main/java/graph/optimizer/partition`.
- Region/fused unit policy: Add under `src/main/java/graph/optimizer/region`; CPU fused policy belongs under `src/main/java/backend/cpu/fused`.
- Memory reuse or binding behavior: Add under `src/main/java/graph/optimizer/memory`.
- Stage wiring: Update `src/main/java/graph/optimizer/OptimizerFactory.java` and stage config in `src/main/java/config/optimizer`.
- Tests: Add focused tests under `src/test/java/graph/optimizer` or root-level optimizer tests.

**New CPU Backend Kernel Or Runtime Plan:**
- Kernel implementation: Add under `src/main/java/backend/cpu/kernels/<family>`.
- Shared family executor: Prefer existing executors such as `src/main/java/backend/cpu/kernels/elementwise/binary/ElementwiseBinaryExecutor.java`, `src/main/java/backend/cpu/kernels/reduction/SumLikeReductionExecutor.java`, or linalg/nn executors.
- Registry: Register in `src/main/java/backend/cpu/registry/CpuKernelResolver.java`.
- Prepare-time plan: Add plan fields or logic under `src/main/java/backend/cpu/kernels/plan` or family `plan` subpackage.
- Workspace/fused metadata: Add logic in `src/main/java/backend/cpu/prepare/CpuNodePreparer.java`.
- Tests: Add execution tests under `src/test/java` or `src/test/java/backend/cpu`.

**New Backend-Agnostic Prepare, Lowering, Or Selection Contract:**
- Prepare orchestration: Add to `src/main/java/backend/prepare` only if backend-neutral.
- Backend-specific preparer: Add to `src/main/java/backend/<target>/prepare`.
- Lowering contract: Add to `src/main/java/backend/lowering`.
- Backend-specific lowerer: Add to `src/main/java/backend/<target>/lowering`.
- Partition descriptor wiring: Add to `src/main/java/backend/partition/BackendPartitionDescriptorRegistry.java`.
- Backend selection policy: Add to `src/main/java/backend/select`.
- Runtime state access: Add to `src/main/java/backend/runtime` or `src/main/java/backend/memory`.
- Tests: Update `src/test/java/SourceTreeHygieneTest.java` only when an intentional architecture boundary changes.

**New Accelerator Feature:**
- Shared DAG or executable contract: Add under `src/main/java/backend/accelerator/dag` or `src/main/java/backend/accelerator/exec`.
- Shared accelerator preparation helpers: Add under `src/main/java/backend/accelerator/prepare`.
- Shared buffer policy: Add under `src/main/java/backend/accelerator/buffer`.
- Metal-specific code: Add under `src/main/java/backend/metal`.
- CUDA-specific code: Add under `src/main/java/backend/cuda`.
- OpenCL-specific code: Add under `src/main/java/backend/opencl`.
- Native Metal shim: Add Objective-C code under `src/main/native/apple` and script support under `scripts`.
- Tests: Add backend tests under `src/test/java/backend/metal`, `src/test/java/backend/cuda`, or `src/test/java/backend/lowering`.

**New Runtime Or Optimizer Config Knob:**
- Optimizer graph policy: Add under `src/main/java/config/optimizer`.
- CPU/runtime backend policy: Add under `src/main/java/config/backend` or `src/main/java/config/runtime`.
- Persisted profile representation: Add under `src/main/java/config/profile`.
- Calibration candidate/value generation: Add under `src/main/java/tuning/calibration` and `src/main/java/tuning/candidate`.
- Profile assembly or persistence: Add under `src/main/java/config/profile`, `src/main/java/tuning/store`, or `src/main/java/tuning/calibration/store`.
- Tests: Add config IO and tuning tests under `src/test/java/config` and `src/test/java/tuning`.

**New Tuning Workflow:**
- Public fluent API: Add under `src/main/java/tuning/api`.
- Autotune session/request/report: Add under `src/main/java/tuning/autotune`.
- Benchmark session/request/report: Add under `src/main/java/tuning/benchmark`.
- Calibration family/run/store: Add under `src/main/java/tuning/calibration`.
- Candidate spaces: Add under `src/main/java/tuning/candidate`.
- Workloads: Add under `src/main/java/tuning/workload`.
- CLI wiring: Add command parsing or orchestration in `src/main/java/synaptik/app/TuningCli.java`.
- Tests: Add coverage under `src/test/java/tuning` and CLI parser tests under `src/test/java/synaptik/app`.

**New Documentation:**
- User-facing overview or guide: Add under `docs`.
- Package-specific ownership notes: Add or update `README.md` inside the package, for example `src/main/java/backend/<target>/README.md`.
- Tensor API details: Update `docs/tensor-api.md` and `src/main/java/tensor/API.md`.
- Architecture or compute lifecycle: Update `docs/architecture.md`, `docs/compute-flow.md`, and `docs/modules.md`.

**Utilities:**
- Shared backend-agnostic helpers: Add under `src/main/java/utils` only when not owned by a specific layer.
- Tensor-specific helpers: Add under `src/main/java/tensor`.
- CPU-specific helpers: Add under `src/main/java/backend/cpu`.
- Tuning-specific helpers: Add under `src/main/java/tuning`.

## Special Directories

**`.planning/codebase`:**
- Purpose: GSD codebase intelligence documents.
- Generated: Yes.
- Committed: Yes when orchestrator commits planning artifacts.

**`.planning/tmp`:**
- Purpose: Temporary verification and manifest artifacts for planning/documentation workflows.
- Generated: Yes.
- Committed: Project-specific; treat as planning workflow output.

**`build`:**
- Purpose: Gradle build outputs, compiled classes, generated native artifacts, and tensor autotune cache paths.
- Generated: Yes.
- Committed: No.

**`.gradle` and `.gradle-userhome`:**
- Purpose: Gradle local caches and wrapper/user-home state.
- Generated: Yes.
- Committed: No.

**`out`:**
- Purpose: IDE output directory.
- Generated: Yes.
- Committed: No.

**`profiles/platform`:**
- Purpose: Platform calibration/report artifacts consumed by tuning workflows.
- Generated: Yes.
- Committed: Yes in this repository for known platform profiles.

**`src/main/native/apple`:**
- Purpose: Native Objective-C source for optional Metal MPS bridge.
- Generated: No.
- Committed: Yes.

**`scripts`:**
- Purpose: Repository helper scripts such as `scripts/build-metal-mps-shim.sh`.
- Generated: No.
- Committed: Yes.

**`logotypes`:**
- Purpose: Project image assets.
- Generated: No.
- Committed: Yes.

**`todo`:**
- Purpose: Roadmap, review, and architecture cleanup notes.
- Generated: No.
- Committed: Yes.

**`.idea` and `.vscode`:**
- Purpose: IDE project metadata.
- Generated: Yes.
- Committed: Partially present in the repository.

**`src/test/resources`:**
- Purpose: JUnit test resources.
- Generated: No.
- Committed: Yes.

**Root `.class` Files:**
- Purpose: Not part of source architecture; compiled artifacts are present at repository root.
- Generated: Yes.
- Committed: Should not be used as source placement; source hygiene tasks in `build.gradle` only scan `src` and `test`.

---

*Structure analysis: 2026-04-29*
