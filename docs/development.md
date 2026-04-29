<!-- generated-by: gsd-doc-writer -->
# Development

Navigation: [Index](index.md) | [Architecture](architecture.md) | [Modules](modules.md) | [Metal Backend](metal-backend.md) | [Testing](testing.md) | [Configuration](configuration.md) | [Troubleshooting](troubleshooting.md)

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
| `src/main/java/graph` | Compilation, optimizer state, prepared graph artifacts, execution trace metadata |
| `src/main/java/graph/optimizer` | Ordered optimizer stages: `AR`, `CSE`, `PART`, `FUSE`, `MEM` |
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
| `scripts/build-metal-mps-shim.sh` | Builds `build/native/apple/libsynaptik_apple_mps.dylib` on macOS |
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
- Backend kernel selection for CPU is centralized in `src/main/java/backend/cpu/registry/CpuKernelResolver.java`.
- CPU preparation and workspace decisions live in `src/main/java/backend/cpu/prepare/CpuNodePreparer.java`.
- Runtime threshold interpretation lives in CPU planning classes such as `src/main/java/backend/cpu/kernels/plan/CpuExecutionPlanner.java`.
- Optimizer stage wiring lives in `src/main/java/graph/optimizer/OptimizerFactory.java`.
- Graph autotune candidates must stay graph-policy-only; `SourceTreeHygieneTest.graphAutotuneCandidatePackageDoesNotImportRuntimeOrBackendConfig` rejects runtime/backend config imports from `src/main/java/tuning/candidate/graph`.

Source hygiene tests also reject legacy package paths such as `graph.fused`, `graph.codegen`, `graph.optimizer.fusion`, `operations.fused`, `backend/kernels/cpu`, `backend/kernels/cuda`, and `backend/kernels/opencl`.

## Adding Tensor Ops

Use an existing family as the template. For a binary elementwise op, compare:

- Descriptor: `src/main/java/operations/elementwise/binary/add.java`
- Builder/backward: `src/main/java/tensor/ops/binary/TensorBinaryOps.java`
- Public static facade: `src/main/java/tensor/TensorOps.java`
- Instance method facade: `src/main/java/tensor/Tensor.java`
- CPU kernel: `src/main/java/backend/cpu/kernels/elementwise/binary/CpuAddKernel.java`
- CPU registry entry: `src/main/java/backend/cpu/registry/CpuKernelResolver.java`
- Coverage: `src/test/java/AllOpsTest.java`, family-specific execution tests, and dtype/broadcast tests when applicable

Checklist for a new primitive:

1. Add an `Operation.OpType` enum value in `src/main/java/operations/Operation.java`.
2. Add an immutable descriptor class under the correct `src/main/java/operations/...` family.
3. Add or extend the matching `src/main/java/tensor/ops/...` builder.
4. Use `TensorPrimitiveBuilder.unary`, `binary`, `ternary`, `nary`, or view helpers instead of directly mutating tensor internals.
5. Attach backward logic with `TensorInternalAccess.setBackwardFunction` when the op participates in autograd.
6. Add a static wrapper in `src/main/java/tensor/TensorOps.java`.
7. Add an instance wrapper in `src/main/java/tensor/Tensor.java` if the op is part of the user-facing fluent API.
8. Add CPU runtime support and register it in `CpuKernelResolver` unless the op is compile-only or descriptor-only.
9. Add tests for forward values, gradients, dtype handling, shape validation, and optimizer interaction if a rewrite can see the op.

Broadcasting should use the existing planners. Binary ops call `TensorBroadcastOps.planBinary(...)`, which delegates to `BroadcastPlanner` and throws `IllegalArgumentException("Broadcast mismatch at dim ...")` when aligned dimensions are incompatible.

## Adding Backend Kernels

The CPU backend is the complete execution backend. New CPU kernels should live under `src/main/java/backend/cpu/kernels/<family>/`, not under old root paths.

CPU kernel checklist:

1. Implement `src/main/java/backend/cpu/kernels/CpuKernel.java`.
2. Override the supported dtype entry points: `forwardF64`, `forwardF32`, `forwardBF16`, `forwardBOOL`, or `forwardI32`.
3. Use the existing family executor where possible, such as `ElementwiseBinaryExecutor`, `ElementwiseUnaryExecutor`, reduction executors, matmul executables, or conv/pool executors.
4. Read prepared metadata from `CpuKernelContext`, not from ad hoc policy logic inside the hot loop.
5. Add a singleton field and switch case in `src/main/java/backend/cpu/registry/CpuKernelResolver.java`.
6. If the op needs workspace or special prepared metadata, update `src/main/java/backend/cpu/prepare/CpuNodePreparer.java` and the relevant planner under `src/main/java/backend/cpu/kernels/.../plan`.
7. Add execution tests that call `CompiledGraph.compile(...).execute(...)` rather than only testing helper methods.

For elementwise kernels, use `CpuAddKernel` as the reference shape: scalar application methods, vector support methods, and direct F64/F32/BF16 implementations live together while dispatch is handled by the shared executor and planner.

For native or accelerator-adjacent paths:

- OpenBLAS FFM lookup checks `-Dopenblas.lib=<path>`, then `OPENBLAS_LIB`, then library name `openblas`.
- Metal MPS lookup checks `-Dsynaptik.metal.mps.lib=<path>`, then `SYNAPTIK_METAL_MPS_LIB`, then library name `synaptik_apple_mps`.
- CUDA lookup checks `-Dsynaptik.cuda.graph.lib=<path>`, then `SYNAPTIK_CUDA_GRAPH_LIB`, then library name `synaptik_cuda_graph`.
- macOS Metal shim build commands:

```bash
./gradlew buildMetalMpsShim
./gradlew nativeBuild
./gradlew metalTest
```

`buildMetalMpsShim` is the low-level task that calls `scripts/build-metal-mps-shim.sh` and writes `build/native/apple/libsynaptik_apple_mps.dylib`. `nativeBuild` is the user-facing optional-native lifecycle task. `metalTest` builds the shim, sets `-Dsynaptik.metal.mps.lib` to the freshly built dylib, and runs only Metal/MPS-focused tests.

Default Java lifecycle tasks stay portable: `classes`, `build`, and `check` do not depend on Metal native compilation. Use `nativeBuild` or `metalTest` when a change touches `src/main/native/apple`, `src/main/java/backend/metal`, or Metal partition/lowering behavior. The native ABI and Objective-C call path are documented in [Metal Backend](metal-backend.md).

## Adding Optimizer Rules

Optimizer stages are defined in `src/main/java/config/optimizer/OptimizerStage.java`:

```text
AR, CSE, PART, FUSE, MEM
```

`src/main/java/config/optimizer/OptimizerConfig.java` validates the ordering:

- `FUSE` requires `PART`
- `PART` must run before `FUSE`
- `MEM` requires `FUSE`
- duplicate stages are rejected

Current source defaults:

- `OptimizerConfig.noOptimization()` uses no stages.
- `OptimizerConfig.trainingDefaults()` uses `AR, CSE, PART, FUSE, MEM`.
- `OptimizerConfig.inferenceDefaults()` uses `AR, CSE, PART, FUSE, MEM`.

Add changes by stage ownership:

| Change | Target path |
|---|---|
| Algebraic identity or lowering | `src/main/java/graph/optimizer/rewrite` |
| Common subexpression behavior | `src/main/java/graph/optimizer/cse/CommonSubexpressionEliminationRule.java` |
| Backend partition intent | `src/main/java/graph/optimizer/partition` |
| Region/fused execution units | `src/main/java/graph/optimizer/region` and CPU-specific fused policy under `src/main/java/backend/cpu/fused` |
| Memory reuse or binding policy | `src/main/java/graph/optimizer/memory` |

`OptimizerFactory.createRule(...)` maps public stages to concrete rules:

- `AR` -> `new RewriteRule(config.rewrite())`
- `CSE` -> `new CommonSubexpressionEliminationRule(config.cse())`
- `PART` -> `new PartitionIntentRule(config.partition())`
- `FUSE` -> `new RegionOptimizationRule(config.fuse())`
- `MEM` -> `new MemoryOptimizerRule(MemoryPlannerPolicy.fromConfig(config.memory()))`

When a new operation has semantic parameters, update CSE signature handling in `CommonSubexpressionEliminationRule.parameterKey(...)`; otherwise structurally different instances may collapse incorrectly or identical instances may fail to collapse.

Use focused tests:

```bash
./gradlew test --no-daemon --tests AlgebraicRewritingPowTest
./gradlew test --no-daemon --tests CommonSubexpressionEliminationRuleTest
./gradlew test --no-daemon --tests graph.optimizer.GraphOptimizerSinglePassTest
./gradlew test --no-daemon --tests graph.optimizer.region.RegionOptimizationRuleTest
./gradlew test --no-daemon --tests graph.optimizer.memory.MemoryPlannerRegionViewTest
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
conv2d-gemm-dispatch
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
