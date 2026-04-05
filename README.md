# Synaptik

Synaptik is a Java tensor and autodiff framework built around explicit compiled-graph execution, optimizer-driven graph rewrites, prepared runtime kernel dispatch, and generated fused kernels. The project combines a tensor runtime, an explicit compiled/prepared execution pipeline, backend-specific kernel dispatch, and a growing operation surface that now includes typed `BOOL` tensors, compare/select operations, and logical bool ops.

## Highlights

- Tensor runtime with explicit shape/stride metadata (`TensorMetadata`)
- Reverse-mode autodiff for a growing set of tensor operations
- Optimizer pipeline with pluggable rewrite and fusion rules
- Runtime fused-kernel generation for element-wise subgraphs
- First-class `BOOL` tensor dtype with compare/select and logical bool ops
- Backend abstraction with CPU kernels and CUDA/OpenCL scaffolding
- Tuning, persisted execution profiles, and numerics diagnostics
- Regression and operation-level test coverage

## Requirements

- JDK 25
- Gradle 9.4.1 compatible environment, or the included Gradle Wrapper
- macOS, Linux, or Windows

Vector API note:

- The project uses `jdk.incubator.vector`
- Gradle adds `--add-modules=jdk.incubator.vector` to compile, test, and runtime tasks

## Build and Run

Using the Gradle Wrapper:

- Run the demo app: `./gradlew run`
- Build the project: `./gradlew build`
- Run tests: `./gradlew test`
- Compile classes for manual entry-point execution: `./gradlew classes`

On Windows, use [`gradlew.bat`](gradlew.bat) instead of [`gradlew`](gradlew).

Alternative main classes can be started from compiled classes, for example:

- default app message entry point: `synaptik.app.Main`
- numerics CLI: `numerics.NumericsCli`

Dependency note:

- the Gradle Wrapper automatically downloads the configured Gradle distribution and declared dependencies from Maven Central, including ASM

## Project Structure

- [`src/main/java/tensor/`](src/main/java/tensor)
  - Core tensor implementation, storage, shape/stride logic, execution state, and autodiff plumbing
  - Module documentation: [`src/main/java/tensor/README.md`](src/main/java/tensor/README.md)
- [`src/main/java/operations/`](src/main/java/operations)
  - Primitive tensor operations such as add, sub, mul, div, pow, exp, log, tanh, relu, sigmoid, contiguous, sum, and newer unary/scalar helpers
  - Module documentation: [`src/main/java/operations/README.md`](src/main/java/operations/README.md)
- [`src/main/java/backend/`](src/main/java/backend)
  - Backend execution layer and per-platform dispatch integration
  - Module documentation: [`src/main/java/backend/README.md`](src/main/java/backend/README.md)
- [`src/main/java/backend/kernels/`](src/main/java/backend/kernels)
  - Backend kernel interfaces and concrete CPU kernel implementations
  - Includes dedicated CPU reduction pipeline in [`src/main/java/backend/kernels/cpu/reduction/`](src/main/java/backend/kernels/cpu/reduction)
- [`src/main/java/backend/registry/`](src/main/java/backend/registry)
  - Operation-to-kernel registries used by CPU, CUDA, and OpenCL backends
- [`src/main/java/graph/`](src/main/java/graph)
  - Compiled graph execution, runtime preparation, and graph-level orchestration
  - Module documentation: [`src/main/java/graph/README.md`](src/main/java/graph/README.md)
- [`src/main/java/graph/optimizer/`](src/main/java/graph/optimizer)
  - Optimizer entry points, factory wiring, rule composition, and optimizer documentation
  - Module documentation: [`src/main/java/graph/optimizer/README.md`](src/main/java/graph/optimizer/README.md)
- [`src/main/java/graph/codegen/`](src/main/java/graph/codegen)
  - Runtime fused code generation for specialized fused operations
- [`src/main/java/tuning/`](src/main/java/tuning)
  - Benchmark, autotune, validation, search, reporting, and persistence surface
  - Module documentation: [`src/main/java/tuning/README.md`](src/main/java/tuning/README.md)
- [`src/main/java/numerics/`](src/main/java/numerics)
  - Standalone numerics A/B harness for stability diagnostics
  - Module documentation: [`src/main/java/numerics/README.md`](src/main/java/numerics/README.md)
- [`src/main/java/config/`](src/main/java/config)
  - Backend and optimizer tuning configuration objects
- [`config/`](config)
  - Persisted execution-profile and tuning data
- [`src/test/java/`](src/test/java)
  - Regression and functional tests

## Core Architecture

Detailed per-module docs:

- Tensor: [`src/main/java/tensor/README.md`](src/main/java/tensor/README.md)
- Operations: [`src/main/java/operations/README.md`](src/main/java/operations/README.md)
- Backend: [`src/main/java/backend/README.md`](src/main/java/backend/README.md)
- Graph: [`src/main/java/graph/README.md`](src/main/java/graph/README.md)
- Optimizer: [`src/main/java/graph/optimizer/README.md`](src/main/java/graph/optimizer/README.md)
- Tuning: [`src/main/java/tuning/README.md`](src/main/java/tuning/README.md)
- Numerics: [`src/main/java/numerics/README.md`](src/main/java/numerics/README.md)

### Tensor Runtime

[`Tensor`](src/main/java/tensor/Tensor.java) is the central runtime object. It carries:

- graph-node state (`operation`, `prevTensors`)
- tensor values for runtime execution
- shape/stride/label/requires-grad metadata in [`TensorMetadata`](src/main/java/tensor/TensorMetadata.java)
- gradient storage and backward propagation helpers
- convenience execution entry points layered over explicit graph artifacts

Operations are represented through the [`Operation`](src/main/java/operations/Operation.java) abstraction and a shared op-type enum. Per-op graph-building and gradient wiring live in tensor helper classes, while backend kernels dispatch from `opType()`.

### Backend Model

The backend layer separates graph execution from device-specific kernels.

- [`CPUBackend`](src/main/java/backend/CPUBackend.java) resolves operation kernels from [`CpuKernelRegistry`](src/main/java/backend/registry/CpuKernelRegistry.java)
- [`CudaBackend`](src/main/java/backend/CudaBackend.java) and [`OpenClBackend`](src/main/java/backend/OpenClBackend.java) follow the same registry-oriented structure
- CPU kernels under [`src/main/java/backend/kernels/cpu/`](src/main/java/backend/kernels/cpu) provide concrete implementations for the currently supported ops
- CUDA/OpenCL kernel packages are currently scaffolding for future implementations

This makes it easier to extend support for new operations without embedding all execution logic directly inside backend classes.

### CPU Dispatch and Execution Modes

CPU execution supports mode-based dispatch for both element-wise and reduction operations:

- `SCALAR`
- `VECTOR` (Vector API via `jdk.incubator.vector`)
- `PARALLEL`
- `PARALLEL_VECTOR`

Dispatch thresholds and parallel chunking behavior are configured through:

- [`CpuKernelConfig`](src/main/java/config/backend/CpuKernelConfig.java)
- [`CpuExecutionPlanner`](src/main/java/backend/kernels/cpu/CpuExecutionPlanner.java)

Reduction (`sum`) also supports configurable numerical-accuracy modes:

- `FAST` (default)
- `KAHAN`
- `NEUMAIER`

Compiled graphs pre-resolve backend and CPU kernels per node to reduce runtime dispatch overhead:

- [`CompiledGraph`](src/main/java/graph/CompiledGraph.java)
- [`PreparedExecution`](src/main/java/graph/execution/PreparedExecution.java)
- [`CPUBackend`](src/main/java/backend/CPUBackend.java)

Parallel CPU execution uses a dedicated pool helper instead of `IntStream.parallel()`:

- [`CpuThreadPool`](src/main/java/backend/kernels/cpu/CpuThreadPool.java)

Non-contiguous execution uses hybrid routing:

- element-wise ops: strided path for small tensors, materialize-to-contiguous for larger tensors
- `sum` reduction: own strided/materialize strategy inside the reduction pipeline

### Optimizer Pipeline

The optimizer was reorganized into a dedicated module rooted at [`GraphOptimizer`](src/main/java/graph/optimizer/GraphOptimizer.java) and built by [`OptimizerFactory`](src/main/java/graph/optimizer/OptimizerFactory.java).

Current rule set includes files such as:

- [`FuseElementWiseRule`](src/main/java/graph/optimizer/rules/FuseElementWiseRule.java)
- [`AlgebraicRewritingRule`](src/main/java/graph/optimizer/rules/AlgebraicRewritingRule.java)
- [`CommonSubexpressionEliminationRule`](src/main/java/graph/optimizer/rules/CommonSubexpressionEliminationRule.java)
- [`MemoryOptimizerRule`](src/main/java/graph/optimizer/rules/MemoryOptimizerRule.java)

The memory optimizer is now backed by an explicit liveness/slot planner with explain and summary metrics rather than a simple ad hoc reuse pool.

Common subexpression elimination now uses structural signatures keyed by `Operation.OpType`, phase, input structure, and explicit operation parameters, which avoids incorrect merges across shape/layout transforms and reduction variants such as `keepDims`.

Additional optimizer-specific notes are documented in [`src/main/java/graph/optimizer/README.md`](src/main/java/graph/optimizer/README.md).

Important stage boundary:

- `OptimizerStage.AR` is the rewrite family stage
- internally it is composed from rewrite passes in [`src/main/java/graph/optimizer/rewrite/`](src/main/java/graph/optimizer/rewrite)
- top-level non-rewrite optimizer stages remain under [`src/main/java/graph/optimizer/rules/`](src/main/java/graph/optimizer/rules)
- rewrite policy is carried through [`RewriteConfig`](src/main/java/config/optimizer/RewriteConfig.java), which now explicitly separates:
  - algebraic rewrite policy
  - linear lowering policy
  - conv2d lowering policy
  - piecewise/select lowering policy

### Fused Code Generation

Fused element-wise regions are materialized through a plan-first codegen path. [`FusedOperationFactory`](src/main/java/operations/FusedOperationFactory.java) converts a fused cluster into a [`FusedExpressionPlan`](src/main/java/graph/codegen/FusedExpressionPlan.java). [`CompiledFusedKernelFactory`](src/main/java/graph/codegen/CompiledFusedKernelFactory.java) then creates a runtime executable through [`FusedKernelGeneratorRouter`](src/main/java/graph/codegen/FusedKernelGeneratorRouter.java), which dispatches to [`FusedOperationGenerator`](src/main/java/graph/codegen/FusedOperationGenerator.java) for `FLOAT32/FLOAT64` and [`HFusedOperationGenerator`](src/main/java/graph/codegen/HFusedOperationGenerator.java) for `FLOAT16`. The compiled fused executable is stored in prepared node metadata and executed by [`CpuFusedKernel`](src/main/java/backend/kernels/cpu/CpuFusedKernel.java).

This path is intended to reduce dispatch overhead and improve locality for chains of simple operations.

## Supported Operation Families

The runtime now includes support for a wider set of operations, including:

- binary arithmetic: add, sub, mul, div, pow
- unary transforms: neg, inv, exp, log, tanh, sqrt
- activations: relu, sigmoid
- layout and utility ops: contiguous, noop
- scalar/broadcast-style helper ops such as mul-scalar
- reduction support via sum
  - `sumAll` and `sum(axis)` through dedicated CPU reduction kernels
- comparison/select support via `greaterThan`, `greaterOrEqual`, `lessThan`, `lessOrEqual`, `equalTo`, `notEqualTo`, and `where`
- logical bool support via `logicalAnd`, `logicalOr`, `logicalNot`
- fused operations generated by the optimizer/codegen path

## Tensor Operation Catalog

Full Tensor public API and operation list is documented in:

- [`src/main/java/tensor/API.md`](src/main/java/tensor/API.md)

Quick operation catalog on `Tensor`:

- Layout / shape: `contiguous`, `reshape`, `expand`, `permute`, `transpose`, `expandDims`, `squeeze`
- Binary: `add`, `sub`, `mul`, `div`, `min`, `max`, `matmul`
- Comparison / select: `greaterThan`, `greaterOrEqual`, `lessThan`, `lessOrEqual`, `equalTo`, `notEqualTo`, `where`, `minimum`, `maximum`
- Bool ops: `logicalAnd`, `logicalOr`, `logicalNot`
- Unary / scalar: `neg`, `inv`, `log`, `exp`, `fastExp`, `tanh`, `fastTanh`, `sqrt`, `sigmoid`, `pow`, `mul(scalar)`, `clamp(min, max)`, `clampMin(min)`, `clampMax(max)`
- Reduction: `sum()`, `sum(axis)`, `mean()`, `mean(axis)`, `min()`, `min(axis)`, `max()`, `max(axis)`, `all()`, `all(axis)`, `any()`, `any(axis)`
- Helper anchor: `forwardOutput()`

Reduction details:

- `sum()` reduces the whole tensor to shape `[1]`
- `sum(int dimension)` reduces one axis and removes that axis from output shape
- `sum(int dimension, boolean keepDims)` can preserve the reduced axis as size `1`
- `mean(...)` follows the same shape policy as `sum(...)`, but divides by the reduced axis size (or total element count for `mean()`)
- `min(...)` and `max(...)` follow the same shape policy as `sum(...)`
- reduction `min/max` backward routes gradient only to winning elements; ties split gradient evenly
- `all(...)` and `any(...)` are `BOOL`-only reductions with the same shape policy as other reductions
- `clamp(min, max)` limits numeric values into the closed interval `[min, max]`
- `clampMin(min)` raises only values below the lower bound
- `clampMax(max)` lowers only values above the upper bound
- `minimum(a, b)` / `maximum(a, b)` are compare/select-based piecewise surfaces with where-style tie semantics

`expand(...)` is also available as an explicit broadcast-shape operation.
It is implemented as a zero-stride alias view.
Use `contiguous()` when you want an explicitly materialized dense tensor.

Broadcasting contract for binary element-wise ops:

- ranks align from the right
- missing leading dimensions behave as `1`
- dimensions are compatible if they are equal or one side is `1`
- gradients are reduced back to the original operand shape during backward execution
- examples:
  - `[2, 3, 4]` + `[3, 4]` -> `[2, 3, 4]`
  - `[3, 4]` + `[2, 1, 4]` -> `[2, 3, 4]`
  - `[1, 1, 2, 4]` + `[2, 3, 1, 4]` -> `[2, 3, 2, 4]`

Comparison ops follow the same binary broadcasting contract and produce `BOOL` tensors.
`where(condition, x, y)` requires a `BOOL` condition, broadcasts all three inputs to a common output shape, and returns the promoted numeric dtype of `x` and `y`.
Comparison ops are nondifferentiable; `where` propagates gradients only through the selected data branches.
Logical bool ops operate only on `BOOL` tensors. `logicalAnd` and `logicalOr` use the same binary broadcasting rules; `logicalNot` is unary. All logical bool ops are nondifferentiable.

## Quick Start Tensor Ops

```java
import backend.runtime.ExecutionMode;
import config.profile.ExecutionProfile;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import tensor.Tensor;

Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{4.0, 5.0, 6.0}, new int[]{3}, null, "b");
a.setRequiresGrad(true);
b.setRequiresGrad(true);

Tensor y = a.add(b).mul(0.5).sum();

ExecutionProfile profile = new ExecutionProfile(
        "default",
        "default",
        y.getDataType(),
        ExecutionMode.FORWARD_BACKWARD,
        OptimizerConfig.trainingDefaults(),
        RuntimeConfig.trainingDefaults()
);

y.compute(profile);
```

## Entry Points

- [`Main`](src/main/java/synaptik/app/Main.java)
  - Minimal packaged application entry point
  - intentionally does not boot a benchmark framework anymore
- [`NumericsCli`](src/main/java/numerics/NumericsCli.java)
  - Standalone numerics comparison CLI

## Benchmarks and Profiles

The benchmark/autotune subsystem now lives under [`src/main/java/tuning/`](src/main/java/tuning).

It provides:

- workload specifications over the public tensor surface
- `ExecutionProfile` candidate generation and mutation
- validation against conservative baseline profiles
- benchmark and autotune sessions
- search strategies, reporting, and persistence
- suite-level reporting with candidate summaries and hotspot aggregation
- preset-oriented request construction through `TuningPreset` and `TuningDefaults`
- workload-aware preset recommendations through `WorkloadPresetFamily`
- curated benchmark etalon suites through `tuning.etalon.FrameworkEtalon`

See:

- [`src/main/java/tuning/README.md`](src/main/java/tuning/README.md)
- [`src/main/java/tuning/ARCHITECTURE.md`](src/main/java/tuning/ARCHITECTURE.md)
- [`src/main/java/tuning/REPORTING.md`](src/main/java/tuning/REPORTING.md)

## Testing

The [`src/test/java/`](src/test/java) directory contains coverage for key execution and regression scenarios, including:

- [`TensorAddTest`](src/test/java/TensorAddTest.java)
- [`AllOpsTest`](src/test/java/AllOpsTest.java)
- [`OptimizerFuseTest`](src/test/java/OptimizerFuseTest.java)
- [`GradientEngineRegressionTest`](src/test/java/GradientEngineRegressionTest.java)

Run them with:

- `./gradlew test`

## Development Notes

- The project depends on ASM for bytecode generation via the Gradle build in [`build.gradle`](build.gradle)
- Fused kernels are generated and loaded at runtime
- CPU execution is the primary implemented backend today
- CUDA and OpenCL support are intentionally incomplete scaffolds
- IntelliJ and Gradle project settings are aligned to JDK 25
- execution profiles are serialized through [`ExecutionProfileIO`](src/main/java/config/profile/ExecutionProfileIO.java)

## Roadmap Ideas

- Complete CUDA/OpenCL kernel implementations
- Broaden broadcast/reduction semantics beyond the current formalized core
- Add more aggressive graph rewrites and cost modeling
- Improve benchmark automation and profile-guided optimizer selection
- Add packaging, publishing, and API documentation

## License

No license file is currently included. Add a dedicated license before public distribution.
