# ComputationalGraph

ComputationalGraph is a lightweight Java computational graph and autodiff playground focused on tensor execution, graph optimization, and runtime fusion of element-wise operations. The project combines a small tensor runtime with an optimizer pipeline, backend-specific kernel dispatch, and generated fused kernels for fast execution experiments.

## Highlights

- Tensor model built on `double[]` storage with shape/stride metadata
- Reverse-mode autodiff for a growing set of tensor operations
- Optimizer pipeline with pluggable rewrite and fusion rules
- Runtime fused-operation generation for element-wise subgraphs
- Backend abstraction with CPU kernels and CUDA/OpenCL scaffolding
- Benchmarking and persisted optimizer profile support
- Regression and operation-level test coverage

## Requirements

- JDK 25
- Gradle 9.x compatible environment, or the included Gradle Wrapper
- macOS, Linux, or Windows

Vector API note:

- The project uses `jdk.incubator.vector`
- Gradle adds `--add-modules=jdk.incubator.vector` to compile, test, and runtime tasks

## Build and Run

Using the Gradle Wrapper:

- Run the demo app: `./gradlew run`
- Build the project: `./gradlew build`
- Run tests: `./gradlew test`
- Run the optimizer benchmark entry point: `./gradlew run --args='benchmark'`

On Windows, use [`gradlew.bat`](gradlew.bat) instead of [`gradlew`](gradlew).

## Project Structure

- [`src/Tensor/`](src/Tensor)
  - Core tensor implementation, storage, shape/stride logic, execution state, and autodiff plumbing
- [`src/Operations/`](src/Operations)
  - Primitive tensor operations such as add, sub, mul, div, pow, exp, log, tanh, relu, sigmoid, contiguous, sum, and newer unary/scalar helpers
- [`src/Backend/`](src/Backend)
  - Backend execution layer and per-platform dispatch integration
- [`src/Backend/kernels/`](src/Backend/kernels)
  - Backend kernel interfaces and concrete CPU kernel implementations
- [`src/Backend/registry/`](src/Backend/registry)
  - Operation-to-kernel registries used by CPU, CUDA, and OpenCL backends
- [`src/Graph/`](src/Graph)
  - Compiled graph execution and graph-level orchestration
- [`src/Graph/optimizer/`](src/Graph/optimizer)
  - Optimizer entry points, factory wiring, rule composition, and optimizer documentation
- [`src/Graph/codegen/`](src/Graph/codegen)
  - Runtime fused code generation for specialized fused operations
- [`src/Benchmark/`](src/Benchmark)
  - Benchmark harness, candidate selection, profile I/O, and tuning utilities
- [`src/Config/`](src/Config)
  - Backend and optimizer tuning configuration objects
- [`config/`](config)
  - Persisted benchmark and optimizer profile data
- [`test/`](test)
  - Regression and functional tests

## Core Architecture

### Tensor Runtime

[`Tensor`](src/Tensor/Tensor.java) is the central runtime object. It carries:

- raw `double[]` storage
- shape and stride metadata
- graph links to producer inputs
- gradient storage and backward propagation helpers
- execution hooks used by optimized and fused graphs

Operations are represented through the [`Operation`](src/Operations/Operation.java) abstraction and a shared op-type enum. This allows the optimizer and backends to reason about operations generically while keeping per-op forward and gradient behavior localized in individual classes.

### Backend Model

The backend layer separates graph execution from device-specific kernels.

- [`CPUBackend`](src/Backend/CPUBackend.java) resolves operation kernels from [`CpuKernelRegistry`](src/Backend/registry/CpuKernelRegistry.java)
- [`CudaBackend`](src/Backend/CudaBackend.java) and [`OpenClBackend`](src/Backend/OpenClBackend.java) follow the same registry-oriented structure
- CPU kernels under [`src/Backend/kernels/cpu/`](src/Backend/kernels/cpu) provide concrete implementations for the currently supported ops
- CUDA/OpenCL kernel packages are currently scaffolding for future implementations

This makes it easier to extend support for new operations without embedding all execution logic directly inside backend classes.

### Optimizer Pipeline

The optimizer was reorganized into a dedicated module rooted at [`GraphOptimizer`](src/Graph/optimizer/GraphOptimizer.java) and built by [`OptimizerFactory`](src/Graph/optimizer/OptimizerFactory.java).

Current rule set includes files such as:

- [`FuseElementWiseRule`](src/Graph/optimizer/rules/FuseElementWiseRule.java)
- [`AlgebraicRewritingRule`](src/Graph/optimizer/rules/AlgebraicRewritingRule.java)
- [`CommonSubexpressionEliminationRule`](src/Graph/optimizer/rules/CommonSubexpressionEliminationRule.java)
- [`MemoryOptimizerRule`](src/Graph/optimizer/rules/MemoryOptimizerRule.java)

Additional optimizer-specific notes are documented in [`src/Graph/optimizer/README.md`](src/Graph/optimizer/README.md).

### Fused Code Generation

Fused element-wise regions are materialized through runtime code generation in [`DFusedOperationGenerator`](src/Graph/codegen/DFusedOperationGenerator.java). Generated fused classes are then used by [`FusedOperation`](src/Operations/FusedOperation.java) during compiled graph execution.

This path is intended to reduce dispatch overhead and improve locality for chains of simple operations.

## Supported Operation Families

The runtime now includes support for a wider set of operations, including:

- binary arithmetic: add, sub, mul, div, pow
- unary transforms: neg, inv, exp, log, tanh, sqrt
- activations: relu, sigmoid
- layout and utility ops: contiguous, noop
- scalar/broadcast-style helper ops such as mul-scalar
- reduction support via sum
- fused operations generated by the optimizer/codegen path

## Entry Points

- [`Main`](src/Main.java)
  - Small runnable demo for building and executing a graph
- [`OptimizerBenchmark`](src/OptimizerBenchmark.java)
  - Benchmark entry point for optimizer and fusion experiments

## Benchmarks and Profiles

The benchmarking subsystem under [`src/Benchmark/`](src/Benchmark) provides:

- optimizer candidate generation
- benchmark orchestration across optimization stages
- tuning knob definitions
- profile serialization/deserialization

A sample persisted profile is stored in [`config/optimizer-profile.json`](config/optimizer-profile.json).

## Testing

The [`test/`](test) directory contains coverage for key execution and regression scenarios, including:

- [`TensorAddTest`](test/TensorAddTest.java)
- [`AllOpsTest`](test/AllOpsTest.java)
- [`OptimizerFuseTest`](test/OptimizerFuseTest.java)
- [`GradientEngineRegressionTest`](test/GradientEngineRegressionTest.java)

Run them with:

- `./gradlew test`

## Development Notes

- The project depends on ASM for bytecode generation via the Gradle build in [`build.gradle`](build.gradle)
- Fused kernels are generated and loaded at runtime
- CPU execution is the primary implemented backend today
- CUDA and OpenCL support are intentionally incomplete scaffolds
- IntelliJ and Gradle project settings are aligned to JDK 25

## Roadmap Ideas

- Complete CUDA/OpenCL kernel implementations
- Expand broadcasting and reduction semantics
- Add more aggressive graph rewrites and cost modeling
- Improve benchmark automation and profile-guided optimizer selection
- Add packaging, publishing, and API documentation

## License

No license file is currently included. Add a dedicated license before public distribution.
