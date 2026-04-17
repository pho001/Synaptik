# Synaptik

Synaptik is a Java framework for tensors, explicit compiled-graph execution, and reverse-mode autodiff. It is not built as an "eager-only" library and it is not a benchmark-first experiment. The core contract is:

- the public `Tensor` API builds a graph
- `CompiledGraph` turns it into an explicit execution artifact
- the optimizer applies purely graph-level transformations
- `PreparedExecution` attaches runtime policy and backend metadata
- the backend runs the concrete CPU kernel path

Today the project primarily targets the CPU backend. GPU backends currently exist only as scaffolding.

## Reading Guide

If you want to understand the project quickly:

1. start with [src/main/java/tensor/README.md](src/main/java/tensor/README.md)
2. then [src/main/java/operations/README.md](src/main/java/operations/README.md)
3. then [src/main/java/graph/README.md](src/main/java/graph/README.md)
4. finally [src/main/java/backend/README.md](src/main/java/backend/README.md) and [src/main/java/tuning/README.md](src/main/java/tuning/README.md)

If you are solving a specific problem:

- public tensor API: [src/main/java/tensor/API.md](src/main/java/tensor/API.md)
- optimizer and rewrite/fusion rules: [src/main/java/graph/optimizer/README.md](src/main/java/graph/optimizer/README.md)
- tuning workflow, persistence, and candidate search: [src/main/java/tuning/README.md](src/main/java/tuning/README.md)
- numerics drift and A/B comparison: [src/main/java/numerics/README.md](src/main/java/numerics/README.md)

## Highlights

- explicit tensor metadata: shape, strides, storage offset, dtype
- compiled/prepared execution pipeline instead of implicit runtime dispatch sprawl
- reverse-mode autodiff over graphs built from `Tensor` operations
- optimizer stage model with rewrites, CSE, fusion, and memory planning
- CPU backend with prepared metadata, dispatch hints, and family-specific executors
- ASM fused backend for hot elementwise fused subgraphs
- specialized primitives for structured kernel families
  - `LINEAR`
  - `SOFTMAX` / `LOG_SOFTMAX` and their gradients
  - `SCALED_DOT_PRODUCT_ATTENTION` forward/backward
  - `CROSS_ENTROPY_LOSS_INDICES`
  - `CONV2D_GEMM`
- tuning stack for benchmark, autotune, and platform calibration

## Requirements

- JDK 25
- a Gradle 9.4.1 compatible environment, or the bundled Gradle Wrapper
- macOS, Linux, or Windows

Vector API note:

- the build and runtime use `jdk.incubator.vector`
- the Gradle wrapper adds `--add-modules=jdk.incubator.vector` to compile, test, and run tasks

## Build And Run

Basic commands:

- `./gradlew classes`
- `./gradlew test`
- `./gradlew run`

On Windows, use `gradlew.bat`.

### Main CLI

The main CLI entry point is [src/main/java/synaptik/app/Main.java](src/main/java/synaptik/app/Main.java).

Supported flows:

```bash
./gradlew run --args="full f64"
./gradlew run --args="calibrate f64"
./gradlew run --args="autotune f64"
./gradlew run --args="benchmark-winner f64"
./gradlew run --args="benchmark-stage-space f64"
```

Meaning:

- `full`
  - convenience flow `calibrate -> autotune -> benchmark-winner`
  - suitable for local iteration
  - not for the cleanest performance numbers
- `calibrate`
  - finds platform runtime defaults for the selected dtype/mode
- `autotune`
  - searches for the graph-level winner for the `abc_sequence_matmul_*` workload
- `benchmark-winner`
  - compares the baseline against the stored winner profile
- `benchmark-stage-space`
  - benchmarks explicit stage-order candidates

### Numerics CLI

The numerics harness runs through [src/main/java/numerics/NumericsCli.java](src/main/java/numerics/NumericsCli.java).

Example:

```bash
java --add-modules jdk.incubator.vector \
  -Dnumerics.dtype=FLOAT32 \
  -Dnumerics.stageA=NONE \
  -Dnumerics.stageB=AR,CSE,FUSE \
  -Dnumerics.size=200000 \
  -cp build/classes/java/main \
  numerics.NumericsCli
```

## Quick Start

### 1. Simple forward/backward computation

```java
import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.profile.ExecutionProfile;
import config.runtime.RuntimeConfig;
import tensor.Tensor;

Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0}, new int[]{3}, null, "a");
Tensor b = new Tensor(new double[]{4.0, 5.0, 6.0}, new int[]{3}, null, "b");
a.setRequiresGrad(true);
b.setRequiresGrad(true);

Tensor y = a.add(b).mul(0.5).sum();

ExecutionProfile profile = new ExecutionProfile(
        "demo",
        "demo",
        y.getDataType(),
        ExecutionMode.FORWARD_BACKWARD,
        OptimizerConfig.trainingDefaults(),
        RuntimeConfig.trainingDefaults()
);

y.compute(profile);
System.out.println(y.scalarAsDouble());
System.out.println(java.util.Arrays.toString(a.getGradient().toDoubleArrayCopy()));
System.out.println(java.util.Arrays.toString(b.getGradient().toDoubleArrayCopy()));
```

What actually happens here:

1. the `Tensor` API builds a DAG from primitives
2. `compute(profile)` internally calls `CompiledGraph.compile(...)`
3. the optimizer applies the stage order from `OptimizerConfig`
4. `prepare(...)` precomputes backend metadata and dispatch hints
5. `PreparedExecution` runs either forward or `FORWARD_BACKWARD`

### 2. Explicit compile/prepare reuse

This is the correct pattern when you want to measure runtime without repeatedly paying compile/prepare overhead:

```java
import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import tensor.Tensor;

Tensor out = input.linear(weight, bias).relu().sum();
CompiledGraph graph = CompiledGraph.compile(out, OptimizerConfig.trainingDefaults());
PreparedExecution prepared = graph.prepare(RuntimeConfig.trainingDefaults());

prepared.execute(ExecutionMode.FORWARD_BACKWARD);
prepared.execute(ExecutionMode.FORWARD_BACKWARD);
```

Use this for:

- steady-state execution benchmarks
- hot-path tracing
- repeated inference/training runs over the same graph

### 3. Broadcasting and select API

```java
Tensor scores = query.matmul(key.transpose());
Tensor masked = Tensor.where(mask, scores, Tensor.scalar(-1.0e9, scores.getDataType()));
Tensor probs = masked.softmax(1);
Tensor out = probs.matmul(value);
```

Important notes:

- `where` requires a `BOOL` condition
- binary ops and compare ops use standard broadcasting
- autograd reduces broadcasted gradients back to the original operand shapes

## Project Structure

- [src/main/java/tensor/](src/main/java/tensor/)
  - public tensor surface, metadata, execution helpers, graph traversal, primitive builders
- [src/main/java/tensor/ops/](src/main/java/tensor/ops/)
  - thematically organized public graph builders
  - `binary`, `unary`, `compare`, `select`, `layout`, `linalg`, `conv`, `pool`, `reduction`, `loss`, `normalization`
- [src/main/java/tensor/options/](src/main/java/tensor/options/)
  - configuration helper records/enums for public higher-level ops
- [src/main/java/operations/](src/main/java/operations/)
  - canonical operation descriptors used in the graph
- [src/main/java/graph/](src/main/java/graph/)
  - compile, prepare, and run orchestration
- [src/main/java/graph/optimizer/](src/main/java/graph/optimizer/)
  - rule pipeline, rewrite family, fusion support, memory planning
- [src/main/java/backend/](src/main/java/backend/)
  - runtime dispatch and backend-specific execution integration
- [src/main/java/backend/kernels/cpu/](src/main/java/backend/kernels/cpu/)
  - CPU kernel families
  - `elementwise`, `reduction`, `linalg`, `nn`, `index`, `layout`, `fused`, `grad`
- [src/main/java/tuning/](src/main/java/tuning/)
  - benchmark, autotune, platform calibration, reporting, persistence, search
- [src/main/java/numerics/](src/main/java/numerics/)
  - numerics A/B harness
- [src/test/java/](src/test/java/)
  - execution, regression, rewrite, tuning, and benchmark contracts

## Core Architecture

### Tensor Layer

`Tensor` is both the public graph node and the runtime container. It carries:

- the `Operation` descriptor
- the list of input tensors
- storage/data views
- `requiresGrad`, `gradient`, and the backward marker
- shape/stride metadata

Today the public API primarily builds graphs through helper layers in `tensor.ops.*`, not through hand-written logic directly inside `Tensor.java`.

### Operation Layer

`operations.*` are not backend kernels. They are graph-level descriptors.

Examples:

- `add`, `mul`, `relu`
- `linear`
- `softmax`
- `scaledDotProductAttention`
- `crossEntropyLossIndices`

The same descriptor:

- defines the node type in the graph
- serves as optimizer input
- is the key for backend kernel resolution

### Graph Layer

`CompiledGraph` performs:

1. topological closure over the forward graph
2. backward graph construction if trainable leaf inputs exist
3. optimizer stages from `OptimizerConfig`
4. separation into forward/backward sections

`PreparedExecution` then performs:

- runtime-specific prepare
- prepared metadata construction for each node
- the actual execution loop
- optional traced runs

### Backend Layer

Today the CPU backend uses prepared metadata instead of repeatedly making runtime decisions directly from `Tensor`.

The prepare phase resolves in particular:

- the compute contract
- dtype conversion/materialization decisions
- dispatch hints
- reduction hints
- matmul hints
- fused executable preparation
- workspace allocation for selected op families

### Optimizer Layer

The optimizer stage order is explicit. The main stage families today are:

- `AR`
  - composite rewrite family
- `CSE`
  - structural common subexpression elimination
- `FUSE`
  - elementwise fused cluster formation
- `MEM`
  - liveness-aware memory planning

The rewrite family today includes more than just algebraic cleanup. It also contains lowering into specialized primitives, for example:

- `matmul + bias -> linear`
- `softmax` / `logSoftmax` backward pattern lowering
- attention forward/backward lowering
- cross-entropy-from-indices lowering
- optional piecewise import canonicalization
- `conv2d -> conv2dGemm` according to policy

## Execution Profiles, Calibration, And Persistence

Runtime is not governed by a single "optimizer profile" JSON anymore. The current flow distinguishes between:

- built-in defaults in code
- a platform runtime profile
  - the output of platform calibration
- a graph-specific best profile
  - the output of autotune for a concrete workload
- the final `ExecutionProfile`
  - the assembled runnable artifact

The preferred layout is:

- `profiles/platform/<platform-id>/calibration/...`
- `profiles/platform/<platform-id>/reports/...`
- `profiles/platform/<platform-id>/tuning/...`

Compatibility fallbacks under `build/...` still exist, but they are not the preferred long-term layout.

## Real Usage Patterns

### When to use only `Tensor.compute(profile)`

Use it for:

- simple integration tests
- sanity checks of a local graph
- small demo programs

Do not use it as your only benchmark harness, because each call recompiles and reprepares the graph.

### When to keep `CompiledGraph`

Use it for:

- inspecting optimizer output
- compile trace / prepare trace
- preparing the same graph repeatedly under different runtime configs

### When to keep `PreparedExecution`

Use it for:

- steady-state benchmarking
- hot-path tracing
- performance experiments over the same graph and the same runtime policy

## Module Docs

- tensor: [src/main/java/tensor/README.md](src/main/java/tensor/README.md)
- tensor API: [src/main/java/tensor/API.md](src/main/java/tensor/API.md)
- operations: [src/main/java/operations/README.md](src/main/java/operations/README.md)
- graph: [src/main/java/graph/README.md](src/main/java/graph/README.md)
- optimizer: [src/main/java/graph/optimizer/README.md](src/main/java/graph/optimizer/README.md)
- backend: [src/main/java/backend/README.md](src/main/java/backend/README.md)
- tuning: [src/main/java/tuning/README.md](src/main/java/tuning/README.md)
- numerics: [src/main/java/numerics/README.md](src/main/java/numerics/README.md)

## Testing

Basic test flow:

- `./gradlew test`
- `./gradlew classes`

Typical coverage areas:

- execution correctness
- dtype coverage
- broadcast contracts
- rewrite/lowering correctness
- fused execution
- tuning/search/report contracts

## Development Notes

- fused kernels are currently generated through ASM codegen during the prepare phase
- CPU is the only fully implemented backend
- CUDA/OpenCL are scaffolding
- the project assumes JDK 25 and the Vector API
- the tuning docs are the source of truth for benchmark/autotune/calibration flow, not older benchmark-only utility layers

## License

There is currently no license file in the repository. Add one before public distribution.
