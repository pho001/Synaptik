# Module Boundaries

This document explains the module responsibilities established by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The contract is authoritative when a summary here is incomplete.

## Shared modules

### `modules/trace`

Owns typed diagnostic DTOs: the event envelope, compile/prepare/run/backend payloads, trace-local identifiers, and typed trace attributes. It contains no graph traversal, execution, business logic, or mutable runtime state and must not import higher layers. This keeps trace data usable from every producer without creating reverse dependencies.

See [Tracing](tracing.md).

### `modules/backend-contract`

Owns the minimal backend vocabulary shared across layers: `BackendId`, `BackendDeviceId`, availability snapshots, declarative requirements, and device classes. Compile-time plans refer to backend identities rather than live backend services.

It does not own capabilities implementation, operation support logic, kernel registries, prepare services, executables, storage, physical buffers, or cost-model implementation.

### `modules/model`

Owns the public tensor model, operation semantics, shape/data type/layout types, host storage abstractions, and immutable graph model. `Tensor` remains public mutable API state; `CompiledGraphModel`, `CompiledNode`, and graph values are compile-time state.

The model does not know backend support, device residency, kernel selection, backend-specific storage, prepared execution, or runtime state. `Operation` expresses semantics and never exposes `supportedBackends()`. Runtime device storage belongs outside this module.

### `modules/config`

Owns declarative compile, prepare, run, publication, and profile configuration. It may describe backend intent and scoring policy, but it contains no live services, concrete backend classes, executable units, runtime state, or kernel class references. A concrete backend interprets its prepare configuration inside that backend.

### `modules/planning`

Owns backend-neutral compile-time planning: intent propagation, capability query contracts and matrices, ownership scoring, node or segment ownership, maximal same-owner partitions, and logical memory/materialization requirements.

Planning answers where work should run. It does not implement fusion or specialization, select a concrete kernel or backend route, allocate physical memory, construct backend DAGs, or create prepared schedules and runtime units.

See [Partition Scoring](partition-scoring.md).

### `modules/compiler`

Owns graph capture and compilation: indexing and ordering, inference and validation, canonicalization and optimization, autograd expansion, publication binding, planning orchestration, compile diagnostics, and `CompileArtifacts`.

Its output is immutable compile-time state. It does not create physical buffers, backend executables, runtime workspaces, prepared schedules, or `PreparedExecution`, and it has no concrete backend dependencies.

### `modules/runtime`

Owns prepared execution contracts and dynamic execution state, including `PreparedExecution`, `PreparedUnit`, `PreparedExecutable`, `PreparedSchedule`, `PreparedMemoryPlan`, slots, `RunState`, resources, transfers, residency, publication, and the prepared execution runner.

Runtime executes already-prepared schedules. It does not optimize graphs, construct autograd, discover backends, look up backend services, lower partitions, or select kernels. The hot path does not use `Operation` or `CompiledNode`.

### `modules/prepare`

Owns shared preparation contracts and validation: `PrepareContext`, `BackendPartitionPreparer`, `PreparedPartition`, partition coverage validation, prepared memory validation, and prepared schedule validation.

Prepare bridges compile artifacts to runtime contracts, but shared prepare code does not contain concrete CPU, Metal, or CUDA lowering, kernel selection, executable implementations, or backend-specific storage. Those belong to concrete backend modules.

### `modules/engine`

Owns the public lifecycle facade and composition root. It wires the compiler, prepare validators and orchestration, runtime, trace sinks, and explicitly registered concrete backends.

Engine does not own kernels, backend internals, graph optimization passes, a runtime service locator, or reflective plugin discovery as the core backend mechanism. Concrete backends never depend on engine.

## Concrete backend modules

`backends/cpu`, `backends/metal`, and `backends/cuda` are vertical implementations. Each backend owns its capability provider, partition preparer, lowering, fusion, specialization, route selection, executable units, storage and workspace, trace contributions, and native integration.

Concrete backend logic belongs in the backend that implements it:

- CPU scalar, Vector API, ASM, and OpenBLAS are routes inside `backends/cpu`, not separate backends.
- MPSGraph and custom Metal kernels, Metal storage, and native bridges belong to `backends/metal`.
- CUDA lowering, kernels, storage, and native integration belong to `backends/cuda`.

Backends may consume model, config, planning, runtime, prepare, backend-contract, and trace contracts. They must not own public tensor APIs or global compiler logic, and must not depend on engine.

`backends/openblas-provider` is a lower-level leaf used by the CPU backend. It owns loading, symbol binding, GEMM calls, and thread control only. It does not interpret config, decide fallback or ownership, participate in planning, manage residency, or expose the Tensor API. Dependency direction is `backends/cpu -> backends/openblas-provider`.

## Extensions

### `extensions/training`

Owns training concepts and optimizer algorithms such as `Optimizer`, `Sgd`, `Adam`, `AdamW`, parameters, sessions, and training steps. It describes the mathematical update but does not access backend storage, select backend kernels, or implement backend-specific optimizer execution.

Training must not depend on concrete backend modules. A fused Adam route on Metal, for example, is a Metal backend prepare/kernel concern rather than a `MetalOptimizerBridge` in training. See [Training Graph](training-graph.md).

### `extensions/onnx`

Owns ONNX import, export, and mapping to and from the model. It remains outside the runtime hot path and does not own backend lowering, kernel selection, execution, or residency.

## Boundary summary

```text
model      = computation semantics and immutable graph model
planning   = backend-neutral ownership and logical requirements
compiler   = graph transformation, autograd, and compile artifacts
prepare    = shared transition contracts and validation
backend    = concrete lowering, implementation choice, and storage
runtime    = prepared execution and dynamic run state
engine     = explicit composition and public lifecycle
trace      = typed diagnostic leaf
```

The dependency implications of these responsibilities are detailed in [Dependency Rules](dependency-rules.md).
