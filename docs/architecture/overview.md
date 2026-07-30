# Architecture Overview

This document explains the architecture defined by the authoritative [`ARCHITECTURE.md`](../../ARCHITECTURE.md). It summarizes the system without replacing the contract.

The overview describes the intended complete system. Only the initial model value foundations are implemented; consult the [roadmap](../planning/roadmap.md) before treating a named module or type as available.

## High-level architecture

Synaptik turns a public tensor expression into an immutable compile-time recipe, prepares that recipe for explicitly registered backends, and then executes a prepared schedule. The architecture keeps semantic graph work separate from backend implementation choices and per-run state.

The main layers are:

- **Model** owns tensors, operation semantics, shape, data type, layout, host storage abstractions, and the immutable graph model. It has no backend or runtime responsibility.
- **Planning** makes backend-neutral compile-time decisions: intent, capability, ownership scoring, partitioning, and logical memory or materialization requirements.
- **Compiler** captures and transforms the graph, expands autograd when requested, coordinates planning, and produces immutable `CompileArtifacts`.
- **Prepare** validates the transition from compile artifacts to executable runtime state and defines the shared backend preparation contract.
- **Backend** implementations lower their assigned partitions, fuse and specialize work, choose concrete kernel routes, and provide backend storage and workspace implementations.
- **Runtime** executes prepared schedules and manages dynamic run state, residency, transfers, materialization, and publication.
- **Engine** is the composition root and public lifecycle facade. It wires the compiler, prepare layer, runtime, and explicitly registered backends.
- **Trace** is a dependency-leaf module containing typed diagnostic DTOs only.

See [Module Boundaries](module-boundaries.md) for detailed ownership and [Dependency Rules](dependency-rules.md) for allowed dependency direction.

## Why compile, prepare, and run are separate

Each stage answers a different kind of question:

1. **Compile** decides what the graph means, how it is optimized, and which backend owns each node or segment. Its output is an immutable recipe, not executable backend state.
2. **Prepare** turns the recipe into executable state. Concrete backends lower partitions and choose implementations; shared orchestration creates and validates memory and schedule structures.
3. **Run** executes only the already-prepared schedule. It handles inputs and mutable state without repeating graph optimization, backend discovery, lowering, or kernel selection.

This separation keeps compile-time reasoning independent of physical buffers, keeps concrete implementation selection inside backends, and keeps the runtime hot path free of `Operation` and `CompiledNode`.

## Lifecycle summary

```text
Tensor expression
  -> GraphCompiler
  -> CompileArtifacts
     - CompiledGraphModel
     - PlannedPartition[]
     - LogicalMemoryPlan
     - PublicationPlan
  -> prepare
  -> BackendPartitionAnalysis[]
  -> PreparedMemoryPlan with assigned slots
  -> PreparedPartition[]
  -> PreparedSchedule
  -> PreparedExecution
  -> RunState
  -> execute
```

The central state distinction is:

- `CompileArtifacts` are immutable compile-time output.
- `PreparedExecution` is prepared runtime state reusable across runs.
- `RunState` is mutable state for an individual run.
- A `PreparedExecutable` computes only the prepared region assigned to it.

The complete stage-by-stage flow is described in [Lifecycle](lifecycle.md). The handoff among prepare, runtime, and concrete backends is described in [Runtime, Prepare, and Backend Boundary](runtime-prepare-backend-boundary.md).

## Boundary scenario

Consider a future matrix multiplication followed by an elementwise addition. Compile may assign both nodes to CPU and place them in one planned partition. CPU prepare may lower that partition to an OpenBLAS matrix multiplication plus a scalar or vectorized addition. Run invokes those prepared units with bound inputs. Moving the OpenBLAS choice into planning would violate backend-owned lowering; passing the original operations back into runtime would violate the hot-path boundary.

## Repository structure

The repository groups responsibilities by architectural role:

```text
modules/       shared model, configuration, planning, compiler, prepare,
               runtime, engine, backend contracts, and tracing
backends/      concrete CPU, Metal, and CUDA backends, plus leaf providers
extensions/    optional training and ONNX functionality
tools/         tuning, benchmarks, and CLI tooling
native/        platform-specific native integration
testing/       architecture, backend-conformance, and integration tests
docs/          explanatory architecture, design, user, backend, and developer docs
```

Production Java packages use the `io.github.pho001.synaptik.*` namespace. Module directories remain concise, such as `modules/compiler`, `backends/cpu`, and `extensions/training`.
