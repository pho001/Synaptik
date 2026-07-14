# Runtime, Prepare, and Backend Boundary

This document explains the boundary defined by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). The root contract remains authoritative.

Runtime, prepare, engine, and concrete backend contracts are not implemented. The types below name the intended architecture and must not be read as a current Java API.

## Boundary in one flow

```text
CompileArtifacts
  -> shared prepare validation and orchestration
  -> concrete backend preparation for each PlannedPartition
  -> PreparedPartition[]
  -> PreparedMemoryPlan
  -> PreparedSchedule
  -> PreparedExecution
  -> runtime execution with RunState
```

Prepare turns an immutable compile recipe into runtime-ready state. Concrete backends supply the executable implementation of their assigned regions. Runtime executes the resulting contracts without rediscovering or lowering backend work.

## What prepare creates

The prepare lifecycle creates:

- `PreparedPartition`, which associates a planned partition and backend identity with its prepared units;
- `PreparedUnit`, which connects a prepared executable to its input and output slots;
- `PreparedExecutable`, the hot-path executable contract for one prepared region;
- `PreparedMemoryPlan`, which defines physical buffer and workspace slots for the prepared execution;
- `PreparedSchedule`, which orders executable, transfer, materialization, and publication work; and
- `PreparedExecution`, the reusable runtime-ready result.

`modules/prepare` owns shared contracts such as `PrepareContext`, `BackendPartitionPreparer`, and `PreparedPartition`, plus partition-coverage, memory-plan, and schedule validation. Engine-level composition coordinates the registered preparers and constructs the complete prepared execution.

Shared prepare code does not implement concrete CPU, Metal, or CUDA lowering and does not own backend-specific executable or storage implementations.

## What runtime does

`modules/runtime` owns the contracts and state needed after preparation. It:

- executes `PreparedSchedule`;
- creates or reuses per-run `RunState` and binds inputs;
- manages runtime slots, resources, and workspaces through prepared contracts;
- performs scheduled residency, transfer, and materialization work;
- invokes `PreparedExecutable.execute(...)`;
- updates residency after execution; and
- publishes outputs and gradients according to the prepared plan and run policy.

Runtime does not optimize the graph, construct autograd, choose backend ownership, discover backends, lower partitions, or select kernels. Its hot path does not operate on `Operation` or `CompiledNode`.

Runtime profiling is passive observation of this execution. Runtime may translate observed facts
into typed trace payloads, but neither profiling nor tracing selects or mutates execution
settings. Runtime performs no model-autotuning search, cache lookup or mutation, or hot-path graph
inspection.

## What a concrete backend does

Each concrete backend owns preparation and execution details for the partitions assigned to it. This includes:

- backend-owned lowering of planned graph regions;
- backend-specific fusion and specialization;
- concrete kernel or executable route selection;
- construction of `PreparedExecutable` implementations;
- backend storage, buffer, and workspace implementations;
- backend-specific materialization and native bridge integration; and
- backend trace contributions.

Planning chooses an owner such as CPU, Metal, or CUDA. The owning backend then chooses scalar, Vector API, OpenBLAS, MPSGraph, a custom Metal kernel, a CUDA kernel, or another backend-internal route during prepare.

Each backend also owns typed, version-controlled, tested candidate generators beside the routes
they configure. A generator uses target capabilities, canonical workload facts, and the tuning
budget to return complete valid configurations. The operation family selects the generator; it is
not a cache key for one universal family-wide setting. Hardware and supported JDK Vector API
species constrain CPU vector candidates, so no candidate can promise an arbitrary physical lane
count.

A future narrow prepare/tuning boundary exposes complete backend candidates opaquely to shared
orchestration. Shared code does not interpret route, vector, thread, tile, kernel, or other private
fields. During ordinary preparation, the backend may reuse a compatible entry from an explicit
workload cache or apply safe heuristics. Model autotuning remains optional for correctness.

The model-specific tuning result is an explicit prepared plan or artifact, not hidden global
state. Its persistent plan record and the reusable workload cache are loaded and updated outside
runtime; incompatible or corrupt entries fall back safely. Physical cache formats and prepared-
executable serialization remain deferred.

## Why there is no shared `backend.lowering` module

Lowering is inseparable from a backend's implementation model, fusion opportunities, specialization, storage, and executable construction. A shared lowering module would either encode concrete backend knowledge in a supposedly common layer or reduce lowering to abstractions that do not own the real decision.

Backend-neutral graph canonicalization and decomposition belong in compiler passes. Backend-specific lowering belongs in each concrete backend's prepare implementation.

## Why there is no runtime service locator

Runtime must execute an already-prepared schedule. Looking up a backend or kernel service during execution would move selection and executable construction into the hot path and hide dependencies.

The engine instead registers backends explicitly and supplies their capability and preparation services before runtime execution. `PreparedExecutable` provides the backend-independent runtime call boundary.

## Why reflective backend discovery is not the core mechanism

Classpath scanning, annotation scanning, reflection, or `ServiceLoader` as the default mechanism would make backend availability implicit and mix composition with execution concerns. Engine is the composition root, so backends are registered explicitly when the engine is built.

An optional convenience layer for plugin discovery would require the architecture update specified by the contract. It must not become a runtime hot-path mechanism.

See [Lifecycle](lifecycle.md) for the complete stage flow, [Module Boundaries](module-boundaries.md) for ownership, and [Dependency Rules](dependency-rules.md) for prohibited dependency edges.
See [Performance Evidence and Tuning](performance-evidence-and-tuning.md) for the optimization
workflow boundaries that feed prepare without entering runtime.
