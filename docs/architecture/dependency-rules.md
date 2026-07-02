# Dependency Rules

This document explains the dependency direction required by [`ARCHITECTURE.md`](../../ARCHITECTURE.md). It is explanatory; the root contract is authoritative.

## Intended direction

The compile-time side builds upward from small shared contracts into planning and compilation:

```text
trace
backend-contract
config
model
  -> planning
  -> compiler
```

The execution side consumes compile-time contracts while moving from shared runtime contracts to concrete backend implementations:

```text
model
config
backend-contract
trace
planning
compiler
  -> runtime
  -> prepare
  -> concrete backends
```

The engine depends on the components it composes:

```text
compiler
runtime
prepare
backends/cpu
backends/metal
backends/cuda
  -> engine
```

These diagrams express architectural direction; individual modules should depend only on the contracts they actually use.

## Concrete forbidden dependencies

- `modules/trace` must not depend on model, planning, compiler, runtime, prepare, engine, or concrete backends.
- `modules/model` must not depend on planning, compiler, runtime, prepare, engine, or concrete backends.
- `modules/config` must not depend on concrete backend implementations.
- `modules/planning` must not depend on concrete backends, runtime, prepare, or engine.
- `modules/compiler` must not depend on runtime, prepare, engine, or concrete backends.
- `modules/runtime` must not depend on concrete backends or engine.
- `modules/prepare` must not depend on concrete backend implementations.
- Concrete backends must not depend on `modules/engine`.
- `backends/openblas-provider` must not depend on compiler, planning, runtime, prepare, engine, or the Tensor API.
- `extensions/training` must not depend on concrete backend modules.
- `extensions/onnx` must not depend on runtime hot-path execution internals.

## Why the boundaries matter

### Trace is a dependency leaf

Every phase may emit diagnostics, so trace DTOs must be safe for every phase to reference. If trace imported graph, runtime, or backend types, diagnostic use would introduce reverse dependencies and make the trace schema inherit business logic. Trace-local identifiers and typed payloads let producers describe events without importing one another.

### Model is backend- and runtime-independent

The model defines computation semantics and the public tensor abstraction. Keeping backend capability, device residency, physical buffers, and prepared execution out of it makes the same graph meaningful before any backend is chosen. It also prevents public mutable `Tensor` state from becoming runtime device state.

### Runtime is independent of concrete backends

Runtime executes `PreparedExecutable` and schedule contracts. Concrete backend preparation supplies implementations before execution. A runtime dependency on concrete backends would pull backend discovery, lowering, or kernel choice into the hot path and break explicit composition.

### Backends are independent of engine

Engine is the outer composition root: it knows and wires concrete backend modules. If a backend depended on engine, composition would become cyclic and backend implementation would be coupled to the public orchestration layer.

## Related semantic dependency rules

Source-level dependency tests should also protect the boundaries that type dependencies alone may miss:

- `Operation` must not expose `supportedBackends()`.
- Runtime hot-path code must not use `Operation` or `CompiledNode`.
- Planning scoring must not reference concrete kernel classes or prepared executables.
- Compile-time plans hold `BackendId`, not live backend services.
- CPU routes such as scalar, Vector API, and OpenBLAS remain implementation routes inside the CPU backend.

## Architecture-test enforcement

Tests under `testing/architecture-tests/` should fail when forbidden module or package edges appear. They should cover every dependency rule above and add focused checks for:

- the trace and model leaf boundaries;
- config independence from concrete backend classes;
- planning and compiler independence from runtime and concrete implementations;
- runtime and prepare independence from concrete backends;
- backend independence from engine;
- the OpenBLAS provider's low-level leaf role;
- absence of backend support APIs on `Operation`;
- absence of compile-time graph types in the runtime hot path; and
- absence of concrete implementation references in partition scoring.

Architecture tests enforce the contract; they do not redefine it. When a dependency rule changes, update [`ARCHITECTURE.md`](../../ARCHITECTURE.md), the relevant explanatory document, an ADR when significant, and the architecture tests in the same change.
