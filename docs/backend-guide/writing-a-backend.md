# Integrate a concrete backend (planned workflow)

## Outcome and supported scope

This guide maps the complete backend lifecycle for future contributors. Shared backend, planning, prepare, runtime, and engine contracts are not implemented, so it is an integration design guide rather than runnable instructions.

## Prerequisites

Read the authoritative [`ARCHITECTURE.md`](../../ARCHITECTURE.md), [module boundaries](../architecture/module-boundaries.md), [dependency rules](../architecture/dependency-rules.md), and the concrete backend's master plan. A backend module may depend on shared contracts but must not depend on engine.

## Integration lifecycle

```text
capability -> compile ownership -> backend prepare -> executable -> runtime
```

1. Implement declarative capability reporting for semantic graph facts.
2. Implement a partition preparer that accepts only partitions owned by this backend.
3. Lower, specialize, fuse, and select routes inside the backend.
4. Construct prepared executable, storage, and workspace implementations with explicit lifetimes.
5. Emit typed backend trace contributions.
6. Expose a backend component that engine composition can register explicitly.

## Conceptual registration

```java
// Conceptual API; engine and backend factories are not implemented.
SynaptikEngine engine = SynaptikEngine.builder()
        .addBackend(cpuBackend())
        .addBackend(metalBackend())
        .build();
```

Each `addBackend` call makes composition visible before compilation and preparation. Runtime must not use classpath scanning, `ServiceLoader`, or a service locator to discover the same components during execution.

## Resources, concurrency, and failures

A backend must define who owns native handles, buffers, and workspaces; when they are released; whether prepared executables are safe to share; and which state is per run. Exact shared concurrency contracts remain to be specified.

Capability rejection occurs before ownership. Lowering or resource-creation failure occurs during prepare. Execution failure occurs during run and must not trigger hidden cross-backend fallback.

## Validation

Future backend work requires focused unit tests, architecture dependency tests, applicable backend-conformance tests, end-to-end integration tests, native cleanup checks, and benchmarks for performance claims. Passing a benchmark never substitutes for correctness tests.

## Related documentation

- [Capability provider](capability-provider.md)
- [Partition preparer](partition-preparer.md)
- [Kernel routes](kernel-routes.md)
- [Runtime/prepare/backend boundary](../architecture/runtime-prepare-backend-boundary.md)
