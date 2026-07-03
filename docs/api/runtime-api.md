# Runtime API

## Purpose and implementation status

This reference explains the planned contracts for reusable prepared execution and one run. Runtime and prepare are not implemented yet; the types and calls below are conceptual.

The key distinction is ownership of state:

```text
CompileArtifacts    PreparedExecution       RunState
immutable recipe -> reusable execution -> mutable invocation
```

## Planned prepared contracts

`PreparedExecution` will contain or reference prepared partitions, executable units, a physical memory plan, and a schedule. Preparation creates it once for a selected set of explicitly registered backends; multiple runs may reuse it.

`PreparedExecutable` will compute only its prepared region. Its hot-path contract will not receive `Operation` or `CompiledNode`, and runtime will not ask it to rediscover a backend or select a kernel.

## Planned run contract

```java
// Conceptual API; not currently runnable.
RunResult result = execution.run(inputs, RunOptions.defaults());
```

- `inputs` will bind invocation values to prepared input bindings.
- `RunOptions` will hold declarative run and publication choices, not live services.
- `RunState` will own per-run mutable slots, resources, and residency facts.
- `RunResult` will expose results published by the prepared publication plan and run policy.

Exact collection types, nullability, concurrency guarantees, ownership of returned values, exception types, and resource-lifetime methods remain open until focused runtime tasks define and test them.

## Boundary and failure model

Run may fail because an input binding is missing or incompatible, a prepared resource cannot be used, transfer or execution fails, or publication fails. It must not recover by discovering another backend and lowering again. Unsupported work must be resolved during compile ownership or fail during prepare.

## Scenario

Suppose preparation maps logical values `ValueId(4)` and `ValueId(9)` to one physical slot because their lifetimes do not overlap. At run time the schedule may reuse that slot, but the two logical identifiers remain distinct. This demonstrates why a graph value ID is not a memory address or buffer slot.

## Related contracts

- [Runtime, prepare, and backend boundary](../architecture/runtime-prepare-backend-boundary.md)
- [Preparing execution](../user-guide/preparing-execution.md)
- [Running models](../user-guide/running-models.md)
- [Glossary](../glossary.md)
