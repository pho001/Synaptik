# Prepare execution (planned workflow)

## Outcome

This guide explains how an immutable compile recipe will become reusable runtime-ready state. Prepare, runtime, engine, and concrete backends are not implemented.

## Planned steps

1. Build an engine with concrete backends registered explicitly.
2. Supply `CompileArtifacts` and declarative prepare configuration.
3. Validate that planned partitions have registered owners.
4. Call the owning backend's partition preparer for each partition.
5. Build and validate prepared partitions, physical memory slots, and a schedule.

```java
// Conceptual API; not currently runnable.
PreparedExecution execution = graph.prepare(PrepareConfig.defaults());
```

During step 4, a concrete backend performs lowering, specialization, fusion, and kernel selection. Shared prepare code validates the handoff but does not contain CPU, Metal, or CUDA lowering.

## Expected result

The result will be a reusable `PreparedExecution`. It will contain executable contracts and storage/schedule plans but no per-run input bindings. Mutable invocation state belongs to `RunState` created or reused for each run.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| A partition has no preparer | Its owning backend was not registered or is unavailable. | Register required backends before prepare and fail before run. |
| Shared prepare selects a Metal kernel | Backend-specific lowering escaped its owner. | Move the choice into the Metal partition preparer. |
| Prepared state is rebuilt every run | Prepare and run lifecycles were mixed. | Reuse `PreparedExecution`; isolate mutable state per run. |

## Related documentation

- [Runtime/prepare/backend boundary](../architecture/runtime-prepare-backend-boundary.md)
- [Partition preparer guide](../backend-guide/partition-preparer.md)
- [Runtime API status](../api/runtime-api.md)
