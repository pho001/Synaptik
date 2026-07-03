# Run a prepared model (planned workflow)

## Outcome

This guide explains the planned invocation lifecycle. There is no runnable execution API yet because runtime, prepare, engine, and public tensor storage are not implemented.

## Planned steps

```java
// Conceptual API; not currently runnable.
RunResult result = execution.run(inputs, RunOptions.defaults());
```

1. Validate and bind inputs to prepared input slots.
2. Create or reuse invocation-local `RunState`.
3. Follow `PreparedSchedule` in order, including prepared transfers or materializations.
4. Invoke each `PreparedExecutable` for only its assigned region.
5. Update residency facts and publish requested results.

## Expected result

`RunResult` will expose results selected by the compile-time publication plan and run policy. Running will not optimize the graph, select another backend, lower a partition, or choose a kernel.

If two logical values have non-overlapping lifetimes, prepare may map them to one reusable memory slot. During run the slot contents change according to the schedule, while their `ValueId` values remain distinct logical identities.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Runtime receives `Operation` or `CompiledNode` | Compile-time graph state leaked into the hot path. | Prepare an executable contract that contains only execution needs. |
| Two concurrent runs share mutable bindings | Per-run state was stored in reusable prepared state. | Isolate each run's `RunState`; exact concurrency guarantees remain to be specified. |
| Runtime searches for a backend | Composition was delayed into execution. | Register and prepare backends before run. |

## Related documentation

- [Runtime API status](../api/runtime-api.md)
- [Lifecycle](../architecture/lifecycle.md)
- [Preparing execution](preparing-execution.md)
