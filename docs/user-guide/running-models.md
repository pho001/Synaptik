# Run a prepared model

## Outcome

This guide runs one current CPU `PreparedExecution`, inspects its ordered publication metadata,
and copies a selected publication into a detached immutable `HostTensorValue`.

## Prerequisites

- One open `Engine.standard()`.
- A prepared handle created by that exact Engine.
- Every Tensor listed by `prepared.compiledGraph().inputs()`, each with compatible live
  caller-owned host storage.

## Run and materialize

The input list may use any order because Engine binds by exact Tensor identity:

```java
HostTensorValue retained;
try (RunResult result = engine.run(prepared, List.of(rightInput, leftInput))) {
    RunResult.Publication first = result.publications().getFirst();
    retained = result.materialize(first, 2L * Float.BYTES);
}

assert retained.byteSize() == 2L * Float.BYTES;
```

`run(...)` snapshots the current host-storage associations after complete logical validation. The
caller continues to own those stores and must keep them live, accessible, and free from
conflicting mutation until the result closes.

The `RunResult` is a lease over completed Runtime publications. Its immutable occurrence list is
ordered forward publications followed by gradient publications. Each occurrence authenticates
its owning result and records logical identity, descriptor, role, index, and gradient metadata;
it does not expose backend storage. `materialize(...)` accepts one exact occurrence from that
result and copies its canonical CPU value. Repeated calls and inward aliases produce independent
detached values.

## Reuse and concurrency

Call `engine.run(prepared, inputs)` repeatedly to amortize compile and prepare work. Each call
receives a distinct mutable `RunState`, so concurrent runs do not share bindings or run-owned
buffers. The immutable prepared recipe is shared.

`Engine.compute(...)` and `Engine.backward(...)` are intentionally different: every call performs
fresh compile, prepare, run, complete byte preflight, materialization, and cleanup. Their returned
values are detached, but they do not reuse prepared state.

## Lifetime result

Closing a `RunResult` releases its lease and Engine-created wrappers, never caller storage. A
completed `HostTensorValue` remains readable after the result, Engine, and input-storage arena
close. Publication metadata remains readable after close, but further materialization fails.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| A required Tensor is missing or repeated | The supplied identities do not match `CompiledGraph.inputs()`. | Supply every required Tensor exactly once; order is irrelevant. |
| Storage is dead or inaccessible | The caller closed or exposed its storage incorrectly before result closure. | Keep caller storage valid for the entire result lease. |
| `materialize(...)` rejects a publication | The occurrence belongs to another result, the result is closed, or the byte limit is too small. | Use an occurrence from the same open result and a checked sufficient bound. |
| Runtime appears to select a backend or kernel | A compile/prepare decision leaked into the run path. | Resolve ownership at compile and the route during preparation. |

## Limitations

Current public execution and materialization use the fixed CPU-only composition. There is no
asynchronous run API, implicit transfer API, persistence format, generic device result, or
mixed-backend execution. `HostTensorValue` is a detached canonical host payload, not a Tensor or a
live backend representation.

## Related documentation

- [Compile a graph](compiling-graphs.md)
- [Prepare execution](preparing-execution.md)
- [Public API status](../api/public-api.md)
- [Lifecycle architecture](../architecture/lifecycle.md)
