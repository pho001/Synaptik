# Run a reusable inference session

## Outcome

This guide compiles one fixed inference graph, opens one `InferenceSession`, and runs the same
prepared execution repeatedly with current caller-owned input storage. Each run returns typed,
ordered publication metadata and supports explicit copying into detached immutable
`HostTensorValue` objects.

## Prerequisites

- One open `Engine.standard()`, or one explicitly composed CPU/Metal Engine.
- A non-empty graph supported by the Engine's registered backends.
- Every Tensor reported by `CompiledGraph.inputs()`, with compatible live caller-owned host
  storage when each run begins.

The Engine remains the composition owner. A session borrows it and must not outlive it.

## Compile and prepare once

Compile the requested output boundary once, then open the session:

```java
CompiledGraph graph = engine.compile(List.of(output));
try (InferenceSession session = engine.session(graph)) {
    assert session.compiledGraph() == graph;
    // Reuse session.run(...) while both session and Engine remain open.
}
```

`engine.session(graph)` performs exactly one preparation before returning. It does not execute the
graph or bind input storage. The graph fixes both boundaries:

- `graph.inputs()` is the stable input membership and Compiler occurrence order; and
- every result uses the graph's stable forward-then-gradient publication occurrence order,
  including distinct occurrences that alias one inward representation.

## Run and materialize

The input list may use any order because Engine binds by exact Tensor identity:

```java
HostTensorValue retained;
try (InferenceSession session = engine.session(graph)) {
    try (RunResult result = session.run(List.of(rightInput, leftInput))) {
        RunResult.Publication first = result.publications().getFirst();
        retained = result.materialize(first, 2L * Float.BYTES);
    }

    // Replace a required Tensor's host-storage association here when the next call
    // should use another compatible caller-owned storage.
    try (RunResult result = session.run(List.of(leftInput, rightInput))) {
        assert result.publications().getFirst().index() == 0;
    }
}

assert retained.byteSize() == 2L * Float.BYTES;
```

Each call verifies exact Tensor identity and the complete compiled descriptor, then snapshots the
current host-storage associations in `graph.inputs()` order. It validates storage data type,
resolved-layout capacity, liveness, and current-thread accessibility before borrowing. The caller
continues to own the storage and must keep it usable and free from conflicting mutation until that
run's result closes.

`RunResult` is the existing publication lease. Each `Publication` records its dense index,
`TensorId`, final descriptor, forward or gradient role, and applicable derivative metadata.
`materialize(...)` accepts only the exact occurrence object from that result and performs a fresh
bounded copy through the publication adapter captured during preparation. A detached
`HostTensorValue` remains readable after the result, session, Engine, and input-storage arena close.

## Reuse and concurrency

Sequential and concurrent calls share the session's immutable prepared recipe and direct captured
input/publication adapters. Every call receives a distinct mutable Runtime `RunState`, borrowed
input wrappers, internal buffers, and workspaces. A run performs no compilation, preparation,
provider or availability interrogation, Engine-registry lookup, backend selection, kernel
selection, or schedule assembly.

The session adds no cache and no result type. It delegates to the existing Engine run contract, so
it adds no per-run allocation or copy beyond that contract. `Engine.compute(...)` and
`Engine.backward(...)` make a different lifetime choice: every call performs fresh compilation,
preparation, execution, materialization, and cleanup.

## Close and in-flight runs

Close the session when its reuse window ends. Close is idempotent, closes the one prepared
execution exactly once, and immediately prevents later run admission. It does not close a
`RunResult` that was already returned.

If close races a run, Runtime's existing prepared lease decides the race:

- a run that acquired the lease first may finish; session close does not wait for it; and
- a close-first run is rejected.

Engine admission is checked first. Under an open Engine, the session checks its prepared close
gate before reading the input list, Tensor metadata, storage, or adapters. A run attempted after
session close therefore reports `prepared execution is closed` without borrowing storage even when
the supplied inputs are invalid. If Engine closure has begun, `advanced engine is closed` wins
before any argument inspection.

Engine closure waits for admitted Engine operations, then closes retained results in reverse
successful-run order, sessions and standalone prepared handles in their shared reverse
preparation-publication order, and finally backend integrations. Engine closure is the final
cleanup boundary for a session the caller leaves open.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Session construction rejects a graph | The graph belongs to another Engine, Engine closure began, or cold preparation rejected its plan. | Compile and open the session with the same open Engine; inspect the prepare-time capability failure. |
| A run reports a missing, duplicate, or unexpected input | The supplied identities do not exactly match `graph.inputs()`. | Supply every required Tensor exactly once; list order is arbitrary. |
| A run rejects a descriptor or storage | The Tensor descriptor changed identity expectations, or storage type, capacity, lifetime, or thread accessibility is incompatible. | Use the exact compiled Tensor and attach compatible accessible live storage before the run. |
| A run reports `prepared execution is closed` | Session close won admission. | Open a new session from the still-valid graph and Engine if more runs are needed. |
| Materialization rejects a publication | The occurrence belongs to another result, the result or Engine is closed, or the byte limit is too small. | Use an occurrence from the same open result and a sufficient checked bound. |

## Limitations

Inference sessions are synchronous and fixed to one compiled graph and one prepared execution.
They do not provide asynchronous submission, dynamic input identities or descriptors, implicit
materialization, persistence, generic device-result handles, or tuning controls. Explicit
CPU/Metal composition supports current single-owner execution and the bounded all-seven-carrier
rank-0..16 static canonical or positive-stride non-overlapping transfer domain; unsupported
capability or transfer combinations fail during compile or session construction rather than
falling back during a run.

## Related documentation

- [Compile a graph](compiling-graphs.md)
- [Prepare execution](preparing-execution.md)
- [Runtime API](../api/runtime-api.md#current-ordinary-engine-boundary)
- [Public API status](../api/public-api.md#current-ordinary-explicit-composition-and-advanced-lifecycle)
- [Lifecycle architecture](../architecture/lifecycle.md#current-public-engine-lifecycle)
