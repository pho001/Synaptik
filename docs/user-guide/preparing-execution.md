# Prepare execution

## Outcome

This guide turns a current ordinary `CompiledGraph` into an immutable reusable
`PreparedExecution`. Preparation analyzes every backend-owned partition, declares and assigns
shared logical slots plus deterministic owner-indexed physical representations, constructs
executable and transfer recipes, and validates one complete schedule. It does not bind caller
inputs or create per-run mutable state.

## Prerequisites

- One open `Engine.standard()`, or one open explicitly composed CPU/Metal Engine.
- A `CompiledGraph` created by that exact Engine.
- One non-empty partition plan whose owners are registered and whose required directed transfers
  are supported.

## Prepare once

```java
CompiledGraph graph = engine.compile(List.of(output));
try (PreparedExecution prepared = engine.prepare(graph)) {
    assert prepared.compiledGraph() == graph;
    // Reuse prepared with engine.run(...) while the handle remains open.
}
```

The returned handle is bound to the exact Engine and graph. It is immutable and may be reused or
shared by concurrent callers because every `run(...)` creates isolated invocation state.

Shared Prepare owns validation, partition-scoped projection, exact logical slot and
representation-position assignment, contribution coverage, schedule assembly, and rollback. CPU
and Metal own their concrete lowering, specialization, route choice, physical storage geometry,
executable construction, and direct transfer endpoints. Runtime receives the completed recipe and
does not repeat those decisions.

## Prefer an inference session for ordinary repeated runs

`InferenceSession` is the thin user-facing owner for the same prepared execution:

```java
CompiledGraph graph = engine.compile(List.of(output));
try (InferenceSession session = engine.session(graph)) {
    try (RunResult result = session.run(inputs)) {
        // Inspect or materialize this run's ordered publications.
    }
}
```

`engine.session(graph)` performs the same ordinary preparation exactly once and keeps its
`PreparedExecution` private. The session adds no compiler, scheduler, runner, cache, result type,
backend discovery, or per-run lookup. Use it when one fixed graph should run repeatedly. Use the
lower-level standalone prepared handle when direct handle ownership is required, including the
existing bounded autotuning handoff.

## Optional bounded CPU-local autotuning

`engine.prepareTuned(graph, request)` is a separate current bounded two-phase CPU-only path. Phase
1 reuses a compatible cache entry or measures candidates for the current single eligible
representative workload. Phase 2 holds that authenticated decision fixed, checks a bounded set of
complete-plan candidates against exact canonical publication bytes, and then returns one fresh
production preparation. The result says explicitly whether it is `TUNED` or an allowed
`SAFE_HEURISTIC_FALLBACK`.

Representative Tensors and their storage remain caller-owned. A strict request fails if tuning
cannot complete; an allowed fallback uses a fresh safe ordinary preparation. This is not generic
multiple-occurrence extraction, Compiler graph-alternative search, Planning owner or partition
search, or mixed-backend tuning. Runtime performs no tuning.

## Expected result

`PreparedExecution` contains reusable recipes and no caller input binding. It is not a serialized
artifact or persistence format. The handle is explicitly and idempotently closeable, preferably
with try-with-resources. Closing it terminally rejects later runs through that handle but does not
close an already-returned `RunResult`, which retains its own lifecycle. Closing the Engine also
prevents new runs and is the final cleanup boundary for a handle the caller leaves open.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Preparation rejects a handle from another Engine | Owner identity is part of the lifecycle contract. | Compile and prepare with the same open Engine. |
| Preparation rejects a zero-node graph | Current composition requires a non-empty partition plan. | Compile an operation supported by a registered owner, not a leaf-only publication. |
| Preparation rejects mixed-owner transfer | One required edge is outside the exact all-six-carrier rank-0..16 static canonical or positive-stride non-overlapping CPU/Metal transfer domain, or an owner is not registered. | Register both integrations and keep cross-owner values inside the checked physical-span and native-storage boundary; there is no fallback or conversion. |
| A prepared recipe is rebuilt for every run | One-shot and reusable lifecycles were confused. | Open one session for ordinary reuse; retain a standalone prepared handle only when direct ownership is required. |

## Related documentation

- [Compile a graph](compiling-graphs.md)
- [Run a reusable inference session](running-models.md)
- [Runtime/Prepare/backend boundary](../architecture/runtime-prepare-backend-boundary.md)
- [Public API status](../api/public-api.md)
