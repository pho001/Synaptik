# Prepare execution

## Outcome

This guide turns a current ordinary `CompiledGraph` into an immutable reusable
`PreparedExecution`. Preparation performs CPU lowering and route selection, declares and assigns
shared resource slots, constructs executable recipes, and validates the schedule. It does not bind
caller inputs or create per-run mutable state.

## Prerequisites

- One open `Engine.standard()`.
- A `CompiledGraph` created by that exact Engine.
- One non-empty maximal CPU-owned partition with supported fully static descriptors.

## Prepare once

```java
CompiledGraph graph = engine.compile(List.of(output));
PreparedExecution prepared = engine.prepare(graph);

assert prepared.compiledGraph() == graph;
```

The returned handle is bound to the exact Engine and graph. It is immutable and may be reused or
shared by concurrent callers because every `run(...)` creates isolated invocation state.

Shared Prepare owns validation, partition-scoped projection, exact shared slot assignment, and
schedule assembly. CPU owns concrete lowering, specialization, route choice, storage geometry,
and executable construction. Runtime receives the completed recipe and does not repeat those
decisions.

## Optional bounded CPU-local autotuning

`engine.prepareTuned(graph, request)` is a separate current path. It consults the current CPU-local
workload handoff, reuses a compatible cache entry or measures bounded candidates, and then returns
a fresh production preparation. The result says explicitly whether it is `TUNED` or an allowed
`SAFE_HEURISTIC_FALLBACK`.

Representative Tensors and their storage remain caller-owned. A strict request fails if tuning
cannot complete; an allowed fallback uses a fresh safe ordinary preparation. This is not generic
multi-occurrence, graph, partition, or complete-plan tuning, and Runtime performs no tuning.

## Expected result

`PreparedExecution` contains reusable recipes and no caller input binding. It is not a serialized
artifact or persistence format. Closing its Engine prevents new runs; the handle itself has no
independent close operation.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Preparation rejects a handle from another Engine | Owner identity is part of the lifecycle contract. | Compile and prepare with the same open Engine. |
| Preparation rejects a zero-node graph | Current CPU composition requires one non-empty maximal partition. | Compile a supported operation such as `input.contiguous()`, not a leaf-only publication. |
| Preparation rejects mixed or multiple partitions | Current composition has one CPU lifecycle adapter and no mixed-owner assembler. | Use a CPU-only graph; Metal/CUDA and mixed-owner execution remain planned. |
| A prepared handle is rebuilt for every run | One-shot and reusable lifecycles were confused. | Retain one prepared handle and call `run(...)` repeatedly. |

## Related documentation

- [Compile a graph](compiling-graphs.md)
- [Run a prepared model](running-models.md)
- [Runtime/Prepare/backend boundary](../architecture/runtime-prepare-backend-boundary.md)
- [Public API status](../api/public-api.md)
