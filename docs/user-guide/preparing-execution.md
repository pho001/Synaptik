# [Prepare](../glossary.md#prepare) execution

## Outcome

This guide turns a current ordinary `CompiledGraph` into an [immutable](../glossary.md#term-immutable) reusable
[`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution). Preparation analyzes every [backend](../glossary.md#backend)-owned [partition](../glossary.md#partition), declares and assigns
shared logical slots plus deterministic [owner](../glossary.md#term-owner)-indexed physical representations, constructs
executable and transfer recipes, and validates one complete schedule. It does not bind caller
inputs or create per-[run](../glossary.md#term-run) [mutable](../glossary.md#term-mutable) state.

Think of [compile](../glossary.md#compile) as fixing what to compute and [prepare](../glossary.md#prepare) as making that plan executable.
A **partition** groups operations assigned to one backend. A logical **slot** names a value's
storage need; a physical **representation** specifies how that value will actually be stored.
A **schedule** orders the already-selected computations and transfers. None of these is this
run's input data or an already-computed answer. Use the
[execution-stage visual](../explanation/packages/execution.md) to follow those handoffs.

## Prerequisites

- One open `Engine.standard()`, or one open explicitly composed [CPU](../glossary.md#term-cpu)/[Metal](../glossary.md#term-metal) [Engine](../glossary.md#term-engine).
- A `CompiledGraph` created by that exact Engine.
- One non-empty partition plan whose owners are registered and whose required directed transfers
  are supported.<span data-glossary-navigation="true"> ([owners glossary](../glossary.md#term-owner))</span><span data-glossary-navigation="true"> ([partition glossary](../glossary.md#partition))</span>

## [Prepare](../glossary.md#prepare) once

```java
CompiledGraph graph = engine.compile(List.of(output));
try (PreparedExecution prepared = engine.prepare(graph)) {
    assert prepared.compiledGraph() == graph;
    // Reuse prepared with engine.run(...) while the handle remains open.
}
```

The returned handle is bound to the exact [Engine](../glossary.md#term-engine) and [graph](../glossary.md#graph). It is [immutable](../glossary.md#term-immutable) and may be reused or
shared by concurrent callers because every `run(...)` creates isolated invocation state.

Shared Prepare owns [validation](../glossary.md#term-validation), [partition](../glossary.md#partition)-scoped projection, exact logical slot and
representation-position assignment, contribution coverage, schedule assembly, and [rollback](../glossary.md#term-rollback). [CPU](../glossary.md#term-cpu)
and [Metal](../glossary.md#term-metal) own their concrete [lowering](../glossary.md#lowering), [specialization](../glossary.md#term-specialization), route choice, physical storage geometry,
executable construction, and direct transfer endpoints. [Runtime](../glossary.md#term-runtime) receives the completed recipe and
does not repeat those decisions.

**Lowering** translates operations into [backend](../glossary.md#backend)-specific work. **Specialization** fixes work for
known types and geometry. A **route** is the concrete implementation chosen within a backend.
**Rollback** [releases](../glossary.md#commit-release-distribution-log-and-seed-meanings) partial setup when preparation fails; it does not mean retrying the
calculation on another backend. Preparing once avoids repeating those setup decisions, not
the numerical operations required for each fresh [batch](../glossary.md#term-batch-machine-learning-and-hyperparameters).

<a id="prefer-an-inference-session-for-ordinary-repeated-runs"></a>

## Reuse the prepared [owner](../glossary.md#term-owner)

Use the same closeable prepared handle for ordinary repeated runs and bounded tuning [handoff](../glossary.md#term-handoff).
`Engine.run(prepared, inputs)` accepts the exact compiled [Tensor](../glossary.md#tensor) [identities](../glossary.md#term-identity-meanings) in arbitrary order;
`Engine.runValues(prepared, values)` accepts fresh fully static exact-[descriptor](../glossary.md#term-descriptor) values in final
`graph.inputs()` order. Neither method compiles or prepares again. See the
[fresh-value example](running-models.md#run-fresh-values-in-compiled-order).

[`InferenceSession`](../glossary.md#former-inference-session--inferencesession) and `Engine.session` were removed. Replace their factory with `Engine.prepare`
and each session [run](../glossary.md#term-run) with `engine.run(prepared, inputs)`, then recompile; there is no shim.

## Optional bounded single-[owner](../glossary.md#term-owner) [autotuning](../glossary.md#autotuning--model-autotuning)

`engine.prepareTuned(graph, request)` is a separate bounded two-phase path for one eligible [CPU](../glossary.md#term-cpu)
[partition](../glossary.md#partition) or an eligible [singleton](../glossary.md#term-singleton) [FLOAT32](../glossary.md#term-float32) [Metal](../glossary.md#term-metal) NEG partition. Metal requires both custom-[kernel](../glossary.md#kernel)
and [MPSGraph](../glossary.md#term-mpsgraph) candidates; an MPSGraph-only plan is not tunable. Phase 1 reuses a compatible [cache](../glossary.md#term-cache)
entry where the [producer](../glossary.md#term-producer) permits [persistence](../glossary.md#term-persistence), or measures the single representative workload.
Metal decisions are session-only and use neither persistent cache. Phase 2 holds the authenticated decision fixed, checks a bounded set of
complete-plan candidates against exact [canonical](../glossary.md#term-canonical-form) [publication](../glossary.md#term-publication) bytes, and then returns one fresh
production preparation. The result says explicitly whether it is `TUNED` or an allowed
`SAFE_HEURISTIC_FALLBACK`.

Representative [Tensors](../glossary.md#tensor) and their storage remain caller-owned. A strict request fails if tuning
cannot complete; an allowed fallback uses a fresh safe ordinary preparation through the same
already-selected [backend](../glossary.md#backend). This is not generic
multiple-[occurrence](../glossary.md#term-occurrence) extraction, [Compiler](../glossary.md#term-compiler) [graph](../glossary.md#graph)-alternative search, [Planning](../glossary.md#term-planning) owner or partition
search, or mixed-backend tuning. [Runtime](../glossary.md#term-runtime) performs no tuning.

A **representative workload** is concrete input chosen to stand for later work during timing.
A **candidate** is an allowed implementation being compared. A **cache** can reuse a compatible
decision; it is not a saved [prediction](../glossary.md#term-prediction) for future data. **Canonical publication bytes** are
outputs copied into the standard comparison representation. A **[heuristic](../glossary.md#term-heuristic) fallback** uses a
permitted rule-selected ordinary preparation when the request allows it, within the same owner.
These are optional tuning concepts, not prerequisites for ordinary [`prepare`](../glossary.md#prepare).

## Expected result

[`PreparedExecution`](../glossary.md#prepared-execution--preparedexecution) contains reusable recipes and no caller input [binding](../glossary.md#term-binding). It is not a serialized
[artifact](../glossary.md#term-artifact) or [persistence](../glossary.md#term-persistence) format. The handle is explicitly and idempotently closeable, preferably
with try-with-resources. Closing it terminally rejects later runs through that handle but does not
close an already-returned [`RunResult`](../glossary.md#run-result--runresult), which retains its own [lifecycle](../glossary.md#lifecycle). Closing the [Engine](../glossary.md#term-engine) also
prevents new runs and is the final cleanup boundary for a handle the caller leaves open.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Preparation rejects a handle from another [Engine](../glossary.md#term-engine) | [Owner](../glossary.md#term-owner) [identity](../glossary.md#term-identity-meanings) is part of the [lifecycle](../glossary.md#lifecycle) contract. | [Compile](../glossary.md#compile) and [prepare](../glossary.md#prepare) with the same open Engine. |
| Preparation rejects a zero-[node](../glossary.md#node) [graph](../glossary.md#graph) | Current composition requires a non-empty [partition](../glossary.md#partition) plan. | Compile an [operation](../glossary.md#operation) supported by a registered owner, not a [leaf](../glossary.md#term-leaf)-only [publication](../glossary.md#term-publication). |
| Preparation rejects mixed-owner transfer | One required edge is outside the exact all-seven-[carrier](../glossary.md#term-storage-carrier) [rank](../glossary.md#term-rank)-0..16 static [canonical](../glossary.md#term-canonical-form) or positive-[stride](../glossary.md#term-stride) non-overlapping [CPU](../glossary.md#term-cpu)/[Metal](../glossary.md#term-metal) transfer domain, or an owner is not registered. | Register both integrations and keep cross-owner values inside the checked physical-span and native-storage boundary; there is no fallback or conversion. |
| A prepared recipe is rebuilt for every [run](../glossary.md#term-run) | One-shot and reusable lifecycles were confused. | Retain one prepared handle for reuse. |

## Related documentation

- [Compile a graph](compiling-graphs.md)<span data-glossary-navigation="true"> ([Compile glossary](../glossary.md#compile); [graph glossary](../glossary.md#graph))</span>
- [Run a prepared graph](running-models.md)<span data-glossary-navigation="true"> ([Run glossary](../glossary.md#term-run))</span>
- [Runtime/Prepare/backend boundary](../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../glossary.md#term-runtime); [Prepare glossary](../glossary.md#prepare); [backend glossary](../glossary.md#backend))</span>
- [Public API status](../reference/index.md)<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span>
