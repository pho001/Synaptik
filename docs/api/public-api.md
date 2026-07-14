# Public API status

## Purpose and status

This page identifies which public contracts a caller can use today and which names are architecture-level plans. It prevents conceptual lifecycle examples from being mistaken for released Java APIs.

Synaptik has no published compatibility guarantee yet. The current implementation contains the
selected public model foundation, tensor-expression metadata surface, and common trace-event
envelope. Compiler, prepare, runtime, backend, and engine APIs remain planned. APIs may change
through the ordered planning process. [`ARCHITECTURE.md`](../../ARCHITECTURE.md) defines module
boundaries, not source or binary compatibility.

## Current public contracts

The implemented `modules:model` surface contains:

- data type metadata, typed scalar values, and numeric promotion;
- BFLOAT16 scalar bit conversion;
- static, named dynamic, and expression dimensions, immutable Shapes, and local broadcasting;
- resolved static layout geometry and host-storage contracts;
- public mutable `Tensor`, eager leaf factories, explicit-source random construction, and
  backend-independent expression metadata;
- typed operation attributes and occurrence signatures, shared multi-output producer provenance,
  and operation-specific result carriers; and
- immutable graph values, nodes, compiled graph-model data, publication bindings, and distinct
  tensor/node/value identifiers.

The [Tensor API reference](tensor-api.md) documents these current contracts, inputs, results,
failures, and examples. The current model surface records meaning and metadata; it does not imply
compiler capture, backend support, kernels, prepared execution, runtime residency, or numerical
execution.

The implemented `modules:trace` surface contains:

- producer-assigned non-negative `TraceEventId` values;
- `TracePhase` lifecycle classification for `COMPILE`, `PREPARE`, and `RUN`;
- `TraceLevel` detail and severity classification;
- the open method-free `TracePayload` marker; and
- the generic `TraceEvent<T extends TracePayload>` envelope with a producer-supplied monotonic
  nanosecond reading.

The [tracing explanation](../architecture/tracing.md) documents the envelope semantics and
ownership boundaries. Concrete payload families, trace-local correlation IDs beyond the event
ID, typed backend attributes, serialization, sinks, and emission remain planned. Backend is a
payload family and producer role, not another lifecycle phase.

## Planned public lifecycle

The architecture uses this conceptual shape:

```java
// Conceptual API: these lifecycle types and methods are not implemented yet.
CompiledGraph graph = CompiledGraph.compile(output, CompileConfig.auto());
PreparedExecution execution = graph.prepare(PrepareConfig.defaults());
RunResult result = execution.run(inputs, RunOptions.defaults());
```

Compile will create immutable graph and ownership artifacts. Prepare will ask explicitly registered concrete backends to lower their assigned partitions. Run will execute the prepared schedule with per-invocation state. See the [compile](compile-api.md) and [runtime](runtime-api.md) reference pages for the planned boundaries.

## Compatibility expectations during development

- Treat Javadoc and implemented tests as the contract for code that exists.
- Treat architecture snippets as conceptual unless a reference page explicitly marks them current.
- Do not depend on draft planning types or package names in external code.
- Check the [roadmap](../planning/roadmap.md) before assuming a planned module is available.

## Related documentation

- [Getting started](../getting-started.md)
- [Architecture overview](../architecture/overview.md)
- [Implementation roadmap](../planning/roadmap.md)
