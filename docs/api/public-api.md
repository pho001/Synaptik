# Public API status

## Purpose and status

This page identifies which public contracts a caller can use today and which names are architecture-level plans. It prevents conceptual lifecycle examples from being mistaken for released Java APIs.

Synaptik has no published compatibility guarantee yet. The current implementation is an early model foundation, and APIs may change through the ordered planning process. [`ARCHITECTURE.md`](../../ARCHITECTURE.md) defines module boundaries, not source or binary compatibility.

## Current public contracts

The implemented `modules:model` surface contains:

- data type metadata and floating-point promotion;
- BFLOAT16 scalar bit conversion;
- static and symbolic dimensions, immutable shapes, and local broadcasting;
- resolved static layout geometry; and
- distinct tensor, graph-node, and graph-value identifiers.

The [Tensor API reference](tensor-api.md) documents these contracts, inputs, results, failures, and examples. Despite that page's name, a public mutable `Tensor` class is not implemented yet.

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
