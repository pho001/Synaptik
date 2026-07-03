# Compile API

## Purpose and implementation status

This reference defines the planned public boundary for turning a tensor expression into immutable compile artifacts. The compiler, configuration, planning, and engine modules are not implemented yet, so all Java names on this page are conceptual unless linked from the current [Tensor API](tensor-api.md).

Compile answers two questions: what the computation means, and which backend identity owns each planned region. It does not create physical buffers, choose concrete kernels, or construct prepared executables.

## Planned inputs and output

Conceptually, compilation receives a requested tensor output and declarative `CompileConfig`:

```java
// Conceptual API; not currently runnable.
CompiledGraph graph = CompiledGraph.compile(output, CompileConfig.auto());
```

- `output` will identify the public tensor expression to capture. A public `Tensor` is planned, not implemented.
- `CompileConfig` will describe compile mode, backend intent, optimization, scoring, and publication policy as data. It will not contain live backend services.
- `CompiledGraph` will be an engine facade over immutable `CompileArtifacts`, not the same object as `CompiledGraphModel`.

The planned artifacts contain an immutable graph model, planned partitions with backend identities, a logical memory plan, publication bindings, and diagnostics. They deliberately contain no device buffers, backend executable objects, runtime residency, or mutable run state.

## Planned lifecycle and failures

```text
expression -> capture -> inference and validation -> optimization
           -> optional autograd -> backend ownership -> logical plans
           -> CompileArtifacts
```

Compilation is expected to reject invalid shapes, data types, operations, graph structure, or unsatisfied backend capabilities before preparation. Exact exception types and callable signatures remain to be specified by the compiler and engine tasks; callers must not code against invented exceptions from this conceptual page.

## Example interpretation

If a future graph contains a matrix multiplication followed by a small elementwise operation, capability analysis may find both CPU and Metal valid. Backend-neutral scoring may assign both nodes to Metal to avoid a transfer boundary. The artifact records only `owner = Metal`; it does not record MPSGraph or a custom Metal kernel. Metal prepare makes that later choice.

## Related contracts

- [Lifecycle](../architecture/lifecycle.md)
- [Partition scoring](../architecture/partition-scoring.md)
- [Compiling graphs user guide](../user-guide/compiling-graphs.md)
- [Roadmap](../planning/roadmap.md)
