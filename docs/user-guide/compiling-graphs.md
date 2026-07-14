# Compile a graph (planned workflow)

## Outcome

This guide explains what graph compilation will produce and how to interpret it. Public `Tensor`
expression construction and the standalone `BackendIntent` configuration value are current. The
compiler, planning, `CompileConfig`, and engine APIs remain planned, so no runnable compile command
exists yet.

The current intent value can record no hard target or retain one current `BackendRequirement`:

```java
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.backend.contract.BackendIdRequirement;
import io.github.pho001.synaptik.config.compile.BackendIntent;

BackendIntent unconstrained = BackendIntent.unconstrained();
BackendIntent requireCpu =
        BackendIntent.requiring(
                new BackendIdRequirement(new BackendId("cpu")));
```

These values are runnable metadata construction, but no current compile entry point accepts them.
`unconstrained` promises neither default selection nor fallback, and `requireCpu` does not verify
that CPU is available or capable.

## Planned steps

1. Build a public tensor expression. Provenance on public tensor state will let graph capture discover producers and inputs without turning `Tensor` into an intermediate-representation node.
2. Choose declarative compile configuration, including the current backend intent and later
   compile mode, optimization, scoring, and profile values.
3. Compile the requested output. The compiler will capture, infer, validate, optimize, optionally expand automatic differentiation, and coordinate backend-neutral planning.
4. Inspect immutable compile artifacts and typed diagnostics.

```java
// Conceptual API; not currently runnable.
CompiledGraph graph = CompiledGraph.compile(output, CompileConfig.auto());
```

## Expected result

Compilation will produce an immutable recipe: a graph model, partitions assigned to backend identities, logical memory requirements, publication bindings, and diagnostics. It will not create physical buffers or choose a CPU, Metal, or CUDA kernel.

For example, a partition may record `owner = CPU`. That means CPU preparation is responsible for it; it does not mean compilation selected scalar, Vector API, or OpenBLAS execution.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| The conceptual classes cannot be imported | The compile lifecycle is planned. | Follow the [roadmap](../planning/roadmap.md) and do not create substitute APIs in another module. |
| A current `BackendIntent` cannot be passed to `CompileConfig` | The aggregate and compiler consumer remain planned. | Keep the intent as declarative metadata until the config and compiler tasks provide that path. |
| A design puts buffers in compile artifacts | Compile-time and prepared state were mixed. | Keep allocation in prepare/backend/runtime layers. |
| A planner chooses a kernel | Ownership and implementation selection were mixed. | Let planning choose a backend identity and backend prepare choose the route. |

## Related documentation

- [Compile API status](../api/compile-api.md)
- [Lifecycle](../architecture/lifecycle.md)
- [Partition scoring](../architecture/partition-scoring.md)
