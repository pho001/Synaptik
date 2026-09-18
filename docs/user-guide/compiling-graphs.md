# Compile a graph

## Outcome

This guide compiles current Tensor expressions into an immutable, Engine-owned
`CompiledGraph`. Compilation captures meaning, validates it, closes eligible fully static
convolution layouts, chooses CPU ownership, and produces logical execution recipes. It does not
allocate physical buffers or run the graph.

## Prerequisites

- JDK 26 and the repository's `:modules:engine` and `:modules:model` projects.
- One open `Engine.standard()` instance.
- A non-empty identity-unique ordered output list.
- For current CPU execution, supported operations with fully static compatible descriptors.

## Compile forward computation

Assume `input` is a caller-owned Tensor leaf with a resolved static descriptor and live host
storage. This current call compiles one non-empty expression:

```java
try (Engine engine = Engine.standard()) {
    Tensor output = input.contiguous();
    CompiledGraph graph = engine.compile(List.of(output));

    assert graph.inputs().size() == 1;
    assert graph.inputs().getFirst().tensorId().equals(input.id());
}
```

`graph.inputs()` is authoritative after Compiler transformations. Each entry reports the exact
caller Tensor identity and final descriptor that a later `run(...)` must bind. The output order
also defines forward publication order.

## Compile reusable gradients

The reusable ordinary overload takes explicit output-aligned cotangent seeds and explicit ordered
gradient targets:

```java
CompiledGraph graph = engine.compile(
        List.of(output),
        List.of(seed),
        List.of(input));
```

The Compiler owns differentiation and validates seed, target, operation, and connectivity
semantics. Engine does not infer targets, add Tensor gradient fields, or provide a no-argument
backward lifecycle. For the narrower fresh scalar-objective case, use `Engine.backward(...)`.

## Ordinary versus advanced compilation

Ordinary `Engine.compile(...)` uses the current fixed settings and CPU-only standard composition.
There is no current `CompileConfig` aggregate facade. `AdvancedEngine.compile(...)` is the
lower-level integration surface for callers that intentionally own the four standalone compile
settings and transfer one `CpuBackendIntegration` to `AdvancedEngine.takeOwnership(...)`.

Neither surface discovers backends, accepts a generic backend registry, or promises Metal, CUDA,
or mixed-owner execution. Compilation may succeed even when current CPU preparation later rejects
the artifact.

## Expected result

The result is an immutable owner-bound handle. Its metadata remains readable after Engine closure,
but it cannot be prepared by another Engine or used to start work after its owner begins closing.
It contains no caller Tensor storage reference and performs no execution.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Compilation rejects an empty or repeated output list | The public boundary must be non-empty and identity-unique. | Supply each exact requested output Tensor once. |
| `prepare(...)` rejects a graph that compiled | Current CPU preparation requires exactly one non-empty maximal CPU partition. | Use supported CPU operations and fully static compatible descriptors; compile success alone is not an execution promise. |
| A caller expects compilation to read input bytes | Compile operates on expression meaning and descriptors. | Keep storage live for `run(...)`, not for compilation itself. |
| A caller expects `CompileConfig.auto()` | Config aggregate facades are not current. | Use ordinary fixed `Engine.compile(...)` or the explicitly advanced standalone settings. |

## Limitations

Current public composition is CPU-only. Model construction leaves Conv2d and Conv3d result layouts
unresolved; Compiler closes only eligible fully static final convolution descriptors before
Planning capability admission. Dynamic or partially dynamic convolution results remain
unresolved. Conv3d forward execution is current, but Conv3d gradients are not.

## Related documentation

- [Prepare execution](preparing-execution.md)
- [Run a prepared model](running-models.md)
- [Public API status](../api/public-api.md)
- [Lifecycle architecture](../architecture/lifecycle.md)
