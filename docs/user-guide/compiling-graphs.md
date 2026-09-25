# Compile a graph

## Outcome

This guide compiles current Tensor expressions into an immutable, Engine-owned
`CompiledGraph`. Compilation captures meaning, validates it, closes eligible fully static
convolution layouts, chooses ownership from the Engine's registered backends, and produces logical
execution recipes. It does not allocate physical buffers or run the graph.

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

Ordinary `Engine.compile(...)` uses fixed compile settings. `Engine.standard()` supplies one CPU
owner; an `Engine.builder()` instance compiles against its frozen explicit CPU/Metal inventory.
There is no current `CompileConfig` aggregate facade. `AdvancedEngine.compile(...)` is the
lower-level integration surface for callers that intentionally own the four standalone compile
settings and transfer one `CpuBackendIntegration` to `AdvancedEngine.takeOwnership(...)`.

Neither surface discovers backends or accepts a public generic plugin registry. Compilation may
succeed even when preparation later finds a missing registered owner, an unsupported cross-owner
transfer, or a backend-specific unsupported partition.

## Expected result

The result is an immutable owner-bound handle. Its metadata remains readable after Engine closure,
but it cannot be prepared by another Engine or used to start work after its owner begins closing.
It contains no caller Tensor storage reference and performs no execution.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| Compilation rejects an empty or repeated output list | The public boundary must be non-empty and identity-unique. | Supply each exact requested output Tensor once. |
| `prepare(...)` rejects a graph that compiled | Preparation requires every planned owner to be registered, every cross-owner edge to have an explicit supported transfer, and each backend to accept its own partitions. | Register the required built-ins and use operations and fully static compatible descriptors in their current domains; compile success alone is not an execution promise. |
| A caller expects compilation to read input bytes | Compile operates on expression meaning and descriptors. | Keep storage live for `run(...)`, not for compilation itself. |
| A caller expects `CompileConfig.auto()` | Config aggregate facades are not current. | Use ordinary fixed `Engine.compile(...)` or the explicitly advanced standalone settings. |


## Numerical profile

Compilation uses the profile captured by the Engine for every capability query and carries it into
preparation. The default is `STRICT_IEEE`. CPU answers the same exact capability matrix under
either profile. Metal's common exact occurrence domain under both profiles contains only canonical
FLOAT32 `NEG`/`ABS`; `RESHAPE`/`EXPAND`/`PERMUTE`/`EXPAND_DIMS`/`SQUEEZE`; `CONTIGUOUS`; bounded
`UNFOLD_AXIS`; exact FLOAT32-data/INT32-index `GATHER`; INT32-to-BOOL `ONE_HOT`; and
FLOAT32/INT32/FLOAT32 `SCATTER_ELEMENTS/NONE`. `ACCELERATOR` additionally admits only tensor
FLOAT32 `ADD`/`SUB`/`MUL`/`DIV`, canonical `SUM`/`MEAN`/`SUM_TO_SHAPE`, and positive static
rank-two `MATMUL`. Strict capability is an accelerator subset, and every other occurrence fails
closed before route selection. Compilation neither changes the requested profile nor falls back
when an occurrence has no eligible owner.

## Limitations

Current public composition supports fixed CPU execution and explicit CPU/Metal mixed-owner
execution. Cross-owner values must be fully static canonical contiguous `FLOAT32`. Both Metal
profiles admit only the common `NEG`/`ABS`, `RESHAPE`/`EXPAND`/`PERMUTE`/`EXPAND_DIMS`/`SQUEEZE`,
`CONTIGUOUS`, `UNFOLD_AXIS`, `GATHER`, `ONE_HOT`, and `SCATTER_ELEMENTS/NONE` families in their
documented exact descriptor domains. Accelerator Metal additionally admits only tensor
`ADD`/`SUB`/`MUL`/`DIV`, canonical
`SUM`/`MEAN`/`SUM_TO_SHAPE`, and positive static rank-two `MATMUL`; strict rejects those additions,
so strict capability remains an accelerator subset. Every other occurrence is fail-closed. No
Blocked operation has a capability or schema row; ABI v4, thirteen exports, node schema 11,
operation wires `1..19`, attribute wires `0..6`, and version-twelve identities remain frozen.
Model construction leaves Conv2d and Conv3d result layouts unresolved; Compiler closes
only eligible fully static final convolution
descriptors before CPU
Planning capability admission. Dynamic or partially
dynamic convolution results remain unresolved. Conv3d forward execution is current on CPU, but
Conv3d gradients are not.

## Related documentation

- [Prepare execution](preparing-execution.md)
- [Run a reusable inference session](running-models.md)
- [Public API status](../api/public-api.md)
- [Lifecycle architecture](../architecture/lifecycle.md)
