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


## Numerical semantics and capability

Compilation has no graph-wide numerical selector. Model defines one operation-family and dtype
contract; Planning queries the actual provider for each complete occurrence. Metal's exact domain
includes its unary/affine/canonicalization/indexing/
BOOL/movement/replacement/fold/aggregate, ordering/top-K/numeric arg-extrema, no-gradient promoted
INT32/INT64 MATMUL, maximum-pooling, and raw INITIAL_STATE rows. Metal also admits
the documented FLOAT32 arithmetic/reduction/scan rows; same-type canonical positive-rank FLOAT32
MSE; every positive-static FLOAT32 MATMUL geometry; no-gradient BFLOAT16/FLOAT32 mixed MATMUL;
convolution, average pooling, and FLOAT32 dropout; and the no-gradient rank-one FLOAT32 L1_NORM,
axis-zero SCATTER_ADD, and singleton VARIANCE custom programs. Metal also admits separate
no-gradient canonical `EXP` occurrences for FLOAT32 on direct MPSGraph and for homogeneous
BFLOAT16/FLOAT16 on typed custom steps. Bounded canonical no-gradient FLOAT32 `SIGMOID` uses a
sign-guarded composed MPSGraph step; low-valued `SIGMOID` remains unsupported. The current ledger
is 88 admitted kinds and 27 remaining false; counts are over operation kinds, not dtype
occurrences. Every unlisted occurrence fails before route selection; compilation does not infer
provider support from Model semantic reachability or fall back to another owner.

## Limitations

Current public composition supports fixed CPU execution and explicit CPU/Metal mixed-owner
execution. Rank-0..16 fully static cross-owner values may use all seven model data types over
canonical or positive-stride non-overlapping layouts with checked physical spans. Every supported
homogeneous FLOAT32 Metal occurrence in the preserved former accelerator baseline has
independently admitted BFLOAT16 and FLOAT16 counterparts. Every
partition containing a low-precision value uses one fixed native custom-program call with compact
run-owned materialized slots and direct targets, including exact homogeneous no-gradient
raw-preserving `RESHAPE`, simple `PERMUTE`, materializing `CONTIGUOUS`, `SLICE`, `CONCAT`, and
`TILE`. Eligible linear canonical FLOAT32 `FLOOR`/`CEIL`/`SIGN`/`RELU` chains use deterministic
bounded generated units. ABI 7 exports thirteen symbols and consumes one bounded schema-20 program
image with type wires `1..7`, operation wires `1..115`, attributes `0..41`, and route wires `1..3`.
Workload, exact-policy, candidate, compatibility, route-policy, and session-codec identities are
version 33; earlier identities, including historical cutover 30 and superseded 31 and 32, fail
closed. Registry presence does not widen capability; unsupported operations fail closed.

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
