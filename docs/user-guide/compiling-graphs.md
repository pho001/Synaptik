# [Compile](../glossary.md#compile) a [graph](../glossary.md#graph)

## Outcome

This guide compiles current [Tensor](../glossary.md#tensor) expressions into an [immutable](../glossary.md#term-immutable), [Engine](../glossary.md#term-engine)-owned
`CompiledGraph`. [Compilation](../glossary.md#compiler-engine-and-execution-stage-names) [captures](../glossary.md#term-capture-meanings) meaning, validates it, closes eligible fully static
[convolution](../glossary.md#term-channels-first-convolution-layers) [layouts](../glossary.md#layout), chooses [ownership](../glossary.md#term-ownership-meanings) from the Engine's registered [backends](../glossary.md#backend), and produces logical
execution recipes. It does not allocate physical [buffers](../glossary.md#term-buffer) or [run](../glossary.md#term-run) the [graph](../glossary.md#graph).

A **graph** records operations and the dependencies linking their values. **[Capture](../glossary.md#term-capture-meanings)** translates
the existing Tensor expression structure into that graph; it is not a recording of computed
data. A **fully static** [descriptor](../glossary.md#term-descriptor) has every [axis](../glossary.md#term-axis) size known before execution. **Layout**
describes how [coordinates](../glossary.md#axis-rank-scalar-vector-and-matrix) map to storage, so “closes a layout” here means resolves its previously
unknown geometry. A logical **recipe** describes later work without performing it.
The [framework stage diagram](../explanation/framework.md) shows where [compile](../glossary.md#compile) stops and [prepare](../glossary.md#prepare)
and run begin.

## Prerequisites

- [JDK](../glossary.md#term-java-development-kit-jdk) 26 and the repository's `:modules:engine` and `:modules:model` projects.
- One open `Engine.standard()` instance.
- A non-empty [identity](../glossary.md#term-identity-meanings)-unique ordered output list.
- For current [CPU](../glossary.md#term-cpu) execution, supported operations with fully static compatible [descriptors](../glossary.md#term-descriptor).

## [Compile](../glossary.md#compile) forward computation

Assume the Java application created `input`, a Tensor with no producing expression (a **leaf**),
and calls the code below. A static descriptor fixes its type and geometry. The application
retains responsibility for attached host storage, but this compile call reads descriptions,
not input bytes; the leaf could instead be a storage-free prototype. Usable storage must be
supplied at execution. This example compiles one non-empty expression:

```java
try (Engine engine = Engine.standard()) {
    Tensor output = input.contiguous();
    // contiguous() describes a conversion; compile does not perform the copy yet.
    CompiledGraph graph = engine.compile(List.of(output));

    assert graph.inputs().size() == 1;
    assert graph.inputs().getFirst().tensorId().equals(input.id());
}
```

`graph.inputs()` is authoritative after [Compiler](../glossary.md#term-compiler) transformations. Each entry reports the exact
caller Tensor [identity](../glossary.md#term-identity-meanings) and final descriptor that a later `run(...)` must bind. The output order
also defines forward [publication](../glossary.md#term-publication) order.

An input **[binding](../glossary.md#term-binding)** later associates each required entry with usable stored data.
A **publication** is an output [occurrence](../glossary.md#term-occurrence) exposed by a [run](../glossary.md#term-run); two occurrences can refer to a
shared internal value without becoming the same publication. Compile fixes these boundaries,
whereas [prepare](../glossary.md#prepare) creates reusable execution resources and run supplies current values.

## [Compile](../glossary.md#compile) reusable [gradients](../glossary.md#term-gradient)

The reusable ordinary overload takes explicit output-aligned [cotangent](../glossary.md#cotangent) [seeds](../glossary.md#commit-release-distribution-log-and-seed-meanings) and explicit ordered
gradient [targets](../glossary.md#term-target):

A **[cotangent seed](../glossary.md#primal-adjoint-and-differentiation-seed)** assigns an output [sensitivity](../glossary.md#term-sensitivity) to combine when computing [derivatives](../glossary.md#term-derivative).
A **gradient target** names the [Tensor](../glossary.md#tensor) whose sensitivity is requested, not the desired-answer
Tensor used by a training [loss](../glossary.md#loss). The [autograd explanation](autograd.md#compile-an-explicitly-seeded-first-order-graph)<span data-glossary-navigation="true"> ([autograd glossary](../glossary.md#autograd))</span>
introduces the [vector-Jacobian product](../glossary.md#term-vector-jacobian-product-vjp) and why [seed](../glossary.md#term-seed) and output must match.

```java
// Explicit targets identify which input derivatives Compiler must construct.
CompiledGraph graph = engine.compile(
        List.of(output),
        List.of(seed),
        List.of(input));
```

The [Compiler](../glossary.md#term-compiler) owns differentiation and validates seed, target, [operation](../glossary.md#operation), and connectivity
semantics. [Engine](../glossary.md#term-engine) does not infer targets, add Tensor gradient fields, or provide a no-[argument](../glossary.md#term-argument)
backward [lifecycle](../glossary.md#lifecycle). For the narrower fresh [scalar](../glossary.md#term-scalar)-[objective](../glossary.md#term-objective) case, use `Engine.backward(...)`.

Here the application calls Engine, and Engine calls Compiler; “Compiler owns differentiation”
means Compiler defines and constructs the derivative expressions. It does not mean Compiler
adopts the application's input memory. The returned graph fixes identities/descriptors and
input order; [binding current values](running-models.md#run-fresh-values-in-compiled-order)
is a separate run-time action.

## Ordinary versus advanced [compilation](../glossary.md#compiler-engine-and-execution-stage-names)

Ordinary `Engine.compile(...)` uses fixed [compile](../glossary.md#compile) settings. `Engine.standard()` supplies one [CPU](../glossary.md#term-cpu)
[owner](../glossary.md#term-owner); an `Engine.builder()` instance compiles against its frozen explicit CPU/[Metal](../glossary.md#term-metal) inventory.
There is no current `CompileConfig` aggregate [facade](../glossary.md#term-facade). `AdvancedEngine.compile(...)` is the
lower-level integration surface for callers that intentionally own the four standalone compile
settings and transfer one `CpuBackendIntegration` to `AdvancedEngine.takeOwnership(...)`.

Neither surface discovers [backends](../glossary.md#backend) or accepts a public generic [plugin](../glossary.md#term-plugin) [registry](../glossary.md#term-registry). Compilation may
succeed even when preparation later finds a missing registered owner, an unsupported cross-owner
transfer, or a backend-specific unsupported [partition](../glossary.md#partition).

## Expected result

The result is an [immutable](../glossary.md#term-immutable) [owner](../glossary.md#term-owner)-bound handle. Its [metadata](../glossary.md#term-metadata) remains readable after [Engine](../glossary.md#term-engine) closure,
but it cannot be prepared by another Engine or used to start work after its owner begins closing.
It contains no caller [Tensor](../glossary.md#tensor) storage reference and performs no execution.

**Owner-bound** means the handle belongs to that exact Engine instance, not any Engine with the
same configuration. The word “compiled” does not imply that caller data has been read or that
results are cached. For the fixed [FLOAT32](../glossary.md#term-float32) `[2,1]` one-[weight](../glossary.md#term-weight) case, the
[reusable tutorial](../tutorials/reusable-execution.md) demonstrates fresh inputs under one
compiled boundary without changing its mathematical rule.

## Common errors

| Symptom | Likely cause | Fix |
|---|---|---|
| [Compilation](../glossary.md#compiler-engine-and-execution-stage-names) rejects an empty or repeated output list | The public boundary must be non-empty and [identity](../glossary.md#term-identity-meanings)-unique. | Supply each exact requested output [Tensor](../glossary.md#tensor) once. |
| `prepare(...)` rejects a [graph](../glossary.md#graph) that compiled | Preparation requires every planned [owner](../glossary.md#term-owner) to be registered, every cross-owner edge to have an explicit supported transfer, and each [backend](../glossary.md#backend) to accept its own partitions. | Register the required built-ins and use operations and fully static compatible [descriptors](../glossary.md#term-descriptor) in their current domains; [compile](../glossary.md#compile) success alone is not an execution promise. |
| A caller expects compilation to read input bytes | Compile operates on expression meaning and descriptors. | Keep storage live for `run(...)`, not for compilation itself. |
| A caller expects `CompileConfig.auto()` | Config aggregate facades are not current. | Use ordinary fixed `Engine.compile(...)` or the explicitly advanced standalone settings. |


## Numerical semantics and capability

[Compilation](../glossary.md#compiler-engine-and-execution-stage-names) has no [graph](../glossary.md#graph)-wide numerical selector. [Model](../glossary.md#term-model) defines one [operation](../glossary.md#operation)-family and [dtype](../glossary.md#data-type--datatype)
contract; [Planning](../glossary.md#term-planning) queries the actual provider for each complete [occurrence](../glossary.md#term-occurrence). [Metal](../glossary.md#term-metal)'s exact domain
includes its [unary](../glossary.md#term-unary)/affine/[canonicalization](../glossary.md#occurrence-identity-and-canonical-form)/indexing/
[BOOL](../glossary.md#data-type--datatype)/movement/replacement/fold/aggregate, ordering/top-K/numeric arg-[extrema](../glossary.md#saturation-pole-and-extremum), no-[gradient](../glossary.md#term-gradient) promoted
[INT32](../glossary.md#data-type--datatype)/[INT64](../glossary.md#data-type--datatype) [MATMUL](../glossary.md#matrix-multiplication--matmul), maximum-[pooling](../glossary.md#term-pooling), and raw INITIAL_STATE rows. Metal also admits
the documented [FLOAT32](../glossary.md#term-float32) arithmetic/reduction/[scan](../glossary.md#term-cumulative-scan) rows; same-type [canonical](../glossary.md#term-canonical-form) positive-[rank](../glossary.md#term-rank) FLOAT32
[MSE](../glossary.md#mean-squared-error--mse); every positive-static FLOAT32 MATMUL geometry; no-gradient [BFLOAT16](../glossary.md#term-bfloat16)/FLOAT32 mixed MATMUL;
[convolution](../glossary.md#term-channels-first-convolution-layers), average pooling, and FLOAT32 [dropout](../glossary.md#dropout--inverted-dropout); and the no-gradient [rank-one](../glossary.md#axis-rank-scalar-vector-and-matrix) FLOAT32 L1_NORM,
[axis](../glossary.md#term-axis)-zero SCATTER_ADD, and [singleton](../glossary.md#term-singleton) [VARIANCE](../glossary.md#term-variance) custom programs. Metal also admits separate
no-gradient canonical [`EXP`](../glossary.md#term-expx) occurrences for FLOAT32 on direct [MPSGraph](../glossary.md#term-mpsgraph) and for [homogeneous](../glossary.md#term-homogeneous)
BFLOAT16/[FLOAT16](../glossary.md#term-float16) on typed custom steps. Bounded canonical no-gradient FLOAT32 [`SIGMOID`](../glossary.md#term-sigmoid) uses a
sign-guarded composed MPSGraph step; homogeneous BFLOAT16 and FLOAT16 `SIGMOID` use separately
qualified typed custom steps. The current ledger
is 88 admitted kinds and 27 remaining false; counts are over operation kinds, not dtype
occurrences. Every unlisted occurrence fails before route selection; compilation does not infer
provider support from Model semantic reachability or fall back to another [owner](../glossary.md#term-owner).

## Limitations

Current public composition supports fixed [CPU](../glossary.md#term-cpu) execution and explicit CPU/[Metal](../glossary.md#term-metal) mixed-[owner](../glossary.md#term-owner)
execution. [Rank](../glossary.md#term-rank)-0..16 fully static cross-owner values may use all seven [model](../glossary.md#term-model) data types over
[canonical](../glossary.md#term-canonical-form) or positive-[stride](../glossary.md#term-stride) non-overlapping [layouts](../glossary.md#layout) with checked physical spans. Every supported
[homogeneous](../glossary.md#term-homogeneous) [FLOAT32](../glossary.md#term-float32) Metal [occurrence](../glossary.md#term-occurrence) in the preserved former [accelerator](../glossary.md#term-accelerator) baseline has
independently admitted [BFLOAT16](../glossary.md#term-bfloat16) and [FLOAT16](../glossary.md#term-float16) counterparts. Every
[partition](../glossary.md#partition) containing a low-[precision](../glossary.md#term-precision) value uses one fixed native custom-program call with compact
[run](../glossary.md#term-run)-owned materialized slots and direct [targets](../glossary.md#term-target), including exact homogeneous no-[gradient](../glossary.md#term-gradient)
raw-preserving `RESHAPE`, simple `PERMUTE`, materializing [`CONTIGUOUS`](../glossary.md#term-contiguous), [`SLICE`](../glossary.md#slice), `CONCAT`, and
`TILE`. Eligible [linear](../glossary.md#term-linear-meanings) canonical FLOAT32 `FLOOR`/`CEIL`/`SIGN`/`RELU` chains use deterministic
bounded generated units. [ABI](../glossary.md#term-application-binary-interface-abi) 7 exports thirteen symbols and consumes one bounded [schema](../glossary.md#term-schema)-20 program
image with type wires `1..7`, [operation](../glossary.md#operation) wires `1..115`, attributes `0..41`, and route wires `1..3`.
Workload, exact-policy, candidate, compatibility, route-policy, and session-[codec](../glossary.md#term-codec) [identities](../glossary.md#term-identity-meanings) are
version 39; earlier identities, including historical cutover 30 and superseded 31 through 38, [fail
closed](../glossary.md#admission-preflight-and-fail-closed-validation). [Registry](../glossary.md#term-registry) presence does not widen capability; unsupported operations fail closed.

Here **ABI** means [application binary interface](../glossary.md#application-binary-interface-and-native-bridge): the calling and representation agreement
between Java and the [native library](../glossary.md#term-native-library). A **schema** defines the permitted structure of an encoded
program, and **wire** numbers are identifiers in that representation. Versioned [identity](../glossary.md#term-identity-meanings) checks
reject incompatible [artifacts](../glossary.md#term-artifact); the numbers above are compatibility facts, not [parameters](../glossary.md#term-parameter) a
caller should tune to enable missing operations.

Model construction leaves [Conv2d](../glossary.md#convolution--conv2d) and [Conv3d](../glossary.md#conv3d--ncdhw) result layouts unresolved; [Compiler](../glossary.md#term-compiler) closes
only eligible fully static final [convolution](../glossary.md#term-channels-first-convolution-layers)
[descriptors](../glossary.md#term-descriptor) before CPU
[Planning](../glossary.md#term-planning) capability [admission](../glossary.md#term-admission). Dynamic or partially
dynamic convolution results remain unresolved. Qualified static CPU FLOAT32/[FLOAT64](../glossary.md#term-float64) and Metal
FLOAT32 Conv3d execute shared input, [weight](../glossary.md#term-weight) and [bias](../glossary.md#term-bias) gradients, including overlap. This does
not admit ordinary named-B Conv3d or unlock low CPU arithmetic. CPU [Pool3d](../glossary.md#pool3d--ncdhw-pooling) independently
retains its `requiresGrad=false` gate; consult the
[production support matrix](../backend-guide/production-support.md#conv3d-training-and-independent-pool3d-boundary)<span data-glossary-navigation="true"> ([support matrix glossary](../glossary.md#term-review-matrix))</span><span data-glossary-navigation="true"> ([matrix glossary](../glossary.md#term-review-matrix))</span>.

## Related documentation

- [Prepare execution](preparing-execution.md)<span data-glossary-navigation="true"> ([Prepare glossary](../glossary.md#prepare))</span>
- [Run a reusable inference session](running-models.md)<span data-glossary-navigation="true"> ([Run glossary](../glossary.md#term-run); [inference glossary](../glossary.md#term-inference))</span>
- [Public API status](../reference/index.md)<span data-glossary-navigation="true"> ([API glossary](../glossary.md#term-application-programming-interface-api))</span>
- [Lifecycle architecture](../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../glossary.md#lifecycle))</span>
