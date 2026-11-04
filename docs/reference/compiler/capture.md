# [Structural capture](../../glossary.md#term-structural-capture) and [model](../../glossary.md#term-model) contracts

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

**[Capture](../../glossary.md#term-capture-meanings)** walks [Tensor](../../glossary.md#tensor) expression dependencies and records operations and their ordered inputs
and outputs as a [graph](../../glossary.md#graph). **[Provenance](../../glossary.md#provenance)** is the association between a Tensor and the [operation](../../glossary.md#operation)
[occurrence](../../glossary.md#term-occurrence) that produced it. A **port** is one input or output position of that occurrence, not a
memory location; a multi-output operation can therefore have several distinct result ports.

## Why a leaf is not necessarily an array

The compiler calls `GraphCapture` with the requested output Tensors. Capture walks each Tensor's
producing expression; when a Tensor has no producer, it is a **leaf** and becomes a graph input
unless explicitly classified as a compiler constant. That leaf may have host storage, or may be
a storage-free prototype describing a future input. Capture inspects neither possibility's bytes.
Its graph uses local value/node IDs and no longer retains the original Tensor objects. The
separate input-ID association lets Engine match later stored inputs without owning the original
storage. Read [cold capture](cold-source.md) for the special API that intentionally also returns
leaf references, and [ordinary input binding](../runtime/engine-boundary.md) for actual value use.

<a id="current-model-contracts"></a>
## Current [model](../../glossary.md#term-model) contracts

Model type support is independent of [backend](../../glossary.md#backend) availability. The temporary Java [CPU](../../glossary.md#term-cpu) low-computation lock is historical and superseded by the [CPU low-precision parity cutover](../../architecture/contracts/backend-execution.md#cpu-low-precision-parity-cutover). [BFLOAT16](../../glossary.md#term-bfloat16), [FLOAT16](../../glossary.md#term-float16) and both [FP8](../../glossary.md#term-fp8) formats reuse ordinary CPU numerical [owners](../../glossary.md#term-owner) and shared [Compiler](../../glossary.md#term-compiler) first-order [AD](../../glossary.md#autograd) under the exact family, promotion, geometry and resource guards. This does not widen [Metal](../../glossary.md#term-metal) [admission](../../glossary.md#term-admission); use the [current backend-qualified matrix](../../backend-guide/production-support.md).

The `io.github.pho001.synaptik.model.operation`, `io.github.pho001.synaptik.model.tensor`, and
`io.github.pho001.synaptik.model.graph` packages now provide compiler-neutral data that later
compiler work can produce and consume:

| Current contract | Meaning | Deliberate boundary |
|---|---|---|
| `OperationSignature` | Exact attributes variant and inclusive local input/output-count bounds | Structural [occurrence](../../glossary.md#term-occurrence) contract, not operand [inference](../../glossary.md#term-inference) or executable support |
| [`TensorProducer`](../../glossary.md#producer--tensorproducer) | [Identity](../../glossary.md#term-identity-meanings) of one pre-[capture](../../glossary.md#term-capture-meanings) expression occurrence with exact [operation](../../glossary.md#operation), ordered input [Tensors](../../glossary.md#tensor), and ordered output [descriptors](../../glossary.md#term-descriptor) | Public-expression origin, not `CompiledNode`, [graph](../../glossary.md#graph) membership, or compiler state |
| `TensorProvenance` | Exact [producer](../../glossary.md#term-producer) plus one zero-based output index, with derived operation, inputs, and descriptor | Indexed result association, not [`NodeId`](../../glossary.md#nodeid), [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot), or graph output [binding](../../glossary.md#term-binding) |
| [`GraphRngState`](../../glossary.md#graph-rng-state--graphrngstate) | [Opaque](../../glossary.md#term-opaque) public wrapper around one explicit key/counter state-expression occurrence | Expression identity and threading boundary, not a generator, exposed numerical Tensor, [runtime](../../glossary.md#term-runtime) state, or bitstream |
| `CompiledNode` | One operation occurrence with ordered input/output `ValueId` snapshots | Local list and [signature](../../glossary.md#term-signature)-[cardinality](../../glossary.md#term-cardinality) [validation](../../glossary.md#term-validation), not graph-wide closure or descriptor compatibility |
| [`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel) | [Immutable](../../glossary.md#term-immutable) ordered graph values, [topological](../../glossary.md#term-topological-order) [nodes](../../glossary.md#node), declared input/output boundaries, and exact node phases | Structural graph state, not compiler passes, partitions, storage, or execution |
| `GraphPhase` | Exactly `FORWARD` or `BACKWARD` [compile](../../glossary.md#compile)-time node classification | Not a [compile mode](../../glossary.md#compile-mode--compilemode), [optimizer](../../glossary.md#term-optimizer) phase, or runtime schedule |
| `ForwardPublicationBinding` | Standalone [`TensorId`](../../glossary.md#tensorid)-to-`ValueId` association used for requested forward outputs inside current compiler [publication](../../glossary.md#term-publication) plans | Not an owning [publication plan](../../glossary.md#publication-plan), a [gradient](../../glossary.md#term-gradient) binding, or a `CompiledGraphModel` component |

`CompiledGraphModel` validates structural closure when constructed. It snapshots its lists and
phase map, requires resolvable references and topological node order, enforces producer and phase
coverage rules, and stores no derived indexes. This validation does not capture an expression,
infer descriptors, transform a graph, perform [autograd](../../glossary.md#autograd), plan [backend ownership](../../glossary.md#backend-ownership), or make the model
executable.

<a id="current-package-private-structural-capture"></a>
### Current package-private [structural capture](../../glossary.md#term-structural-capture)

The [compiler](../../glossary.md#term-compiler) [module](../../glossary.md#term-module) now contains package-private `GraphCapture`. Its package-private static entry
point accepts one non-null, non-empty ordered `List<Tensor>` and returns a
[`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel). This is an internal compiler implementation boundary, not a public [compile](../../glossary.md#compile)
[API](../../glossary.md#term-application-programming-interface-api) or a callable [lifecycle](../../glossary.md#lifecycle) [facade](../../glossary.md#term-facade).

[Capture](../../glossary.md#term-capture-meanings) traverses exact [Tensor](../../glossary.md#tensor) and [`TensorProducer`](../../glossary.md#producer--tensorproducer) references in requested-result order and
[producer](../../glossary.md#term-producer)-input order. A first-seen [provenance](../../glossary.md#provenance)-free Tensor becomes one [graph](../../glossary.md#graph) input. A first-seen
producer is emitted once after its reachable input producers, and every ordered producer output
[descriptor](../../glossary.md#term-descriptor) becomes a [graph value](../../glossary.md#graph-value) even when no public Tensor wrapper exposes that position.
Produced Tensors resolve through `TensorProvenance.outputIndex()`. Requested graph outputs retain
caller order, while every captured [node](../../glossary.md#node) is classified `GraphPhase.FORWARD`.

[`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) and [`NodeId`](../../glossary.md#nodeid) allocation restarts at zero for each call and follows deterministic
depth-first postorder encounter. Capture preserves repeated operand positions, distinct
[identity](../../glossary.md#term-identity-meanings)-separated producer occurrences, exact [operation](../../glossary.md#operation) and descriptor references, zero-input
semantic source nodes, and reachable [opaque](../../glossary.md#term-opaque) [RNG](../../glossary.md#graph-rng-state--graphrngstate)-state edges. The returned graph retains none of
the traversed Tensor, producer, provenance, or [`GraphRngState`](../../glossary.md#graph-rng-state--graphrngstate) references.

The request rejects a null list, an empty list, a null [element](../../glossary.md#term-element), a repeated exact Tensor reference,
or distinct wrappers that resolve to one graph output value. No caller collection is mutated.
The returned graph owns [immutable](../../glossary.md#term-immutable) collection snapshots through `CompiledGraphModel`.

An overload accepts a package-private immutable [constant](../../glossary.md#term-constant)-ingress request. Each [binding](../../glossary.md#term-binding) pairs one
exact provenance-free, non-[gradient](../../glossary.md#term-gradient) Tensor [leaf](../../glossary.md#term-leaf) identity with one exact typed [logical splat](../../glossary.md#logical-splat). A
logical splat means that the same [`ScalarValue`](../../glossary.md#typed-scalar-value--scalarvalue) applies at every logical [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix); it contains no
[Shape](../../glossary.md#shape), dense [payload](../../glossary.md#artifact-envelope-and-atomic-publication), storage, Tensor, [buffer](../../glossary.md#term-buffer), or [backend](../../glossary.md#backend) value. Capture maps a binding only at the
existing Tensor-identity-to-`ValueId` boundary and rejects a binding that is not encountered as a
reachable leaf. It does not read `Tensor.hostStorage()` or infer a constant from a factory method,
descriptor, [layout](../../glossary.md#layout), label, storage contents, or the mere absence of provenance.

The result of that overload is an internal immutable sidecar around the unchanged
`CompiledGraphModel`. Every fixed source remains structurally a graph input. For every other
captured leaf, the sidecar records only its immutable [`TensorId`](../../glossary.md#tensorid) against the current graph-local
`ValueId`; it retains no Tensor or provenance. [Canonicalization](../../glossary.md#occurrence-identity-and-canonical-form) and optimization remap this
association with the graph inputs, and complete [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) projects it in final graph-input
order.
Consequently an explicitly fixed source cannot also be supplied later as a caller input. An
otherwise identical leaf absent from ingress remains bindable, including a leaf created by a
[scalar](../../glossary.md#term-scalar), zero, one, full, import, allocation, or random factory path. There is no public constant-
binding entry point; `GraphCompilationPort` deliberately supplies empty ingress.

The original capture entries remain forward-only. One package-private combined entry receives
forward outputs, ordered [target](../../glossary.md#term-target)/gradient roles, the exact identity set of original forward
producers, and merged explicit splat ingress. It traverses forward and gradient roots together,
assigns graph-local IDs once, and classifies original producer occurrences as `FORWARD` and
generated occurrences as `BACKWARD`. The graph-output boundary contains the distinct forward
values first, then each gradient value absent from that prefix at its first role [occurrence](../../glossary.md#term-occurrence).
Target roles retain boundary ordinals, so two targets can share one gradient `ValueId` without an
identity node.

Capture itself performs no [inference](../../glossary.md#term-inference), transformation, or [derivative](../../glossary.md#term-derivative)-rule selection. The
following package-private boundaries revalidate, canonicalize, and optionally optimize the
captured graph. The original graph-stage entry stops there; the complete entry derives
[publication](../../glossary.md#term-publication), invokes [Planning](../../glossary.md#term-planning), and constructs `CompileArtifacts` only after final graph
[validation](../../glossary.md#term-validation). Neither entry prepares backends, allocates physical memory, or executes values.
`GraphCompiler` and both implementation entries remain package-private. The public compile
surfaces are ordinary `Engine.compile(...)`, which returns an [owner](../../glossary.md#term-owner)-bound `CompiledGraph`, and
`AdvancedEngine.compile(...)`, which returns an owner-bound `AdvancedCompiledGraph`.
`GraphCompilationPort` remains the narrow constant-free integration [SPI](../../glossary.md#term-service-provider-interface-spi) rather than the sole
public lifecycle facade. Only the bounded cold source capture below is public; no general
validation, [optimizer](../../glossary.md#term-optimizer), or compile-configuration aggregate method exists.


---

[Previous topic](entry-points.md) · [Next topic](cold-source.md)
