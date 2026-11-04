# Structural capture and model contracts

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

**Capture** walks Tensor expression dependencies and records operations and their ordered inputs
and outputs as a graph. **Provenance** is the association between a Tensor and the operation
occurrence that produced it. A **port** is one input or output position of that occurrence, not a
memory location; a multi-output operation can therefore have several distinct result ports.

<a id="current-model-contracts"></a>
## Current model contracts

Model type support is independent of backend availability. The temporary Java CPU low-computation lock is historical and superseded by the [CPU low-precision parity cutover](../../architecture/contracts/backend-execution.md#cpu-low-precision-parity-cutover). BFLOAT16, FLOAT16 and both FP8 formats reuse ordinary CPU numerical owners and shared Compiler first-order AD under the exact family, promotion, geometry and resource guards. This does not widen Metal admission; use the [current backend-qualified matrix](../../backend-guide/production-support.md).

The `io.github.pho001.synaptik.model.operation`, `io.github.pho001.synaptik.model.tensor`, and
`io.github.pho001.synaptik.model.graph` packages now provide compiler-neutral data that later
compiler work can produce and consume:

| Current contract | Meaning | Deliberate boundary |
|---|---|---|
| `OperationSignature` | Exact attributes variant and inclusive local input/output-count bounds | Structural occurrence contract, not operand inference or executable support |
| `TensorProducer` | Identity of one pre-capture expression occurrence with exact operation, ordered input Tensors, and ordered output descriptors | Public-expression origin, not `CompiledNode`, graph membership, or compiler state |
| `TensorProvenance` | Exact producer plus one zero-based output index, with derived operation, inputs, and descriptor | Indexed result association, not `NodeId`, `ValueId`, or graph output binding |
| `GraphRngState` | Opaque public wrapper around one explicit key/counter state-expression occurrence | Expression identity and threading boundary, not a generator, exposed numerical Tensor, runtime state, or bitstream |
| `CompiledNode` | One operation occurrence with ordered input/output `ValueId` snapshots | Local list and signature-cardinality validation, not graph-wide closure or descriptor compatibility |
| `CompiledGraphModel` | Immutable ordered graph values, topological nodes, declared input/output boundaries, and exact node phases | Structural graph state, not compiler passes, partitions, storage, or execution |
| `GraphPhase` | Exactly `FORWARD` or `BACKWARD` compile-time node classification | Not a compile mode, optimizer phase, or runtime schedule |
| `ForwardPublicationBinding` | Standalone `TensorId`-to-`ValueId` association used for requested forward outputs inside current compiler publication plans | Not an owning publication plan, a gradient binding, or a `CompiledGraphModel` component |

`CompiledGraphModel` validates structural closure when constructed. It snapshots its lists and
phase map, requires resolvable references and topological node order, enforces producer and phase
coverage rules, and stores no derived indexes. This validation does not capture an expression,
infer descriptors, transform a graph, perform autograd, plan backend ownership, or make the model
executable.

<a id="current-package-private-structural-capture"></a>
### Current package-private structural capture

The compiler module now contains package-private `GraphCapture`. Its package-private static entry
point accepts one non-null, non-empty ordered `List<Tensor>` and returns a
`CompiledGraphModel`. This is an internal compiler implementation boundary, not a public compile
API or a callable lifecycle facade.

Capture traverses exact Tensor and `TensorProducer` references in requested-result order and
producer-input order. A first-seen provenance-free Tensor becomes one graph input. A first-seen
producer is emitted once after its reachable input producers, and every ordered producer output
descriptor becomes a graph value even when no public Tensor wrapper exposes that position.
Produced Tensors resolve through `TensorProvenance.outputIndex()`. Requested graph outputs retain
caller order, while every captured node is classified `GraphPhase.FORWARD`.

`ValueId` and `NodeId` allocation restarts at zero for each call and follows deterministic
depth-first postorder encounter. Capture preserves repeated operand positions, distinct
identity-separated producer occurrences, exact operation and descriptor references, zero-input
semantic source nodes, and reachable opaque RNG-state edges. The returned graph retains none of
the traversed Tensor, producer, provenance, or `GraphRngState` references.

The request rejects a null list, an empty list, a null element, a repeated exact Tensor reference,
or distinct wrappers that resolve to one graph output value. No caller collection is mutated.
The returned graph owns immutable collection snapshots through `CompiledGraphModel`.

An overload accepts a package-private immutable constant-ingress request. Each binding pairs one
exact provenance-free, non-gradient Tensor leaf identity with one exact typed logical splat. A
logical splat means that the same `ScalarValue` applies at every logical coordinate; it contains no
Shape, dense payload, storage, Tensor, buffer, or backend value. Capture maps a binding only at the
existing Tensor-identity-to-`ValueId` boundary and rejects a binding that is not encountered as a
reachable leaf. It does not read `Tensor.hostStorage()` or infer a constant from a factory method,
descriptor, layout, label, storage contents, or the mere absence of provenance.

The result of that overload is an internal immutable sidecar around the unchanged
`CompiledGraphModel`. Every fixed source remains structurally a graph input. For every other
captured leaf, the sidecar records only its immutable `TensorId` against the current graph-local
`ValueId`; it retains no Tensor or provenance. Canonicalization and optimization remap this
association with the graph inputs, and complete compilation projects it in final graph-input
order.
Consequently an explicitly fixed source cannot also be supplied later as a caller input. An
otherwise identical leaf absent from ingress remains bindable, including a leaf created by a
scalar, zero, one, full, import, allocation, or random factory path. There is no public constant-
binding entry point; `GraphCompilationPort` deliberately supplies empty ingress.

The original capture entries remain forward-only. One package-private combined entry receives
forward outputs, ordered target/gradient roles, the exact identity set of original forward
producers, and merged explicit splat ingress. It traverses forward and gradient roots together,
assigns graph-local IDs once, and classifies original producer occurrences as `FORWARD` and
generated occurrences as `BACKWARD`. The graph-output boundary contains the distinct forward
values first, then each gradient value absent from that prefix at its first role occurrence.
Target roles retain boundary ordinals, so two targets can share one gradient `ValueId` without an
identity node.

Capture itself performs no inference, transformation, or derivative-rule selection. The
following package-private boundaries revalidate, canonicalize, and optionally optimize the
captured graph. The original graph-stage entry stops there; the complete entry derives
publication, invokes Planning, and constructs `CompileArtifacts` only after final graph
validation. Neither entry prepares backends, allocates physical memory, or executes values.
`GraphCompiler` and both implementation entries remain package-private. The public compile
surfaces are ordinary `Engine.compile(...)`, which returns an owner-bound `CompiledGraph`, and
`AdvancedEngine.compile(...)`, which returns an owner-bound `AdvancedCompiledGraph`.
`GraphCompilationPort` remains the narrow constant-free integration SPI rather than the sole
public lifecycle facade. Only the bounded cold source capture below is public; no general
validation, optimizer, or compile-configuration aggregate method exists.


---

[Previous topic](entry-points.md) · [Next topic](cold-source.md)
