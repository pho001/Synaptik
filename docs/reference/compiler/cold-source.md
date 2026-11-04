# Cold source [capture](../../glossary.md#term-structural-capture) and reconstruction

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

**Cold source** is a computation description that can be inspected and rebuilt without running
its numeric operations. Reconstruction uses compatible replacement inputs to recreate ordinary
[Tensor](../../glossary.md#tensor) expressions; it is not loading a compiled executable. The raw source preserves construction
structure, whereas complete [compilation](../../glossary.md#compiler-engine-and-execution-stage-names) may optimize it and complete logical [layout](../../glossary.md#layout) [metadata](../../glossary.md#term-metadata).

## Which references cross the cold API?

Checkpoint or another integration caller supplies output expressions to `captureSource` and
receives both immutable graph metadata and borrowed original leaf references. This leaf list is
not a detached snapshot of bytes. When that caller supplies replacement Tensors to
`reconstructSource`, Compiler validates their descriptors and builds new producer objects that
reference them. It copies no stored values, creates no Arena and needs nothing to close. A
replacement can itself be a produced expression; its dependencies remain attached for later
compilation. [GraphCheckpoint.copy](../checkpoint/graph.md#inputs-outputs-and-identity), not this
Compiler constructor, is responsible for acquiring private stored state.

<a id="current-cold-source-capture-and-reconstruction"></a>
### Current cold source [capture](../../glossary.md#term-structural-capture) and reconstruction

`GraphCompilationPort.captureSource(List<Tensor>)` returns `SourceCapture(graph, leaves)`: a
validated **raw** [`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel) and an [immutable](../../glossary.md#term-immutable) [snapshot](../../glossary.md#term-snapshot) of exact [provenance](../../glossary.md#provenance)-free [leaves](../../glossary.md#term-leaf).
The existing traversal collects each leaf when it assigns that input's ID, so leaf position and
`graph.inputs()` position agree even with sharing or equal-looking distinct [Tensors](../../glossary.md#tensor). Requested
output order and distinct [producer](../../glossary.md#term-producer) occurrences survive. Empty, null or duplicate output requests
retain ordinary [capture](../../glossary.md#term-capture-meanings) rejection. Direct record construction also checks source [admission](../../glossary.md#term-admission),
input count/order, full [descriptors](../../glossary.md#term-descriptor), provenance absence and unique exact leaf references.

`GraphCompilationPort.validateSource(CompiledGraphModel)` runs the same complete [metadata](../../glossary.md#term-metadata)-only
cold [preflight](../../glossary.md#term-preflight) without creating a Tensor or consuming Tensor IDs, reading/copying [host storage](../../glossary.md#host-storage),
or allocating tensor storage. It checks all stored [nodes](../../glossary.md#node) and bodies, including dead work and T=0,
but leaves leaf [identity](../../glossary.md#term-identity-meanings)/substitution checks to capture or reconstruction. [Checkpoint](../../glossary.md#term-checkpoint) decode calls
it before any candidate Tensor in every supported version; it is not an execution capability query.

`reconstructSource(graph, leaves)` snapshots positional substitutions and checks the complete
source and every contained body before creating any formal or producer. Every value must have a fully static [Shape](../../glossary.md#shape) and checked logical
[element](../../glossary.md#term-element) count; [scalar](../../glossary.md#term-scalar) and empty shapes remain valid. Every stored node, including dead work, must
be FORWARD and belong to the closed
[flat source inventory](../../architecture/contracts/compiler-autograd.md#compile-lifecycle):
ADD/MUL/NEG/[MATMUL](../../glossary.md#matrix-multiplication--matmul)/RELU/[TANH](../../glossary.md#term-tanh)/[SIGMOID](../../glossary.md#term-sigmoid), [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) PERMUTE `[1,0]`, and the selected floating SUB,
[CNN](../../glossary.md#term-convolutional-neural-network-cnn)/static recurrent mappings and fixed scans listed there. All-valid [general SCAN](../../glossary.md#general-differentiable-scan) additionally carries
an independently scoped body; its extra constructors remain body-only. Existing family [inference](../../glossary.md#term-inference-meanings)
verifies promotion, [broadcasting](../../glossary.md#broadcasting), MATMUL [rank](../../glossary.md#term-rank)/[batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters)
semantics, attributes and exact output descriptors, with no deferred obligations. [Model](../../glossary.md#term-model) owns
structural closure, [signature](../../glossary.md#term-signature)/attribute checks and [layout](../../glossary.md#layout) geometry. Unsupported families [fail
closed](../../glossary.md#admission-preflight-and-fail-closed-validation); there is no generic constructor or [registry](../../glossary.md#term-registry). The three [activations](../../glossary.md#term-activation) retain Model's
floating-only guards across all six floating types; admission is not an execution
capability claim for every [occurrence](../../glossary.md#term-occurrence).

General bodies reuse the flat constructors except scans. Their bounded additions are tensor DIV
and integral SUB; exact typed scalar ADD/SUB/MUL/DIV; full, single-[axis](../../glossary.md#term-axis), ordered multi-axis and
[target](../../glossary.md#term-target)-Shape SUM; static EXPAND; scalar/empty RESHAPE; any normalized EXPAND_DIMS/SQUEEZE axis;
[CONTIGUOUS](../../glossary.md#term-contiguous); and floating-to-floating CAST (or nonfloating identity CAST). Model family guards
and applicable [dtypes](../../glossary.md#data-type--datatype) remain authoritative. Low cold support is independent of the implemented
[CPU](../../glossary.md#term-cpu) low-[precision](../../glossary.md#term-precision) execution cutover. Other unlisted [PURE](../../glossary.md#term-pure) operations, nonfloating cross-dtype
casts, arbitrary permutations and nested scans remain outside this bounded cold inventory.
See the [exact body inventory](../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction)
for attributes and constructor restrictions. This is not an all-PURE copier or a [persistence](../../glossary.md#term-persistence) promise.

MAX_POOL3D and AVERAGE_POOL3D admit exact typed Model attributes with positive [kernel](../../glossary.md#kernel)/[stride](../../glossary.md#term-stride)/[dilation](../../glossary.md#term-dilation),
nonnegative symmetric [padding](../../glossary.md#padding) and literal floor/ceil geometry, not just builder defaults.
They also enter otherwise-qualified all-valid bodies and add only these two constructors to
the length whitelist. Fully static per-role floating descriptors, ROW/SHARED, axis and
terminal-SHARED restrictions remain; actual/formal and next/initial pairs match exactly, but
different roles may use different floating types with explicit casts retained. [CONV2D](../../glossary.md#convolution--conv2d)/[CONV3D](../../glossary.md#conv3d--ncdhw) still require biased defaults; MAX_POOL2D keeps its narrow
kernel-stride, zero-padding, unit-dilation floor boundary.

Complete preflight includes dead outer scans, unused declarations and T=0 bodies. Raw bodies must
have empty `demandedOutputIndices`, with [graph](../../glossary.md#graph) outputs exactly the first-encounter-unique
concatenation of next-[carry](../../glossary.md#term-carry), emission and [residual](../../glossary.md#scan-residual-port-and-history-demand) roles, in that order. Ordinary compiled demand
is not narrowed by these cold-only checks. Static element counts and descriptor-layout arithmetic
are checked, not physical bytes, resource limits or CPU layout [qualification](../../glossary.md#term-qualification).

The supplied list must have the graph-input count and full `TensorDescriptor.equals` agreement,
including resolved/unresolved layout and `requiresGrad`. Tensor IDs need not match the originals.
Replacements may be ordinary produced expressions: their upstream provenance is an [opaque](../../glossary.md#term-opaque)
substitution, retained for later ordinary [compile](../../glossary.md#compile)/[autograd](../../glossary.md#autograd), not newly admitted source families.
Repeated supplied references are explicit substitution and may yield repeated output references;
ordinary compile still rejects duplicate resolved outputs. Position is the association contract,
so same-descriptor swaps are undetectable.

After complete preflight, separate call-local [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot)-to-Tensor tables rebuild each [scope](../../glossary.md#term-scope)'s stored occurrence
once through its selected ordinary public family constructor, retaining
[canonical](../../glossary.md#term-canonical-form) output wrappers and exact [operation](../../glossary.md#operation) kinds rather than decomposing activations.
Each general occurrence creates fresh distinct canonical storage-free formals in declaration order,
including unused positions, one `ScanBody` and one `Scan.apply`. Time count does not multiply body
objects. Every final-carry, emission and latent [residual port](../../glossary.md#scan-residual-port-and-history-demand) is mapped, including unrequested ports
and repeated role [bindings](../../glossary.md#term-binding). Reusing an original template does not merge reconstructed scopes;
local IDs never resolve against an outer table. [Constants](../../glossary.md#term-constant) enter through explicit outer [invariant](../../glossary.md#term-invariant)
bindings or scalar attributes, not hidden Tensor leaves. Tables are discarded on return/failure;
immutable outputs retain only ordinary provenance, not original templates or formals.
Zero-node pass-through, arbitrary graph IDs and unused inputs are valid source structure, not
an [Engine](../../glossary.md#term-engine) executability promise. No storage is read, allocated, rebound or owned, even when a
[borrowed](../../glossary.md#ownership-borrowing-and-lease) leaf's storage is absent or dead. There is no [close lifecycle](../../glossary.md#ownership-borrowing-and-lease) or [cache](../../glossary.md#term-cache). Null [arguments](../../glossary.md#term-argument)
or elements throw `NullPointerException`; unsupported/invalid source and association throw
contextual `IllegalArgumentException`, wrapping checked geometry [overflow](../../glossary.md#term-overflow). Failures publish no
partial outputs and do not roll back consumed Tensor IDs.

For a storage-free symbolic example, start with a [FLOAT32](../../glossary.md#term-float32) scalar `x` and source `x * x`:

```java
// The replacement keeps its NEG dependency; reconstruction substitutes expressions, not values.
import io.github.pho001.synaptik.compiler.GraphCompilationPort;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.util.List;
import java.util.Optional;

Tensor x = TensorFactory.create(new TensorDescriptor(
        DataType.FLOAT32, Shape.scalar(), Optional.empty(), true));
GraphCompilationPort.SourceCapture source = GraphCompilationPort.captureSource(List.of(x.mul(x)));
Tensor upstream = TensorFactory.create(x.descriptor());
Tensor replacement = upstream.neg();
List<Tensor> outputs = GraphCompilationPort.reconstructSource(source.graph(), List.of(replacement));
```

The raw source has one input and one MUL node. Both reconstructed MUL inputs are the exact
`replacement`; later ordinary differentiation can reach `upstream` through its NEG provenance.
This constructs symbolic metadata only, not numeric values. Replacing the scalar by Shape `[1]`
fails descriptor association even though ordinary arithmetic could broadcast it.

This boundary deliberately skips [canonicalization](../../glossary.md#occurrence-identity-and-canonical-form), folding, dead-code/[common-subexpression
elimination](../../glossary.md#term-common-subexpression-elimination-cse), [publication](../../glossary.md#term-publication), [planning](../../glossary.md#term-planning) and final logical-layout closure. Parameterless operation results
remain layout-unresolved; PERMUTE keeps its constructor's resolved-[view](../../glossary.md#term-view-meanings) or unresolved descriptor.
`CompileArtifacts.graph()` is not source: a closed output descriptor is rejected rather than
stripped or forced into a fabricated Tensor. Ordinary `compile()` and autograd are unchanged.
Checkpoint copy/[rebind](../../glossary.md#capture-graph-ports-and-reconstruction) and [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) acquisition consume the accepted bounded cold body inventory,
including its static [homogeneous](../../glossary.md#term-homogeneous) FLOAT32/[FLOAT64](../../glossary.md#term-float64) length intersection. Persistence independently
uses v4 for all-valid bodies and v5 for that length intersection, or v6 when an outer/body [Pool3d](../../glossary.md#pool3d--ncdhw-pooling)
occurrence requires it; unchanged sources retain smallest v1–v5 bytes. V1/v2/v3 remain body-free.
Cold [Compiler](../../glossary.md#term-compiler) reconstruction itself grants no numerical workflow or wider wire admission.


---

[Previous topic](capture.md) · [Next topic](verification-scan.md)
