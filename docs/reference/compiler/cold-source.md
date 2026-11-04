# Cold source capture and reconstruction

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="current-cold-source-capture-and-reconstruction"></a>
### Current cold source capture and reconstruction

`GraphCompilationPort.captureSource(List<Tensor>)` returns `SourceCapture(graph, leaves)`: a
validated **raw** `CompiledGraphModel` and an immutable snapshot of exact provenance-free leaves.
The existing traversal collects each leaf when it assigns that input's ID, so leaf position and
`graph.inputs()` position agree even with sharing or equal-looking distinct Tensors. Requested
output order and distinct producer occurrences survive. Empty, null or duplicate output requests
retain ordinary capture rejection. Direct record construction also checks source admission,
input count/order, full descriptors, provenance absence and unique exact leaf references.

`GraphCompilationPort.validateSource(CompiledGraphModel)` runs the same complete metadata-only
cold preflight without creating a Tensor or consuming Tensor IDs, reading/copying host storage,
or allocating tensor storage. It checks all stored nodes and bodies, including dead work and T=0,
but leaves leaf identity/substitution checks to capture or reconstruction. Checkpoint decode calls
it before any candidate Tensor in every supported version; it is not an execution capability query.

`reconstructSource(graph, leaves)` snapshots positional substitutions and checks the complete
source and every contained body before creating any formal or producer. Every value must have a fully static Shape and checked logical
element count; scalar and empty shapes remain valid. Every stored node, including dead work, must
be FORWARD and belong to the closed
[flat source inventory](../../architecture/contracts/compiler-autograd.md#compile-lifecycle):
ADD/MUL/NEG/MATMUL/RELU/TANH/SIGMOID, rank-two PERMUTE `[1,0]`, and the selected floating SUB,
CNN/static recurrent mappings and fixed scans listed there. All-valid general SCAN additionally carries
an independently scoped body; its extra constructors remain body-only. Existing family inference
verifies promotion, broadcasting, MATMUL rank/batch
semantics, attributes and exact output descriptors, with no deferred obligations. Model owns
structural closure, signature/attribute checks and layout geometry. Unsupported families fail
closed; there is no generic constructor or registry. The three activations retain Model's
floating-only guards across FLOAT64, FLOAT32, BFLOAT16 and FLOAT16; admission is not an execution
capability claim for every occurrence.

General bodies reuse the flat constructors except scans. Their bounded additions are tensor DIV
and integral SUB; exact typed scalar ADD/SUB/MUL/DIV; full, single-axis, ordered multi-axis and
target-Shape SUM; static EXPAND; scalar/empty RESHAPE; any normalized EXPAND_DIMS/SQUEEZE axis;
CONTIGUOUS; and identity CAST. Model family guards and applicable dtypes remain authoritative:
FLOAT16/BFLOAT16 TANH round-trip cold while Java low-type computation remains locked. Other PURE
operations, cross-dtype casts, arbitrary permutations and nested scans remain unsupported here.
See the [exact body inventory](../../architecture/contracts/compiler-autograd.md#bounded-general-scan-cold-reconstruction)
for attributes and constructor restrictions. This is not an all-PURE copier or a persistence promise.

MAX_POOL3D and AVERAGE_POOL3D admit exact typed Model attributes with positive kernel/stride/dilation,
nonnegative symmetric padding and literal floor/ceil geometry, not just builder defaults.
They also enter otherwise-qualified all-valid bodies and add only these two constructors to
the length whitelist. Static homogeneous FLOAT32/FLOAT64, ROW/SHARED, axis and terminal-SHARED
restrictions remain. CONV2D/CONV3D still require biased defaults; MAX_POOL2D keeps its narrow
kernel-stride, zero-padding, unit-dilation floor boundary.

Complete preflight includes dead outer scans, unused declarations and T=0 bodies. Raw bodies must
have empty `demandedOutputIndices`, with graph outputs exactly the first-encounter-unique
concatenation of next-carry, emission and residual roles, in that order. Ordinary compiled demand
is not narrowed by these cold-only checks. Static element counts and descriptor-layout arithmetic
are checked, not physical bytes, resource limits or CPU layout qualification.

The supplied list must have the graph-input count and full `TensorDescriptor.equals` agreement,
including resolved/unresolved layout and `requiresGrad`. Tensor IDs need not match the originals.
Replacements may be ordinary produced expressions: their upstream provenance is an opaque
substitution, retained for later ordinary compile/autograd, not newly admitted source families.
Repeated supplied references are explicit substitution and may yield repeated output references;
ordinary compile still rejects duplicate resolved outputs. Position is the association contract,
so same-descriptor swaps are undetectable.

After complete preflight, separate call-local `ValueId`-to-Tensor tables rebuild each scope's stored occurrence
once through its selected ordinary public family constructor, retaining
canonical output wrappers and exact operation kinds rather than decomposing activations.
Each general occurrence creates fresh distinct canonical storage-free formals in declaration order,
including unused positions, one `ScanBody` and one `Scan.apply`. Time count does not multiply body
objects. Every final-carry, emission and latent residual port is mapped, including unrequested ports
and repeated role bindings. Reusing an original template does not merge reconstructed scopes;
local IDs never resolve against an outer table. Constants enter through explicit outer invariant
bindings or scalar attributes, not hidden Tensor leaves. Tables are discarded on return/failure;
immutable outputs retain only ordinary provenance, not original templates or formals.
Zero-node pass-through, arbitrary graph IDs and unused inputs are valid source structure, not
an Engine executability promise. No storage is read, allocated, rebound or owned, even when a
borrowed leaf's storage is absent or dead. There is no close lifecycle or cache. Null arguments
or elements throw `NullPointerException`; unsupported/invalid source and association throw
contextual `IllegalArgumentException`, wrapping checked geometry overflow. Failures publish no
partial outputs and do not roll back consumed Tensor IDs.

For a storage-free symbolic example, start with a FLOAT32 scalar `x` and source `x * x`:

```java
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

This boundary deliberately skips canonicalization, folding, dead-code/common-subexpression
elimination, publication, planning and final logical-layout closure. Parameterless operation results
remain layout-unresolved; PERMUTE keeps its constructor's resolved-view or unresolved descriptor.
`CompileArtifacts.graph()` is not source: a closed output descriptor is rejected rather than
stripped or forced into a fabricated Tensor. Ordinary `compile()` and autograd are unchanged.
Checkpoint copy/rebind and NN acquisition consume the accepted bounded cold body inventory,
including its static homogeneous FLOAT32/FLOAT64 length intersection. Persistence independently
uses v4 for all-valid bodies and v5 for that length intersection, or v6 when an outer/body Pool3d
occurrence requires it; unchanged sources retain smallest v1–v5 bytes. V1/v2/v3 remain body-free.
Cold Compiler reconstruction itself grants no numerical workflow or wider wire admission.


---

[Previous topic](capture.md) · [Next topic](verification-scan.md)
