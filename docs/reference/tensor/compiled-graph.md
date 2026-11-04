# Graph values and compiled nodes

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

<a id="graph-values-and-compiled-nodes"></a>
## Graph values and compiled nodes

The public graph-model contracts live in `io.github.pho001.synaptik.model.graph`. They keep data,
computation, graph boundaries, node phase, and publication association separate:

| Contract | Stored state | Meaning |
|---|---|---|
| `GraphValue` | `ValueId id`, `TensorDescriptor descriptor` | One described logical input, intermediate result, or output. |
| `CompiledNode` | `NodeId id`, `Operation operation`, ordered `inputs`, ordered `outputs`, optional `scanBody` | One occurrence of computation and the value positions it consumes and produces; a general scan has one independently scoped compiled attachment. |
| `CompiledScanBody` | `graph`, ordered `nextCarry`, `emissions`, `residuals`, increasing `demandedOutputIndices` | Structural body/result/demand metadata, not capture or execution. |
| `GraphPhase` | Exactly `FORWARD` or `BACKWARD` | A node's compile-time classification, not a compile mode or runtime schedule. |
| `CompiledGraphModel` | Ordered `values`, topological `nodes`, ordered `inputs` and `outputs`, exact `nodePhases` | One immutable, structurally closed compile-time graph. |
| `ForwardPublicationBinding` | `TensorId tensorId`, `ValueId valueId` | A standalone requested-forward association used inside current compiler-owned publication plans. |

Both identity types are local to an owning graph context. Equal numeric IDs in another graph do
not establish a relationship, and a `NodeId` is never interchangeable with a `ValueId`.

`GraphValue` stores no producer or consumer. This allows an input value to use the same record as a
produced value. `CompiledGraphModel` derives producer relationships while validating construction,
but it stores no producer, consumer, node, value, or phase index.

`CompiledNode` copies its input and output lists with `List.copyOf`, preserving encounter order and
preventing later caller mutation from changing the node. Input positions may be empty or repeat a
value ID. Output positions must contain at least one value and must be unique within that one node.
After those structural checks, the node validates both final list sizes against the operation's
selected signature. Empty inputs or multiple outputs therefore remain available to a kind that
declares them, while a unary signature rejects two inputs. The lists have value semantics: callers
rely on their contents and order, not container identity.

The records use generated structural equality and hashing. For `CompiledNode`, list order and
repeated input positions are part of equality. Generated text is useful for diagnostics but is not
a serialization format, graph-validation result, or execution-dispatch key.

The old four-argument `CompiledNode` constructor remains available and supplies an empty
attachment for ordinary/fixed operations. The five-argument form takes a non-null
`Optional<CompiledScanBody>`, present exactly for `ScanKind.SCAN`. Attachment contents participate
in record equality, hashing and diagnostic text. A scan node checks outer input/output counts,
body graph input count and result-group sizes against `ScanAttrs`; missing/wrong-kind or
incompatible attachments reject. `ScanAttrs` contains `timeCount`, carry/sequence/invariant/
emission/residual counts, `reverse`, explicit `lengthAware`, `rowCarryCount` and immutable ordered
`List<OptionalInt> sequenceRowAxes`. The seven-argument constructor supplies
`lengthAware=false, rowCarryCount=0, sequenceRowAxes=List.of()` for unchanged all-valid mode.
The former nine-argument sentinel form is removed. Explicit length mode allows zero ROW carries;
current qualified consumers still require their independently supported topology.
Length mode has checked outer input count `c+s+i+1` but local
formal count `c+s+i`; output counts/order are unchanged. It never retains source Tensors,
graphs or runtime state. Constructor-call compatibility does not promise the old record
reflection, equality/hash algorithm or diagnostic text shape; text is not a wire/cache key.

`CompiledScanBody(graph, nextCarry, emissions, residuals, demandedOutputIndices)` retains one
existing immutable `CompiledGraphModel`, not another graph representation. Its graph inputs list
all formals in carry/step/invariant declaration order, including unused ones; local inputs are
unique even when several actual outer positions refer to the same Tensor. Ordered result-role
bindings may repeat local output IDs, while their distinct union must equal `graph.outputs()`.
Thus one body value may serve carry, emission and residual roles without duplicate graph values
or merged outer wrappers. All role lists are immutable snapshots. Demand indices are zero-based
positions into the concatenated final-carry/emission/residual outer ports, strictly increasing,
unique and in range; an empty list is valid. Model validates supplied demand metadata but does
not compute liveness or allocate histories.

Nested general scans reject. Body IDs are interpreted only within the contained graph, so
numeric coincidence with an outer ID neither aliases values nor causes rejection. These checks
establish local structure/cardinality, not source-to-IR capture correctness, descriptor inference,
derivative support or backend admission.

`CompiledGraphModel` snapshots all four lists with `List.copyOf`; list order is part of the record
value. It snapshots `nodePhases` with `Map.copyOf`; the mapping is structural, but iteration order
is not an API contract. Construction requires unique value and node IDs, resolvable boundaries and
node references, producer-free inputs, exactly one producer for every non-input value, topological
node order, at least one graph output, and exactly one phase for every node. Repeated node inputs,
zero-input nodes, unused graph inputs, and a zero-node pass-through graph are valid.

`GraphPhase` contains only `FORWARD` and `BACKWARD`, in that declaration order. It does not encode
an optimizer phase, compile mode, or runtime schedule. `ForwardPublicationBinding` remains
separate from `CompiledGraphModel` and carries no `Tensor`, gradient role, publication policy or
target, storage, backend, or execution state. Current `PublicationPlan` supplies exact graph
context and separate forward/gradient binding-list membership without changing the model record.

<a id="complete-test-local-graph-model-example"></a>
### Complete test-local graph-model example

<a id="goal-and-inputs-49"></a>
#### Goal and inputs

Connect two scalar graph values through one sample operation occurrence, validate them as a
compiled graph model, and create a separate publication binding. The input has
`ValueId(0)`, the output has `ValueId(1)`, the node has `NodeId(0)`, and both values use an
unresolved-layout `FLOAT32` scalar descriptor. `SampleKind` is test-local and is not a production
operation family.

```java
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.graph.CompiledGraphModel;
import io.github.pho001.synaptik.model.graph.CompiledNode;
import io.github.pho001.synaptik.model.graph.GraphPhase;
import io.github.pho001.synaptik.model.graph.GraphValue;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.model.graph.ForwardPublicationBinding;
import io.github.pho001.synaptik.model.graph.ValueId;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.OperationKind;
import io.github.pho001.synaptik.model.operation.OperationSignature;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorId;
import java.util.List;
import java.util.Map;
import java.util.Optional;

public final class GraphElementExample {
    private enum SampleKind implements OperationKind {
        IDENTITY;

        private static final List<OperationSignature> SIGNATURES = List.of(
                OperationSignature.fixed(NoOperationAttrs.class, 1, 1));

        @Override
        public List<OperationSignature> signatures() {
            return SIGNATURES;
        }
    }

    public static void main(String[] args) {
        TensorDescriptor scalar = new TensorDescriptor(
                DataType.FLOAT32, Shape.scalar(), Optional.empty(), false);
        GraphValue input = new GraphValue(new ValueId(0), scalar);
        GraphValue output = new GraphValue(new ValueId(1), scalar);

        Operation identity = new Operation(
                SampleKind.IDENTITY, NoOperationAttrs.INSTANCE);
        CompiledNode node = new CompiledNode(
                new NodeId(0),
                identity,
                List.of(input.id()),
                List.of(output.id()));

        CompiledGraphModel graph = new CompiledGraphModel(
                List.of(input, output),
                List.of(node),
                List.of(input.id()),
                List.of(output.id()),
                Map.of(node.id(), GraphPhase.FORWARD));
        ForwardPublicationBinding publication = new ForwardPublicationBinding(
                new TensorId(9), output.id());

        System.out.println(node.operation().kind().name());
        System.out.println(graph.inputs().stream().map(ValueId::value).toList());
        System.out.println(graph.outputs().stream().map(ValueId::value).toList());
        System.out.println(graph.nodePhases().get(node.id()));
        System.out.println(publication.tensorId());
    }
}
```

<a id="meaningful-lines-12"></a>
#### Meaningful lines

- The one `TensorDescriptor` describes the logical scalar facts. Reusing that immutable value does
  not allocate shared storage or claim that input and output are the same graph value.
- The two `GraphValue` constructions give the data positions distinct `ValueId` values. Neither
  construction names a producer.
- `Operation` describes the sample identity semantics. `CompiledNode` gives that operation one
  occurrence and connects its ordered input position to value `0` and output position to value `1`.
- Passing `input.id()` and `output.id()` uses typed data identities. The node snapshots both lists;
  it does not store either `GraphValue` object or infer their descriptors.
- `CompiledGraphModel` snapshots the graph elements and boundaries, verifies that node `0` is in
  topological order, and requires its one `FORWARD` phase entry.
- `ForwardPublicationBinding` associates tensor identity `9` with output value `1`. It remains a
  separate object because the graph model does not own publication policy or execution.

<a id="result-and-interpretation-46"></a>
#### Result and interpretation

The program prints:

```text
IDENTITY
[0]
[1]
FORWARD
TensorId[value=9]
```

The result shows the operation kind, validated graph boundaries, compile-time node phase, and
public tensor identity selected by the standalone binding. Model construction proves structural
graph closure and local operation occurrence cardinality; it does not itself validate descriptor
compatibility. Current package-private compiler verification can perform that semantic check. The
current complete compiler entry can place a binding in an exact-graph `PublicationPlan` and
`CompileArtifacts`, but model construction itself does not prove that relationship, choose
storage or a backend, prepare execution, or run the operation.

<a id="failures-and-useful-variations-37"></a>
#### Failures and useful variations

- A null `GraphValue` component fails with `NullPointerException` and message `id` or
  `descriptor`.
- A null node component fails with `NullPointerException` and its component name. A null list
  element reports its zero-based position, such as `inputs[1]` or `outputs[2]`.
- An empty output list fails with `IllegalArgumentException` and message
  `outputs must not be empty`. A repeated output reports its later position and duplicate ID.
- `List.of()` is a valid input list, and repeated input IDs remain in their original positions.
  Mutating either list returned by the node fails with `UnsupportedOperationException`.
- An empty graph-output list is invalid. A zero-node graph is valid when its listed value is both
  an input and an output and its phase map is empty.
- A node cannot read its own or a later node's output. Every non-input value needs exactly one
  producer, and the phase map must cover exactly the listed nodes.

`GraphValue` and `CompiledNode` validation is deliberately local. `CompiledGraphModel` adds
graph-wide existence, ID uniqueness, producer, boundary, topology, and phase-coverage validation.
The selected operation signature validates local input and output counts. Descriptor agreement
remains outside the model container and is current only through package-private compiler
verification. The bounded compiler transformations described above are current; broader
transformation and execution remain planned.


---

[Previous topic](operation-elementwise.md) · [Next topic](historical-status.md)
