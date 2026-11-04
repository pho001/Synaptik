# [Graph](../../glossary.md#graph) values and compiled [nodes](../../glossary.md#node)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/model/src/main/java/io/github/pho001/synaptik/model/tensor/Tensor.java)

## Who supplies the graph records?

The Compiler normally translates Tensor expressions into these records; the example's application
constructs them directly to teach the same structural boundary. `CompiledNode` receives IDs and
an operation description, not current input arrays. Its list copies freeze the connections,
not tensor values. `CompiledGraphModel` then checks that those connections form a closed graph.
A graph-local value ID names a data position, not a physical memory slot.
Read [operation signatures](operation-foundation.md) for the node's port-count checks and
[Tensor identity and provenance](identity.md) for the source objects that capture translates.

**[Reader model](../../glossary.md#term-reader-model).** A computation graph separates data from the operations using it:
a **value** is one described input or result, and a **node** is one particular use of an [operation](../../glossary.md#operation).
Two nodes may perform the same kind of calculation without being the same [occurrence](../../glossary.md#term-occurrence).
**[Topological order](../../glossary.md#term-topological-order)** puts each [producer](../../glossary.md#term-producer) before nodes that consume its outputs.
An **[intermediate representation](../../glossary.md#term-intermediate-representation-ir)** is the structured form used to describe a program between
source construction and execution. These graph records describe that structure; they are not
an execution schedule or allocated [tensor](../../glossary.md#tensor) memory. **[Publication](../../glossary.md#term-publication)** separately associates a
selected graph result with the public Tensor that should receive it.

<a id="graph-values-and-compiled-nodes"></a>
## [Graph](../../glossary.md#graph) values and compiled [nodes](../../glossary.md#node)

The public graph-[model](../../glossary.md#term-model) contracts live in `io.github.pho001.synaptik.model.graph`. They keep data,
computation, graph boundaries, node phase, and [publication](../../glossary.md#term-publication) association separate:

| Contract | Stored state | Meaning |
|---|---|---|
| `GraphValue` | `ValueId id`, `TensorDescriptor descriptor` | One described logical input, intermediate result, or output. |
| `CompiledNode` | `NodeId id`, `Operation operation`, ordered `inputs`, ordered `outputs`, optional `scanBody` | One [occurrence](../../glossary.md#term-occurrence) of computation and the value positions it consumes and produces; a [general scan](../../glossary.md#general-differentiable-scan) has one independently scoped compiled attachment. |
| `CompiledScanBody` | `graph`, ordered `nextCarry`, `emissions`, `residuals`, increasing `demandedOutputIndices` | Structural body/result/demand [metadata](../../glossary.md#term-metadata), not [capture](../../glossary.md#term-capture-meanings) or execution. |
| `GraphPhase` | Exactly `FORWARD` or `BACKWARD` | A node's [compile](../../glossary.md#compile)-time classification, not a [compile mode](../../glossary.md#compile-mode--compilemode) or [runtime](../../glossary.md#term-runtime) schedule. |
| [`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel) | Ordered `values`, [topological](../../glossary.md#term-topological-order) `nodes`, ordered `inputs` and `outputs`, exact `nodePhases` | One [immutable](../../glossary.md#term-immutable), structurally closed compile-time graph. |
| `ForwardPublicationBinding` | `TensorId tensorId`, `ValueId valueId` | A standalone requested-forward association used inside current [compiler](../../glossary.md#term-compiler)-owned publication plans. |

Both [identity](../../glossary.md#term-identity-meanings) types are local to an owning graph context. Equal numeric IDs in another graph do
not establish a relationship, and a [`NodeId`](../../glossary.md#nodeid) is never interchangeable with a [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot).

`GraphValue` stores no [producer](../../glossary.md#term-producer) or consumer. This allows an input value to use the same record as a
produced value. `CompiledGraphModel` derives producer relationships while validating construction,
but it stores no producer, consumer, node, value, or phase index.

`CompiledNode` copies its input and output lists with `List.copyOf`, preserving encounter order and
preventing later caller mutation from changing the node. Input positions may be empty or repeat a
value ID. Output positions must contain at least one value and must be unique within that one node.
After those structural checks, the node validates both final list sizes against the [operation](../../glossary.md#operation)'s
selected [signature](../../glossary.md#term-signature). Empty inputs or multiple outputs therefore remain available to a kind that
declares them, while a [unary](../../glossary.md#term-unary) signature rejects two inputs. The lists have value semantics: callers
rely on their contents and order, not container identity.

The records use generated [structural equality](../../glossary.md#term-structural-equality) and hashing. For `CompiledNode`, list order and
repeated input positions are part of equality. Generated text is useful for diagnostics but is not
a [serialization](../../glossary.md#term-serialization) format, graph-[validation](../../glossary.md#term-validation) result, or execution-[dispatch](../../glossary.md#term-dispatch) key.

The old four-[argument](../../glossary.md#term-argument) `CompiledNode` constructor remains available and supplies an empty
attachment for ordinary/fixed operations. The five-argument form takes a non-null
`Optional<CompiledScanBody>`, present exactly for `ScanKind.SCAN`. Attachment contents participate
in record equality, hashing and diagnostic text. A [scan](../../glossary.md#term-cumulative-scan) node checks outer input/output counts,
body graph input count and result-group sizes against `ScanAttrs`; missing/wrong-kind or
incompatible attachments reject. `ScanAttrs` contains `timeCount`, [carry](../../glossary.md#term-carry)/sequence/[invariant](../../glossary.md#term-invariant)/
emission/[residual](../../glossary.md#scan-residual-port-and-history-demand) counts, `reverse`, explicit `lengthAware`, `rowCarryCount` and immutable ordered
`List<OptionalInt> sequenceRowAxes`. The seven-argument constructor supplies
`lengthAware=false, rowCarryCount=0, sequenceRowAxes=List.of()` for unchanged all-valid mode.
The former nine-argument sentinel form is removed. Explicit length mode allows zero ROW carries;
current qualified consumers still require their independently supported [topology](../../glossary.md#term-topology).
Length mode has checked outer input count `c+s+i+1` but local
formal count `c+s+i`; output counts/order are unchanged. It never retains source [Tensors](../../glossary.md#tensor),
graphs or runtime state. Constructor-call compatibility does not promise the old record
reflection, equality/[hash](../../glossary.md#term-hash) algorithm or diagnostic text shape; text is not a wire/[cache key](../../glossary.md#term-cache-key).

`CompiledScanBody(graph, nextCarry, emissions, residuals, demandedOutputIndices)` retains one
existing immutable `CompiledGraphModel`, not another graph representation. Its graph inputs list
all formals in carry/step/invariant declaration order, including unused ones; local inputs are
unique even when several actual outer positions refer to the same Tensor. Ordered result-role
[bindings](../../glossary.md#term-binding) may repeat local output IDs, while their distinct union must equal `graph.outputs()`.
Thus one body value may serve carry, emission and residual roles without duplicate graph values
or merged outer wrappers. All role lists are immutable snapshots. Demand indices are zero-based
positions into the concatenated final-carry/emission/residual outer ports, strictly increasing,
unique and in range; an empty list is valid. Model validates supplied demand metadata but does
not compute [liveness](../../glossary.md#term-liveness) or allocate histories.

Nested general scans reject. Body IDs are interpreted only within the contained graph, so
numeric coincidence with an outer ID neither [aliases](../../glossary.md#aliasing-and-arena) values nor causes rejection. These checks
establish local structure/[cardinality](../../glossary.md#term-cardinality), not source-to-[IR](../../glossary.md#term-intermediate-representation-ir) capture correctness, [descriptor](../../glossary.md#term-descriptor) [inference](../../glossary.md#term-inference),
[derivative](../../glossary.md#term-derivative) support or [backend](../../glossary.md#backend) [admission](../../glossary.md#term-admission).

`CompiledGraphModel` snapshots all four lists with `List.copyOf`; list order is part of the record
value. It snapshots `nodePhases` with `Map.copyOf`; the mapping is structural, but iteration order
is not an [API](../../glossary.md#term-application-programming-interface-api) contract. Construction requires unique value and node IDs, resolvable boundaries and
node references, producer-free inputs, exactly one producer for every non-input value, topological
node order, at least one graph output, and exactly one phase for every node. Repeated node inputs,
zero-input nodes, unused graph inputs, and a zero-node pass-through graph are valid.

`GraphPhase` contains only `FORWARD` and `BACKWARD`, in that declaration order. It does not encode
an [optimizer](../../glossary.md#term-optimizer) phase, compile mode, or runtime schedule. `ForwardPublicationBinding` remains
separate from `CompiledGraphModel` and carries no `Tensor`, [gradient](../../glossary.md#term-gradient) role, publication policy or
[target](../../glossary.md#term-target), storage, backend, or execution state. Current `PublicationPlan` supplies exact graph
context and separate forward/gradient binding-list membership without changing the model record.

<a id="complete-test-local-graph-model-example"></a>
### Complete test-local [graph](../../glossary.md#graph)-[model](../../glossary.md#term-model) example

<a id="goal-and-inputs-49"></a>
#### Goal and inputs

Connect two [scalar](../../glossary.md#term-scalar) [graph](../../glossary.md#graph) values through one [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) [operation](../../glossary.md#operation) [occurrence](../../glossary.md#term-occurrence), validate them as a
[compiled graph](../../glossary.md#compiled-graph--compiledgraphmodel) [model](../../glossary.md#term-model), and create a separate [publication](../../glossary.md#term-publication) [binding](../../glossary.md#term-binding). The input has
`ValueId(0)`, the output has `ValueId(1)`, the [node](../../glossary.md#node) has `NodeId(0)`, and both values use an
unresolved-[layout](../../glossary.md#layout) [`FLOAT32`](../../glossary.md#term-float32) scalar [descriptor](../../glossary.md#term-descriptor). `SampleKind` is test-local and is not a production
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
        // Reusing a descriptor shares immutable facts, not input/output value identity.
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

        // Graph construction checks the connections; it does not execute IDENTITY.
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
  not allocate shared storage or claim that input and output are the same graph value.<span data-glossary-navigation="true"> ([scalar glossary](../../glossary.md#term-scalar); [immutable glossary](../../glossary.md#term-immutable); [graph value glossary](../../glossary.md#graph-value))</span>
- The two `GraphValue` constructions give the data positions distinct [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) values. Neither
  construction names a producer.<span data-glossary-navigation="true"> ([producer glossary](../../glossary.md#term-producer))</span>
- [`Operation`](../../glossary.md#operation) describes the [sample](../../glossary.md#term-sample-machine-learning-and-hyperparameters) [identity](../../glossary.md#term-identity-meanings) semantics. `CompiledNode` gives that operation one
  occurrence and connects its ordered input position to value <span data-glossary-navigation="true"> ([occurrence glossary](../../glossary.md#term-occurrence))</span>`0` and output position to value `1`.
- Passing `input.id()` and `output.id()` uses typed data identities. The node snapshots both lists;
  it does not store either `GraphValue` object or infer their [descriptors](../../glossary.md#term-descriptor).
- [`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel) snapshots the [graph](../../glossary.md#graph) [elements](../../glossary.md#term-element) and boundaries, verifies that [node](../../glossary.md#node) `0` is in
  topological order, and requires its one <span data-glossary-navigation="true"> ([topological order glossary](../../glossary.md#term-topological-order))</span>`FORWARD` phase entry.
- `ForwardPublicationBinding` associates [tensor](../../glossary.md#tensor) identity `9` with output value `1`. It remains a
  separate object because the graph model does not own publication policy or execution.<span data-glossary-navigation="true"> ([model glossary](../../glossary.md#term-model); [publication glossary](../../glossary.md#term-publication))</span>

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

The result shows the [operation](../../glossary.md#operation) kind, validated [graph](../../glossary.md#graph) boundaries, [compile](../../glossary.md#compile)-time [node](../../glossary.md#node) phase, and
public [tensor](../../glossary.md#tensor) [identity](../../glossary.md#term-identity-meanings) selected by the standalone [binding](../../glossary.md#term-binding). [Model](../../glossary.md#term-model) construction proves structural
graph closure and local operation [occurrence](../../glossary.md#term-occurrence) [cardinality](../../glossary.md#term-cardinality); it does not itself validate [descriptor](../../glossary.md#term-descriptor)
compatibility. Current package-private [compiler](../../glossary.md#term-compiler) verification can perform that semantic check. The
current complete compiler entry can place a binding in an exact-graph `PublicationPlan` and
`CompileArtifacts`, but model construction itself does not prove that relationship, choose
storage or a [backend](../../glossary.md#backend), [prepare](../../glossary.md#prepare) execution, or [run](../../glossary.md#term-run) the operation.

<a id="failures-and-useful-variations-37"></a>
#### Failures and useful variations

- A null `GraphValue` component fails with `NullPointerException` and message `id` or
  [`descriptor`](../../glossary.md#term-descriptor).
- A null [node](../../glossary.md#node) component fails with `NullPointerException` and its component name. A null list
  element reports its zero-based position, such as <span data-glossary-navigation="true"> ([element glossary](../../glossary.md#term-element))</span>`inputs[1]` or `outputs[2]`.
- An empty output list fails with `IllegalArgumentException` and message
  `outputs must not be empty`. A repeated output reports its later position and duplicate ID.
- `List.of()` is a valid input list, and repeated input IDs remain in their original positions.
  Mutating either list returned by the node fails with `UnsupportedOperationException`.
- An empty graph-output list is invalid. A zero-node graph is valid when its listed value is both
  an input and an output and its phase map is empty.
- A node cannot read its own or a later node's output. Every non-input value needs exactly one
  producer, and the phase map must cover exactly the listed nodes.

`GraphValue` and `CompiledNode` [validation](../../glossary.md#term-validation) is deliberately local. [`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel) adds
[graph](../../glossary.md#graph)-wide existence, ID uniqueness, [producer](../../glossary.md#term-producer), boundary, [topology](../../glossary.md#term-topology), and phase-coverage validation.
The selected [operation signature](../../glossary.md#operation-signature) validates local input and output counts. Descriptor agreement
remains outside the [model](../../glossary.md#term-model) container and is current only through package-private [compiler](../../glossary.md#term-compiler)
verification. The bounded compiler transformations described above are current. This graph
container does not execute them or [run](../../glossary.md#term-run) values; [compilation](../../glossary.md#compiler-engine-and-execution-stage-names), preparation and execution already
have separate [owners](../../glossary.md#term-owner). Its structural validity grants neither an additional rewrite nor
[backend](../../glossary.md#backend) support for an arbitrary graph.


---

[Previous topic](operation-elementwise.md)
