# Public capability and [planning](../../glossary.md#term-planning) values

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

A capability query describes one exact [operation](../../glossary.md#operation) and its input/output [metadata](../../glossary.md#term-metadata). A provider answers
whether its [backend](../../glossary.md#backend) can own that [occurrence](../../glossary.md#term-occurrence); planning then records the selected [owner](../../glossary.md#term-owner) and groups
consecutive same-owner [nodes](../../glossary.md#node) into **partitions**. These public values are plans, not allocations,
selected devices or executable [kernels](../../glossary.md#kernel).

The public `modules:planning` surface contains these backend-neutral [compile](../../glossary.md#compile)-time declarations:

- [`OperationCapabilityQuery`](../../glossary.md#operation-capability-query--operationcapabilityquery), an immutable operation occurrence consisting of one exact
  backend-independent `Operation` reference, ordered immutable membership snapshots of input and
  output `TensorDescriptor` references, an optional independently scoped `CompiledScanBody`,
  and an optional `ScanActivityPlan`;
- `ScanActivityPlan`, immutable structural presence and ordered-merge facts for one exact local
  Scan body, with distinct carry/seed roots and temporal accumulators;
- [`BackendCapabilityProvider`](../../glossary.md#backend-capability-provider--backendcapabilityprovider), an explicitly supplied collaboration with a stable non-null
  [`BackendId`](../../glossary.md#backend-identity--backendid) and a deterministic [boolean](../../glossary.md#data-type--datatype) capability answer; and
- `BackendOwnerPlanning`, a stateless one-method collaboration that composes internal hard
  eligibility and baseline owner comparison for one occurrence;
- `PlannedPartition`, an immutable owner-plus-node-ID recipe for one non-empty consecutive region
  of an owning compiled graph;
- `MaximalSameOwnerPartitioning`, the stateless generator over one complete compiler-assembled
  node-to-owner map;
- `LogicalMemoryRequirement`, one immutable graph-value recipe retaining logical descriptor,
  optional producer partition, distinct consumer partitions, and graph-output obligation; and
- `LogicalMemoryPlan`, an [immutable](../../glossary.md#term-immutable) ordered [snapshot](../../glossary.md#term-snapshot) of distinct per-value requirements; and
- `LogicalMemoryPlanning`, the stateless derivation over one closed graph and its ordered complete
  partitions.

The query validates non-null references, [signature](../../glossary.md#term-signature) occurrence counts and attachment applicability.
A present body requires `ScanKind.SCAN`; a present activity plan requires that exact attached
body [graph](../../glossary.md#graph) and applicable header positions, next-presence [bindings](../../glossary.md#term-binding) and ordered temporal ADDs.
Its [canonical](../../glossary.md#term-canonical-form) constructor has five [arguments](../../glossary.md#term-argument), including both optionals. The genuine
three-argument ordinary/fixed-operation constructor supplies empty body and activity attachments;
there is no four-argument body-bearing constructor. Missing-body [Scan](../../glossary.md#general-differentiable-scan) questions remain
representable for provider rejection. Attachment checks do not infer ordinary [descriptors](../../glossary.md#term-descriptor) or
establish backend support. [Mutable](../../glossary.md#term-mutable) source lists cannot change a constructed query, while the
exact operation, descriptor [elements](../../glossary.md#term-element) and optional body/activity references are retained.
For scans, provider true requires the whole body and loop binder, not merely supported children.
The [complete capability contract](planning.md#current-operation-capability-contracts) separates
implemented transport, [Compiler](../../glossary.md#term-compiler) query/demand emission and [Prepare](../../glossary.md#prepare) scoped projection/static
binding/demand [validation](../../glossary.md#term-validation) from actual backend support. CPU0030 now qualifies bounded [FLOAT32](../../glossary.md#term-float32)
same-[shape](../../glossary.md#shape) [pointwise](../../glossary.md#term-pointwise) scans, including automatic reverse only when its actual generated body also
qualifies. CPU0031 adds qualified [homogeneous](../../glossary.md#term-homogeneous) FLOAT32 broadcast, SUM/sumToShape and bounded
shape/[layout](../../glossary.md#layout) closure, including independently qualified actual reverse bodies. CPU0032 adds static
homogeneous FLOAT32 [rank-two](../../glossary.md#axis-rank-scalar-vector-and-matrix) [MATMUL](../../glossary.md#matrix-multiplication--matmul) under that same whole-body/binder [qualification](../../glossary.md#term-qualification). CPU0033
implements homogeneous [FLOAT64](../../glossary.md#term-float64) across the complete bounded declared algebra and actual
first-order reverse closure. Those task inventories are historical qualification layers.
Current [general Scan](../../glossary.md#general-differentiable-scan) composes every ordinary [CPU](../../glossary.md#term-cpu)-supported [PURE](../../glossary.md#term-pure) unit under its ordinary guards,
not a [finite](../../glossary.md#term-finite) Scan-kind whitelist. All-valid WholeTensor differs from proved length-aware
ROW([axis](../../glossary.md#term-axis)) and SHARED roles; SHARED [tensors](../../glossary.md#tensor) are not restricted to [scalars](../../glossary.md#term-scalar). Original length-[AD](../../glossary.md#autograd)
carries/sequences and emissions stay ROW(0), while intermediate ROW axes follow the proof.
Row-mixing requires materialized same-length support, not Tensor [identity](../../glossary.md#term-identity-meanings).
Selected ordinary VJPs retain [ND](../../glossary.md#term-nd) cuts and finite-path guards; absence is not a present zero [seed](../../glossary.md#term-seed).
CPU [BFLOAT16](../../glossary.md#term-bfloat16), [FLOAT16](../../glossary.md#term-float16) and both [OFP8](../../glossary.md#term-ofp8) types use these same ordinary and shared Compiler AD owners,
with FLOAT32 working values and final logical [narrowing](../../glossary.md#term-narrowing) under existing family/geometry guards.
This adds neither [Metal](../../glossary.md#term-metal) [FP8](../../glossary.md#term-fp8) nor Metal general Scan support. Unlike low operands promote to
FLOAT32 and FLOAT64 dominates; Metal retains its independent mixed-low restrictions.
Cold/wire and [NN](../../glossary.md#neural-network-module-parameter-buffer-and-forward-context) workflow inventories remain separately bounded. The
[CPU guide](../../backend-guide/cpu-backend.md#bounded-float32-general-scan) states the execution [scope](../../glossary.md#term-scope).

This current example asks an illustrative local provider about one [unary](../../glossary.md#term-unary) `ABS` occurrence:

```java
// This illustrative provider recognizes a name only; production support also checks exact occurrence guards.
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.operation.NoOperationAttrs;
import io.github.pho001.synaptik.model.operation.Operation;
import io.github.pho001.synaptik.model.operation.elementwise.unary.UnaryElementwiseKind;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.planning.capability.BackendCapabilityProvider;
import io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

TensorDescriptor vector =
        new TensorDescriptor(DataType.FLOAT32, Shape.of(4), Optional.empty(), false);
Operation abs = new Operation(UnaryElementwiseKind.ABS, NoOperationAttrs.INSTANCE);
OperationCapabilityQuery query =
        new OperationCapabilityQuery(abs, List.of(vector), List.of(vector));

BackendId cpu = new BackendId("cpu");
BackendCapabilityProvider illustrativeCpu = new BackendCapabilityProvider() {
    @Override
    public BackendId backendId() {
        return cpu;
    }

    @Override
    public boolean supports(OperationCapabilityQuery candidate) {
        Objects.requireNonNull(candidate, "query");
        return candidate.operation().kind() == UnaryElementwiseKind.ABS;
    }
};

boolean semanticOwnershipSupported = illustrativeCpu.supports(query);
```

The concrete inputs are one FLOAT32 descriptor with Shape `[4]`, the `ABS` operation, and the
[backend identity](../../glossary.md#backend-identity--backendid) `"cpu"`. The result is `true` because this illustrative provider recognizes that
operation kind. It proves only semantic [ownership](../../glossary.md#term-ownership-meanings) support for this immutable occurrence; it does
not prove CPU registration or availability, evaluate [`BackendIntent`](../../glossary.md#backend-intent--backendintent), choose a [device](../../glossary.md#term-device) or CPU
route, prepare work, or execute values. Production CPU and Metal providers exist for their
qualified occurrence domains; this example is not either provider and grants no general-scan
support. Package-private Compiler is the current consumer, while no public graph-wide
planning consumer exists.

Provider implementations must reject a null query with `NullPointerException("query")`. A false
answer carries no diagnostic reason. Inside the same package, current implementation code
validates complete provider/snapshot associations and combines backend-level support, non-empty
availability, and an optional exact hard requirement into an immutable provider-ordered
`BackendId` list. A second package-private step validates equal-ID snapshot associations and uses
that list directly as the complete candidate set. It returns the first preferred-class match, or
the first eligible identity when no preference or match exists; provider order resolves ties. It
returns the exact eligibility reference, allows extra unique snapshots, treats an empty matching
snapshot as a preference nonmatch, and selects no device. Empty eligibility fails internally
before snapshot elements are read.

The eligibility result and selector remain package-private. Public
`BackendOwnerPlanning.selectOwner(...)` composes them once for a non-null query, intent, provider
list, snapshot list, and scoring config. The intermediate does not escape. Public
`MaximalSameOwnerPartitioning.partition(...)` consumes one [`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel) and a complete
`Map<NodeId, BackendId>`. It validates exact owner-map coverage, then groups consecutive nodes in
the graph's stored [topological order](../../glossary.md#term-topological-order) while equal `BackendId` values continue.

The following current code constructs the public recipe directly:

```java
// Direct construction snapshots IDs but does not prove they belong to one consecutive graph region.
import io.github.pho001.synaptik.backend.contract.BackendId;
import io.github.pho001.synaptik.model.graph.NodeId;
import io.github.pho001.synaptik.planning.partition.PlannedPartition;
import java.util.List;

BackendId cpu = new BackendId("cpu");
NodeId first = new NodeId(10);
NodeId second = new NodeId(11);
PlannedPartition partition =
        new PlannedPartition(cpu, List.of(first, second));
```

`partition.owner()` returns the exact `cpu` reference. `partition.nodeIds()` is an immutable,
non-empty ordered snapshot that contains the exact `first` and `second` references. Construction
rejects a null owner or list, an empty list, the first null element, and the first later duplicate.
This direct example proves only the [DTO](../../glossary.md#data-transfer-object--dto) contract; it does not prove that those node IDs belong to
one graph or share an owner. `MaximalSameOwnerPartitioning` establishes those graph-relative
facts.

The generator defines adjacency only by consecutive positions in `CompiledGraphModel.nodes()`.
Graph edges, phases, [fan-out](../../glossary.md#term-fan-out), merges, repeated inputs, graph input/output values, and multiple
outputs from one [producer](../../glossary.md#term-producer) do not independently split an equal-owner [run](../../glossary.md#term-run). A zero-node graph yields
no partitions. Each generated [partition](../../glossary.md#partition) retains the exact graph [`NodeId`](../../glossary.md#nodeid) references and the exact
owner reference associated with its first node, and both the result list and membership lists are
immutable.

Public `LogicalMemoryPlanning.plan(...)` accepts the closed graph and ordered complete partition
recipes. It validates partition nulls, membership, exact graph-order coverage, and adjacent-owner
maximality before deriving one requirement per [graph value](../../glossary.md#graph-value) in `CompiledGraphModel.values()`
order. Generated producer and consumer entries retain exact supplied partition references;
consumers are distinct and follow partition order. `graph.outputs()` alone supplies the generated
graph-output flag.

The two logical-memory records are public data and can also be constructed directly. A standalone
`LogicalMemoryPlan` may be empty and validates only that its requirements are non-null and have
distinct [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) values. A standalone requirement validates and snapshots its components but
has no owning graph against which to prove producer, consumer, or output relationships. The
public generator establishes those graph-relative facts.

These records retain `TensorDescriptor` rather than calculating element or byte counts, so
dynamic and expression [dimensions](../../glossary.md#dimension) remain representable. They carry no
`ForwardPublicationBinding`, `GradientPublicationBinding`, [`TensorId`](../../glossary.md#tensorid), [lifetime](../../glossary.md#term-lifetime), slot,
allocation, transfer, selected device, route, [kernel](../../glossary.md#kernel), executable, [prepared schedule](../../glossary.md#prepared-schedule--preparedschedule), or [runtime](../../glossary.md#term-runtime)
[residency](../../glossary.md#residency). `graphOutput` is a logical preservation obligation, not a [publication](../../glossary.md#term-publication) [target](../../glossary.md#term-target) or
policy.

Reusable or public capability [matrices](../../glossary.md#term-review-matrix), public eligibility [evaluation](../../glossary.md#term-evaluation), a public graph-wide
Planning workflow and numeric or cost scoring are not exposed here. Numerical profiles are
not a current selector. Physical memory, preparation, Runtime and execution already have
separate owners; these logical records do not implement them. Compiler currently owns owner-map assembly and immutable [artifact](../../glossary.md#term-artifact)
construction.


---

[Previous topic](configuration-values.md) · [Next topic](artifact-values.md)
## Construction does not answer the whole planning question

An integration author can construct the records below directly, but Compiler is the caller that
supplies final graph facts to Planning in the normal pipeline. A direct `PlannedPartition` value
only snapshots an owner and node IDs; partition generation establishes whether those IDs really
form one region of this graph. Likewise a query contains descriptors, not the current arrays.
Providers inspect those descriptions and return support without acquiring storage. Read
[the complete Planning collaboration](planning.md) before treating a local illustrative provider
as a production capability proof, and [physical assignment](../runtime/prepare.md) for the later
transition from logical facts to resource slots.

