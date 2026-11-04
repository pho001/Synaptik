# Public capability and planning values

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

> **Scope:** Model construction, Compiler admission, backend execution, and checkpoint persistence are separate gates. Execution-status sentences retained in this detailed material record earlier qualification slices; they are not an exhaustive current support matrix. The [current backend matrix](../../backend-guide/production-support.md) and its incorporated contracts supersede narrower historical CPU/Metal lists. Package-private implementation sketches are not supported application API.

The public `modules:planning` surface contains these backend-neutral compile-time declarations:

- `OperationCapabilityQuery`, an immutable operation occurrence consisting of one exact
  backend-independent `Operation` reference, ordered immutable membership snapshots of input and
  output `TensorDescriptor` references, an optional independently scoped `CompiledScanBody`,
  and an optional `ScanActivityPlan`;
- `ScanActivityPlan`, immutable structural presence and ordered-merge facts for one exact local
  Scan body, with distinct carry/seed roots and temporal accumulators;
- `BackendCapabilityProvider`, an explicitly supplied collaboration with a stable non-null
  `BackendId` and a deterministic boolean capability answer; and
- `BackendOwnerPlanning`, a stateless one-method collaboration that composes internal hard
  eligibility and baseline owner comparison for one occurrence;
- `PlannedPartition`, an immutable owner-plus-node-ID recipe for one non-empty consecutive region
  of an owning compiled graph;
- `MaximalSameOwnerPartitioning`, the stateless generator over one complete compiler-assembled
  node-to-owner map;
- `LogicalMemoryRequirement`, one immutable graph-value recipe retaining logical descriptor,
  optional producer partition, distinct consumer partitions, and graph-output obligation; and
- `LogicalMemoryPlan`, an immutable ordered snapshot of distinct per-value requirements; and
- `LogicalMemoryPlanning`, the stateless derivation over one closed graph and its ordered complete
  partitions.

The query validates non-null references, signature occurrence counts and attachment applicability.
A present body requires `ScanKind.SCAN`; a present activity plan requires that exact attached
body graph and applicable header positions, next-presence bindings and ordered temporal ADDs.
Its canonical constructor has five arguments, including both optionals. The genuine
three-argument ordinary/fixed-operation constructor supplies empty body and activity attachments;
there is no four-argument body-bearing constructor. Missing-body Scan questions remain
representable for provider rejection. Attachment checks do not infer ordinary descriptors or
establish backend support. Mutable source lists cannot change a constructed query, while the
exact operation, descriptor elements and optional body/activity references are retained.
For scans, provider true requires the whole body and loop binder, not merely supported children.
The [complete capability contract](planning.md#current-operation-capability-contracts) separates
implemented transport, Compiler query/demand emission and Prepare scoped projection/static
binding/demand validation from actual backend support. CPU0030 now qualifies bounded FLOAT32
same-shape pointwise scans, including automatic reverse only when its actual generated body also
qualifies. CPU0031 adds qualified homogeneous FLOAT32 broadcast, SUM/sumToShape and bounded
shape/layout closure, including independently qualified actual reverse bodies. CPU0032 adds static
homogeneous FLOAT32 rank-two MATMUL under that same whole-body/binder qualification. CPU0033
implements homogeneous FLOAT64 across the complete bounded declared algebra and actual
first-order reverse closure. Those task inventories are historical qualification layers.
Current general Scan composes every ordinary CPU-supported PURE unit under its ordinary guards,
not a finite Scan-kind whitelist. All-valid WholeTensor differs from proved length-aware
ROW(axis) and SHARED roles; SHARED tensors are not restricted to scalars. Original length-AD
carries/sequences and emissions stay ROW(0), while intermediate ROW axes follow the proof.
Row-mixing requires materialized same-length support, not Tensor identity.
Selected ordinary VJPs retain ND cuts and finite-path guards; absence is not a present zero seed.
CPU BFLOAT16, FLOAT16 and both OFP8 types use these same ordinary and shared Compiler AD owners,
with FLOAT32 working values and final logical narrowing under existing family/geometry guards.
This adds neither Metal FP8 nor Metal general Scan support. Unlike low operands promote to
FLOAT32 and FLOAT64 dominates; Metal retains its independent mixed-low restrictions.
Cold/wire and NN workflow inventories remain separately bounded. The
[CPU guide](../../backend-guide/cpu-backend.md#bounded-float32-general-scan) states the execution scope.

This current example asks an illustrative local provider about one unary `ABS` occurrence:

```java
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
backend identity `"cpu"`. The result is `true` because this illustrative provider recognizes that
operation kind. It proves only semantic ownership support for this immutable occurrence; it does
not prove CPU registration or availability, evaluate `BackendIntent`, choose a device or CPU
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
`MaximalSameOwnerPartitioning.partition(...)` consumes one `CompiledGraphModel` and a complete
`Map<NodeId, BackendId>`. It validates exact owner-map coverage, then groups consecutive nodes in
the graph's stored topological order while equal `BackendId` values continue.

The following current code constructs the public recipe directly:

```java
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
This direct example proves only the DTO contract; it does not prove that those node IDs belong to
one graph or share an owner. `MaximalSameOwnerPartitioning` establishes those graph-relative
facts.

The generator defines adjacency only by consecutive positions in `CompiledGraphModel.nodes()`.
Graph edges, phases, fan-out, merges, repeated inputs, graph input/output values, and multiple
outputs from one producer do not independently split an equal-owner run. A zero-node graph yields
no partitions. Each generated partition retains the exact graph `NodeId` references and the exact
owner reference associated with its first node, and both the result list and membership lists are
immutable.

Public `LogicalMemoryPlanning.plan(...)` accepts the closed graph and ordered complete partition
recipes. It validates partition nulls, membership, exact graph-order coverage, and adjacent-owner
maximality before deriving one requirement per graph value in `CompiledGraphModel.values()`
order. Generated producer and consumer entries retain exact supplied partition references;
consumers are distinct and follow partition order. `graph.outputs()` alone supplies the generated
graph-output flag.

The two logical-memory records are public data and can also be constructed directly. A standalone
`LogicalMemoryPlan` may be empty and validates only that its requirements are non-null and have
distinct `ValueId` values. A standalone requirement validates and snapshots its components but
has no owning graph against which to prove producer, consumer, or output relationships. The
public generator establishes those graph-relative facts.

These records retain `TensorDescriptor` rather than calculating element or byte counts, so
dynamic and expression dimensions remain representable. They carry no
`ForwardPublicationBinding`, `GradientPublicationBinding`, `TensorId`, lifetime, slot,
allocation, transfer, selected device, route, kernel, executable, prepared schedule, or runtime
residency. `graphOutput` is a logical preservation obligation, not a publication target or
policy.

Reusable or public capability matrices, public eligibility evaluation, a public graph-wide
Planning workflow, numeric or cost scoring, profiles, physical memory, preparation, runtime, and
execution remain planned. Compiler currently owns owner-map assembly and immutable artifact
construction.


---

[Previous topic](configuration-values.md) · [Next topic](artifact-values.md)
