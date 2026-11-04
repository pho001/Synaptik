# Capability queries and planned partitions

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/compiler/src/main/java/io/github/pho001/synaptik/compiler/CompileArtifacts.java)

**Capability** asks whether an implementation supports the exact described operation, not just
whether an operation name appears in a list. **Eligibility** also accounts for availability and
hard requirements. A **partition** is a consecutive graph region assigned to one backend; it does
not promise that the region becomes one fused numerical kernel.

<a id="current-operation-capability-contracts"></a>
## Current operation-capability contracts

`io.github.pho001.synaptik.planning.capability.OperationCapabilityQuery` describes one immutable
operation occurrence for a compile-time capability question. Its exact ordered components are
`Operation operation`, `List<TensorDescriptor> inputs`, `List<TensorDescriptor> outputs`,
`Optional<CompiledScanBody> scanBody`, and `Optional<ScanActivityPlan> scanActivityPlan`.
Body-bearing construction supplies both optionals explicitly; there is no four-argument
constructor. The genuine three-argument ordinary/fixed-operation constructor supplies empty
body and activity attachments. Missing-body Scan questions remain representable for provider
rejection.

The canonical constructor checks operation and list references, scans and snapshots input
membership, scans and snapshots output membership, and then validates both occurrence counts
against `operation.signature()`. It next checks the body optional and restricts a present body to
`ScanKind.SCAN`, then checks the activity optional. A present activity plan requires the exact
attached body graph reference. Its declared numeric-carry, sequence-seed and temporal-accumulator
positions, next-presence bindings and ordered temporal ADDs must apply to that body and header.
These attachment checks do not infer ordinary descriptors, compute differentiation demand or
establish backend support.

The immutable snapshots retain exact descriptor element references, including repeated
positions. The operation, optional containers and contained immutable body/activity references
are retained exactly. All five components participate in record equality, hashing and diagnostic
text; the text is not a wire format. Signature-valid header/body/descriptor mismatches not covered
by attachment validation remain questions for independent provider rejection. IDs are never
resolved across the containing and local graph scopes.

`ScanActivityPlan` contains one exact local `CompiledGraphModel`, ordered `Carry` declarations,
ordered seed-formal positions, scope memberships, complete node-to-scope associations, optional
ordered `Merge` declarations and distinct temporal `Accumulator` declarations. Presence roots
are carry slots followed by seed slots; tied numeric values do not merge their roots. A scope is
the disjunction of its root memberships, and an empty scope is absent. Every executable body node,
including primal-only work, has a scope. Optional ADD merges require identical operand/result
types and Shapes: a sole present operand is copied by raw identity, not added to a numeric zero.
Temporal accumulation remains a separate ordered ADD, rather than an optional list merge.
The plan snapshots and validates intrinsic local structure; Compiler, Prepare and the selected
backend independently validate its applicability. It is structural metadata, not a derivative
formula, runtime value, executable callback or cross-operation sparse-gradient protocol.

`BackendCapabilityProvider` names one stable non-null `BackendId` through `backendId()` and answers
`supports(query)` for that same backend. For an immutable query and unchanged immutable provider
configuration, the boolean answer is deterministic. Implementations reject a null query with
`NullPointerException("query")`. `true` means only that the backend can semantically own the
described occurrence; `false` carries no reason.

For a general scan, true must qualify the complete header, every body occurrence and descriptor,
ordered bindings, residual/demand requirements and generated topology together with the loop
binder. Missing, malformed, unknown or unsupported semantics return false, including at zero steps
or empty demand. Supported children alone do not establish support. `BackendOwnerPlanning`
forwards the exact query unchanged and selects one owner for the whole scan, never mixed-owner
body partitions. Transport and Compiler scoped capture, first-order AD and final body-bearing
query/demand emission are implemented, as are Prepare scoped projection, static binding checks and
complete-graph demand validation before analysis. CPU0030 supplies real backend support for a
bounded FLOAT32 same-shape pointwise body and binder. The actual forward and Compiler-generated
reverse bodies must each satisfy the predicate; a forward-only true answer is not an arbitrary
backward guarantee. CPU0031 extends the bounded FLOAT32 predicate to broadcast, SUM/sumToShape and
shape/layout closure, with the same independent actual-reverse check. CPU0032 adds static
homogeneous FLOAT32 rank-two MATMUL under that check. CPU0033 extends the complete bounded
declared algebra and actual first-order reverse closure to homogeneous FLOAT64, with static T,
canonical boundaries and proved zero-offset all-zero-stride scalar reads. A true
answer from a test provider is not a real-backend support claim. See the
[bounded CPU route](../../backend-guide/cpu-backend.md#bounded-float32-general-scan).

The provider is supplied explicitly to compile-time planning. The public contract performs no
registry, discovery, classpath scan, `ServiceLoader` lookup, availability or hard-requirement
evaluation, scoring, route or kernel selection, preparation, or execution. The CPU backend
supplies the production `CpuCapabilityProvider`. Current ordinary `Engine.compile(...)` and
`AdvancedEngine.compile(...)` workflows explicitly pass that provider from their fixed CPU
composition to public cross-module `GraphCompilationPort`. The port delegates the complete
pipeline to package-private `GraphCompiler`, which supplies providers once per final graph node
and retains only a selected `BackendId`, not a provider object. The port is an integration SPI,
not a standalone ordinary compiler facade, and the current Engine exposes no generic provider-
registration workflow.

Current package-private planning implementation validates a complete association between ordered
providers and caller-supplied availability snapshots by equal `BackendId`, then applies non-empty
availability and the optional exact hard target before invoking each still-possible provider once.
It returns only an immutable provider-ordered list of eligible `BackendId` references; an empty
list is a valid internal no-match result. Exact-device and device-class requirements consult the
matching snapshot only for availability. They do not make the boolean answer device-level, select
a device, or retain one.

Current package-private baseline selection treats that list as the complete candidate set. It
validates the full snapshot input for nulls and duplicate equal identities, requires one equal-ID
snapshot for each eligible backend, and permits extra unique snapshots. Empty eligibility fails
before snapshot-element reads. A neutral configuration returns the first eligible identity; a
present preference returns the first provider-order class match or falls back to the first
eligible identity. Empty snapshots are preference nonmatches, and the exact eligibility identity
reference is returned. No provider is called again and no device, route, or kernel is selected.
Public `BackendOwnerPlanning.selectOwner(...)` validates its five top-level references in
declaration order and composes these two internal steps exactly once for one occurrence. The
eligibility result remains package-private and does not escape.

Reusable or public capability matrices, a public eligibility result/evaluator, public graph-wide
planning orchestration, numeric or cost scoring, device-level capability/selection, and a
public eligibility diagnostic schema are not provided by this API. There is no numerical
profile selector. Preparation, Runtime and execution are implemented in their separate owners,
not inside these declarative Planning values.
Compiler currently owns owner-map assembly and contextualizes only the terminal no-hard-eligible
failure.

<a id="current-planned-partition-recipe"></a>
## Current planned-partition recipe

`io.github.pho001.synaptik.planning.partition.PlannedPartition` is current public compile-time
data with exactly two ordered components:

```java
BackendId owner
List<NodeId> nodeIds
```

The owner is one non-null backend ownership identity. The node list is non-null, non-empty,
contains no null or duplicate identity by equality, and is copied into an immutable ordered
snapshot. The record retains the exact owner and element references, uses ordinary record
equality and hashing, and carries no graph values, partition ID, boundary edges, transfer or
memory facts, selected device, route, kernel, executable, or runtime state.

Public stateless `MaximalSameOwnerPartitioning.partition(...)` accepts one immutable
`CompiledGraphModel` and one complete `Map<NodeId, BackendId>`. Equal map keys associate with graph
nodes, but output membership always uses the exact `NodeId` references from `graph.nodes()`. It
validates all keys, graph coverage, and owners before constructing a result. For each maximal run
it retains the exact owner reference mapped to the first node and compares later owners with
`BackendId.equals`.

For graph order `[n0, n1, n2, n3, n4]` and owner values
`[cpu, cpu, metal, metal, cpu]`, the current result meaning is:

```text
PlannedPartition(cpu,   [n0, n1])
PlannedPartition(metal, [n2, n3])
PlannedPartition(cpu,   [n4])
```

The final CPU node stays separate because only consecutive positions in the stored validated
topological order are adjacent. Conversely, consecutive independent nodes with equal owners join.
Graph edges, phase changes, fan-out, merges, repeated input positions, graph input/output values,
and multiple output values from one producer do not independently create boundaries. A valid
zero-node pass-through graph produces an immutable empty list.

Compiler currently assembles the complete owner map from per-occurrence results and invokes this
operation directly. Public visibility serves that concrete cross-package consumer; it does not
create a graph-wide Planning facade or public compile workflow. A partition is an immutable
ownership recipe, not a promise of one fused kernel, one executable, or any prepare/runtime
behavior.


---

[Previous topic](configuration.md) · [Next topic](memory-output.md)
