# Prepare declarations, assignment and finalization

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

The public `modules:prepare` surface now spans the existing `prepare.analysis` package, four
root-package finalization contracts, and five complete-graph orchestration declarations.
Analysis receives one immutable `PartitionDag` for exactly one planned partition and retains one
typed opaque selected plan plus exact buffer/workspace declarations. The DAG keeps exact nodes in
stable topological order and exposes immutable producer/output-port, consumer/input-port, edge,
external-input-occurrence, and local-sink facts. Repeated input ports and every output of a
multi-output node remain separate occurrences in node-and-port order. External inputs and local
sinks are topology-only: they imply no cross-partition owner, publication, transfer,
materialization, memory, fusion, route, or schedule policy.

`PrepareContext` has six canonical record components: `partitionDag`, `values`,
`memoryRequirements`, `constants`, `backendInputs`, and `Map<NodeId, ScanActivityPlan>
scanActivityPlans`. Activity keys name containing Scan nodes in this projection and each plan
retains that occurrence's exact local body graph. Shared `GraphPreparation` projects these
associations from `artifacts.derivatives().scanActivityByNode()`; copying or subsetting contexts
preserves or filters applicable associations explicitly. Ordinary/synthetic constructors supply
an empty map, not a way to discard an annotated Scan's metadata. Its derived `partition()` and
`nodes()` views delegate to the DAG. No binary compatibility is promised for a changed record
descriptor, component reflection, equality, hash code or text form. The context exposes no
complete containing `CompiledGraphModel`, graph region, callback or Compiler-owned aggregate.

`PreparationResourceAssignment` associates each exact declaration
with an assigned Runtime slot and dense plan index. `BackendPartitionFinalization` carries one
typed analysis, the exact shared `PreparedMemoryPlan`, and assignments in declaration order to a
`BackendPartitionFinalizer`. The finalizer returns one
`BackendPartitionFinalizationResult`, which snapshots its exact-plan executable and persistent
resources in physical acquisition order. `PreparedPartition` retains only the exact planned
partition and returned `PreparedExecutable`; resource ownership is never inferred from that
association or from schedule occurrences. `PartitionPreparation` keeps each backend's typed
inputs, preparer, and finalizer positionally associated. `PreparedBufferAssignment` translates one
graph `ValueId` to its exact `BufferSlot` and dense plan index. `PreparedScheduleContext` exposes
only validated stable Model, Planning, Prepare, and Runtime facts to one explicit
`PreparedScheduleAssembler`: planned partitions, graph values, bindable source IDs, constants,
ordered publication IDs, the memory plan, prepared partitions, and buffer assignments. It
contains no Compiler aggregate.
`GraphPreparation.prepare(...)` rejects repeated resource identities, rolls successful results
back after any later preparation failure, and transfers the ordered unique resources only by
returning the exact validated `PreparedExecution`. Shared Prepare invokes cleanup but neither
looks up a resource nor performs its physical allocation.

The ordinary three-argument preparation form identifies every producerless, consumerless
published graph input, classified by the existing caller or constant source plan. It asks the
supplied assembler for a `ProducerlessPublishedValueResource`, retaining five components:
the exact stable value and logical requirement, backend owner, existing Runtime `ByteSize size`,
and byte alignment. The default geometry hook is
`producerlessPublishedValue(GraphValue, LogicalMemoryRequirement, Optional<ScalarValue>)`;
the scalar is present only for an authenticated constant, empty only for a caller. No payload
is read or representation allocated by this contribution. The four-argument form accepts the
complete explicit contribution list and uses the same exact-reference, role and coverage checks.

Source-only callers receive real slots in Compiler bindable occurrence order, before remaining
ordinary declarations; source-only constants remain last in final graph-value order. Neither
source receives a partition-local analysis or finalizer assignment. Caller payloads remain borrowed;
constants retain existing fresh per-run initialization. Static affine layouts use Model referenced
span; named templates use checked existing binding recipes, including structurally symbolic
zero-byte geometry. Complete source facts participate in explicit batch mode. A genuinely fixed
CPU unit within that complete batch plan authenticates its fixed selected geometry without changing
its lowering or relaxing Runtime defaults.

`PrepareContext.validateValueShapes(List<GraphValue>)` shares the existing fixed-or-sole-named-B
shape proof and returns the common symbol, if any. It does not establish caller-axis authority,
ROW provenance or backend admission.

The complete-set operation that validates coverage and source identity, assigns slots, constructs
the shared memory plan, and invokes finalizers remains package-private behind the public graph
operation. Its current conservative
policy shares one buffer slot across declarations of the same `ValueId` using maximum geometry
and gives every workspace declaration a distinct slot. It performs no physical allocation,
closeable prepared-resource acquisition, run-state creation, invocation binding/execution,
transfer, residency, constant initialization or materialization, or publication. Producerless
contributions still require at least one non-empty planned partition and therefore do not make a
zero-node graph executable. The supplied assembler constructs an
immutable schedule recipe once; Prepare validates exact source, execution, coordinate, and
publication coverage before constructing the prepared root. Ordinary Engine composes this
operation from all exact partition owners, backend physical contributions, and explicit transfer
recipes captured during cold preparation.


---

[Previous topic](contracts.md) · [Next topic](../engine/lifecycle.md)
