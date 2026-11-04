# [Prepare](../../glossary.md#prepare) declarations, assignment and finalization

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

Prepare bridges a logical computation plan and reusable execution recipes. A [directed acyclic
graph](../../glossary.md#term-directed-acyclic-graph-dag) (**[DAG](../../glossary.md#intermediate-representation-dag-and-ssa)**) records [operation](../../glossary.md#operation) dependencies without cycles; its input/output **ports** are
ordered operand and result positions. [Backend](../../glossary.md#backend) analysis translates that [partition](../../glossary.md#partition) into concrete
work and declares its memory needs before shared preparation assigns slots and finalizes recipes.

The public `modules:prepare` surface now spans the existing `prepare.analysis` package, four
root-package finalization contracts, and five complete-[graph](../../glossary.md#graph) orchestration declarations.
Analysis receives one [immutable](../../glossary.md#term-immutable) [`PartitionDag`](../../glossary.md#partition-dag-partitiondag) for exactly one planned partition and retains one
typed [opaque](../../glossary.md#term-opaque) selected plan plus exact [buffer](../../glossary.md#term-buffer)/workspace declarations. The DAG keeps exact [nodes](../../glossary.md#node) in
stable [topological order](../../glossary.md#term-topological-order) and exposes immutable [producer](../../glossary.md#term-producer)/output-port, consumer/input-port, edge,
external-input-[occurrence](../../glossary.md#term-occurrence), and local-sink facts. Repeated input ports and every output of a
multi-output node remain separate occurrences in node-and-port order. External inputs and local
sinks are [topology](../../glossary.md#term-topology)-only: they imply no cross-partition [owner](../../glossary.md#term-owner), [publication](../../glossary.md#term-publication), transfer,
[materialization](../../glossary.md#materialization), memory, [fusion](../../glossary.md#term-fusion), route, or schedule policy.

[`PrepareContext`](../../glossary.md#prepare-context-preparecontext) has six [canonical](../../glossary.md#term-canonical-form) record components: `partitionDag`, `values`,
`memoryRequirements`, [`constants`](../../glossary.md#term-constant), `backendInputs`, and `Map<NodeId, ScanActivityPlan>
scanActivityPlans`. Activity keys name containing [Scan](../../glossary.md#term-cumulative-scan) nodes in this projection and each plan
retains that occurrence's exact local body graph. Shared [`GraphPreparation`](../../glossary.md#graph-preparation--graphpreparation) projects these
associations from `artifacts.derivatives().scanActivityByNode()`; copying or subsetting contexts
preserves or filters applicable associations explicitly. Ordinary/synthetic constructors supply
an empty map, not a way to discard an annotated Scan's [metadata](../../glossary.md#term-metadata). Its derived `partition()` and
`nodes()` [views](../../glossary.md#term-view-meanings) delegate to the DAG. No [binary](../../glossary.md#term-binary-descriptor-and-metadata) compatibility is promised for a changed record
[descriptor](../../glossary.md#term-descriptor), component reflection, equality, [hash code](../../glossary.md#term-hash-code) or text form. The context exposes no
complete containing [`CompiledGraphModel`](../../glossary.md#compiled-graph--compiledgraphmodel), graph region, [callback](../../glossary.md#term-callback) or [Compiler](../../glossary.md#term-compiler)-owned aggregate.

`PreparationResourceAssignment` associates each exact declaration
with an assigned [Runtime](../../glossary.md#term-runtime) slot and dense plan index. `BackendPartitionFinalization` carries one
typed analysis, the exact shared [`PreparedMemoryPlan`](../../glossary.md#prepared-memory-plan--preparedmemoryplan), and assignments in declaration order to a
`BackendPartitionFinalizer`. The finalizer returns one
`BackendPartitionFinalizationResult`, which snapshots its exact-plan executable and persistent
resources in physical acquisition order. [`PreparedPartition`](../../glossary.md#prepared-partition--preparedpartition) retains only the exact planned
partition and returned [`PreparedExecutable`](../../glossary.md#prepared-executable--preparedexecutable); resource [ownership](../../glossary.md#term-ownership) is never inferred from that
association or from schedule occurrences. `PartitionPreparation` keeps each backend's typed
inputs, preparer, and finalizer positionally associated. [`PreparedBufferAssignment`](../../glossary.md#prepared-buffer-assignment--preparedbufferassignment) translates one
graph [`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) to its exact [`BufferSlot`](../../glossary.md#buffer-slot--bufferslot) and dense plan index. [`PreparedScheduleContext`](../../glossary.md#prepared-schedule-context--preparedschedulecontext) exposes
only validated stable [Model](../../glossary.md#term-model), [Planning](../../glossary.md#term-planning), Prepare, and Runtime facts to one explicit
`PreparedScheduleAssembler`: planned partitions, graph values, bindable source IDs, constants,
ordered publication IDs, the memory plan, prepared partitions, and buffer assignments. It
contains no Compiler aggregate.
`GraphPreparation.prepare(...)` rejects repeated resource [identities](../../glossary.md#term-identity-meanings), rolls successful results
back after any later preparation failure, and transfers the ordered unique resources only by
returning the exact validated [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution). Shared Prepare invokes cleanup but neither
looks up a resource nor performs its physical allocation.

The ordinary three-[argument](../../glossary.md#term-argument) preparation form identifies every producerless, consumerless
published graph input, classified by the existing caller or constant source plan. It asks the
supplied assembler for a [`ProducerlessPublishedValueResource`](../../glossary.md#producerless-published-value-resource--producerlesspublishedvalueresource), retaining five components:
the exact stable value and logical requirement, backend owner, existing Runtime `ByteSize size`,
and [byte alignment](../../glossary.md#term-byte-alignment). The default geometry hook is
`producerlessPublishedValue(GraphValue, LogicalMemoryRequirement, Optional<ScalarValue>)`;
the [scalar](../../glossary.md#term-scalar) is present only for an authenticated constant, empty only for a caller. No [payload](../../glossary.md#artifact-envelope-and-atomic-publication)
is read or representation allocated by this contribution. The four-argument form accepts the
complete explicit contribution list and uses the same exact-reference, role and coverage checks.

Source-only callers receive real slots in Compiler bindable occurrence order, before remaining
ordinary declarations; source-only constants remain last in final graph-value order. Neither
source receives a partition-local analysis or finalizer assignment. Caller payloads remain [borrowed](../../glossary.md#ownership-borrowing-and-lease);
constants retain existing fresh per-[run](../../glossary.md#term-run) initialization. Static affine [layouts](../../glossary.md#layout) use Model [referenced
span](../../glossary.md#referenced-element-span); named templates use checked existing [binding](../../glossary.md#term-binding) recipes, including structurally symbolic
zero-byte geometry. Complete source facts participate in explicit [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) mode. A genuinely fixed
[CPU](../../glossary.md#term-cpu) unit within that complete batch plan authenticates its fixed selected geometry without changing
its [lowering](../../glossary.md#lowering) or relaxing Runtime defaults.

`PrepareContext.validateValueShapes(List<GraphValue>)` shares the existing fixed-or-sole-named-B
[shape](../../glossary.md#shape) proof and returns the common [symbol](../../glossary.md#term-symbol), if any. It does not establish caller-[axis](../../glossary.md#term-axis) authority,
ROW [provenance](../../glossary.md#provenance) or backend [admission](../../glossary.md#term-admission).

The complete-set operation that validates coverage and source [identity](../../glossary.md#term-identity-meanings), assigns slots, constructs
the shared memory plan, and invokes finalizers remains package-private behind the public graph
operation. Its current conservative
policy shares one [buffer slot](../../glossary.md#buffer-slot--bufferslot) across declarations of the same `ValueId` using maximum geometry
and gives every workspace declaration a distinct slot. It performs no physical allocation,
closeable prepared-resource acquisition, run-state creation, invocation binding/execution,
transfer, [residency](../../glossary.md#residency), constant initialization or materialization, or publication. Producerless
contributions still require at least one non-empty planned partition and therefore do not make a
zero-node graph executable. The supplied assembler constructs an
immutable schedule recipe once; Prepare validates exact source, execution, [coordinate](../../glossary.md#axis-rank-scalar-vector-and-matrix), and
publication coverage before constructing the prepared root. Ordinary [Engine](../../glossary.md#term-engine) composes this
operation from all exact partition owners, backend physical contributions, and explicit transfer
recipes captured during cold preparation.


---

[Previous topic](contracts.md) · [Next topic](../engine/lifecycle.md)
## The two calls on either side of Prepare

Engine calls `GraphPreparation.prepare` with Compiler's immutable artifact and explicit backend
collaborators. Prepare calls each selected backend's analyzer to describe resource needs, assigns
slots, then calls its finalizer with those exact assignments. A successful finalizer transfers
its acquired persistent resources to Prepare; Prepare either rolls them back on a later failure
or transfers them to the returned Runtime `PreparedExecution`. No application input bytes cross
this setup boundary. At run time, backend creators supply fresh per-run representations under
[the creation plan](creation.md); [Engine composition](../engine/composition.md) owns the longer-lived
opened integrations.

