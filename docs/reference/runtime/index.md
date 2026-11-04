<a id="runtime-api"></a>
# [Runtime](../../glossary.md#term-runtime) purpose and [mental model](../../glossary.md#term-mental-model)

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md)<span data-glossary-navigation="true"> ([Javadoc glossary](../../glossary.md#term-javadoc); [signatures glossary](../../glossary.md#term-signature))</span> · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

Think of Runtime as following an already prepared work list, not interpreting a [Tensor](../../glossary.md#tensor) [graph](../../glossary.md#graph).
A **schedule** gives the order of actions, **slots** name planned storage roles, and
**representations** supply the [backend](../../glossary.md#backend)'s concrete resources for those roles. One **[run state](../../glossary.md#run-state--runstate)**
tracks a single invocation's resources and which [buffer](../../glossary.md#term-buffer) copies currently hold usable values.

<a id="purpose-and-implementation-status"></a>
## Purpose and implementation status

This reference explains the implemented [Runtime](../../glossary.md#term-runtime) foundations for prepared-memory geometry,
reusable representation creation descriptions, one [run](../../glossary.md#term-run)'s physical-representation [lifecycle](../../glossary.md#lifecycle) and
validity, checked [binding](../../glossary.md#term-binding) of [backend](../../glossary.md#backend) recipes, ordered scheduling, [prepared publication](../../glossary.md#prepared-publication--preparedpublication), the
completed-result [lease](../../glossary.md#term-lease), the [immutable](../../glossary.md#term-immutable)
prepared-execution root, and the current [Prepare](../../glossary.md#prepare)-owned orchestration that constructs geometry,
finalizes backend recipes, validates the complete schedule, and returns that root. The current
public surface contains:

- `runtime.memory`: [`BufferSlot`](../../glossary.md#buffer-slot--bufferslot), [`WorkspaceSlot`](../../glossary.md#workspace-slot--workspaceslot), and [`PreparedMemoryPlan`](../../glossary.md#prepared-memory-plan--preparedmemoryplan) with its nested
  `BufferEntry` and `WorkspaceEntry` records;
- `runtime.resource`: the [nominal](../../glossary.md#term-nominal) `BufferRepresentation` and `WorkspaceRepresentation` lifecycle
  roles implemented by concrete backends, the persistent [`PreparedResource`](../../glossary.md#prepared-resource--preparedresource) cleanup role, plus
  [`PreparedRepresentationPlan`](../../glossary.md#prepared-representation-plan--preparedrepresentationplan) and its nested preparation and creator contracts;
- `runtime.run`: `RunResourceOwnership`, `BufferRepresentationBinding`, [`RunState`](../../glossary.md#run-state--runstate),
  [`PreparedPublication`](../../glossary.md#prepared-publication--preparedpublication), [`BoundPublication`](../../glossary.md#bound-publication--boundpublication), [`RunResult`](../../glossary.md#run-result--runresult), and stateless
  [`PreparedExecutionRunner`](../../glossary.md#prepared-execution-runner--preparedexecutionrunner);
- `runtime.execution`: [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution), [`PreparedExecutable`](../../glossary.md#prepared-executable--preparedexecutable), its nested `BufferSelection`,
  `WorkspaceSelection`, and `BufferAccess` contracts, [`BoundInvocation`](../../glossary.md#bound-invocation--boundinvocation),
  [`PreparedBufferTransfer`](../../glossary.md#prepared-buffer-transfer--preparedbuffertransfer), and [`BoundBufferTransfer`](../../glossary.md#bound-buffer-transfer--boundbuffertransfer); and
- `runtime.schedule`: [`PreparedSchedule`](../../glossary.md#prepared-schedule--preparedschedule), its [sealed](../../glossary.md#term-sealed) nested `Step` contract,
  `RepresentationCreationStep`, `ExecutionStep`, `BufferTransferStep`, and `PublicationStep`.

The geometry, reusable creation description, package-private all-or-cleaned setup, structural
[residency](../../glossary.md#residency), explicit per-[buffer](../../glossary.md#term-buffer)-copy validity, cold-[bound invocation](../../glossary.md#bound-invocation--boundinvocation) and transfer contracts,
creation, execution, transfer, and dense [publication](../../glossary.md#term-publication)-suffix schedule recipes, the whole-state
result lease, lifecycle-bearing prepared-execution root, Prepare-owned resource assignments,
typed backend finalization, [`PreparedPartition`](../../glossary.md#prepared-partition--preparedpartition), and complete [graph preparation](../../glossary.md#graph-preparation--graphpreparation) are current.
Ordinary [compile](../../glossary.md#compile), shared single-[owner](../../glossary.md#term-owner) or mixed-owner prepare, logical [host](../../glossary.md#term-host)-input binding,
[synchronous](../../glossary.md#term-synchronous) run, typed publication [metadata](../../glossary.md#term-metadata), and explicit bounded [detached](../../glossary.md#detached-model-results) host [materialization](../../glossary.md#materialization)
are current through [`Engine`](../../glossary.md#term-engine). `Engine.standard()` supplies fixed [CPU](../../glossary.md#term-cpu) [ownership](../../glossary.md#term-ownership-meanings);
`Engine.builder()` can explicitly compose CPU and [Metal](../../glossary.md#term-metal). Their exact integrations implement
physical allocation, storage access, finalization and contribution, execution, transfer endpoints,
ingress, and materialization for their supported domains.

<a id="mental-model"></a>
## [Mental model](../../glossary.md#term-mental-model)

```text
compile facts             prepare handoff               Runtime state
ValueId / requirement -> source-to-slot assignment -> PreparedMemoryPlan
current                  current                      current
                                                              |
                         PreparedRepresentationPlan ----------+
                         current reusable creation description |
                                                              v
                         cold setup -> RunState + validity
                         current internal  current per run
     PreparedMemoryPlan + PreparedSchedule + persistent resources -> PreparedExecution
     current              current            current contract       current reusable owner

        PreparedExecutionRunner -> lease -> create/bind all -> ordered direct traversal
        current public runner      current  cold setup          current synchronous run

        BufferTransferStep -> cold bind -> BoundBufferTransfer -> transfer + validity
        current recipe        current       current per run      current contract

        PublicationStep -> cold bind -> BoundPublication -> publish -> RunResult
        current recipe      current       current per run     current    current lease
```

Read the flow from logical source facts toward invocation state. [Prepare](../../glossary.md#prepare) exposes exact [buffer](../../glossary.md#term-buffer) and
workspace requirements through [`BackendPartitionAnalysis`](../../glossary.md#backend-partition-analysis-backendpartitionanalysis), then its current package-internal
[handoff](../../glossary.md#term-handoff) retains source-to-slot associations and constructs the [Runtime](../../glossary.md#term-runtime) geometry. Public
`GraphPreparation.prepare(...)` [coordinates](../../glossary.md#axis-rank-scalar-vector-and-matrix) that handoff and validates one explicitly supplied
schedule recipe before returning [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution). Current [`PreparedRepresentationPlan`](../../glossary.md#prepared-representation-plan--preparedrepresentationplan)
describes [borrowed](../../glossary.md#ownership-borrowing-and-lease) inputs and concrete-[backend](../../glossary.md#backend) creators in that geometry's encounter order.
Package-private cold setup validates all caller inputs, creates fresh [run](../../glossary.md#term-run)-owned representation
objects and workspaces, and constructs one [`RunState`](../../glossary.md#run-state--runstate). A creator may return a fresh read-only
[binding](../../glossary.md#term-binding) whose child [lease](../../glossary.md#term-lease) refers to [immutable](../../glossary.md#term-immutable) prepared physical storage; object freshness and
run-owned cleanup remain unchanged. A current [`PreparedExecutable`](../../glossary.md#prepared-executable--preparedexecutable) selects those resident
representations by dense position, checks backend compatibility during cold binding, and creates a
per-run [`BoundInvocation`](../../glossary.md#bound-invocation--boundinvocation). A current [`PreparedBufferTransfer`](../../glossary.md#prepared-buffer-transfer--preparedbuffertransfer) selects two distinct already-created
representations of one buffer, checks their concrete compatibility during cold binding, and creates
a per-run [`BoundBufferTransfer`](../../glossary.md#bound-buffer-transfer--boundbuffertransfer) that orchestrates the explicit validity transition around
backend-owned physical transfer work. The shared contracts still implement no concrete allocation
or storage access.
The current schedule can retain one first-only creation prefix
followed by executable or transfer occurrences and then a dense [publication](../../glossary.md#term-publication)-only suffix. It does
not invoke or execute any step. Current publication names an already-created valid copy and
leases the complete state to a result. While that exact lease remains open, the inward Runtime
result can return the borrowed [nominal](../../glossary.md#term-nominal) representation selected for one result [occurrence](../../glossary.md#term-occurrence); this
is not output-value or storage access. The runner composes these contracts without backend
discovery or [graph](../../glossary.md#graph) interpretation.


---

[Next topic](engine-boundary.md)

## Topic navigation

- [Runtime public contract relationships](contracts.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime))</span>
- [Prepare declarations, assignment and finalization](prepare.md)<span data-glossary-navigation="true"> ([Prepare glossary](../../glossary.md#prepare))</span>
- [Public Engine binding and results](engine-boundary.md)<span data-glossary-navigation="true"> ([Engine glossary](../../glossary.md#term-engine); [binding glossary](../../glossary.md#term-binding))</span>
- [Advanced Engine and prepared execution](prepared-execution.md)<span data-glossary-navigation="true"> ([prepared execution glossary](../../glossary.md#prepared-execution--preparedexecution))</span>
- [Slots and prepared-memory geometry](memory.md)
- [Per-run representations and validity](representations.md)<span data-glossary-navigation="true"> ([run glossary](../../glossary.md#term-run))</span>
- [Representation creation, run-state closure and failures](creation.md)
- [Prepared executables and cold invocation binding](executables.md)
- [Prepared buffer transfers](transfers.md)<span data-glossary-navigation="true"> ([buffer glossary](../../glossary.md#term-buffer))</span>
- [Ordered prepared schedules](schedules.md)
- [Prepared publication and result leases](publication.md)<span data-glossary-navigation="true"> ([Prepared publication glossary](../../glossary.md#prepared-publication--preparedpublication); [leases glossary](../../glossary.md#term-lease))</span>
- [Prepare finalization and run orchestration](orchestration.md)
