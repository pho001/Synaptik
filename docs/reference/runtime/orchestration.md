# [Prepare](../../glossary.md#prepare) finalization and [run](../../glossary.md#term-run) orchestration

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

Orchestration means arranging existing stages in the right order. **Analysis** declares the
[backend](../../glossary.md#backend)'s resource needs; **assignment** gives those needs stable slots; **finalization** builds
recipes using the assignments. The runner later binds and executes them for one invocation.
Preparation failure cleanup [releases](../../glossary.md#commit-release-distribution-log-and-seed-meanings) acquired resources rather than returning a partial [owner](../../glossary.md#term-owner).

## Separate the preparation transaction from one run

Engine calls shared Prepare with artifacts and explicit backend collaborators. Prepare calls
backend analyzers/finalizers and returns a reusable owner only after schedule validation; on a
later setup failure it closes persistent resources already handed to it. This is cleanup of
acquired resources, not reversal of arbitrary backend effects. A later Engine run calls Runtime,
which invokes creators, binds recipes and executes one isolated state. Closing a result cleans
that state's owned resources; closing preparation cleans its persistent resources once admitted
run leases permit it. Read [creation failure cleanup](creation.md) and
[prepared-owner leases](prepared-execution.md#current-prepared-execution) for the two different
failure/lifetime boundaries.

<a id="current-prepare-orchestration-assignment-and-backend-finalization"></a>
## Current [Prepare](../../glossary.md#prepare) orchestration, assignment, and [backend](../../glossary.md#backend) finalization

The public root of `io.github.pho001.synaptik.prepare` contains the finalization contracts and the
complete [graph](../../glossary.md#graph)-preparation [operation](../../glossary.md#operation) that bridge analysis declarations to current [Runtime](../../glossary.md#term-runtime)
recipes:

- `PreparationResourceAssignment.Buffer` and `.Workspace` retain one exact analysis requirement,
  its exact assigned slot, and the dense index of that slot's entry in [`PreparedMemoryPlan`](../../glossary.md#prepared-memory-plan--preparedmemoryplan);
- `BackendPartitionFinalization<P>` retains one exact typed analysis, the exact shared memory
  plan, and an immutable assignment list in analysis-requirement order;
- `BackendPartitionFinalizer<P>` is implemented by the owning concrete backend to construct one
  immutable [`PreparedExecutable`](../../glossary.md#prepared-executable--preparedexecutable) recipe after assignment; and
- [`PreparedPartition`](../../glossary.md#prepared-partition--preparedpartition) retains the exact `PlannedPartition` and finalized executable references.

`PartitionPreparation<I,P>` positionally associates backend inputs, a typed preparer, and its
matching finalizer. `GraphPreparation.prepare(...)` accepts one complete `CompileArtifacts`, one
such value per planned [partition](../../glossary.md#partition), and one explicit `PreparedScheduleAssembler`. Its ordinary
three-[argument](../../glossary.md#term-argument) form identifies each producerless, consumerless published caller or [constant](../../glossary.md#term-constant) input
and requests a [`ProducerlessPublishedValueResource`](../../glossary.md#producerless-published-value-resource--producerlesspublishedvalueresource) from the assembler's default
`producerlessPublishedValue(value, logicalRequirement, Optional<ScalarValue>)` hook. The [scalar](../../glossary.md#term-scalar) is
present exactly for a constant, empty exactly for a caller. The five-component contribution retains
the exact [graph value](../../glossary.md#graph-value) and logical requirement, backend [owner](../../glossary.md#term-owner), an existing [immutable](../../glossary.md#term-immutable) `ByteSize`
recipe and [alignment](../../glossary.md#term-alignment). The four-argument form accepts the complete explicit contribution list.

Before backend work, Prepare snapshots the lists and validates exact references, authenticated
disjoint source roles, required [publication](../../glossary.md#term-publication), complete unique coverage and no [producer](../../glossary.md#term-producer) or consumer.
The shared `PrepareContext.validateValueShapes(List<GraphValue>)` [guard](../../glossary.md#term-guard) retains the existing
fixed-or-sole-named-B vocabulary; actual caller-[axis](../../glossary.md#term-axis) authority and backend [admission](../../glossary.md#term-admission) remain separate.
Callers receive leading slots in exact [Compiler](../../glossary.md#term-compiler) [occurrence](../../glossary.md#term-occurrence) order, including source-only callers;
remaining ordinary declarations retain encounter order and producerless constants remain last.
No source contribution is added to a partition-local [`PrepareContext`](../../glossary.md#prepare-context-preparecontext) or finalizer requirement.

ByteSize recipes remain unevaluated until existing [run](../../glossary.md#term-run) geometry resolves actual B once before
allocation. Complete source named facts determine [batch](../../glossary.md#term-batch-machine-learning-and-hyperparameters) mode even for zero bytes. Fixed [CPU](../../glossary.md#term-cpu) units
within that complete batch plan retain their static [lowering](../../glossary.md#lowering) and authenticate fixed selected
geometry through existing cold hooks; Runtime defaults remain unchanged. Caller [borrowing](../../glossary.md#term-borrowing),
constant initialization and publication use existing owners. At least one ordinary non-empty
planned partition remains required; this [handoff](../../glossary.md#term-handoff) does not enable zero-[node](../../glossary.md#node) execution.

The operation constructs all partition contexts before analysis, invokes each preparer and
finalizer once in partition order, then gives the assembler one immutable
[`PreparedScheduleContext`](../../glossary.md#prepared-schedule-context--preparedschedulecontext). That context contains no Compiler aggregate. It retains the validated
stable planned partitions, graph values, bindable-source IDs, constants, ordered publication IDs,
shared memory plan, prepared partitions, and [`PreparedBufferAssignment`](../../glossary.md#prepared-buffer-assignment--preparedbufferassignment) values that map graph
[`ValueId`](../../glossary.md#logical-valueid-versus-physical-memory-slot) values to dense Runtime [buffer](../../glossary.md#term-buffer) positions.

The package-internal complete-set handoff validates expected partition coverage, exact projected
source references, and [backend ownership](../../glossary.md#backend-ownership) before assignment. It first [seeds](../../glossary.md#commit-release-distribution-log-and-seed-meanings) caller buffer slots in
exact Compiler bindable occurrence order, using each caller's ordinary declaration or authenticated
source-only contribution. It then traverses ordinary partitions and requirements in stored order.
Each remaining first-seen buffer `ValueId` receives the next dense slot; later declarations share
that exact slot and retain maximum declared size and alignment. Every workspace declaration receives
a fresh dense slot with unchanged geometry. Only [canonical](../../glossary.md#term-canonical-form) source-only constants are appended after
ordinary declarations, with their supplied geometry unchanged. Producerless contributions receive
no finalizer assignment because no partition owns them. No [lifetime](../../glossary.md#term-lifetime), interference, [aliasing](../../glossary.md#term-aliasing), or
reuse [model](../../glossary.md#term-model) is inferred.

Every typed finalization is constructed before any backend is invoked. Finalizers then run once
in partition order, and each returned finalization result must contain a non-null executable that
retains the exact shared plan object. Finalization may construct immutable reusable recipe state
and acquire [identity](../../glossary.md#term-identity-meanings)-unique persistent prepared resources. The backend rolls those resources back
if finalization cannot return successfully; after return, Prepare owns [rollback](../../glossary.md#term-rollback) until the complete
acquisition-ordered resource list transfers exactly once to Runtime's [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution).
Finalization does not create a [`RunState`](../../glossary.md#run-state--runstate), bind or execute an invocation, or create a schedule.

The complete-set operation and its batch result remain package-private behind
[`GraphPreparation`](../../glossary.md#graph-preparation--graphpreparation). The assembler returns a reusable recipe only. Prepare validates exact plan
identity, caller-input order, constant initialization, executable coverage and order,
representation coordinates, and ordered forward-then-[gradient](../../glossary.md#term-gradient) publication before returning
`PreparedExecution`. A [compile-time constant source](../../glossary.md#compile-time-constant-source) must use at least one `InitializedBuffer` and
no caller-input occurrence; an initialized representation is rejected for a nonconstant value.
Repeated gradient values and forward/gradient [aliases](../../glossary.md#aliasing-and-arena) remain separate publication occurrences
over the same assigned buffer.

[Graph preparation](../../glossary.md#graph-preparation--graphpreparation) performs no physical allocation, creator invocation, [binding](../../glossary.md#term-binding), execution,
constant initialization or [materialization](../../glossary.md#materialization), transfer or publication.
On preparation failure, it closes already-acquired persistent resources as described above. It contains the
Compiler aggregate only in shared Prepare; [concrete backend](../../glossary.md#concrete-backend)-facing `PrepareContext` values remain
Compiler-free. The current ordinary [Engine](../../glossary.md#term-engine) maps logical caller inputs and publication roles to
these Runtime coordinates without exposing them. Prepare 0005 supplies only the shared handoff and
assignment capability. CPU 0010H now contributes the physical geometry and initialized
representation recipe, and Engine 0006 wires the supported source-only positive-one [seed](../../glossary.md#term-seed) into the
ordinary one-shot backward [lifecycle](../../glossary.md#lifecycle). No new Runtime contract was required.

<a id="current-aggregate-and-run-orchestration"></a>
## Current aggregate and [run](../../glossary.md#term-run) orchestration

The current [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution) privately retains the exact memory plan, exact same-plan schedule,
and an [identity](../../glossary.md#term-identity-meanings)-unique [snapshot](../../glossary.md#term-snapshot) of persistent prepared resources in acquisition order. Executable
recipes remain reachable through schedule occurrences, while [`PreparedPartition`](../../glossary.md#prepared-partition--preparedpartition) remains a
[Prepare](../../glossary.md#prepare)-owned association and does not cross into this [Runtime](../../glossary.md#term-runtime) aggregate. Schedule or executable
repetition never duplicates resource [ownership](../../glossary.md#term-ownership). There is no distinct `PreparedUnit`: list
position is the [occurrence](../../glossary.md#term-occurrence), and the exact executable supplies the work recipe and memory-plan
association.

Public Prepare orchestration now passes the acquisition-ordered, identity-unique resources from
successful [backend](../../glossary.md#backend) finalization results to the three-[argument](../../glossary.md#term-argument) constructor. Until that constructor
succeeds, shared Prepare rolls the resources back after any later finalizer, association,
schedule-assembly, schedule-[validation](../../glossary.md#term-validation), or aggregate-construction failure. [CPU](../../glossary.md#term-cpu) finalization
returns the existing executable with an empty resource list. Ordinary and advanced [Engine](../../glossary.md#term-engine)
prepared handles now each own and close one such inward [owner](../../glossary.md#term-owner). Temporary one-shot and
representative preparations close lexically after their results, while published handles are
retained for explicit close or Engine shutdown.

<a id="current-prepared-runner"></a>
## Current prepared runner

```java
// Runtime borrows caller representations; the returned result now owns cleanup of completed run state.
PreparedExecutionRunner runner = new PreparedExecutionRunner();
RunResult result = runner.run(execution, callerInputs);
```

- The runner acquires an [opaque](../../glossary.md#term-opaque) prepared-execution [lease](../../glossary.md#term-lease) before inspecting `callerInputs` or
  creating a [`RunState`](../../glossary.md#run-state--runstate), and [releases](../../glossary.md#commit-release-distribution-log-and-seed-meanings) it after the complete [synchronous](../../glossary.md#term-synchronous) call.
- `callerInputs` supplies dense [borrowed](../../glossary.md#ownership-borrowing-and-lease) representations in creation-plan encounter order.
- Exactly one isolated `RunState` is created and consumed for the complete [heterogeneous](../../glossary.md#term-heterogeneous) [run](../../glossary.md#term-run).
- Every executable, transfer, and [publication occurrence](../../glossary.md#publication-occurrence) cold-binds before the first action.
- Traversal uses direct bound references and precomputed primitive executable coordinates.
- The current [Runtime](../../glossary.md#term-runtime) [`RunResult`](../../glossary.md#run-result--runresult) leases the whole run state and permits borrowed access to one
  selected nominal representation while that lease is open. It exposes no host or Tensor value;
  ordinary Engine separately composes the current CPU copy into a detached host value.
- A run failure remains primary if deferred persistent-resource cleanup also fails. If execution
  succeeded but lease cleanup fails, the new result is closed and is not returned; any distinct
  result-cleanup failure is suppressed on the persistent-resource failure.

Successful runner return therefore exposes completed state, not pending [device](../../glossary.md#term-device) work. The result
may outlive the call because completion does not [release](../../glossary.md#term-release) its run-owned output/[publication](../../glossary.md#term-publication) resources.
Concurrent calls remain independent without any overlap, order, fairness, queue, or [throughput](../../glossary.md#term-throughput)
guarantee. [Materialization](../../glossary.md#materialization) and transfers remain synchronous at their existing boundaries.

Current [ownership](../../glossary.md#term-ownership-meanings) distinguishes borrowed inputs from run-owned internal resources, and current
per-copy validity is explicit within `RunState`. Current publication leases the complete state to
`RunResult`, while [immutable](../../glossary.md#term-immutable) persistent prepared resources stay with [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution).
Current cold checked [binding](../../glossary.md#term-binding) creates [backend](../../glossary.md#backend)-owned typed invocation and
transfer objects with direct references. One exact [prepared buffer transfer](../../glossary.md#prepared-buffer-transfer--preparedbuffertransfer) and its success-only
destination-valid transition are current. [Prepared publication](../../glossary.md#prepared-publication--preparedpublication), its suffix ordering, one-shot
validity check, [alias](../../glossary.md#aliasing-and-arena) preservation, empty result, and whole-state lease are also current.
Cold preparation supplies the selected transfer recipes; the runner does not discover or
select a transfer route during execution. It validates declared reads, invalidates every copy
of each declared output [buffer](../../glossary.md#term-buffer) before backend work, validates exact writes only after success,
and closes state after a post-creation failure while preserving the original unchecked failure.
The runner itself performs no [Trace](../../glossary.md#trace) emission.

<a id="boundary-and-failure-model"></a>
## Boundary and failure [model](../../glossary.md#term-model)

[Run](../../glossary.md#term-run) may fail because an input [binding](../../glossary.md#term-binding) is missing or incompatible, a [prepared resource](../../glossary.md#prepared-resource--preparedresource) cannot be used, transfer or execution fails, or [publication](../../glossary.md#term-publication) fails. It must not recover by discovering another [backend](../../glossary.md#backend) and [lowering](../../glossary.md#lowering) again. Unsupported work must be resolved during [compile](../../glossary.md#compile) [ownership](../../glossary.md#term-ownership-meanings) or fail during [prepare](../../glossary.md#prepare).

<a id="related-contracts"></a>
## Related contracts

- [Runtime, prepare, and backend boundary](../../architecture/runtime-prepare-backend-boundary.md)
- [Preparing execution](../../user-guide/preparing-execution.md)
- [Running models](../../user-guide/running-models.md)
- [Glossary](../../glossary.md)

<a id="numerical-profile-boundary"></a>
## Numerical-profile boundary

[Runtime](../../glossary.md#term-runtime) APIs deliberately expose no numerical-profile selector or per-[run](../../glossary.md#term-run) override. [Engine](../../glossary.md#term-engine) [captures](../../glossary.md#term-capture-meanings)
the [graph](../../glossary.md#graph)-wide profile before [compilation](../../glossary.md#compiler-engine-and-execution-stage-names), and [backend](../../glossary.md#backend) preparation resolves it before a
[`PreparedExecutable`](../../glossary.md#prepared-executable--preparedexecutable) reaches Runtime. This keeps [run state](../../glossary.md#run-state--runstate), [bindings](../../glossary.md#term-binding), and [Trace](../../glossary.md#trace) payloads free of
numerical-policy branches.

---

[Previous topic](publication.md)
