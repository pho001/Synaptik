# Prepare finalization and run orchestration

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

Orchestration means arranging existing stages in the right order. **Analysis** declares the
backend's resource needs; **assignment** gives those needs stable slots; **finalization** builds
recipes using the assignments. The runner later binds and executes them for one invocation.
Preparation failure cleanup releases acquired resources rather than returning a partial owner.

<a id="current-prepare-orchestration-assignment-and-backend-finalization"></a>
## Current Prepare orchestration, assignment, and backend finalization

The public root of `io.github.pho001.synaptik.prepare` contains the finalization contracts and the
complete graph-preparation operation that bridge analysis declarations to current Runtime
recipes:

- `PreparationResourceAssignment.Buffer` and `.Workspace` retain one exact analysis requirement,
  its exact assigned slot, and the dense index of that slot's entry in `PreparedMemoryPlan`;
- `BackendPartitionFinalization<P>` retains one exact typed analysis, the exact shared memory
  plan, and an immutable assignment list in analysis-requirement order;
- `BackendPartitionFinalizer<P>` is implemented by the owning concrete backend to construct one
  immutable `PreparedExecutable` recipe after assignment; and
- `PreparedPartition` retains the exact `PlannedPartition` and finalized executable references.

`PartitionPreparation<I,P>` positionally associates backend inputs, a typed preparer, and its
matching finalizer. `GraphPreparation.prepare(...)` accepts one complete `CompileArtifacts`, one
such value per planned partition, and one explicit `PreparedScheduleAssembler`. Its ordinary
three-argument form identifies each producerless, consumerless published caller or constant input
and requests a `ProducerlessPublishedValueResource` from the assembler's default
`producerlessPublishedValue(value, logicalRequirement, Optional<ScalarValue>)` hook. The scalar is
present exactly for a constant, empty exactly for a caller. The five-component contribution retains
the exact graph value and logical requirement, backend owner, an existing immutable `ByteSize`
recipe and alignment. The four-argument form accepts the complete explicit contribution list.

Before backend work, Prepare snapshots the lists and validates exact references, authenticated
disjoint source roles, required publication, complete unique coverage and no producer or consumer.
The shared `PrepareContext.validateValueShapes(List<GraphValue>)` guard retains the existing
fixed-or-sole-named-B vocabulary; actual caller-axis authority and backend admission remain separate.
Callers receive leading slots in exact Compiler occurrence order, including source-only callers;
remaining ordinary declarations retain encounter order and producerless constants remain last.
No source contribution is added to a partition-local `PrepareContext` or finalizer requirement.

ByteSize recipes remain unevaluated until existing run geometry resolves actual B once before
allocation. Complete source named facts determine batch mode even for zero bytes. Fixed CPU units
within that complete batch plan retain their static lowering and authenticate fixed selected
geometry through existing cold hooks; Runtime defaults remain unchanged. Caller borrowing,
constant initialization and publication use existing owners. At least one ordinary non-empty
planned partition remains required; this handoff does not enable zero-node execution.

The operation constructs all partition contexts before analysis, invokes each preparer and
finalizer once in partition order, then gives the assembler one immutable
`PreparedScheduleContext`. That context contains no Compiler aggregate. It retains the validated
stable planned partitions, graph values, bindable-source IDs, constants, ordered publication IDs,
shared memory plan, prepared partitions, and `PreparedBufferAssignment` values that map graph
`ValueId` values to dense Runtime buffer positions.

The package-internal complete-set handoff validates expected partition coverage, exact projected
source references, and backend ownership before assignment. It first seeds caller buffer slots in
exact Compiler bindable occurrence order, using each caller's ordinary declaration or authenticated
source-only contribution. It then traverses ordinary partitions and requirements in stored order.
Each remaining first-seen buffer `ValueId` receives the next dense slot; later declarations share
that exact slot and retain maximum declared size and alignment. Every workspace declaration receives
a fresh dense slot with unchanged geometry. Only canonical source-only constants are appended after
ordinary declarations, with their supplied geometry unchanged. Producerless contributions receive
no finalizer assignment because no partition owns them. No lifetime, interference, aliasing, or
reuse model is inferred.

Every typed finalization is constructed before any backend is invoked. Finalizers then run once
in partition order, and each returned finalization result must contain a non-null executable that
retains the exact shared plan object. Finalization may construct immutable reusable recipe state
and acquire identity-unique persistent prepared resources. The backend rolls those resources back
if finalization cannot return successfully; after return, Prepare owns rollback until the complete
acquisition-ordered resource list transfers exactly once to Runtime's `PreparedExecution`.
Finalization does not create a `RunState`, bind or execute an invocation, or create a schedule.

The complete-set operation and its batch result remain package-private behind
`GraphPreparation`. The assembler returns a reusable recipe only. Prepare validates exact plan
identity, caller-input order, constant initialization, executable coverage and order,
representation coordinates, and ordered forward-then-gradient publication before returning
`PreparedExecution`. A compile-time constant source must use at least one `InitializedBuffer` and
no caller-input occurrence; an initialized representation is rejected for a nonconstant value.
Repeated gradient values and forward/gradient aliases remain separate publication occurrences
over the same assigned buffer.

Graph preparation performs no physical allocation, creator invocation, binding, execution,
constant initialization or materialization, transfer, publication, or cleanup. It contains the
Compiler aggregate only in shared Prepare; concrete backend-facing `PrepareContext` values remain
Compiler-free. The current ordinary Engine maps logical caller inputs and publication roles to
these Runtime coordinates without exposing them. Prepare 0005 supplies only the shared handoff and
assignment capability. CPU 0010H now contributes the physical geometry and initialized
representation recipe, and Engine 0006 wires the supported source-only positive-one seed into the
ordinary one-shot backward lifecycle. No new Runtime contract was required.

<a id="current-aggregate-and-run-orchestration"></a>
## Current aggregate and run orchestration

The current `PreparedExecution` privately retains the exact memory plan, exact same-plan schedule,
and an identity-unique snapshot of persistent prepared resources in acquisition order. Executable
recipes remain reachable through schedule occurrences, while `PreparedPartition` remains a
Prepare-owned association and does not cross into this Runtime aggregate. Schedule or executable
repetition never duplicates resource ownership. There is no distinct `PreparedUnit`: list
position is the occurrence, and the exact executable supplies the work recipe and memory-plan
association.

Public Prepare orchestration now passes the acquisition-ordered, identity-unique resources from
successful backend finalization results to the three-argument constructor. Until that constructor
succeeds, shared Prepare rolls the resources back after any later finalizer, association,
schedule-assembly, schedule-validation, or aggregate-construction failure. CPU finalization
returns the existing executable with an empty resource list. Ordinary and advanced Engine
prepared handles now each own and close one such inward owner. Temporary one-shot and
representative preparations close lexically after their results, while published handles are
retained for explicit close or Engine shutdown.

<a id="current-prepared-runner"></a>
## Current prepared runner

```java
PreparedExecutionRunner runner = new PreparedExecutionRunner();
RunResult result = runner.run(execution, callerInputs);
```

- The runner acquires an opaque prepared-execution lease before inspecting `callerInputs` or
  creating a `RunState`, and releases it after the complete synchronous call.
- `callerInputs` supplies dense borrowed representations in creation-plan encounter order.
- Exactly one isolated `RunState` is created and consumed for the complete heterogeneous run.
- Every executable, transfer, and publication occurrence cold-binds before the first action.
- Traversal uses direct bound references and precomputed primitive executable coordinates.
- The current Runtime `RunResult` leases the whole run state and permits borrowed access to one
  selected nominal representation while that lease is open. It exposes no host or Tensor value;
  ordinary Engine separately composes the current CPU copy into a detached host value.
- A run failure remains primary if deferred persistent-resource cleanup also fails. If execution
  succeeded but lease cleanup fails, the new result is closed and is not returned; any distinct
  result-cleanup failure is suppressed on the persistent-resource failure.

Successful runner return therefore exposes completed state, not pending device work. The result
may outlive the call because completion does not release its run-owned output/publication resources.
Concurrent calls remain independent without any overlap, order, fairness, queue, or throughput
guarantee. Materialization and transfers remain synchronous at their existing boundaries.

Current ownership distinguishes borrowed inputs from run-owned internal resources, and current
per-copy validity is explicit within `RunState`. Current publication leases the complete state to
`RunResult`, while immutable persistent prepared resources stay with `PreparedExecution`.
Current cold checked binding creates backend-owned typed invocation and
transfer objects with direct references. One exact prepared buffer transfer and its success-only
destination-valid transition are current. Prepared publication, its suffix ordering, one-shot
validity check, alias preservation, empty result, and whole-state lease are also current.
Cold preparation supplies the selected transfer recipes; the runner does not discover or
select a transfer route during execution. It validates declared reads, invalidates every copy
of each declared output buffer before backend work, validates exact writes only after success,
and closes state after a post-creation failure while preserving the original unchecked failure.
The runner itself performs no Trace emission.

<a id="boundary-and-failure-model"></a>
## Boundary and failure model

Run may fail because an input binding is missing or incompatible, a prepared resource cannot be used, transfer or execution fails, or publication fails. It must not recover by discovering another backend and lowering again. Unsupported work must be resolved during compile ownership or fail during prepare.

<a id="related-contracts"></a>
## Related contracts

- [Runtime, prepare, and backend boundary](../../architecture/runtime-prepare-backend-boundary.md)
- [Preparing execution](../../user-guide/preparing-execution.md)
- [Running models](../../user-guide/running-models.md)
- [Glossary](../../glossary.md)

<a id="numerical-profile-boundary"></a>
## Numerical-profile boundary

Runtime APIs deliberately expose no numerical-profile selector or per-run override. Engine captures
the graph-wide profile before compilation, and backend preparation resolves it before a
`PreparedExecutable` reaches Runtime. This keeps run state, bindings, and Trace payloads free of
numerical-policy branches.

---

[Previous topic](publication.md)
