# [Runtime](../../glossary.md#term-runtime) public contract relationships

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

Runtime connects reusable work recipes to the resources of one invocation. **Prepared** objects
describe repeated work; **bound** objects hold that [run](../../glossary.md#term-run)'s concrete references. A **[lease](../../glossary.md#term-lease)** keeps
resources usable for a defined [lifetime](../../glossary.md#term-lifetime) without making the borrower their independent [owner](../../glossary.md#term-owner).
Closing a result ends that result's lease; it does not close the caller's [borrowed](../../glossary.md#ownership-borrowing-and-lease) input storage.

The public `modules:runtime` surface now contains five focused packages:

- `runtime.memory` defines [`BufferSlot`](../../glossary.md#buffer-slot--bufferslot), [`WorkspaceSlot`](../../glossary.md#workspace-slot--workspaceslot), and [immutable](../../glossary.md#term-immutable) ordered
  [`PreparedMemoryPlan`](../../glossary.md#prepared-memory-plan--preparedmemoryplan) byte geometry;
- `runtime.resource` defines the [nominal](../../glossary.md#term-nominal) [backend](../../glossary.md#backend)-implemented `BufferRepresentation` and
  `WorkspaceRepresentation` cleanup roles, the persistent [`PreparedResource`](../../glossary.md#prepared-resource--preparedresource) cleanup role, plus
  immutable [`PreparedRepresentationPlan`](../../glossary.md#prepared-representation-plan--preparedrepresentationplan) origins, caller-input occurrences, ordinary and
  initialized buffer origins, and typed buffer/workspace creators;
- `runtime.run` defines borrowed/run-owned [buffer](../../glossary.md#term-buffer) [bindings](../../glossary.md#term-binding), the array-backed one-run [`RunState`](../../glossary.md#run-state--runstate),
  dense-coordinate [`PreparedPublication`](../../glossary.md#prepared-publication--preparedpublication), per-run [`BoundPublication`](../../glossary.md#bound-publication--boundpublication), the whole-state
  [`RunResult`](../../glossary.md#run-result--runresult) lease, and stateless [`PreparedExecutionRunner`](../../glossary.md#prepared-execution-runner--preparedexecutionrunner); and
- `runtime.execution` defines the immutable-recipe, explicitly closeable [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution)
  owner, reusable
  [`PreparedExecutable`](../../glossary.md#prepared-executable--preparedexecutable) recipes, their dense buffer/representation and workspace selections,
  aligned buffer-access declarations, and
  per-run [`BoundInvocation`](../../glossary.md#bound-invocation--boundinvocation) objects, plus reusable [`PreparedBufferTransfer`](../../glossary.md#prepared-buffer-transfer--preparedbuffertransfer) recipes and per-run
  [`BoundBufferTransfer`](../../glossary.md#bound-buffer-transfer--boundbuffertransfer) actions; and
- `runtime.schedule` defines immutable [`PreparedSchedule`](../../glossary.md#prepared-schedule--preparedschedule) recipes, their [sealed](../../glossary.md#term-sealed) plan-associated
  `Step` family, the optional first-only `RepresentationCreationStep`, `ExecutionStep`,
  `BufferTransferStep`, and the dense final `PublicationStep` suffix.

`PreparedRepresentationPlan` snapshots dense buffer origins and workspace creators against one
exact memory plan. Concrete backends implement immutable [thread-safe](../../glossary.md#term-thread-safe) creator callbacks that return
fresh physical results for each run. Package-private cold setup validates every borrowed caller
input before callbacks, creates buffers then workspaces, and rolls back successful created results
in reverse order if setup fails. The public runner is the narrow composition seam for that setup;
it is not an [Engine](../../glossary.md#term-engine) or value [facade](../../glossary.md#term-facade).

Every representation bound into an open `RunState` is structurally resident until closure. Each
buffer copy has one independent explicit validity bit: borrowed inputs start valid, ordinary
newly created run-owned buffers start invalid, and initialized run-owned buffers start valid.
The initialized creator, not Runtime, materializes the logical value. Workspaces are run-owned
scratch without logical validity. Query and mutation are [constant](../../glossary.md#term-constant)-time dense-array operations and
perform no physical copy, backend call, [ownership](../../glossary.md#term-ownership-meanings) change, or implicit coherence.

`PreparedExecutable.bind` requires the exact same `PreparedMemoryPlan` reference as the open
`RunState`. It resolves selections in supplied order and delegates explicit checked
representation compatibility to the [concrete backend](../../glossary.md#concrete-backend). The backend then creates a
`BoundInvocation` with direct concrete typed fields and the exact state association. The hot
`execute()` call checks only that the retained state is open before delegating to the backend;
execution after state closure is rejected.

Each executable buffer selection declares `READ_ONLY`, `WRITE_ONLY`, or `READ_WRITE` access. The
runner validates reads before invalidating every copy of each output buffer, then validates only
the exact declared writes after successful backend return. A failed action leaves all copies of
its output buffers invalid.

`PreparedBufferTransfer.bind` similarly requires an exact matching open state but selects two
distinct, already-created representations of one logical buffer. Concrete compatibility checks
remain cold, and the resulting `BoundBufferTransfer` stores direct concrete typed fields. An
already-valid destination is a no-op. Otherwise a valid source permits one backend action, and
only successful return marks the destination valid. Backend failure leaves Runtime validity
unchanged. [Materialization](../../glossary.md#materialization) is this same explicit transfer to an equivalent already-created
destination, not another [operation](../../glossary.md#operation), allocation path, or route search.

One schedule retains an exact `PreparedMemoryPlan` and an immutable ordered [snapshot](../../glossary.md#term-snapshot) of creation,
executable, buffer-transfer, and [publication](../../glossary.md#term-publication) occurrences. Every step must report that same plan by [reference
identity](../../glossary.md#term-reference-identity). The
creation [occurrence](../../glossary.md#term-occurrence) is optional for compatibility but, when present, is sole and first. Empty and
executable-only or transfer-only schedules and repeated executable or transfer occurrences remain
valid. Publication occurrences, when present, form a dense `0..N-1` final suffix; distinct result
positions may [alias](../../glossary.md#aliasing-and-arena) one exact representation. Construction performs
no [callback](../../glossary.md#term-callback), binding, execution, allocation, resource action, or ownership transfer. No
`PreparedUnit` is needed because list position is the occurrence and each current step supplies
its recipe and plan association.

`PreparedPublication` cold-binds one exact already-created buffer representation by dense buffer
and representation positions. `BoundPublication.publish()` requires that selected copy to be
valid at the publication moment and changes only a one-shot local flag. It performs no [lookup](../../glossary.md#term-lookup),
fallback, transfer, conversion, or backend work. A complete ordered set of published occurrences
constructs `RunResult`, which privately preserves result aliases and leases cleanup of the entire
`RunState`. Empty results are valid; partial publication transfers no cleanup responsibility.
While that exact Runtime lease is open, `RunResult.publicationRepresentation(resultIndex)` returns
the borrowed exact `BufferRepresentation` selected for one dense occurrence. Repeated access and
aliased occurrences preserve exact reference identity. The caller must not close, transfer, or
retain the reference beyond result closure and must externally prevent access from racing result
or state activity because Runtime `RunResult` is not thread-safe. The method performs no copy,
wrapper construction, cast, validity query, transfer, or physical access; closed-state failure
precedes index [validation](../../glossary.md#term-validation).

This method is an inward Runtime service-provider interface ([SPI](../../glossary.md#term-service-provider-interface-spi)), not an ordinary result-value
[API](../../glossary.md#term-application-programming-interface-api). A `BufferRepresentation` is only a nominal backend-owned [lifecycle](../../glossary.md#lifecycle) role: it is not [host
storage](../../glossary.md#host-storage), a [Tensor](../../glossary.md#tensor) value, or guaranteed to be [host](../../glossary.md#term-host)-accessible. The ordinary Engine `RunResult`
continues to expose [metadata](../../glossary.md#term-metadata) only, and `AdvancedRunResult` continues to expose only result count
and lifecycle. Neither Engine result exposes the inward representation.

Completion and lifetime are separate: a successful [synchronous](../../glossary.md#term-synchronous) run has finished its work before
returning, while the returned Engine/Runtime result may keep completed output resources and
publication state alive until close. Engine serializes materialization against outward result
close; Runtime `RunResult` and `RunState` remain non-thread-safe.

One `PreparedExecution` now supplies the current reusable Runtime root and unique persistent-
[resource owner](../../glossary.md#ownership-borrowing-and-lease). It retains the exact memory-plan and same-plan schedule references. The
two-[argument](../../glossary.md#term-argument) constructor remains resource-free; the three-argument constructor privately
snapshots non-null resources that are unique by exact [identity](../../glossary.md#term-identity-meanings). The final class intentionally
uses `Object` identity rather than the former record's [structural equality](../../glossary.md#term-structural-equality) and exposes no
resource aggregate or lookup.

Close is explicit, [idempotent](../../glossary.md#term-idempotent), and non-waiting. It rejects new run leases immediately and closes
resources in reverse acquisition order when no admitted run remains; the last lease performs
deferred cleanup after a close/run [race](../../glossary.md#term-race). Cleanup attempts all resources, runs outside lifecycle
[synchronization](../../glossary.md#term-synchronization), and preserves the first unchecked failure with later distinct failures
suppressed. Immutable recipe sharing remains safe, while each invocation still uses a distinct
[mutable](../../glossary.md#term-mutable) `RunState`.

`PreparedExecutionRunner.run(execution, callerInputs)` first acquires the [opaque](../../glossary.md#term-opaque) execution lease,
then creates one isolated state, cold-binds
every non-creation occurrence before the first action, traverses direct bound references in
schedule order, and either returns the whole-state result lease or closes the state after failure.
The runner is stateless and may serve concurrent calls; each call remains synchronous and owns
distinct mutable state. Existing run failure remains primary over lease-cleanup failure. A
cleanup failure after successful execution closes the new result and prevents its return.

Successful public or Runtime run return is a synchronous completion barrier: all actions and native
work for that invocation have completed, or no result is returned. `RunResult` is therefore a
completed-state resource/publication lease, not a future, event, fence, or in-flight completion
handle. Concurrent callers receive isolated mutable state but gain no promise of [device](../../glossary.md#term-device) overlap,
cross-run order, fairness, queue [topology](../../glossary.md#term-topology), or [throughput](../../glossary.md#term-throughput). The current APIs expose no [asynchronous](../../glossary.md#term-asynchronous)
run, [cancellation](../../glossary.md#term-cancellation), timeout, polling, callback, or completion-stage surface.

These contracts contain no concrete backend implementation and do not provide physical access,
host/Tensor value access, or Engine behavior. Creator callbacks leave physical allocation mechanics to
the implementing backend, and binding owns no auxiliary closeable resource or ownership change.
See the [Runtime API](index.md) for creation, validity, scheduling, selection, lifecycle,
failure, and thread-safety details.


---

[Previous topic](../engine/cpu-integration.md) · [Next topic](prepare.md)
## Keep the outer and inner callers distinct

At the application boundary, the caller lends stored Tensor inputs to Engine and receives an
Engine result. Inside Engine, the caller is Engine itself: it lends backend-specific
representations to `PreparedExecutionRunner`. Runtime calls backend binding/execution hooks and
returns its own result lease to Engine. That inward lease is not a public Tensor value, and the
application never owns the representation obtained through Runtime's SPI. To obtain independent
bytes, the application asks its **Engine** result to materialize a publication while it remains
open. Follow [ordinary binding](engine-boundary.md) for that outer lifetime and
[publication](publication.md) for the inner lease.

