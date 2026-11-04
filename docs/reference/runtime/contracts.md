# Runtime public contract relationships

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

The public `modules:runtime` surface now contains five focused packages:

- `runtime.memory` defines `BufferSlot`, `WorkspaceSlot`, and immutable ordered
  `PreparedMemoryPlan` byte geometry;
- `runtime.resource` defines the nominal backend-implemented `BufferRepresentation` and
  `WorkspaceRepresentation` cleanup roles, the persistent `PreparedResource` cleanup role, plus
  immutable `PreparedRepresentationPlan` origins, caller-input occurrences, ordinary and
  initialized buffer origins, and typed buffer/workspace creators;
- `runtime.run` defines borrowed/run-owned buffer bindings, the array-backed one-run `RunState`,
  dense-coordinate `PreparedPublication`, per-run `BoundPublication`, the whole-state
  `RunResult` lease, and stateless `PreparedExecutionRunner`; and
- `runtime.execution` defines the immutable-recipe, explicitly closeable `PreparedExecution`
  owner, reusable
  `PreparedExecutable` recipes, their dense buffer/representation and workspace selections,
  aligned buffer-access declarations, and
  per-run `BoundInvocation` objects, plus reusable `PreparedBufferTransfer` recipes and per-run
  `BoundBufferTransfer` actions; and
- `runtime.schedule` defines immutable `PreparedSchedule` recipes, their sealed plan-associated
  `Step` family, the optional first-only `RepresentationCreationStep`, `ExecutionStep`,
  `BufferTransferStep`, and the dense final `PublicationStep` suffix.

`PreparedRepresentationPlan` snapshots dense buffer origins and workspace creators against one
exact memory plan. Concrete backends implement immutable thread-safe creator callbacks that return
fresh physical results for each run. Package-private cold setup validates every borrowed caller
input before callbacks, creates buffers then workspaces, and rolls back successful created results
in reverse order if setup fails. The public runner is the narrow composition seam for that setup;
it is not an Engine or value facade.

Every representation bound into an open `RunState` is structurally resident until closure. Each
buffer copy has one independent explicit validity bit: borrowed inputs start valid, ordinary
newly created run-owned buffers start invalid, and initialized run-owned buffers start valid.
The initialized creator, not Runtime, materializes the logical value. Workspaces are run-owned
scratch without logical validity. Query and mutation are constant-time dense-array operations and
perform no physical copy, backend call, ownership change, or implicit coherence.

`PreparedExecutable.bind` requires the exact same `PreparedMemoryPlan` reference as the open
`RunState`. It resolves selections in supplied order and delegates explicit checked
representation compatibility to the concrete backend. The backend then creates a
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
unchanged. Materialization is this same explicit transfer to an equivalent already-created
destination, not another operation, allocation path, or route search.

One schedule retains an exact `PreparedMemoryPlan` and an immutable ordered snapshot of creation,
executable, buffer-transfer, and publication occurrences. Every step must report that same plan by reference
identity. The
creation occurrence is optional for compatibility but, when present, is sole and first. Empty and
executable-only or transfer-only schedules and repeated executable or transfer occurrences remain
valid. Publication occurrences, when present, form a dense `0..N-1` final suffix; distinct result
positions may alias one exact representation. Construction performs
no callback, binding, execution, allocation, resource action, or ownership transfer. No
`PreparedUnit` is needed because list position is the occurrence and each current step supplies
its recipe and plan association.

`PreparedPublication` cold-binds one exact already-created buffer representation by dense buffer
and representation positions. `BoundPublication.publish()` requires that selected copy to be
valid at the publication moment and changes only a one-shot local flag. It performs no lookup,
fallback, transfer, conversion, or backend work. A complete ordered set of published occurrences
constructs `RunResult`, which privately preserves result aliases and leases cleanup of the entire
`RunState`. Empty results are valid; partial publication transfers no cleanup responsibility.
While that exact Runtime lease is open, `RunResult.publicationRepresentation(resultIndex)` returns
the borrowed exact `BufferRepresentation` selected for one dense occurrence. Repeated access and
aliased occurrences preserve exact reference identity. The caller must not close, transfer, or
retain the reference beyond result closure and must externally prevent access from racing result
or state activity because Runtime `RunResult` is not thread-safe. The method performs no copy,
wrapper construction, cast, validity query, transfer, or physical access; closed-state failure
precedes index validation.

This method is an inward Runtime service-provider interface (SPI), not an ordinary result-value
API. A `BufferRepresentation` is only a nominal backend-owned lifecycle role: it is not host
storage, a Tensor value, or guaranteed to be host-accessible. The ordinary Engine `RunResult`
continues to expose metadata only, and `AdvancedRunResult` continues to expose only result count
and lifecycle. Neither Engine result exposes the inward representation.

Completion and lifetime are separate: a successful synchronous run has finished its work before
returning, while the returned Engine/Runtime result may keep completed output resources and
publication state alive until close. Engine serializes materialization against outward result
close; Runtime `RunResult` and `RunState` remain non-thread-safe.

One `PreparedExecution` now supplies the current reusable Runtime root and unique persistent-
resource owner. It retains the exact memory-plan and same-plan schedule references. The
two-argument constructor remains resource-free; the three-argument constructor privately
snapshots non-null resources that are unique by exact identity. The final class intentionally
uses `Object` identity rather than the former record's structural equality and exposes no
resource aggregate or lookup.

Close is explicit, idempotent, and non-waiting. It rejects new run leases immediately and closes
resources in reverse acquisition order when no admitted run remains; the last lease performs
deferred cleanup after a close/run race. Cleanup attempts all resources, runs outside lifecycle
synchronization, and preserves the first unchecked failure with later distinct failures
suppressed. Immutable recipe sharing remains safe, while each invocation still uses a distinct
mutable `RunState`.

`PreparedExecutionRunner.run(execution, callerInputs)` first acquires the opaque execution lease,
then creates one isolated state, cold-binds
every non-creation occurrence before the first action, traverses direct bound references in
schedule order, and either returns the whole-state result lease or closes the state after failure.
The runner is stateless and may serve concurrent calls; each call remains synchronous and owns
distinct mutable state. Existing run failure remains primary over lease-cleanup failure. A
cleanup failure after successful execution closes the new result and prevents its return.

Successful public or Runtime run return is a synchronous completion barrier: all actions and native
work for that invocation have completed, or no result is returned. `RunResult` is therefore a
completed-state resource/publication lease, not a future, event, fence, or in-flight completion
handle. Concurrent callers receive isolated mutable state but gain no promise of device overlap,
cross-run order, fairness, queue topology, or throughput. The current APIs expose no asynchronous
run, cancellation, timeout, polling, callback, or completion-stage surface.

These contracts contain no concrete backend implementation and do not provide physical access,
host/Tensor value access, or Engine behavior. Creator callbacks leave physical allocation mechanics to
the implementing backend, and binding owns no auxiliary closeable resource or ownership change.
See the [Runtime API](index.md) for creation, validity, scheduling, selection, lifecycle,
failure, and thread-safety details.


---

[Previous topic](../engine/cpu-integration.md) · [Next topic](prepare.md)
