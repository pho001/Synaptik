# [ADR](../../glossary.md#term-architecture-decision-record-adr) 0013: Prepared-execution persistent-resource [lifecycle](../../glossary.md#lifecycle)


## Editorial reading context

A backend finalizer can acquire a reusable native executable. Before successful return it must clean up its own failed acquisition; afterward shared Prepare owns rollback until the complete PreparedExecution takes ownership. Runtime borrows that prepared state under a run lease. Closing a completed result releases a different, run-owned lifetime, not the reusable recipe.

Use the [prepared leases and result lifetime](../../architecture/lifecycle.md) for focused guidance. The original status, decision
and evidence below remain historical records; this reading aid does not amend them.

## Status

Accepted — 2026-09-20

## Context

[Backend](../../glossary.md#backend) finalization can produce [immutable](../../glossary.md#term-immutable), reusable native state whose [lifetime](../../glossary.md#term-lifetime) spans runs. The
first concrete consumer is a [shape](../../glossary.md#shape)-specialized `MPSGraphExecutable`. It is compiled once during
[Metal](../../glossary.md#term-metal) preparation, reused by [synchronous](../../glossary.md#term-synchronous) runs, and released explicitly through [Objective-C](../../glossary.md#native-packaging-and-platform-vocabulary)
[ownership](../../glossary.md#term-ownership-meanings). The existing [Runtime](../../glossary.md#term-runtime) root carries only a memory plan and schedule, while the current
[Prepare](../../glossary.md#prepare) contract forbids a finalizer from returning a closeable [prepared resource](../../glossary.md#prepared-resource--preparedresource).

This resource is not per-[run](../../glossary.md#term-run) scratch. It must survive every isolated [`RunState`](../../glossary.md#run-state--runstate), but it must not
outlive the [prepared execution](../../glossary.md#prepared-execution--preparedexecution) that makes it reachable. Runtime cannot know Metal, [CUDA](../../glossary.md#term-cuda), native
handle types, or physical [release](../../glossary.md#term-release) mechanics.

## Decision drivers

- one deterministic [owner](../../glossary.md#term-owner) for persistent prepared resources;
- [immutable](../../glossary.md#term-immutable) reusable recipe semantics with an explicit finite [resource lifetime](../../glossary.md#ownership-borrowing-and-lease);
- safe concurrent [synchronous](../../glossary.md#term-synchronous) runs and a close protocol without indefinite waiting;
- transactional preparation and complete partial-failure [rollback](../../glossary.md#term-rollback);
- [backend](../../glossary.md#backend)-owned physical mechanics with no concrete-backend [dependency](../../glossary.md#term-dependency) in [Runtime](../../glossary.md#term-runtime); and
- no raw `Object`, unchecked generic access, registry, string dispatch, concrete-type switch,
  cleaner, or garbage-collection correctness dependency.<span data-glossary-navigation="true"> ([registry glossary](../../glossary.md#term-registry))</span>

## Options considered

### [PreparedExecution](../../glossary.md#prepared-execution--preparedexecution)-owned resource aggregate

[Runtime](../../glossary.md#term-runtime) owns a narrow [nominal](../../glossary.md#term-nominal) prepared-resource contract and retains an [identity](../../glossary.md#term-identity-meanings)-unique private
aggregate in `PreparedExecution`. [Backends](../../glossary.md#backend) implement physical close. [Prepare](../../glossary.md#prepare) transfers acquired
resources transactionally into the completed execution.

### Direct [ownership](../../glossary.md#term-ownership-meanings) by [PreparedExecutable](../../glossary.md#prepared-executable--preparedexecutable)

This puts a native resource near its user, but schedule occurrences may repeat one executable or
several executables may share one compiled resource. Closing the [occurrence](../../glossary.md#term-occurrence) graph would therefore
duplicate ownership, require reference counting in [backend](../../glossary.md#backend) recipes, or make cleanup depend on
schedule [topology](../../glossary.md#term-topology).

### [Backend](../../glossary.md#backend) integration or global [ownership](../../glossary.md#term-ownership-meanings)

An integration [owner](../../glossary.md#term-owner) can keep native state alive, but then a [prepared execution](../../glossary.md#prepared-execution--preparedexecution) cannot [release](../../glossary.md#term-release) it
independently. [Resource lifetime](../../glossary.md#ownership-borrowing-and-lease) expands to an [engine](../../glossary.md#term-engine), adapter, [cache](../../glossary.md#term-cache), or process, and hidden
global ownership risks service location and leaks between otherwise independent preparations.

### Per-[run](../../glossary.md#term-run) recompilation or workspace [ownership](../../glossary.md#term-ownership-meanings)

Recompiling an [MPS](../../glossary.md#term-mps) [graph](../../glossary.md#graph) for every run violates staged cold preparation and adds avoidable hot-path
work. A workspace belongs to one [`RunState`](../../glossary.md#run-state--runstate); using it for a reusable executable either recompiles
or incorrectly shares run-owned [mutable](../../glossary.md#term-mutable) state.

### Cleaner or garbage-collection cleanup

A cleaner can be a last-resort diagnostic safeguard, but [garbage collection](../../glossary.md#term-garbage-collection-gc) is nondeterministic,
does not define failure reporting, and cannot be the correctness mechanism for scarce native
resources.

## Decision

Synaptik selects one central private prepared-resource aggregate owned by [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution).
[Runtime](../../glossary.md#term-runtime) defines only a narrow [nominal](../../glossary.md#term-nominal) [`PreparedResource`](../../glossary.md#prepared-resource--preparedresource) [lifecycle](../../glossary.md#lifecycle) contract. A [concrete backend](../../glossary.md#concrete-backend)
implements that contract and performs the physical [release](../../glossary.md#term-release) of its native state. The aggregate is
not a public resource [lookup](../../glossary.md#term-lookup) [API](../../glossary.md#term-application-programming-interface-api) and exposes no [backend](../../glossary.md#backend) payload.

One prepared [owner](../../glossary.md#term-owner) may publish a backend-private child [lease](../../glossary.md#term-lease) to a fresh [run](../../glossary.md#term-run)-owned representation
only when the underlying physical state is [immutable](../../glossary.md#term-immutable) and the representation is read-only. In that
case `PreparedResource.close()` rejects new child acquisition and relinquishes the prepared-owner
reference; an already-issued child may defer the exact-once physical release until its [run
state](../../glossary.md#run-state--runstate)/result closes. The child is not another prepared-resource entry and cannot expose a Runtime
lookup or backend payload.

`PreparedExecution` becomes a final lifecycle-bearing class implementing `AutoCloseable`. Its
memory plan, schedule, executable recipes, and retained resource [identities](../../glossary.md#term-identity-meanings) remain immutable. It
is no longer a record: records cannot keep private lifecycle state outside their components, and
making a resource [scope](../../glossary.md#term-scope) a record component would expose [ownership](../../glossary.md#term-ownership-meanings) and include it in [structural
equality](../../glossary.md#term-structural-equality). The class retains the existing component-style accessors and a resource-free
constructor for current callers. It uses object [identity](../../glossary.md#term-reference-identity) rather than value equality; two owners
of distinct native resources must not compare equal because their logical recipe components are
equal.

The private aggregate snapshots resources in acquisition order, rejects a repeated exact identity,
and owns each accepted resource exactly once. Repeated executable or schedule occurrences do not
add resource entries. Physical cleanup is deterministic reverse acquisition order and attempt-all.
The first unchecked exception or error is primary; later distinct failures are suppressed in
encounter order, while the same exact primary throwable is not self-suppressed.

Each [synchronous](../../glossary.md#term-synchronous) runner call acquires a prepared-execution lease before creating a [`RunState`](../../glossary.md#run-state--runstate) or
performing backend work. The close transition and lease [admission](../../glossary.md#term-admission) are serialized:

```text
OPEN --close begins--> CLOSING (new leases rejected)
  |                         |
  +-- active lease ---------+-- last lease releases resources --> CLOSED
                            +-- no active lease: close releases resources --> CLOSED
```

Close is [idempotent](../../glossary.md#term-idempotent) and does not wait. A run admitted before close may finish. If no run is active,
the closing caller performs cleanup and receives any failure. Otherwise close returns after
preventing new runs, and the last lease releaser performs cleanup and receives any failure. A run
failure remains primary when lease cleanup also fails. If deferred cleanup fails after otherwise
successful execution, the run fails and its newly created [`RunResult`](../../glossary.md#run-result--runresult) is closed rather than
returned.

Preparation uses a three-stage ownership [transaction](../../glossary.md#term-transaction). A backend finalizer owns and reverses any
resource acquired before it successfully returns. After return, shared [Prepare](../../glossary.md#prepare) owns the returned
unique resources and reverses them if any later finalizer, association check, schedule assembly,
schedule [validation](../../glossary.md#term-validation), or aggregate construction fails. Successful `PreparedExecution`
construction is the sole ownership-transfer point. Cleanup at each stage uses the same reverse,
attempt-all, primary/suppression, and self-suppression rules.

A finalizer returns resources in physical acquisition order. Prepare concatenates those lists in
[partition](../../glossary.md#partition)-finalization order and rejects a repeated exact identity across or within results. It
tracks one ownership [occurrence](../../glossary.md#term-occurrence) per identity, so even an invalid duplicate [handoff](../../glossary.md#term-handoff) is physically
closed once. Executable sharing and repeated schedule occurrences never contribute ownership.
When preparation already has a primary failure, [rollback](../../glossary.md#term-rollback) failures are suppressed on that primary
in reverse-cleanup encounter order; a repeated reference to the exact primary throwable is skipped.

## Rationale

The selected [owner](../../glossary.md#term-owner) matches reachability: one [prepared execution](../../glossary.md#prepared-execution--preparedexecution) makes the persistent state usable
and is the smallest object that can close it exactly once regardless of schedule [topology](../../glossary.md#term-topology). A
[nominal](../../glossary.md#term-nominal) [Runtime](../../glossary.md#term-runtime) contract gives shared orchestration enough information to close a resource without
learning its type or mechanics. Deferred [release](../../glossary.md#term-release) makes close bounded and avoids waiting on code
that may itself need to finish through the same [prepared resource](../../glossary.md#prepared-resource--preparedresource).

This bounded child-[lease](../../glossary.md#term-lease) rule supports prepared [immutable](../../glossary.md#term-immutable) [constants](../../glossary.md#term-constant) without extending the
[synchronous](../../glossary.md#term-synchronous) execution lease through [`RunResult`](../../glossary.md#run-result--runresult). It does not authorize a pool: [mutable](../../glossary.md#term-mutable) outputs
outlive the synchronous call and mutable address workspaces are rewritten during cold [binding](../../glossary.md#term-binding).
Pooling them requires a separately reviewed async/result-[lifetime](../../glossary.md#term-lifetime) contract for exclusive return,
reset/validity, bounded capacity/eviction, context close, and cleanup failure.

## Consequences

### Positive

- Native compiled executables have a deterministic [lifetime](../../glossary.md#term-lifetime) independent of [backend](../../glossary.md#backend)-global state.
- Logical recipes remain [immutable](../../glossary.md#term-immutable) and reusable while [lifecycle](../../glossary.md#lifecycle) state stays explicit.
- Concurrent runs are safe, and close cannot [deadlock](../../glossary.md#term-deadlock) by waiting for them.
- [Prepare](../../glossary.md#prepare) failures cannot leak resources acquired by an earlier successful finalizer.
- [CPU](../../glossary.md#term-cpu) preparations remain resource-free and use the compatible constructor.

### Negative and risks

- [Runtime](../../glossary.md#term-runtime) gains synchronized cold [lifecycle](../../glossary.md#lifecycle) bookkeeping and one [lease](../../glossary.md#term-lease) action per [run](../../glossary.md#term-run).
- [`PreparedExecution`](../../glossary.md#prepared-execution--preparedexecution) changes from record value semantics to class [identity](../../glossary.md#term-identity-meanings) semantics.
- The runner needs a narrow public Runtime integration seam for a lease because it currently lives
  in a sibling package; that seam must not expose resources or permit backend lookup.<span data-glossary-navigation="true"> ([backend glossary](../../glossary.md#backend); [lookup glossary](../../glossary.md#term-lookup))</span>
- A close caller may return before physical cleanup when an admitted run is active; any cleanup
  failure is then observed by the last lease-releasing run rather than that close call.
- A child-delayed physical release failure is observed by the last run-owned binding close rather
  than replayed through the already-closed prepared owner.<span data-glossary-navigation="true"> ([release glossary](../../glossary.md#term-release); [binding glossary](../../glossary.md#term-binding); [owner glossary](../../glossary.md#term-owner))</span>

### Migration, testing, and follow-up

[Runtime](../../glossary.md#term-runtime) task 0016 implements the [nominal](../../glossary.md#term-nominal) resource, execution [lifecycle](../../glossary.md#lifecycle), runner [lease](../../glossary.md#term-lease), and focused
architecture-source inventory update. [Prepare](../../glossary.md#prepare) 0006 then adds finalizer resource [handoff](../../glossary.md#term-handoff) and
transactional [rollback](../../glossary.md#term-rollback). [Engine](../../glossary.md#term-engine) 0010 must give ordinary and advanced prepared handles deterministic
[ownership](../../glossary.md#term-ownership-meanings), close temporary/tuning preparations, and close retained preparations before [backend](../../glossary.md#backend)
integration shutdown. [Metal](../../glossary.md#term-metal) 0002 remains blocked until all three shared lifecycle tasks complete.

The decision changes no [module](../../glossary.md#term-module) [dependency direction](../../glossary.md#java-platform-and-build-tools). The Runtime [architecture test](../../glossary.md#term-architecture-test) must classify
the new source and continue enforcing concrete-backend independence; no [dependency](../../glossary.md#term-dependency)-rule test
changes are required.

## Related documentation

- [Architecture contract](../../../ARCHITECTURE.md)<span data-glossary-navigation="true"> ([Architecture contract glossary](../../glossary.md#architecture-contract))</span>
- [Lifecycle](../../architecture/lifecycle.md)<span data-glossary-navigation="true"> ([Lifecycle glossary](../../glossary.md#lifecycle))</span>
- [Runtime, Prepare, and Backend Boundary](../../architecture/runtime-prepare-backend-boundary.md)<span data-glossary-navigation="true"> ([Runtime glossary](../../glossary.md#term-runtime); [Prepare glossary](../../glossary.md#prepare); [Backend glossary](../../glossary.md#backend))</span>
- [ADR 0010: Staged backend preparation](0010-staged-backend-preparation.md)<span data-glossary-navigation="true"> ([ADR glossary](../../glossary.md#term-architecture-decision-record-adr))</span>
- [ADR 0011: Per-run Runtime resource ownership and cold binding](0011-per-run-runtime-resource-ownership.md)<span data-glossary-navigation="true"> ([resource ownership glossary](../../glossary.md#term-resource-ownership))</span><span data-glossary-navigation="true"> ([run glossary](../../glossary.md#term-run); [ownership glossary](../../glossary.md#term-ownership); [binding glossary](../../glossary.md#term-binding))</span>
- [Runtime master plan](../../planning/modules/runtime/master-plan.md)
- [Metal master plan](../../planning/backends/metal/master-plan.md)<span data-glossary-navigation="true"> ([Metal glossary](../../glossary.md#term-metal))</span>
