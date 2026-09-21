# ADR 0013: Prepared-execution persistent-resource lifecycle

## Status

Accepted — 2026-09-20

## Context

Backend finalization can produce immutable, reusable native state whose lifetime spans runs. The
first concrete consumer is a shape-specialized `MPSGraphExecutable`. It is compiled once during
Metal preparation, reused by synchronous runs, and released explicitly through Objective-C
ownership. The existing Runtime root carries only a memory plan and schedule, while the current
Prepare contract forbids a finalizer from returning a closeable prepared resource.

This resource is not per-run scratch. It must survive every isolated `RunState`, but it must not
outlive the prepared execution that makes it reachable. Runtime cannot know Metal, CUDA, native
handle types, or physical release mechanics.

## Decision drivers

- one deterministic owner for persistent prepared resources;
- immutable reusable recipe semantics with an explicit finite resource lifetime;
- safe concurrent synchronous runs and a close protocol without indefinite waiting;
- transactional preparation and complete partial-failure rollback;
- backend-owned physical mechanics with no concrete-backend dependency in Runtime; and
- no raw `Object`, unchecked generic access, registry, string dispatch, concrete-type switch,
  cleaner, or garbage-collection correctness dependency.

## Options considered

### PreparedExecution-owned resource aggregate

Runtime owns a narrow nominal prepared-resource contract and retains an identity-unique private
aggregate in `PreparedExecution`. Backends implement physical close. Prepare transfers acquired
resources transactionally into the completed execution.

### Direct ownership by PreparedExecutable

This puts a native resource near its user, but schedule occurrences may repeat one executable or
several executables may share one compiled resource. Closing the occurrence graph would therefore
duplicate ownership, require reference counting in backend recipes, or make cleanup depend on
schedule topology.

### Backend integration or global ownership

An integration owner can keep native state alive, but then a prepared execution cannot release it
independently. Resource lifetime expands to an engine, adapter, cache, or process, and hidden
global ownership risks service location and leaks between otherwise independent preparations.

### Per-run recompilation or workspace ownership

Recompiling an MPS graph for every run violates staged cold preparation and adds avoidable hot-path
work. A workspace belongs to one `RunState`; using it for a reusable executable either recompiles
or incorrectly shares run-owned mutable state.

### Cleaner or garbage-collection cleanup

A cleaner can be a last-resort diagnostic safeguard, but garbage collection is nondeterministic,
does not define failure reporting, and cannot be the correctness mechanism for scarce native
resources.

## Decision

Synaptik selects one central private prepared-resource aggregate owned by `PreparedExecution`.
Runtime defines only a narrow nominal `PreparedResource` lifecycle contract. A concrete backend
implements that contract and performs the physical release of its native state. The aggregate is
not a public resource lookup API and exposes no backend payload.

`PreparedExecution` becomes a final lifecycle-bearing class implementing `AutoCloseable`. Its
memory plan, schedule, executable recipes, and retained resource identities remain immutable. It
is no longer a record: records cannot keep private lifecycle state outside their components, and
making a resource scope a record component would expose ownership and include it in structural
equality. The class retains the existing component-style accessors and a resource-free
constructor for current callers. It uses object identity rather than value equality; two owners
of distinct native resources must not compare equal because their logical recipe components are
equal.

The private aggregate snapshots resources in acquisition order, rejects a repeated exact identity,
and owns each accepted resource exactly once. Repeated executable or schedule occurrences do not
add resource entries. Physical cleanup is deterministic reverse acquisition order and attempt-all.
The first unchecked exception or error is primary; later distinct failures are suppressed in
encounter order, while the same exact primary throwable is not self-suppressed.

Each synchronous runner call acquires a prepared-execution lease before creating a `RunState` or
performing backend work. The close transition and lease admission are serialized:

```text
OPEN --close begins--> CLOSING (new leases rejected)
  |                         |
  +-- active lease ---------+-- last lease releases resources --> CLOSED
                            +-- no active lease: close releases resources --> CLOSED
```

Close is idempotent and does not wait. A run admitted before close may finish. If no run is active,
the closing caller performs cleanup and receives any failure. Otherwise close returns after
preventing new runs, and the last lease releaser performs cleanup and receives any failure. A run
failure remains primary when lease cleanup also fails. If deferred cleanup fails after otherwise
successful execution, the run fails and its newly created `RunResult` is closed rather than
returned.

Preparation uses a three-stage ownership transaction. A backend finalizer owns and reverses any
resource acquired before it successfully returns. After return, shared Prepare owns the returned
unique resources and reverses them if any later finalizer, association check, schedule assembly,
schedule validation, or aggregate construction fails. Successful `PreparedExecution`
construction is the sole ownership-transfer point. Cleanup at each stage uses the same reverse,
attempt-all, primary/suppression, and self-suppression rules.

A finalizer returns resources in physical acquisition order. Prepare concatenates those lists in
partition-finalization order and rejects a repeated exact identity across or within results. It
tracks one ownership occurrence per identity, so even an invalid duplicate handoff is physically
closed once. Executable sharing and repeated schedule occurrences never contribute ownership.
When preparation already has a primary failure, rollback failures are suppressed on that primary
in reverse-cleanup encounter order; a repeated reference to the exact primary throwable is skipped.

## Rationale

The selected owner matches reachability: one prepared execution makes the persistent state usable
and is the smallest object that can close it exactly once regardless of schedule topology. A
nominal Runtime contract gives shared orchestration enough information to close a resource without
learning its type or mechanics. Deferred release makes close bounded and avoids waiting on code
that may itself need to finish through the same prepared resource.

## Consequences

### Positive

- Native compiled executables have a deterministic lifetime independent of backend-global state.
- Logical recipes remain immutable and reusable while lifecycle state stays explicit.
- Concurrent runs are safe, and close cannot deadlock by waiting for them.
- Prepare failures cannot leak resources acquired by an earlier successful finalizer.
- CPU preparations remain resource-free and use the compatible constructor.

### Negative and risks

- Runtime gains synchronized cold lifecycle bookkeeping and one lease action per run.
- `PreparedExecution` changes from record value semantics to class identity semantics.
- The runner needs a narrow public Runtime integration seam for a lease because it currently lives
  in a sibling package; that seam must not expose resources or permit backend lookup.
- A close caller may return before physical cleanup when an admitted run is active; any cleanup
  failure is then observed by the last lease-releasing run rather than that close call.

### Migration, testing, and follow-up

Runtime task 0016 implements the nominal resource, execution lifecycle, runner lease, and focused
architecture-source inventory update. Prepare 0006 then adds finalizer resource handoff and
transactional rollback. Engine 0010 must give ordinary and advanced prepared handles deterministic
ownership, close temporary/tuning preparations, and close retained preparations before backend
integration shutdown. Metal 0002 remains blocked until all three shared lifecycle tasks complete.

The decision changes no module dependency direction. The Runtime architecture test must classify
the new source and continue enforcing concrete-backend independence; no dependency-rule test
changes are required.

## Related documentation

- [Architecture contract](../../../ARCHITECTURE.md)
- [Lifecycle](../../architecture/lifecycle.md)
- [Runtime, Prepare, and Backend Boundary](../../architecture/runtime-prepare-backend-boundary.md)
- [ADR 0010: Staged backend preparation](0010-staged-backend-preparation.md)
- [ADR 0011: Per-run Runtime resource ownership and cold binding](0011-per-run-runtime-resource-ownership.md)
- [Runtime master plan](../../planning/modules/runtime/master-plan.md)
- [Metal master plan](../../planning/backends/metal/master-plan.md)
