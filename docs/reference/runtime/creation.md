# Representation creation, run-state closure and failures

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

A creation plan says which representations are borrowed and which a backend must create for a
run. A **representation** is a backend's concrete form of a logical buffer or scratch space.
**Rollback** here means closing successfully created owned resources if setup cannot complete,
not rewinding arbitrary backend effects or closing borrowed inputs.

<a id="current-prepared-representation-creation"></a>
## Current prepared representation creation

`PreparedRepresentationPlan` retains the exact `PreparedMemoryPlan`, immutable snapshots of both
buffer-list levels, and an immutable workspace-creator list. Each buffer preparation is a
zero-component `CallerInput`, a `CreatedBuffer(BufferCreator)` whose new representation starts
invalid, or an `InitializedBuffer(BufferCreator)` whose new representation starts valid. The two
created variants use the same creator ownership, encounter order, identity, rollback, and cleanup
rules. Runtime does not know which graph value or scalar an initialized creator materializes.
Dense caller-input encounter order is buffer position first and representation position second.
Each workspace position has one `WorkspaceCreator`. Callback implementations must be immutable
and thread-safe, and each successful call must return a fresh non-null representation object for
that run. The object may own a read-only child lease into immutable prepared physical storage; it
must not expose that storage as a writable destination or permit new leases after prepared-owner
close.

The package-private `RunStateCreation` operation validates the complete caller count, non-null
elements, and caller identity uniqueness before invoking any callback. It then creates buffers in
dense buffer/representation order and workspaces in workspace order. A successful result becomes
run-owned only when complete `RunState` construction succeeds. If a creator, result validation,
or state construction reports a `RuntimeException` or `Error`, setup closes every successfully
created result once in reverse creation order, preserves the original failure unchanged, and adds
cleanup failures as suppressed exceptions in cleanup encounter order. It never closes borrowed
inputs or closes a duplicate callback result as a second owned object.

Plan construction invokes no callback. A null or duplicate callback result is rejected during
cold setup. No creation callback performs transfer, materialization, execution, publication, or
validity mutation as part of the shared contract.

<a id="run-state-lifecycle-and-failures"></a>
## Run-state lifecycle and failures

`RunState.close()` marks the state closed before physical cleanup. It closes workspaces from last
position to first, then run-owned buffer representations from last buffer position to first and
last representation to first. Borrowed buffers are skipped. Every owned representation is
attempted once. The first `RuntimeException` or `Error` is rethrown after all attempts. Later
failure objects distinct from that primary object are attached in cleanup encounter order as
suppressed exceptions. If a distinct resource throws the same exact primary object, Runtime does
not attach that object to itself because Java forbids self-suppression; the repeated occurrence
does not replace the primary failure, stop traversal, or prevent a still-later distinct failure
from being suppressed. Repeated closure performs no cleanup and does not rethrow an earlier
failure.

Representation and validity access or mutation after closure fails first with
`IllegalStateException("run state is closed")`. The retained plan, slot counts, per-buffer
representation counts, and `isClosed()` remain inspectable. `RunState` is not thread-safe;
callers must not race access, validity mutation, and closure on one instance. Concurrent runs may
share the immutable plan but have independent validity arrays and require distinct run-owned
representations. Borrowed representations may be shared only when the caller guarantees their
lifetime, safe access, and external synchronization.

Construction validates top-level nulls, plan-sized counts, each ordered buffer position, and then
each ordered workspace position. Null failures identify the argument or indexed position. Count,
empty-list, and duplicate-identity failures are `IllegalArgumentException`; every duplicate exact
identity reports `representation is already bound to this run`. Invalid access indices use
`IndexOutOfBoundsException` and identify `bufferIndex`, `representationIndex`, or
`workspaceIndex` with the rejected value.

<a id="failures"></a>
## Failures

Entry construction checks slot nullness, then byte size, then byte alignment. A null slot fails
with `NullPointerException("slot")`; a negative size fails with
`IllegalArgumentException("byteSize must be non-negative")`; and an invalid alignment fails with
`IllegalArgumentException("byteAlignment must be a positive power of two")`.

Plan construction checks both top-level lists before scanning entries. It then validates and
snapshots buffers before workspaces. A null entry reports its supplied zero-based list position.
The first later duplicate reports that position and the diagnostic slot text. These failures
reject ambiguous geometry; they do not perform assignment, deduplication, or recovery.


---

[Previous topic](representations.md) · [Next topic](executables.md)
