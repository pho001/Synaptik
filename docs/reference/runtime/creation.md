# Representation creation, [run](../../glossary.md#term-run)-state closure and failures

[Reference goals](../index.md) · [Javadoc and exact signatures](../javadoc.md) · [Source entry point](../../../modules/runtime/src/main/java/io/github/pho001/synaptik/runtime/execution/PreparedExecution.java)

A creation plan says which representations are [borrowed](../../glossary.md#ownership-borrowing-and-lease) and which a [backend](../../glossary.md#backend) must create for a
run. A **representation** is a backend's concrete form of a logical [buffer](../../glossary.md#term-buffer) or scratch space.
**[Rollback](../../glossary.md#term-rollback)** here means closing successfully created owned resources if setup cannot complete,
not rewinding arbitrary backend effects or closing borrowed inputs.

## The runner is the caller of creators

Engine calls Runtime's runner; the runner's setup calls the concrete backend creators retained
in `PreparedRepresentationPlan`. The application does not have to create each internal output or
workspace itself. Caller-input representations are borrowed and remain the lender's cleanup
responsibility; successful creator results become run-owned only after complete state
construction. Setup failure closes the successfully created resources but not borrowed input
storage. The returned result later owns cleanup of that complete state. Read
[result leasing](publication.md) alongside [per-copy validity](representations.md), because an
allocated object is not necessarily initialized or publishable.

<a id="current-prepared-representation-creation"></a>
## Current prepared representation creation

[`PreparedRepresentationPlan`](../../glossary.md#prepared-representation-plan--preparedrepresentationplan) retains the exact [`PreparedMemoryPlan`](../../glossary.md#prepared-memory-plan--preparedmemoryplan), [immutable](../../glossary.md#term-immutable) snapshots of both
[buffer](../../glossary.md#term-buffer)-list levels, and an immutable workspace-creator list. Each buffer preparation is a
zero-component `CallerInput`, a `CreatedBuffer(BufferCreator)` whose new representation starts
invalid, or an `InitializedBuffer(BufferCreator)` whose new representation starts valid. The two
created variants use the same creator [ownership](../../glossary.md#term-ownership-meanings), encounter order, [identity](../../glossary.md#term-identity-meanings), [rollback](../../glossary.md#term-rollback), and cleanup
rules. [Runtime](../../glossary.md#term-runtime) does not know which [graph value](../../glossary.md#graph-value) or [scalar](../../glossary.md#term-scalar) an initialized creator materializes.
Dense caller-input encounter order is buffer position first and representation position second.
Each workspace position has one `WorkspaceCreator`. [Callback](../../glossary.md#term-callback) implementations must be immutable
and [thread-safe](../../glossary.md#term-thread-safe), and each successful call must return a fresh non-null representation object for
that [run](../../glossary.md#term-run). The object may own a read-only child [lease](../../glossary.md#term-lease) into immutable prepared physical storage; it
must not expose that storage as a writable destination or permit new leases after prepared-[owner](../../glossary.md#term-owner)
close.

The package-private `RunStateCreation` [operation](../../glossary.md#operation) validates the complete caller count, non-null
[elements](../../glossary.md#term-element), and caller identity uniqueness before invoking any callback. It then creates buffers in
dense buffer/representation order and workspaces in workspace order. A successful result becomes
run-owned only when complete [`RunState`](../../glossary.md#run-state--runstate) construction succeeds. If a creator, result [validation](../../glossary.md#term-validation),
or state construction reports a `RuntimeException` or `Error`, setup closes every successfully
created result once in reverse creation order, preserves the original failure unchanged, and adds
cleanup failures as suppressed exceptions in cleanup encounter order. It never closes [borrowed](../../glossary.md#ownership-borrowing-and-lease)
inputs or closes a duplicate callback result as a second owned object.

Plan construction invokes no callback. A null or duplicate callback result is rejected during
cold setup. No creation callback performs transfer, [materialization](../../glossary.md#materialization), execution, [publication](../../glossary.md#term-publication), or
validity mutation as part of the shared contract.

<a id="run-state-lifecycle-and-failures"></a>
## [Run](../../glossary.md#term-run)-state [lifecycle](../../glossary.md#lifecycle) and failures

`RunState.close()` marks the state closed before physical cleanup. It closes workspaces from last
position to first, then run-owned [buffer](../../glossary.md#term-buffer) representations from last buffer position to first and
last representation to first. [Borrowed](../../glossary.md#ownership-borrowing-and-lease) buffers are skipped. Every owned representation is
attempted once. The first `RuntimeException` or `Error` is rethrown after all attempts. Later
failure objects distinct from that primary object are attached in cleanup encounter order as
suppressed exceptions. If a distinct resource throws the same exact primary object, [Runtime](../../glossary.md#term-runtime) does
not attach that object to itself because Java forbids self-suppression; the repeated [occurrence](../../glossary.md#term-occurrence)
does not replace the primary failure, stop traversal, or prevent a still-later distinct failure
from being suppressed. Repeated closure performs no cleanup and does not rethrow an earlier
failure.

Representation and validity access or mutation after closure fails first with
`IllegalStateException("run state is closed")`. The retained plan, slot counts, per-buffer
representation counts, and `isClosed()` remain inspectable. [`RunState`](../../glossary.md#run-state--runstate) is not [thread-safe](../../glossary.md#term-thread-safe);
callers must not [race](../../glossary.md#term-race) access, validity mutation, and closure on one instance. Concurrent runs may
share the [immutable](../../glossary.md#term-immutable) plan but have independent validity arrays and require distinct run-owned
representations. Borrowed representations may be shared only when the caller guarantees their
[lifetime](../../glossary.md#term-lifetime), safe access, and external [synchronization](../../glossary.md#term-synchronization).

Construction validates top-level nulls, plan-sized counts, each ordered buffer position, and then
each ordered workspace position. Null failures identify the [argument](../../glossary.md#term-argument) or indexed position. Count,
empty-list, and duplicate-[identity](../../glossary.md#term-identity-meanings) failures are `IllegalArgumentException`; every duplicate exact
identity reports `representation is already bound to this run`. Invalid access indices use
`IndexOutOfBoundsException` and identify `bufferIndex`, `representationIndex`, or
`workspaceIndex` with the rejected value.

<a id="failures"></a>
## Failures

Entry construction checks slot nullness, then byte size, then [byte alignment](../../glossary.md#term-byte-alignment). A null slot fails
with `NullPointerException("slot")`; a negative size fails with
`IllegalArgumentException("byteSize must be non-negative")`; and an invalid [alignment](../../glossary.md#term-alignment) fails with
`IllegalArgumentException("byteAlignment must be a positive power of two")`.

Plan construction checks both top-level lists before scanning entries. It then validates and
snapshots [buffers](../../glossary.md#term-buffer) before workspaces. A null entry reports its supplied zero-based list position.
The first later duplicate reports that position and the diagnostic slot text. These failures
reject ambiguous geometry; they do not perform assignment, deduplication, or recovery.


---

[Previous topic](representations.md) · [Next topic](executables.md)
